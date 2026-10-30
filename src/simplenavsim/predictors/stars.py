"""The multi-agent STARS forecaster as the navigation policy's pedestrian predictor.

Every environment's visible pedestrians form one graph from their observed history (8 steps at
0.4 s, the STEPS rate the forecaster was trained at; ``SimConfig.history_steps / history_stride``),
built for all environments at once on the device (``stars_ma_batch.graph_batch_from_padded``). The
forecaster predicts every pedestrian jointly 4.8 s ahead; predictions are linearly resampled to
the planner's time grid (constant velocity extrapolation past 4.8 s). Pedestrians with fewer than
two observed positions fall back to constant velocity.

The CVAE decoder is evaluated at the prior mean (``mode="mean"``, a single most-likely joint
future) or averaged over ``samples`` prior draws (``mode="average"``). The robot is not part of the
graph: STEPS contains no robot, so the forecaster predicts pedestrians as if moving among
themselves, and the controller's costs handle the robot's influence. ``predict_every`` reuses a
forecast for several control steps to save compute.
"""
from __future__ import annotations

import torch

from baselines.stars_ma_batch import graph_batch_from_padded
from baselines.stars_ma_graph import to_scene
from baselines.train_stars_ma import load_forecaster
from simplenavsim.policies.base import ConstantVelocityPredictor, PedestrianPredictor

MIN_HISTORY_POINTS = 2


class StarsForecastPredictor(PedestrianPredictor):
    def __init__(self, checkpoint, *, device="cpu", mode: str = "mean", samples: int = 8, predict_every: int = 1,
                 amp: bool = True) -> None:
        self.device = torch.device(device)
        self.model, payload = load_forecaster(checkpoint, self.device)
        self.model.eval()
        self.knn, self.radius = payload["graph"]["knn"], payload["graph"]["radius"]
        self.dt_model = float(payload["window"]["dt"])
        self.steps_obs = int(payload["window"]["observation_steps"])
        self.mode, self.samples, self.predict_every, self.amp = mode, samples, max(1, predict_every), amp
        self.fallback = ConstantVelocityPredictor()
        self._cache, self._calls = None, 0

    def forecast(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        """Model-rate forecasts ``[B, P, 12, 2]`` and which pedestrians got one ``[B, P]``."""
        if obs.ped_history is None or obs.ped_history.shape[2] != self.steps_obs:
            raise ValueError(f"The STARS predictor needs {self.steps_obs} history steps at {self.dt_model} s "
                             "(SimConfig.history_steps / history_stride)")
        history, valid = obs.ped_history.to(self.device), obs.history_valid.to(self.device)
        valid = valid & obs.ped_visible.to(self.device)[..., None]
        node_mask = valid[..., -1] & (valid.sum(-1) >= MIN_HISTORY_POINTS)
        B, P = node_mask.shape
        out = history.new_zeros(B, P, self.model.prediction_steps, 2)
        if not bool(node_mask.any()):
            return out, node_mask
        # Pack each environment's forecast nodes to the front (graph_batch_from_padded expects that).
        order = torch.argsort((~node_mask).to(torch.int8), dim=1, stable=True)
        packed_mask = torch.gather(node_mask, 1, order)
        gather = lambda value: torch.gather(value, 1, order.view(B, P, *([1] * (value.dim() - 2))).expand_as(value))
        batch = graph_batch_from_padded(gather(history), gather(valid), packed_mask, dt=self.dt_model, knn=self.knn,
                                        radius=self.radius, prediction_steps=self.model.prediction_steps)
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                             enabled=self.amp and self.device.type == "cuda"):
            encoded = self.model.encode(batch)
            if self.model.latent_dim and self.mode == "mean":
                z = self.model.sample_latent(encoded["context"], 1, mean=True)
            elif self.model.latent_dim:
                z = self.model.sample_latent(encoded["context"], self.samples)
            else:
                z = None
            local = self.model.decode(batch, encoded, z).float().mean(dim=0)        # [Nt, 12, 2]
        scene = to_scene(local, batch.origin, batch.heading)
        packed = batch.to_padded(scene)                                               # [B, P, 12, 2] packed order
        out.scatter_(1, order[..., None, None].expand_as(packed), packed)
        return out, node_mask

    def predict(self, obs, steps: int, dt: float) -> torch.Tensor:
        self._calls += 1
        if self._cache is None or (self._calls - 1) % self.predict_every == 0 or self._cache[0].shape[:2] != obs.ped_pos.shape[:2]:
            self._cache = (*self.forecast(obs), obs.ped_pos.clone(), obs.time)
        forecast, have, position, time = self._cache
        forecast, have = forecast.to(obs.ped_pos.device), have.to(obs.ped_pos.device)
        shift = obs.time - time                                                       # forecast age (s)
        # Model grid: t = 0 (the position the forecast started from), dt_model, ..., 12 dt_model.
        grid = torch.cat([position[:, :, None], forecast], dim=2)                     # [B, P, 13, 2]
        query = shift + torch.arange(1, steps + 1, device=grid.device, dtype=grid.dtype) * dt
        index = query / self.dt_model
        low = index.floor().clamp(0, grid.shape[2] - 2).long()
        frac = (index - low.to(index.dtype)).clamp(min=0.0)
        a, b = grid[:, :, low], grid[:, :, low + 1]
        resampled = a + frac[None, None, :, None] * (b - a)                          # extrapolates past the end
        cv = self.fallback.predict(obs, steps, dt)
        return torch.where(have[..., None, None], resampled, cv)

    def reset(self) -> None:
        self._cache, self._calls = None, 0
