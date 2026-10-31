"""Episode rollouts with full trajectory recording, and the robot-absent counterfactual."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from simplenavsim.core.state import ScenarioBatch


@dataclass
class Rollout:
    """Recorded episode for a batch; time index 0 is the initial state."""

    robot_pose: torch.Tensor      # [B, K, 3]
    robot_v: torch.Tensor         # [B, K]
    robot_w: torch.Tensor         # [B, K]
    command: torch.Tensor         # [B, K, 2] command applied going into step k
    emergency: torch.Tensor       # [B, K] bool, policy emergency stop
    ped_pos: torch.Tensor         # [B, P, K, 2]
    ped_active: torch.Tensor      # [B, P, K]
    success: torch.Tensor         # [B]
    ped_collision: torch.Tensor   # [B]
    obstacle_collision: torch.Tensor
    end_step: torch.Tensor        # [B]
    robot_present: bool
    dt: float
    ped_influence: torch.Tensor | None = None  # [B, P, K] robot-attributable forecast change (learned crowds)

    @property
    def steps(self) -> int:
        return int(self.ped_pos.shape[2])


def rollout(env, batch: ScenarioBatch, policy=None, params: torch.Tensor | None = None, *,
            robot_present: bool = True) -> Rollout:
    """Run every environment of ``batch`` for its whole horizon.

    Pedestrians are always simulated to the end of the time grid. With ``robot_present=False``
    the robot is removed (the counterfactual baseline) and ``policy`` is not called.
    """
    state = env.reset(batch, robot_present=robot_present)
    if policy is not None:
        policy.reset(batch.size)
    poses, vs, ws, commands, emergencies = [state.robot.pose], [state.robot.v], [state.robot.w], [state.command], []
    peds, active = [state.peds.pos], [state.peds.active]
    influence = [torch.zeros_like(state.peds.active, dtype=state.peds.pos.dtype)]
    tracks_influence = hasattr(env.pedestrians, "last_influence")
    emergencies.append(torch.zeros(batch.size, dtype=torch.bool, device=state.robot.pose.device))
    with torch.no_grad():
        for _ in range(batch.steps - 1):
            command = None
            if robot_present and policy is not None and not bool(state.done.all()):
                command = policy.act(env.observe(state, batch), params)
                command = torch.where(state.done[:, None], torch.zeros_like(command), command)
            emergency = policy.last_info.get("emergency_stop") if (command is not None and policy is not None) else None
            state = env.step(state, batch, command)
            poses.append(state.robot.pose), vs.append(state.robot.v), ws.append(state.robot.w)
            commands.append(state.command), peds.append(state.peds.pos), active.append(state.peds.active)
            emergencies.append(emergency if emergency is not None else torch.zeros_like(emergencies[0]))
            if tracks_influence:
                value = env.pedestrians.last_influence
                influence.append(value.clone() if value is not None else torch.zeros_like(influence[0]))
    return Rollout(torch.stack(poses, 1), torch.stack(vs, 1), torch.stack(ws, 1), torch.stack(commands, 1),
                   torch.stack(emergencies, 1), torch.stack(peds, 2), torch.stack(active, 2), state.success,
                   state.ped_collision, state.obstacle_collision, state.end_step, robot_present, batch.dt,
                   torch.stack(influence, 2) if tracks_influence else None)
