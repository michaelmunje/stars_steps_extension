"""ETH/UCY closed-loop windows: fixed cohorts, oracle goals, no future leakage.

Recordings are the converted 2.5 Hz world-coordinate tracks (``datasets/{eth,ucy}/converted``, metres,
one step = 0.4 s). Benchmark scenes follow the usual five-scene convention: eth = seq_eth, hotel =
seq_hotel, univ = students001 + students003, zara1 = crowds_zara01, zara2 = crowds_zara02.
``CALIBRATION`` (crowds_zara03, arxiepiskopi1, uni_examples) are UCY recordings outside the five scenes,
used only to calibrate the classical baselines.

A window starts at ``start``; its last observed step is ``t0 = start + OBS - 1``. The cohort is every
track present at ``t0``; all of them are simulated jointly from their observed history (missing
history steps are masked) for up to ``MAX_STEPS`` steps. An agent is *scored* at horizon ``H`` only if
it has all ``OBS`` history steps, a continuous recorded track through ``t0 + H`` and a track that
continues past it (so its oracle goal, the track's last recorded position, lies beyond every scored
position); the others are context agents (simulated, never scored or drawn). Tracks that appear after
``t0`` are not simulated (fixed cohort). Simulators only ever see ``Window.inputs()``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DT = 0.4
OBS = 8
HORIZONS = {2.0: 5, 4.0: 10, 4.8: 12, 8.0: 20, 12.0: 30}     # seconds -> scored steps
MAX_STEPS = 30
ROOT = Path(__file__).resolve().parents[3] / "datasets"
SCENES = {"eth": ["seq_eth"], "hotel": ["seq_hotel"], "univ": ["students001", "students003"],
          "zara1": ["crowds_zara01"], "zara2": ["crowds_zara02"]}
CALIBRATION = ["crowds_zara03", "arxiepiskopi1", "uni_examples"]
SOURCE = {"seq_eth": "eth", "seq_hotel": "eth"}            # everything else is under ucy/
# Split of the six-dataset checkpoint (cvae_all_lr3e-4_seed42), whole recordings
# (datasets/{eth,ucy}/split_manifests/grouped-seed42.csv): it trained on every window of these.
POOLED_SPLIT = {"seq_hotel": "train", "crowds_zara01": "train", "students001": "train",
                "students003": "val", "seq_eth": "test", "crowds_zara02": "test"}


@dataclass
class Recording:
    name: str
    tracks: dict[int, tuple[np.ndarray, np.ndarray]]     # id -> (steps [T], xy [T, 2]), steps sorted
    last_step: int


@dataclass
class Window:
    scene_id: str
    dataset: str
    recording: str
    start: int
    steps: int                                            # simulated steps (<= MAX_STEPS, recording permitting)
    agent_ids: np.ndarray                                 # [N] cohort
    obs: np.ndarray                                       # [N, OBS, 2] (NaN where missing)
    obs_valid: np.ndarray                                 # [N, OBS]
    future: np.ndarray                                    # [N, steps, 2] ground truth, scoring only (NaN where missing)
    future_valid: np.ndarray                              # [N, steps]
    goal: np.ndarray                                      # [N, 2] oracle goal: last recorded position
    end_step: np.ndarray                                  # [N] last recorded step of each track
    goal_at_boundary: np.ndarray                          # [N] track ends at the recording's last step
    entry_steps: np.ndarray                               # first step of tracks entering after t0 (not simulated)
    meta: dict = field(default_factory=dict)

    @property
    def t0(self) -> int:
        return self.start + OBS - 1

    def scored(self, horizon: int) -> np.ndarray:
        """[N] agents scored at ``horizon`` steps."""
        if horizon > self.steps:
            return np.zeros(len(self.agent_ids), bool)
        return self.obs_valid.all(1) & self.future_valid[:, :horizon].all(1) & (self.end_step > self.t0 + horizon)

    def late_arrivals(self, horizon: int) -> int:
        return int((self.entry_steps <= self.t0 + horizon).sum())

    def inputs(self) -> dict:
        """Everything a simulator may see: observed history, its mask and the oracle goals."""
        return {"obs": self.obs.copy(), "obs_valid": self.obs_valid.copy(), "goal": self.goal.copy()}


def load_recording(name: str) -> Recording:
    frame = pd.read_parquet(ROOT / SOURCE.get(name, "ucy") / "converted" / f"{name}.traj.parquet")
    frame = frame.sort_values(["track_id", "step"])
    tracks = {int(i): (g["step"].to_numpy(np.int64), g[["x", "y"]].to_numpy(np.float64))
              for i, g in frame.groupby("track_id")}
    return Recording(name, tracks, int(frame["step"].max()))


def _positions(recording: Recording, track_id: int, steps: np.ndarray):
    s, xy = recording.tracks[track_id]
    idx = np.clip(np.searchsorted(s, steps), 0, len(s) - 1)
    ok = s[idx] == steps
    return np.where(ok[:, None], xy[idx], np.nan), ok


def windows(recording: Recording, dataset: str, *, stride: int = 10, min_steps: int = 5) -> list[Window]:
    """Windows of ``recording`` every ``stride`` steps (cohort present at ``t0``, >= ``min_steps`` to simulate)."""
    first = min(s[0] for s, _ in recording.tracks.values())
    out = []
    for start in range(first, recording.last_step - OBS - min_steps + 2, stride):
        t0 = start + OBS - 1
        steps = min(MAX_STEPS, recording.last_step - t0)
        ids = [i for i, (s, _) in recording.tracks.items() if s[0] <= t0 <= s[-1] and np.any(s == t0)]
        if not ids or steps < min_steps:
            continue
        hist, fut = np.arange(start, t0 + 1), np.arange(t0 + 1, t0 + 1 + steps)
        obs, ov = zip(*(_positions(recording, i, hist) for i in ids))
        future, fv = zip(*(_positions(recording, i, fut) for i in ids))
        end = np.array([recording.tracks[i][0][-1] for i in ids])
        entry = np.array([s[0] for s, _ in recording.tracks.values() if t0 < s[0] <= t0 + steps], dtype=np.int64)
        out.append(Window(f"{dataset}-{recording.name}-{start}", dataset, recording.name, start, steps, np.array(ids),
                          np.stack(obs), np.stack(ov), np.stack(future), np.stack(fv),
                          np.stack([recording.tracks[i][1][-1] for i in ids]), end, end >= recording.last_step, entry,
                          {"pooled_split": POOLED_SPLIT.get(recording.name, "unseen")}))
    return out


def benchmark_windows(*, scenes=None, stride: int = 10, recordings=None) -> list[Window]:
    """Windows over the five scenes (or ``recordings`` given as ``{dataset: [names]}``)."""
    groups = recordings or {k: v for k, v in SCENES.items() if scenes is None or k in scenes}
    return [w for dataset, names in groups.items() for name in names
            for w in windows(load_recording(name), dataset, stride=stride)]


def horizon_counts(wins: list[Window]) -> list[dict]:
    """Windows / scored agents / exclusions per dataset and horizon (for the report)."""
    rows = []
    for (label, h) in HORIZONS.items():
        for dataset in dict.fromkeys(w.dataset for w in wins):
            ws = [w for w in wins if w.dataset == dataset]
            scored = [w.scored(h) for w in ws]
            full_hist = sum(int(w.obs_valid.all(1).sum()) for w in ws)
            rows.append({"horizon_s": label, "dataset": dataset, "windows": sum(int(s.any()) for s in scored),
                         "candidate_windows": len(ws), "cohort_agents": sum(len(w.agent_ids) for w, s in zip(ws, scored) if s.any()),
                         "scored_agents": sum(int(s.sum()) for s in scored),
                         "excluded_short_history": sum(len(w.agent_ids) for w in ws) - full_hist,
                         "excluded_track_ends_by_horizon": full_hist - sum(int(s.sum()) for s in scored),
                         "scored_goal_at_boundary": sum(int((w.goal_at_boundary & s).sum()) for w, s in zip(ws, scored)),
                         "late_arrivals_not_simulated": sum(w.late_arrivals(h) for w, s in zip(ws, scored) if s.any())})
    return rows
