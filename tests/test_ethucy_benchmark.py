"""ETH/UCY closed-loop benchmark: data, leakage, closed loop, metrics, seeds and the preference manifest."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from simplenavsim.realism import data as D
from simplenavsim.realism import metrics as M
from simplenavsim.realism import sim as S

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simplenavsim" / "scripts"))
import ethucy_benchmark as B  # noqa: E402

pytestmark = pytest.mark.skipif(not (D.ROOT / "eth" / "converted" / "seq_eth.traj.parquet").exists(),
                                reason="ETH/UCY converted data not available")
needs_stars = pytest.mark.skipif(not S.CHECKPOINTS["stars_steps"].exists(), reason="STARS checkpoint not available")


@pytest.fixture(scope="module")
def zara():
    rec = D.load_recording("crowds_zara01")
    return rec, D.windows(rec, "zara1")


def test_units_and_rate():
    rec = D.load_recording("seq_eth")
    speeds = np.concatenate([np.linalg.norm(np.diff(xy, axis=0), axis=1)[np.diff(s) == 1] / D.DT
                             for s, xy in rec.tracks.values() if len(s) > 1])
    assert 0.5 < np.median(speeds) < 2.0                     # walking speed in m/s at 0.4 s per step
    assert all(np.all(np.diff(s) > 0) for s, _ in rec.tracks.values())


def test_agent_ids_and_time_alignment(zara):
    rec, wins = zara
    for w in wins[:20]:
        assert len(set(w.agent_ids.tolist())) == len(w.agent_ids)
        for i, a in enumerate(w.agent_ids):
            steps, xy = rec.tracks[int(a)]
            k = np.flatnonzero(steps == w.t0)[0]
            assert np.allclose(w.obs[i, -1], xy[k])                       # last observed step is t0
            if k + 1 < len(steps) and steps[k + 1] == w.t0 + 1:
                assert np.allclose(w.future[i, 0], xy[k + 1])             # first future step is t0 + 1
            assert np.allclose(w.goal[i], xy[-1])                         # oracle goal: last recorded position
        for h in D.HORIZONS.values():
            s = w.scored(h)
            assert np.all(w.future_valid[s, :h]) and np.all(w.end_step[s] > w.t0 + h)


def _small(wins, n=3):
    return [w for w in wins if w.scored(12).sum() >= 2][:n]


def _sims():
    sims = {"cv": S.cv, "goal_straight": S.goal_straight, "sfm": S.sfm, "orca": S.orca}
    if S.CHECKPOINTS["stars_steps"].exists():
        model = S.Stars(S.CHECKPOINTS["stars_steps"])
        sims.update(stars=model, stars_goal=model.with_goal(0.3))
    return sims


def test_no_future_leakage(zara):
    wins = _small(zara[1])
    changed = copy.deepcopy(wins)
    for w in changed:
        w.future = w.future + 100.0                                       # ground truth future is irrelevant
    for name, fn in _sims().items():
        a, b = fn(S.pad(wins), 8, seed=0), fn(S.pad(changed), 8, seed=0)
        assert np.array_equal(a, b), name
    assert set(wins[0].inputs()) == {"obs", "obs_valid", "goal"}


@needs_stars
def test_closed_loop_feedback(zara):
    """Perturbing an executed position changes every later STARS input (and so the rollout)."""
    wins = _small(zara[1], 1)
    x, model = S.pad(wins), S.Stars(S.CHECKPOINTS["stars_steps"])

    def push(k, pos):
        if k == 2:
            pos = pos.clone()
            pos[0, 0] += torch.tensor([0.5, 0.0])
        return pos

    base, pushed = model(x, 8, seed=0), model(x, 8, seed=0, hook=push)
    assert np.array_equal(base[:, :, :2], pushed[:, :, :2])               # before the perturbation: identical
    assert not np.allclose(base[0, 0, 3:], pushed[0, 0, 3:] - [0.5, 0.0], atol=1e-4)  # not a mere offset
    others = np.abs(base[0, 1:, 3:] - pushed[0, 1:, 3:]).max()
    assert others > 0                                                     # neighbours react


def test_displacement_metrics():
    w = D.Window("t", "zara1", "r", 0, 4, np.array([1, 2]), np.zeros((2, D.OBS, 2)), np.ones((2, D.OBS), bool),
                 np.stack([np.stack([np.arange(1, 5), np.zeros(4)], -1), np.stack([np.arange(1, 5), np.full(4, 5.0)], -1)]),
                 np.ones((2, 4), bool), np.array([[9.0, 0], [9.0, 5]]), np.array([100, 100]), np.zeros(2, bool), np.array([]))
    w.obs[1] = [0.0, 5.0]
    pred = w.future + [0.0, 1.0]
    row, agents, _ = M.window_metrics(pred, w, 4)
    assert row["agents"] == 2 and np.isclose(row["ade_sum"], 2.0) and np.isclose(row["fde_sum"], 2.0)
    assert np.isclose(row["sde_sum"], 2.0) and row["nonlinear_agents"] == 0
    assert np.isclose(row["mde_end_sum"], 2.0)                            # 1 m from each ground-truth endpoint


def test_collision_metrics():
    a = np.zeros((4, 2))
    b = np.array([[0.3, 0], [0.3, 0], [1.0, 0], [1.0, 0]])
    c = M.collisions(np.stack([a, b]))
    assert c["colliding_pairs"] == 1 and c["colliding_pair_steps"] == 2 and c["agents_in_collision"] == 2
    assert np.isclose(c["hacs_sum"], 4)                                   # 2 ordered pairs x 2 steps
    assert np.isclose(c["sacs_sum"], 4 * (0.4 - 0.3) / 0.4)
    assert np.isclose(c["min_separation"], 0.3) and c["agent_steps"] == 8


@needs_stars
def test_seed_repeatability(zara):
    wins, model = _small(zara[1], 2), S.Stars(S.CHECKPOINTS["stars_steps"])
    x = S.pad(wins)
    assert np.array_equal(model(x, 6, seed=3), model(x, 6, seed=3))
    assert not np.array_equal(model(x, 6, seed=3), model(x, 6, seed=4))
    assert np.array_equal(model.with_goal(0.0)(x, 6, seed=1), model(x, 6, seed=1))


def _manifest(tmp_path, clip_name=None):
    (tmp_path / "clips").mkdir()
    cfg = {"width": 960, "height": 720, "fps": 12, "trail_s": 2.0, "show_goals": False, "show_ids": False,
           "background": "#ffffff", "render_config_hash": "sha256:" + hashlib.sha256(b"r").hexdigest()}
    clips = []
    for m in B.STUDY_MODELS:
        cid = "c_" + hashlib.sha256(m.encode()).hexdigest()[:6]
        name = clip_name(m) if clip_name else cid
        for ext in ("mp4", "gif"):
            (tmp_path / "clips" / f"{name}.{ext}").write_bytes(b"x")
        clips.append({"clip_id": cid, "model": m, "seed": 0, "mp4": f"clips/{name}.mp4", "gif": f"clips/{name}.gif",
                      "duration_s": 8.0, "fps": 12, "init_hash": "sha256:i", "render_config_hash": cfg["render_config_hash"]})
    m = {"schema_version": "1.0", "benchmark_run_id": "x", "created_utc": "2026-11-05T00:00:00Z", "study_type": "factual",
         "render": cfg, "scenarios": [{"scenario_id": "eth-0001", "dataset": "eth", "recording": "seq_eth", "start_frame": 1,
                                       "start_time_s": 0.4, "num_pedestrians": 3, "horizon_s": 8.0, "sim_dt": 0.4,
                                       "goal_rule": "oracle", "bounds_m": [0, 0, 4, 3], "init_hash": "sha256:i",
                                       "counterfactual": None, "clips": clips}]}
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    return tmp_path / "manifest.json"


def test_manifest_schema(tmp_path):
    assert B.validate_manifest(_manifest(tmp_path)) == []
    bad = tmp_path / "bad"
    bad.mkdir()
    assert B.validate_manifest(_manifest(bad, clip_name=lambda m: m))      # named by model, not clip_id


def test_render_hash_stable():
    from simplenavsim.realism import render as R
    assert R.render_config() == R.render_config()
    assert R.render_config()["render_config_hash"].startswith("sha256:")


def test_cli_help():
    with pytest.raises(SystemExit) as e:
        B.main(["--help"])
    assert e.value.code == 0
