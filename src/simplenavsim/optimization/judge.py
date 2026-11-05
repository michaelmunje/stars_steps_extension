"""Judged fitness: an evaluator labels rendered trajectories, and the labels are CMA-ES's fitness.

This replaces the hand-written objective ``J`` (``evaluator.Objective``) with AutoEval's recipe
(Zhou et al. 2025, arXiv 2503.24278): roll out, show the outcome to a *judge*, and turn its
classification into the score. AutoEval's judge is a fine-tuned PaliGemma looking at the final
camera frame; here the judge is an agent (a Claude model, or a person) looking at a top-down
picture of the whole episode, so there is no classifier to train and the rubric can say what
"good social navigation" means in words.

Labels (``LABELS``) and their fitness:

    excellent   +1.0  reaches the goal, direct and smooth, comfortable clearance throughout
    acceptable  +0.5  reaches the goal, with a flaw: a close pass, a detour, hesitation or wiggling
    stalled      0.0  does not reach the goal in time, without contact (froze, looped, backed off)
    unsafe      -1.0  touches a person or a wall

A candidate's fitness is the mean over its episodes. The picture (``render_episode``) shows only
what a camera above the scene would: paths with time ticks, the robot and the nearest person at
their closest approach, the goal circle and when the episode ended. The simulator's own
success / collision flags and metrics are *not* shown; they are kept beside the labels so the
judge can be compared with the hand-written objective afterwards (AutoEval's Fig. 7 analogue).

Judges (same interface as AutoEval's ``BaseSuccessDetector``: images in, labels out, inputs and
outputs saved):

* ``FileJudge``: labels are written to ``labels.json`` by someone else, an agent in the loop or a
  person, between ``judged_cmaes.py ask`` and ``judged_cmaes.py tell``.
* ``ClaudeCodeJudge``: a headless Claude Code agent (``claude -p``) opens every episode picture with its
  Read tool and writes ``labels.json``; it may only read the pictures and write that one file. Uses
  whatever account the ``claude`` CLI is logged in with (no API key needed).
* ``ClaudeJudge``: the Claude API labels each generation in one request (all episode images, the
  rubric, forced JSON through a tool). Needs ``ANTHROPIC_API_KEY`` and ``pip install anthropic``.
* ``RuleJudge``: labels from the simulator's metrics, for dry runs and tests without an API key.
"""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

LABELS = {
    "excellent": (1.0, "Reaches the goal circle on a direct, smooth path, keeps comfortable clearance (gap about "
                       "0.45 m or more) from every person, and does not make anyone swerve out of its way."),
    "acceptable": (0.5, "Reaches the goal circle, but with a flaw: a close pass (gap under ~0.45 m), making a person "
                        "swerve around it, a noticeable detour, hesitation, wiggling, or a slow arrival."),
    "stalled": (0.0, "Does not reach the goal circle before the episode ends, and never touches anyone: it froze, "
                     "looped, circled or backed away."),
    "unsafe": (-1.0, "Touches a person or a wall (gap at or below 0 m), or the episode ends abruptly next to a "
                     "person or wall short of the goal."),
}
FITNESS = {name: value for name, (value, _) in LABELS.items()}

RUBRIC = """You are evaluating a mobile robot navigating among pedestrians, from a top-down picture of one episode.

How to read a picture:
- Blue thick line: the robot's path from its start (hollow square) to where it ended (filled blue disc with a heading tick).
- Green star with a dashed green circle: the goal; the robot succeeds when its centre enters the dashed circle (1 m).
- Thin grey lines: pedestrians' paths. Dots on every path, robot and pedestrians alike, mark whole seconds and share
  one colour scale (dark purple = start, yellow = the end of the colour bar), so equal colours mean the same moment.
- Black lines: walls.
- Red outlined disc and blue outlined disc joined by a dotted line: where the nearest pedestrian and the robot were at their
  closest approach; the caption gives that gap (surface to surface) and its time.
- "ended at" in the caption: when the episode stopped. Episodes stop when the robot reaches the goal, touches something,
  or at the 30 s time limit.

Classify the episode into exactly one label:
""" + "\n".join(f"- {name} ({fit:+.1f}): {text}" for name, (fit, text) in LABELS.items()) + """

Judge what you see, as a careful person watching the robot would: safety first, then whether it got there, then
comfort and how natural the path looks. Give a one-sentence reason that names what you saw."""


# ---------------------------------------------------------------------------------------------- rendering

def episode_records(run, batch, robot_radius: float, goal_tolerance: float) -> list[dict]:
    """Numpy view of every row of a rollout, plus its closest approach (what a picture needs)."""
    n = lambda t: t.detach().float().cpu().numpy()
    pose, ped, active = n(run.robot_pose), n(run.ped_pos), run.ped_active.cpu().numpy()
    radius, mask = n(batch.ped_radius), batch.ped_mask.cpu().numpy()
    segments, seg_mask = n(batch.segments), batch.segment_mask.cpu().numpy()
    out = []
    for b in range(pose.shape[0]):
        end = int(run.end_step[b])
        gap = np.linalg.norm(ped[b, :, : end + 1] - pose[b, None, : end + 1, :2], axis=-1) \
            - radius[b, :, None] - robot_radius
        gap = np.where(active[b, :, : end + 1] & mask[b, :, None], gap, np.inf)
        k_min = p_min = None
        if np.isfinite(gap).any():
            p_min, k_min = np.unravel_index(np.argmin(gap), gap.shape)
        out.append({"pose": pose[b, : end + 1], "ped_pos": ped[b], "ped_active": active[b] & mask[b][:, None],
                     "ped_radius": radius[b], "segments": segments[b][seg_mask[b]], "goal": n(batch.robot_goal)[b],
                     "start": n(batch.robot_start)[b], "end_step": end, "dt": run.dt, "robot_radius": robot_radius,
                     "goal_tolerance": goal_tolerance, "closest": None if k_min is None else
                     {"ped": int(p_min), "step": int(k_min), "gap": float(gap[p_min, k_min])}})
    return out


def view_for(episodes: list[dict], margin: float = 1.5, aspect: float = 4 / 3):
    """One (lo, hi) window covering every robot path, start and goal in ``episodes`` (a scene's column)."""
    pts = np.concatenate([np.concatenate([e["pose"][:, :2], e["goal"][None], e["start"][None, :2]]) for e in episodes])
    lo, hi = pts.min(0) - margin, pts.max(0) + margin
    lo[1], hi[1] = min(lo[1], -3.0), max(hi[1], 3.0)
    span, centre = hi - lo, (lo + hi) / 2
    span = np.maximum(span, [span[1] * aspect, span[0] / aspect])
    return centre - span / 2, centre + span / 2


def draw_episode(ax, e: dict, view, *, duration: float = 30.0, title: str | None = None) -> None:
    (lo, hi), cmap = view, plt.cm.plasma
    ax.set_xlim(lo[0], hi[0]), ax.set_ylim(lo[1], hi[1]), ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]), ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color("#bbbbbb")
    for x0, y0, x1, y1 in e["segments"]:
        ax.plot([x0, x1], [y0, y1], color="black", lw=2.5, zorder=2)
    ax.add_patch(Circle(e["goal"], e["goal_tolerance"], fill=False, ls="--", lw=1.2, color="#2e8b57", zorder=2))
    ax.plot(*e["goal"], marker="*", color="#2e8b57", ms=13, zorder=5)
    end, dt = e["end_step"], e["dt"]
    every = int(round(1.0 / dt))
    for p in range(e["ped_pos"].shape[0]):
        live = e["ped_active"][p, : end + 1]
        if not live.any():
            continue
        steps = np.flatnonzero(live)
        ax.plot(*e["ped_pos"][p, steps].T, color="#888888", lw=1.0, alpha=0.7, zorder=3)
        ticks = steps[steps % every == 0]
        ax.scatter(*e["ped_pos"][p, ticks].T, c=ticks * dt, cmap=cmap, vmin=0, vmax=duration, s=9, zorder=4)
    ax.plot(*e["pose"][:, :2].T, color="#1f4e9c", lw=2.4, alpha=0.75, zorder=5)
    ticks = np.arange(0, end + 1, every)
    ax.scatter(*e["pose"][ticks, :2].T, c=ticks * dt, cmap=cmap, vmin=0, vmax=duration, s=22, edgecolors="#1f4e9c",
               linewidths=0.6, zorder=6)
    ax.plot(*e["start"][:2], marker="s", mfc="none", mec="#1f4e9c", ms=9, mew=1.5, zorder=6)
    x, y, th = e["pose"][-1]
    ax.add_patch(Circle((x, y), e["robot_radius"], color="#1f4e9c", zorder=7))
    ax.plot([x, x + 0.45 * np.cos(th)], [y, y + 0.45 * np.sin(th)], color="white", lw=1.5, zorder=8)
    caption = f"ended at {end * dt:.1f} s"
    c = e["closest"]
    if c is not None:
        k, p = c["step"], c["ped"]
        rp, pp = e["pose"][k, :2], e["ped_pos"][p, k]
        ax.add_patch(Circle(pp, e["ped_radius"][p], fill=False, color="#d62728", lw=1.8, zorder=9))
        ax.add_patch(Circle(rp, e["robot_radius"], fill=False, color="#1f4e9c", lw=1.8, zorder=9))
        ax.plot(*np.stack([rp, pp]).T, color="#d62728", ls=":", lw=1.2, zorder=9)
        caption += f"   closest gap {c['gap']:.2f} m at {k * dt:.1f} s"
    ax.text(0.02, 0.97, caption, transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5), zorder=10)
    if title:
        ax.set_title(title, fontsize=9.5, pad=3)


def time_scale(episodes: list[dict]) -> float:
    """Colour-scale end: the longest episode, rounded up to 5 s (so short episodes still span the colours)."""
    longest = max(e["end_step"] * e["dt"] for e in episodes)
    return float(max(10.0, 5.0 * np.ceil(longest / 5.0)))


def _colorbar(fig, axes, duration: float) -> None:
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(0, duration))
    fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.01, label="time (s)")


def render_episode(e: dict, path, *, title: str, view=None, duration: float | None = None) -> None:
    duration = duration or time_scale([e])
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=110, layout="constrained")
    draw_episode(ax, e, view or view_for([e]), duration=duration, title=title)
    _colorbar(fig, ax, duration)
    fig.savefig(path)
    plt.close(fig)


def render_sheet(grid: list[list[dict]], row_labels, col_labels, path, *, title: str, duration: float | None = None,
                 cell_titles=None, cell_colours=None) -> None:
    """Contact sheet: one row per candidate, one column per scene (each column shares its view and every cell
    the colour scale). ``cell_titles`` / ``cell_colours`` [R][C] replace the default ``row | column`` titles."""
    R, C = len(grid), len(grid[0])
    duration = duration or time_scale([e for row in grid for e in row])
    views = [view_for([grid[r][c] for r in range(R)]) for c in range(C)]
    fig, axes = plt.subplots(R, C, figsize=(4.4 * C + 0.9, 3.1 * R + 0.6), dpi=95, squeeze=False, layout="constrained")
    for r in range(R):
        for c in range(C):
            text = cell_titles[r][c] if cell_titles else f"{row_labels[r]} | {col_labels[c]}"
            draw_episode(axes[r, c], grid[r][c], views[c], duration=duration, title=text)
            if cell_colours:
                axes[r, c].title.set_color(cell_colours[r][c])
    fig.suptitle(title, fontsize=13)
    _colorbar(fig, axes, duration)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- judges

@dataclass
class Verdict:
    label: str
    reason: str = ""

    @property
    def fitness(self) -> float:
        return FITNESS[self.label]


def check_label(label: str) -> str:
    label = str(label).strip().lower()
    if label not in LABELS:
        raise ValueError(f"unknown label {label!r}; choose from {list(LABELS)}")
    return label


class Judge:
    """Labels a generation: ``episodes`` are ``{"id", "image", ...}``; returns ``{id: Verdict}``."""

    name = "judge"

    def label(self, episodes: list[dict], generation_dir: Path) -> dict[str, Verdict]:
        raise NotImplementedError


class FileJudge(Judge):
    """Labels written by an agent or a person to ``<generation>/labels.json``:
    ``{"c0_head_on-12": {"label": "acceptable", "reason": "..."}, ...}`` (a bare label string also works)."""

    name = "file"

    def label(self, episodes, generation_dir):
        path = Path(generation_dir) / "labels.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not written yet: label the episodes in pending.json first")
        raw = json.loads(path.read_text())
        missing = [e["id"] for e in episodes if e["id"] not in raw]
        if missing:
            raise ValueError(f"{path} is missing labels for {missing}")
        out = {}
        for e in episodes:
            v = raw[e["id"]]
            v = {"label": v} if isinstance(v, str) else v
            out[e["id"]] = Verdict(check_label(v["label"]), v.get("reason", ""))
        return out


class RuleJudge(Judge):
    """Thresholds on the simulator's metrics (dry runs, tests): not what the judge is meant to be."""

    name = "rule"

    def label(self, episodes, generation_dir):
        out = {}
        for e in episodes:
            m = e["sim"]
            if m["collision"] > 0:
                label = "unsafe"
            elif m["success"] < 1:
                label = "stalled"
            elif m["min_ped_clearance"] >= 0.45 and m["path_efficiency"] >= 0.9:
                label = "excellent"
            else:
                label = "acceptable"
            out[e["id"]] = Verdict(label, "rule on simulator metrics")
        return out


class ClaudeJudge(Judge):
    """Claude labels every episode of a generation in one request (the images, the rubric, JSON by tool)."""

    name = "claude"

    def __init__(self, model: str = "claude-opus-5-5", max_tokens: int = 8192, attempts: int = 3):
        import anthropic

        self.anthropic = anthropic
        self.client, self.model, self.max_tokens, self.attempts = anthropic.Anthropic(), model, max_tokens, attempts

    def label(self, episodes, generation_dir):
        content = [{"type": "text", "text": RUBRIC + f"\n\nThere are {len(episodes)} episodes below, each preceded by "
                                                     "its id. Label every one with the record_labels tool."}]
        for e in episodes:
            data = base64.standard_b64encode(Path(e["image"]).read_bytes()).decode()
            content += [{"type": "text", "text": f"Episode {e['id']}:"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}]
        tool = {"name": "record_labels", "description": "Record one label per episode.",
                "input_schema": {"type": "object", "required": ["labels"], "properties": {"labels": {
                    "type": "array", "items": {"type": "object", "required": ["id", "label", "reason"], "properties": {
                        "id": {"type": "string"}, "label": {"type": "string", "enum": list(LABELS)},
                        "reason": {"type": "string"}}}}}}}
        for attempt in range(self.attempts):
            try:
                reply = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, tools=[tool],
                                                    tool_choice={"type": "tool", "name": "record_labels"},
                                                    messages=[{"role": "user", "content": content}])
                break
            except (self.anthropic.APIConnectionError, self.anthropic.RateLimitError,
                    self.anthropic.InternalServerError):
                if attempt + 1 == self.attempts:
                    raise
                time.sleep(20 * (attempt + 1))
        records = next(b.input for b in reply.content if b.type == "tool_use")["labels"]
        (Path(generation_dir) / "judge_response.json").write_text(json.dumps(records, indent=1))
        out = {r["id"]: Verdict(check_label(r["label"]), r.get("reason", "")) for r in records}
        missing = [e["id"] for e in episodes if e["id"] not in out]
        if missing:
            raise ValueError(f"judge returned no label for {missing}")
        return out


class ClaudeCodeJudge(Judge):
    """A headless Claude Code agent judges the generation: ``claude -p`` run in the generation's folder,
    allowed only to read the episode pictures and to write ``labels.json`` (``sim.json``, with the
    simulator's metrics, stays out of reach). Its transcript is kept in ``judge_transcript.json``."""

    name = "claude-code"

    def __init__(self, model: str | None = None, command: str = "claude", timeout: int = 1800):
        if shutil.which(command) is None:
            raise FileNotFoundError(f"`{command}` not found: install Claude Code and log in, or use --judge claude / file")
        self.model, self.command, self.timeout = model, command, timeout

    def label(self, episodes, generation_dir):
        folder = Path(generation_dir).resolve()
        (folder / "labels.json").unlink(missing_ok=True)
        listing = "\n".join(f"- {e['id']}: {Path(e['image']).resolve()}" for e in episodes)
        prompt = (RUBRIC + f"""

You are the judge for one generation of a policy search. Open each of the {len(episodes)} pictures below with the
Read tool, look at it carefully, and label it. Judge only from the pictures.

{listing}

When you have looked at every picture, use the Write tool once to create {folder / "labels.json"}: one JSON object
mapping every episode id above to {{"label": <one of {", ".join(LABELS)}>, "reason": <one sentence on what you saw>}}.
""")
        command = [self.command, "-p", prompt, "--output-format", "json", "--permission-mode", "acceptEdits",
                   "--allowedTools", "Read(./episodes/**)", "Write(./labels.json)",
                   "--disallowedTools", "Bash", "Edit", "Read(./sim.json)", "WebFetch", "WebSearch", "Task"]
        if self.model:
            command += ["--model", self.model]
        done = subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=self.timeout,
                              stdin=subprocess.DEVNULL)
        (folder / "judge_transcript.json").write_text(done.stdout or done.stderr)
        if done.returncode:
            raise RuntimeError(f"claude -p failed ({done.returncode}): {(done.stderr or done.stdout)[-2000:]}")
        return FileJudge().label(episodes, folder)


def make_judge(name: str, **options) -> Judge:
    return {"file": FileJudge, "rule": RuleJudge, "claude": ClaudeJudge, "claude-code": ClaudeCodeJudge}[name](**options)
