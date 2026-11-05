"""Clips of ETH/UCY rollouts: MP4 (H.264, yuv420p) and GIF, single panels or labelled grids.

Rendering is identical for every model: same view (``bounds``), discs, trail and frame rate; the
0.4 s simulation steps are linearly interpolated for display only. Anonymous clips carry no title,
ids, goals or model label.
"""
from __future__ import annotations

import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from .data import DT, OBS  # noqa: E402

try:                                   # bundled ffmpeg when no system one (PYTHONPATH=<dir with imageio_ffmpeg>)
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass

RENDER = {"width": 960, "height": 720, "fps": 12, "trail_s": 2.0, "show_goals": False, "show_ids": False,
          "background": "#ffffff"}
STYLE = {"disc_radius_m": 0.25, "agent_color": "#2b6cb0", "trail_color": "#90b4de", "trail_width": 1.6,
         "axis": "off", "scale_bar_m": 2.0, "interpolation": "linear", "dpi": 120, "renderer": "realism.render v1"}


def render_config() -> dict:
    blob = json.dumps({**RENDER, **STYLE}, sort_keys=True).encode()
    return {**RENDER, "render_config_hash": "sha256:" + hashlib.sha256(blob).hexdigest()}


def init_hash(agent_ids, obs, goal) -> str:
    h = hashlib.sha256()
    for a in (np.asarray(agent_ids, np.int64), np.round(np.asarray(obs, np.float64), 6), np.round(np.asarray(goal, np.float64), 6)):
        h.update(a.tobytes())
    return "sha256:" + h.hexdigest()


def bounds(paths, pad: float = 2.0, aspect: float = 4 / 3) -> list[float]:
    """View box around ``paths [A, T, 2]`` (padded, widened to ``aspect``)."""
    lo, hi = np.nanmin(paths.reshape(-1, 2), 0) - pad, np.nanmax(paths.reshape(-1, 2), 0) + pad
    c, size = (lo + hi) / 2, hi - lo
    w = max(size[0], size[1] * aspect)
    h = w / aspect
    return [float(c[0] - w / 2), float(c[1] - h / 2), float(c[0] + w / 2), float(c[1] + h / 2)]


def _frames(path, horizon_s, fps):
    """``path [A, OBS + H, 2]`` (history then rollout) at display rate from -trail to the horizon."""
    t_steps = (np.arange(path.shape[1]) - (OBS - 1)) * DT
    t = np.arange(-int(RENDER["trail_s"] * fps), int(round(horizon_s * fps)) + 1) / fps
    out = np.stack([np.stack([np.interp(t, t_steps, path[a, :, d]) for d in (0, 1)], -1) for a in range(path.shape[0])])
    return out, int(RENDER["trail_s"] * fps)


def animate(panels, view, horizon_s, mp4=None, gif=None, *, titles=None, shape=(1, 1), size=None, suptitle=None):
    """Write an animation of ``panels`` (list of ``[A, OBS + H, 2]``) laid out as ``shape``."""
    fps, dpi = RENDER["fps"], STYLE["dpi"]
    width, height = size or (RENDER["width"], RENDER["height"])
    fig, axes = plt.subplots(*shape, figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=RENDER["background"],
                             squeeze=False)
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.93 if (titles or suptitle) else 0.99, 0.03, 0.12 if titles else 0.03)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    data, trail = zip(*(_frames(p, horizon_s, fps) for p in panels))
    trail = trail[0]
    artists = []
    for i, ax in enumerate(axes.flat):
        if i >= len(panels):
            ax.axis("off")
            continue
        ax.set_facecolor(RENDER["background"])
        ax.set_xlim(view[0], view[2]), ax.set_ylim(view[1], view[3]), ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#d0d0d0")
        if titles:
            ax.set_title(titles[i], fontsize=10)
        bar = STYLE["scale_bar_m"]
        x0, y0 = view[0] + 0.4, view[1] + 0.4
        ax.plot([x0, x0 + bar], [y0, y0], color="#555555", lw=2)
        ax.text(x0 + bar / 2, y0 + 0.15, f"{bar:g} m", ha="center", va="bottom", fontsize=7, color="#555555")
        lines = LineCollection([], colors=STYLE["trail_color"], linewidths=STYLE["trail_width"])
        ax.add_collection(lines)
        discs = [plt.Circle((np.nan, np.nan), STYLE["disc_radius_m"], color=STYLE["agent_color"]) for _ in range(len(data[i]))]
        for d in discs:
            ax.add_patch(d)
        artists.append((lines, discs, data[i]))
    n_frames = data[0].shape[1] - trail

    def update(f):
        out = []
        for lines, discs, xy in artists:
            k = f + trail
            lines.set_segments([seg for seg in xy[:, k - trail:k + 1]])
            for d, p in zip(discs, xy[:, k]):
                d.center = p
            out += [lines, *discs]
        return out

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    if mp4:
        anim.save(mp4, writer=FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]), dpi=dpi)
    if gif:
        anim.save(gif, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return n_frames / fps
