"""TrajNet++ scene-type characterization of the canonical datasets.

Reads the per-dataset scene indices written by :mod:`preprocess` (which store
the TrajNet++ tag produced by :func:`categorization.get_type`) and reports the
type distribution per dataset, plus the type-3 interaction sub-tags.

Population
----------
The reported distribution covers *every categorized forecast window* in the
index, before the TrajNet++ type-1 acceptance down-sampling and before the
train-budget subsampling.  Those two are sampling choices for training and
would otherwise be read as a property of the data.  The accepted-only
distribution is emitted alongside, under ``accepted``, for transparency.

Windows are dense (stride 1 at 2.5 Hz), so a single pedestrian contributes many
overlapping windows; the distribution is over windows, not over pedestrians.

Observation length
------------------
Upstream TrajNet++ characterizes 9 observed + 12 predicted frames.  This
repository uses the 8 + 12 convention from ``configs/task.json``; the
categorization code is otherwise identical to upstream, which
``tests/test_trajnet_categorization_parity.py`` checks at both lengths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from artifact_provenance import git_provenance

ROOT = Path(__file__).resolve().parents[2]

REFERENCE_IMPLEMENTATION = "https://" + "github.com/vita-epfl/trajnetplusplustools"

# Display order and labels; keys are ``configs/task.json`` dataset names.
DATASETS = (
    ("eth", "ETH"),
    ("ucy", "UCY"),
    ("steps", "STEPS"),
    ("ptdpes", "PTDPES"),
    ("madras", "MADRAS"),
    ("grand_central", "Grand Central"),
)

TYPE_LABELS = {1: "Static", 2: "Linear", 3: "Interacting", 4: "Non-interacting"}
SUB_TAG_LABELS = {1: "Leader-follower", 2: "Collision avoidance", 3: "Group", 4: "Other"}


def dataset_directory(dataset_root: Path, name: str) -> Path:
    """Match ``configs/task.json`` names to on-disk directories case-insensitively."""
    matches = [path for path in dataset_root.iterdir() if path.is_dir() and path.name.lower() == name.lower()]
    if len(matches) > 1:
        raise ValueError(f"Multiple dataset directories match {name}: {matches}")
    return matches[0] if matches else dataset_root / name


def scene_index_path(dataset_dir: Path, data_seed: int) -> Path:
    """The scene index for this dataset, whichever split strategy produced it."""
    candidates = sorted((dataset_dir / "indices").glob(f"scenes-seed{data_seed}*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No scene index under {dataset_dir / 'indices'}; run preprocessing first")
    if len(candidates) > 1:
        raise ValueError(f"Multiple scene indices under {dataset_dir / 'indices'}: {candidates}")
    return candidates[0]


def _distribution(tags) -> dict:
    counts = {tag: 0 for tag in TYPE_LABELS}
    for pair in pc.value_counts(tags).to_pylist():
        counts[int(pair["values"])] = int(pair["counts"])
    total = sum(counts.values())
    return {
        "total_windows": total,
        "counts": {str(tag): counts[tag] for tag in sorted(counts)},
        "fractions": {str(tag): (counts[tag] / total if total else 0.0) for tag in sorted(counts)},
    }


def _sub_tag_distribution(table) -> dict:
    """Sub-tag counts over type-3 windows.  A window may carry several."""
    type_three = table.filter(pc.equal(table.column("scene_tag"), 3))
    counts = {tag: 0 for tag in SUB_TAG_LABELS}
    if type_three.num_rows:
        flattened = pc.list_flatten(type_three.column("sub_tags").combine_chunks())
        for pair in pc.value_counts(flattened).to_pylist():
            counts[int(pair["values"])] = int(pair["counts"])
    total = type_three.num_rows
    return {
        "type3_windows": total,
        "counts": {str(tag): counts[tag] for tag in sorted(counts)},
        # Sub-tags are not mutually exclusive, so these share a denominator of
        # type-3 windows and need not sum to one.
        "fractions_of_type3": {str(tag): (counts[tag] / total if total else 0.0) for tag in sorted(counts)},
    }


def characterize(dataset: str, dataset_root: Path, data_seed: int, root: Path = ROOT) -> dict:
    index_path = scene_index_path(dataset_directory(dataset_root, dataset), data_seed)
    table = pq.read_table(index_path, columns=["scene_tag", "sub_tags", "split", "accepted"])
    accepted = table.filter(table.column("accepted"))
    result = {
        "dataset": dataset,
        "scene_index": str(index_path.relative_to(root) if index_path.is_relative_to(root) else index_path),
        "all_windows": _distribution(table.column("scene_tag")),
        "accepted": _distribution(accepted.column("scene_tag")),
        "interaction_sub_tags": _sub_tag_distribution(table),
        "by_split": {},
    }
    for split in ("train", "val", "test"):
        rows = table.filter(pc.equal(table.column("split"), split))
        result["by_split"][split] = _distribution(rows.column("scene_tag"))
    return result


def render_table(rows: list[dict], labels: dict[str, str], provenance: dict[str, object]) -> str:
    """LaTeX table of the type distribution, mirroring TrajNet++ Table 1."""
    lines = [
        r"% Generated by src/paper/generate_trajnet_characterization.py -- do not edit by hand.",
        f"% git_commit={provenance['git_commit']} git_dirty={provenance['git_dirty']}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Dataset & Windows & Type I (\%) & Type II (\%) & Type III (\%) & Type IV (\%) \\",
        r"\midrule",
    ]
    for row in rows:
        fractions = row["all_windows"]["fractions"]
        percentages = " & ".join(f"{100 * fractions[str(tag)]:.1f}" for tag in (1, 2, 3, 4))
        lines.append(f"{labels[row['dataset']]} & {row['all_windows']['total_windows']:,} & {percentages} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def render_sub_tag_table(rows: list[dict], labels: dict[str, str], provenance: dict[str, object]) -> str:
    lines = [
        r"% Generated by src/paper/generate_trajnet_characterization.py -- do not edit by hand.",
        f"% git_commit={provenance['git_commit']} git_dirty={provenance['git_dirty']}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Dataset & Type III windows & Leader-follower (\%) & Collision avoidance (\%) & Group (\%) & Other (\%) \\",
        r"\midrule",
    ]
    for row in rows:
        sub = row["interaction_sub_tags"]
        percentages = " & ".join(f"{100 * sub['fractions_of_type3'][str(tag)]:.1f}" for tag in (1, 2, 3, 4))
        lines.append(f"{labels[row['dataset']]} & {sub['type3_windows']:,} & {percentages} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def generate(root: Path = ROOT, data_seed: int = 42, datasets=DATASETS) -> dict:
    task = json.loads((root / "configs" / "task.json").read_text())
    rows = [characterize(name, root / "datasets", data_seed, root) for name, _ in datasets]
    labels = dict(datasets)
    provenance = git_provenance(root)
    result = {
        "reference_implementation": REFERENCE_IMPLEMENTATION,
        "observation_steps": int(task["history_steps"]),
        "prediction_steps": int(task["future_steps"]),
        "sample_hz": float(task["sample_hz"]),
        "data_seed": data_seed,
        "type_labels": {str(tag): label for tag, label in TYPE_LABELS.items()},
        "sub_tag_labels": {str(tag): label for tag, label in SUB_TAG_LABELS.items()},
        "population_note": (
            "Distributions cover every categorized forecast window in the scene index, before "
            "TrajNet++ type-1 acceptance down-sampling and before train-budget subsampling. "
            "Windows are dense (stride 1), so one pedestrian contributes many overlapping windows."
        ),
        "observation_length_note": (
            "Upstream TrajNet++ observes 9 frames; this repository observes "
            f"{int(task['history_steps'])} per configs/task.json. The categorization is otherwise "
            "identical to upstream (see tests/test_trajnet_categorization_parity.py)."
        ),
        "datasets": rows,
        "git_commit": provenance["git_commit"],
        "git_dirty": provenance["git_dirty"],
    }
    stats_dir = root / "paper" / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / "trajnet_characterization.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "paper" / "table_trajnet_characterization.tex").write_text(render_table(rows, labels, provenance))
    (root / "paper" / "table_trajnet_interaction_sub_tags.tex").write_text(
        render_sub_tag_table(rows, labels, provenance)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the TrajNet++ scene-type characterization tables")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--root", type=Path, default=ROOT)
    arguments = parser.parse_args()
    result = generate(arguments.root, arguments.data_seed)
    for row in result["datasets"]:
        fractions = row["all_windows"]["fractions"]
        percentages = "  ".join(f"{TYPE_LABELS[tag][:4]} {100 * fractions[str(tag)]:5.1f}%" for tag in (1, 2, 3, 4))
        print(f"{row['dataset']:<14} {row['all_windows']['total_windows']:>12,}  {percentages}")


if __name__ == "__main__":
    main()
