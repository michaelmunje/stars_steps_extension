from __future__ import annotations

import json
from pathlib import Path


def write_sizing_table(stats_dir: Path, output: Path, datasets: set[str] | None = None) -> Path:
    rows = []
    for path in sorted(stats_dir.glob("*_sizing.json")):
        stats = json.loads(path.read_text())
        if datasets is not None and stats["dataset"] not in datasets:
            continue
        rows.append(
            " & ".join(
                str(value)
                for value in (
                    stats["dataset"],
                    f"{stats['hours']:.3f}" if "hours" in stats else "--",
                    stats["T_d_raw"],
                    stats["T_d_post_categorization"],
                    stats["L_med"],
                    stats["s_d"],
                    stats["w_d"],
                    f"{stats['p_d']:.4f}",
                    stats["final_train_scenes"],
                    stats["final_val_scenes"],
                    stats["final_test_scenes"],
                )
            ) + r" \\"
        )
    latex = "\n".join(
        [
            r"\begin{tabular}{lrrrrrrrrrr}",
            r"Dataset & Hours & Raw tracks & Post-cat. tracks & $L_{med}$ & $s_d$ & $w_d$ & $p_d$ & Train & Val & Test \\",
            r"\hline",
            *rows,
            r"\end{tabular}",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(latex, encoding="utf-8")
    return output
