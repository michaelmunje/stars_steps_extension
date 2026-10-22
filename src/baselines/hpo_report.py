"""Summarise an Optuna study from ``hpo_stars_ma``: best trials, parameter importances, plots.

    python -m baselines.hpo_report experiments/stars_ma/hpo/steps_cvae_scratch --output docs/results/hpo_steps_cvae_scratch.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import optuna


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()
    journal = str(args.study_dir / "journal.log")
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(
        journal, lock_obj=optuna.storages.journal.JournalFileOpenLock(journal)))
    study = optuna.load_study(study_name=args.study_dir.name, storage=storage)
    trials = study.trials
    complete = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in trials)
    complete.sort(key=lambda t: t.value)
    names = sorted({k for t in complete for k in t.params})
    lines = [f"# Optuna study `{study.study_name}`", "",
             f"{len(trials)} trials: {len(complete)} complete, {pruned} pruned. Objective: validation macro agent "
             "ADE (best-of-K, K from the study command), lower is better.", "",
             f"## Top {min(args.top, len(complete))} trials", "",
             "| trial | value | " + " | ".join(names) + " |", "| ---: | ---: |" + " --- |" * len(names)]
    for t in complete[: args.top]:
        cells = [f"{t.params[n]:.3g}" if isinstance(t.params.get(n), float) else str(t.params.get(n, "")) for n in names]
        lines.append(f"| {t.number} | {t.value:.4f} | " + " | ".join(cells) + " |")
    if len(complete) >= 10:
        try:
            importance = optuna.importance.get_param_importances(study)
            lines += ["", "## Parameter importance (fANOVA)", "", "| parameter | importance |", "| --- | ---: |"]
            lines += [f"| {k} | {v:.3f} |" for k, v in importance.items()]
        except Exception as error:  # importance needs enough variation
            lines += ["", f"(importance unavailable: {error})"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        values = [t.value for t in sorted(complete, key=lambda t: t.number)]
        numbers = [t.number for t in sorted(complete, key=lambda t: t.number)]
        best = [min(values[: i + 1]) for i in range(len(values))]
        ax.scatter(numbers, values, s=10, alpha=0.6, label="complete trial")
        ax.plot(numbers, best, color="C3", label="best so far")
        ax.set_xlabel("trial"), ax.set_ylabel("val agent ADE (m)"), ax.grid(alpha=0.3), ax.legend()
        ax.set_title(study.study_name)
        fig.tight_layout()
        fig.savefig(args.output.with_suffix(".png"), dpi=130)
    except Exception:
        pass
    print(args.output)


if __name__ == "__main__":
    main()
