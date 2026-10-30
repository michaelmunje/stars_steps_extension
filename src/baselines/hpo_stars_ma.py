"""Optuna hyperparameter search for the multi-agent STARS forecaster.

Every worker process runs trials on one GPU and shares the study through an Optuna journal file,
so a node runs four workers (``scripts/horizon/gpu_jobs.slurm`` with four copies of the same
command) and more nodes simply add workers. A trial trains ``--trial-epochs`` epochs with the
sampled hyperparameters (``train_stars_ma.train``), reports the validation selection metric
(macro agent ADE) every epoch and is pruned by a median rule when it falls behind earlier
trials. Results: ``<study dir>/trials.csv``, ``best.yaml`` (flags to rerun the best trial at full
length) and one run directory per trial.

    python -m baselines.hpo_stars_ma --config configs/stars_ma/steps.yaml --study steps_cvae \\
        --decoder cvae --encoder-checkpoint experiments/stars_ma/runs/ssl_steps/checkpoints/best.pt \\
        --n-trials 25 --trial-epochs 30
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import optuna
import torch
import yaml

from baselines import train_stars_ma as T


def search_space(trial: optuna.Trial, options) -> list[str]:
    """Trainer flags for one trial."""
    flags = [
        "--learning-rate", f"{trial.suggest_float('learning_rate', 1e-4, 3e-3, log=True):.3g}",
        "--max-nodes", str(trial.suggest_categorical("max_nodes", [4096, 8192, 16384])),
        "--weight-decay", f"{trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True):.3g}",
        "--decoder-hidden-dim", str(trial.suggest_categorical("decoder_hidden_dim", [128, 256, 512])),
        "--decoder-dropout", f"{trial.suggest_float('decoder_dropout', 0.0, 0.3):.3f}",
        "--representation", trial.suggest_categorical("representation", ["mu", "hidden"]),
        "--warmup-epochs", f"{trial.suggest_float('warmup_epochs', 0.5, 5.0):.2f}",
    ]
    decoder = options.decoder if options.decoder != "search" else trial.suggest_categorical("decoder", ["deterministic", "cvae"])
    flags += ["--decoder", decoder]
    if decoder == "cvae":
        flags += ["--kl-weight", f"{trial.suggest_float('kl_weight', 0.05, 3.0, log=True):.3g}",
                  "--cvae-latent-dim", str(trial.suggest_categorical("cvae_latent_dim", [8, 16, 32, 64]))]
    if options.encoder_checkpoint:
        mode = trial.suggest_categorical("encoder_mode", ["frozen", "finetune"])
        flags += ["--encoder-mode", mode, "--encoder-checkpoint", str(options.encoder_checkpoint)]
        if mode == "finetune":
            flags += ["--encoder-lr-ratio", f"{trial.suggest_float('encoder_lr_ratio', 0.01, 1.0, log=True):.3g}"]
    else:
        flags += ["--encoder-mode", "scratch",
                  "--hidden-dim", str(trial.suggest_categorical("hidden_dim", [128, 256])),
                  "--dropout", f"{trial.suggest_float('dropout', 0.0, 0.3):.3f}",
                  "--knn", str(trial.suggest_categorical("knn", [8, 16, 24]))]
    return flags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="Dataset config (configs/stars_ma/*.yaml)")
    parser.add_argument("--study", required=True, help="Study name; shared by every worker")
    parser.add_argument("--study-dir", type=Path, default=Path("experiments/stars_ma/hpo"))
    parser.add_argument("--decoder", choices=("deterministic", "cvae", "search"), default="cvae")
    parser.add_argument("--encoder-checkpoint", type=Path, help="Phase-1 checkpoint (search frozen vs finetune)")
    parser.add_argument("--n-trials", type=int, default=20, help="Trials run by this worker")
    parser.add_argument("--trial-epochs", type=int, default=30)
    parser.add_argument("--timeout", type=float, help="Seconds this worker may run")
    parser.add_argument("--samples", type=int, default=6, help="K during validation (cheaper than 20)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="More trainer flags for every trial")
    options = parser.parse_args()

    root = options.study_dir / options.study
    root.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(root / "journal.log"),
                                                   lock_obj=optuna.storages.journal.JournalFileOpenLock(str(root / "journal.log"))))
    worker = int(os.environ.get("SLURM_PROCID", "0")) * 100 + int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or 0)
    sampler = optuna.samplers.TPESampler(seed=options.seed + worker, multivariate=True, n_startup_trials=8)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=5)
    study = optuna.create_study(study_name=options.study, storage=storage, direction="minimize", sampler=sampler,
                                pruner=pruner, load_if_exists=True)

    def objective(trial: optuna.Trial) -> float:
        flags = search_space(trial, options)
        run_dir = root / f"trial_{trial.number:04d}"
        argv = ["--config", str(options.config), "--phase", "forecast", "--run-dir", str(run_dir),
                "--epochs", str(options.trial_epochs), "--samples", str(options.samples), "--patience", "0",
                *flags, *options.extra]
        args = T.parse_args(argv)
        T.setup_distributed(args)

        def report(epoch, selection, metrics):
            trial.report(selection, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned at epoch {epoch} (selection {selection:.4f})")

        args.epoch_callback = report
        trial.set_user_attr("run_dir", str(run_dir))
        trial.set_user_attr("flags", flags)
        started = time.time()
        try:
            result = T.train(args)
        finally:
            trial.set_user_attr("seconds", time.time() - started)
            torch.cuda.empty_cache()
        return float(result["best"])

    study.optimize(objective, n_trials=options.n_trials, timeout=options.timeout, catch=(RuntimeError,),
                   gc_after_trial=True)
    frame = study.trials_dataframe()
    frame.to_csv(root / "trials.csv", index=False)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if complete:
        best = min(complete, key=lambda t: t.value)
        (root / "best.yaml").write_text(yaml.safe_dump({
            "trial": best.number, "value": best.value, "params": best.params,
            "flags": best.user_attrs.get("flags"), "run_dir": best.user_attrs.get("run_dir")}, sort_keys=False))
        print(f"best trial {best.number}: {best.value:.4f} {best.params}")


if __name__ == "__main__":
    main()
