"""Run the official ETH/UCY test evaluations of SocialVAE and MoFlow (pinned submodules, official checkpoints).

    python -m forecast_bench.official_eval socialvae --fold eth [--no-fpc] [--device cpu]
    python -m forecast_bench.official_eval moflow --fold eth          # needs CUDA (the trainer times with CUDA events)

SocialVAE: ``references/social-vae/main.py --test data/<fold>/test --ckpt models/<fold>`` (README command). Without
``--no-fpc`` it uses the FPC rate stored in the checkpoint. ``torch.utils.tensorboard`` is stubbed (only used for
training; tensorboard is not in our venv).
MoFlow: ``references/moflow/eval_eth.py`` with the README teacher command, minus ``--eval_on_train --save_samples``
(those evaluate on the *training* split to dump IMLE distillation targets). Checkpoints from
https://huggingface.co/fyxfelixfu/moflow, laid out as ``<moflow-ckpts>/<fold>/{cor_fm.yml,models/checkpoint_best.pt}``.
"""
from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOCIALVAE = ROOT / "references/social-vae"
MOFLOW = ROOT / "references/moflow"
MOFLOW_CKPTS = Path(os.environ.get("MOFLOW_CKPTS", "/scratch/09865/mmunje/.cache/moflow_ckpts/eth_ucy/moflow"))
SVAE_FOLD = {"zara1": "zara01", "zara2": "zara02"}


def socialvae(fold: str, no_fpc: bool, device: str | None) -> None:
    name = SVAE_FOLD.get(fold, fold)
    os.chdir(SOCIALVAE)
    sys.path.insert(0, str(SOCIALVAE))
    stub = types.ModuleType("torch.utils.tensorboard")
    stub.SummaryWriter = None
    sys.modules["torch.utils.tensorboard"] = stub
    sys.argv = ["main.py", "--test", f"data/{name}/test", "--ckpt", f"models/{name}", "--config", f"config/{name}.py"]
    sys.argv += ["--no-fpc"] if no_fpc else []
    sys.argv += ["--device", device] if device else []
    # exec instead of runpy so spawned data-loading workers do not re-run main.py as their __main__
    exec(compile(Path("main.py").read_text(), "main.py", "exec"), {"__name__": "__main__"})


def moflow(fold: str) -> None:
    ckpt = MOFLOW_CKPTS / fold
    sys.argv = ["eval_eth.py", "--ckpt_path", str(ckpt / "models/checkpoint_best.pt"), "--cfg", str(ckpt / "cor_fm.yml"),
                "--subset", fold, "--rotate", "--rotate_time_frame", "6", "--batch_size", "1000",
                "--sampling_steps", "100", "--solver", "lin_poly", "--lin_poly_p", "5", "--lin_poly_long_step", "1000"]
    os.chdir(MOFLOW)
    # MoFlow's data/, models/, utils/ are namespace packages, which lose to our src/data.py and src/models.py
    for name in ("data", "models", "utils"):
        package = types.ModuleType(name)
        package.__path__ = [str(MOFLOW / name)]
        sys.modules[name] = package
    sys.path.insert(0, str(MOFLOW))
    exec(compile(Path("eval_eth.py").read_text(), "eval_eth.py", "exec"), {"__name__": "__main__"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=["socialvae", "moflow"])
    parser.add_argument("--fold", required=True, choices=["eth", "hotel", "univ", "zara1", "zara2"])
    parser.add_argument("--no-fpc", action="store_true")
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.model == "socialvae":
        socialvae(args.fold, args.no_fpc, args.device)
    else:
        moflow(args.fold)
