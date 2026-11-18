"""Run the official Social-GAN scripts/evaluate_model.py on CPU under modern PyTorch.

Shims only: AttrDict (the attrdict package is broken on Python >= 3.10), .cuda() -> no-op, safe
torch.load, and get_dset_path -> the identical SGAN-format files in references/social-vae/data.
"""
import os, runpy, sys, types
import torch

REPO = "/scratch/09865/mmunje/stars-forecast-benchmark/references"
sys.path.insert(0, f"{REPO}/social-gan")
attrdict = types.ModuleType("attrdict")
class AttrDict(dict):
    __getattr__ = dict.__getitem__
attrdict.AttrDict = AttrDict
sys.modules["attrdict"] = attrdict
torch.Tensor.cuda = lambda self, *a, **k: self.contiguous()  # old .cuda() copies were contiguous
torch.nn.Module.cuda = lambda self, *a, **k: self
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{"map_location": "cpu", "weights_only": True, **k})
import sgan.utils
sgan.utils.get_dset_path = lambda name, split: os.path.join(
    REPO, "social-vae/data", {"zara1": "zara01", "zara2": "zara02"}.get(name, name), split)
torch.manual_seed(int(os.environ.get("SEED", 0)))
sys.argv[0] = "evaluate_model.py"
runpy.run_path(f"{REPO}/social-gan/scripts/evaluate_model.py", run_name="__main__")
