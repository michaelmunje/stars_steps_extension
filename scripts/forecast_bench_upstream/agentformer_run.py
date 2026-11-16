"""Run the pinned 2021 AgentFormer test.py under modern PyTorch (alias shims only, as in src/baselines/agentformer.py)."""
import builtins, runpy, sys
from typing import Optional, Tuple
import numpy as np, torch, torch.nn.modules.linear as L
for _n, _t in (("int", int), ("float", float), ("bool", bool), ("object", object)):
    if not hasattr(np, _n): setattr(np, _n, _t)
if not hasattr(L, "_LinearWithBias"): L._LinearWithBias = L.Linear
builtins.Tensor = torch.Tensor; builtins.Optional = Optional; builtins.Tuple = Tuple
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{"weights_only": False, **k})  # 2021 pickles (checkpoint dicts)
import pathlib, types
script = sys.argv[1]; sys.argv = sys.argv[1:]; ref = pathlib.Path(script).resolve().parent
sys.path.insert(0, str(ref))
for name in ("data", "model", "utils"):  # the benchmark venv's editable install also provides a top-level `data`
    pkg = types.ModuleType(name); pkg.__path__ = [str(ref / name)]; sys.modules[name] = pkg
runpy.run_path(script, run_name="__main__")
