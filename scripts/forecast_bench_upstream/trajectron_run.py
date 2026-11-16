"""Run a pinned 2020 Trajectron++ script under NumPy 2 / Python 3.12 (alias shims only).

OLD_GRADIENT=1 restores derivative_of as it was before upstream commit 03712b7 (np.gradient, i.e.
central differences), the preprocessing the pretrained ETH/UCY weights were trained and published with.
"""
import collections, collections.abc, os, runpy, sys
import numpy as np
if not hasattr(np, "float"): np.float = float
if not hasattr(collections, "Sequence"): collections.Sequence = collections.abc.Sequence
script = sys.argv[1]; sys.argv = sys.argv[1:]
if os.environ.get("OLD_GRADIENT") == "1":
    sys.path.insert(0, "/scratch/09865/mmunje/stars-forecast-benchmark/references/trajectron-plus-plus/trajectron")
    import environment, environment.data_utils
    def derivative_of(x, dt=1, radian=False):
        if radian:
            x = environment.data_utils.make_continuous_copy(x)
        if x[~np.isnan(x)].shape[-1] < 2:
            return np.zeros_like(x)
        dx = np.full_like(x, np.nan)
        dx[~np.isnan(x)] = np.gradient(x[~np.isnan(x)], dt)
        return dx
    environment.derivative_of = environment.data_utils.derivative_of = derivative_of
runpy.run_path(script, run_name="__main__")
