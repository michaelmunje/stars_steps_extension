# STEPS transfer results — minADE20 / minFDE20

Generated 2026-09-14T16:25:10+00:00

Learned rows are mean ± sample standard deviation over **3 seeds** (0, 1, 2), K=20.
126 downstream runs + 84 direct evaluations, 0 failures, before the STEPS-pretrained rows were removed. Target fraction 1.0 throughout.
Constant velocity is deterministic (K=1, no seed), so its minADE20 == ADE and minFDE20 == FDE.

**5-fold** = mean over the five ETH/UCY folds. **all-7** = mean over all seven scopes. Lower is better.

### minADE20

| condition | eth | hotel | univ | zara1 | zara2 | grand_central | madras | 5-fold | all-7 |
|---|---|---|---|---|---|---|---|---|---|
| Constant velocity (K=1) | 0.663 | 0.503 | 0.566 | 0.462 | 0.406 | 0.692 | 0.623 | **0.520** | **0.559** |
| SocialVAE scratch | 0.353 ± 0.002 | 0.234 ± 0.004 | 0.332 ± 0.003 | 0.289 ± 0.005 | 0.262 ± 0.006 | 0.310 ± 0.000 | 0.262 ± 0.026 | **0.294** | **0.292** |
| SocialVAE direct ← PTDPES | 0.268 ± 0.013 | 0.224 ± 0.016 | 0.305 ± 0.003 | 0.263 ± 0.005 | 0.233 ± 0.007 | 0.313 ± 0.006 | 0.246 ± 0.003 | **0.259** | **0.264** |
| SocialVAE fine-tuned ← PTDPES | 0.270 ± 0.006 | 0.205 ± 0.008 | 0.276 ± 0.003 | 0.231 ± 0.005 | 0.213 ± 0.001 | 0.277 ± 0.000 | 0.219 ± 0.004 | **0.239** | **0.242** |
| MoFlow scratch | 0.277 ± 0.011 | 0.208 ± 0.004 | 0.273 ± 0.012 | 0.251 ± 0.009 | 0.219 ± 0.007 | 0.284 ± 0.016 | 0.296 ± 0.016 | **0.246** | **0.258** |
| MoFlow direct ← PTDPES | 0.325 ± 0.077 | 0.225 ± 0.031 | 0.356 ± 0.055 | 0.312 ± 0.034 | 0.254 ± 0.036 | 0.320 ± 0.008 | 0.308 ± 0.043 | **0.294** | **0.300** |
| MoFlow fine-tuned ← PTDPES | 0.269 ± 0.009 | 0.218 ± 0.018 | 0.364 ± 0.025 | 0.281 ± 0.033 | 0.232 ± 0.017 | 0.323 ± 0.017 | 0.276 ± 0.005 | **0.273** | **0.280** |

### minFDE20

| condition | eth | hotel | univ | zara1 | zara2 | grand_central | madras | 5-fold | all-7 |
|---|---|---|---|---|---|---|---|---|---|
| Constant velocity (K=1) | 1.281 | 0.987 | 1.247 | 1.028 | 0.917 | 1.392 | 1.174 | **1.092** | **1.147** |
| SocialVAE scratch | 0.554 ± 0.007 | 0.392 ± 0.002 | 0.592 ± 0.002 | 0.562 ± 0.021 | 0.479 ± 0.014 | 0.507 ± 0.001 | 0.407 ± 0.046 | **0.516** | **0.499** |
| SocialVAE direct ← PTDPES | 0.409 ± 0.028 | 0.363 ± 0.046 | 0.542 ± 0.011 | 0.481 ± 0.022 | 0.425 ± 0.019 | 0.487 ± 0.010 | 0.348 ± 0.014 | **0.444** | **0.436** |
| SocialVAE fine-tuned ← PTDPES | 0.418 ± 0.004 | 0.327 ± 0.025 | 0.508 ± 0.003 | 0.422 ± 0.027 | 0.390 ± 0.010 | 0.443 ± 0.002 | 0.318 ± 0.007 | **0.413** | **0.404** |
| MoFlow scratch | 0.446 ± 0.025 | 0.347 ± 0.007 | 0.504 ± 0.028 | 0.441 ± 0.014 | 0.392 ± 0.015 | 0.458 ± 0.033 | 0.492 ± 0.033 | **0.426** | **0.440** |
| MoFlow direct ← PTDPES | 0.563 ± 0.181 | 0.390 ± 0.077 | 0.649 ± 0.114 | 0.570 ± 0.069 | 0.458 ± 0.091 | 0.537 ± 0.016 | 0.524 ± 0.092 | **0.526** | **0.527** |
| MoFlow fine-tuned ← PTDPES | 0.429 ± 0.012 | 0.350 ± 0.033 | 0.681 ± 0.066 | 0.511 ± 0.067 | 0.424 ± 0.039 | 0.549 ± 0.042 | 0.446 ± 0.008 | **0.479** | **0.484** |

### Notes

- **Constant velocity** follows Schöller et al. (2020): extrapolate the displacement over the final observed timestep. Computed on the *identical* test populations as the learned rows — same `downstream_test_dataset(target, fold, cfg)`, same world coordinate frame, same 8+12 window. Scene counts match the models' `n_predictions` exactly on all seven scopes (294 / 133 / 1550 / 177 / 373 / 19036 / 225). Note this is **not** `src/paper/generate_tables.py::evaluate_cv`, which evaluates each dataset's own grouped test split and would give a different ETH/UCY population than the five-fold LOSO protocol used here.
- `direct` = pretrained checkpoint evaluated on the target with **no** target training.
- `scratch` = random initialisation trained on the same target data with the **identical** optimisation recipe as fine-tuning; the control that isolates the value of pretraining.
- **madras, SocialVAE rows only**: `batch_size=32` instead of the recipe's 128 (SocialVAE OOMs on madras's up-to-506-agent scenes); recorded in each run's `optimization_hash`. Internally consistent but not strictly comparable to the other six scopes on optimisation budget.
- Pretraining corpora are capped to ~100k retained training windows by `n_target` in `configs/task.json` (PTDPES 101,413).
- STEPS-pretrained rows were removed; rerun them on the current `steps` data.
- MoFlow's pretrained checkpoints were all selected at **epoch 2 of 150**, the likely reason every MoFlow pretrained variant underperforms MoFlow scratch.

