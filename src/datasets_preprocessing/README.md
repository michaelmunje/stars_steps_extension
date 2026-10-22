# Dataset preprocessing

This committed directory contains the lightweight, reproducible definition of each
dataset adapter. Every dataset folder contains:

- `source.json`: raw field and unit mapping
- `README.md`: conversion and calibration notes
- `convert.py`: raw-to-canonical converter
- `splits.json`: a per-dataset **split override**, read only by `--split-strategy temporal` and
  `--split-strategy held_out_recording`. Runtime manifests are generated under `datasets/`.

  The reader is strict and strategy-specific:

  | requested strategy | keys | `ratios` | unknown keys |
  | --- | --- | --- | --- |
  | `temporal` | `strategy`, `ratios`, `comment` | **required**, exactly `train`/`val`/`test`, positive, summing to 1 | **rejected** |
  | `held_out_recording` | `strategy`, `held_out_recording`, optional `ratios` | optional, defaults to `{"train": 0.9, "val": 0.1}`; exactly `train`/`val` — **no `test` key** | **ignored** (no allowlist) |
  | `grouped_scene_sequence`, `leave_one_scene_out` | *file not read at all* | — | — |

  The two readers are **not symmetric**, which is easy to get wrong: `temporal` enforces an allowlist and
  raises `unknown split override key(s) [...]` — a check that runs **before** the strategy check — while
  `held_out_recording` has no allowlist and silently ignores extra keys. Their `ratios` shapes differ too:
  three-way for `temporal`, two-way for `held_out_recording`, because the held-out recording *is* the test
  split and the ratios divide only what remains.

  Consequences worth knowing before editing one of these files:

  - Under the **grouped** and **LOSO** strategies the file is never opened. The ratios declared in
    `{atc,eth,madras,PTDPES,ucy}/splits.json` are therefore **not authoritative** — the grouped path
    uses the hardcoded `GROUPED_DEFAULT_RATIOS` in `src/split_manifests.py`, and editing those
    declarations changes nothing.
  - Those same five files carry `source` / `note` / `seed` keys that the reader rejects, so
    `--split-strategy temporal` on any of them fails with `unknown split override key(s)` rather than
    running. `grand_central/splits.json` is the one file that satisfies the schema, and its ratios are
    live (see the `comment` field in it for the decision behind them).
  - A local `datasets/<name>/splits.json` takes precedence over the copy here (`dataset_paths.metadata_path`),
    which is where `atc_subset`'s lives.
  - For `held_out_recording` the file is **required**, not optional: it names the recording that becomes
    the test split.

Large and generated files live separately under the gitignored `datasets/<name>/`
tree:

- `raw/`: manually downloaded source data
- `converted/`: files matching the canonical schema defined in
  [`src/canonical.py`](../canonical.py)
- `indices/`: preprocessed scene indices and category caches
- `analysis/`: non-destructive OpenTraj smoothing views, trajlet indices, and reports
- `split_manifests/`: deterministic JSON/CSV/Markdown grouped assignments and lock files

Converters take the output dataset directory as their first argument. Their defaults
already point to the corresponding directory under `datasets/`. Runtime code reads
local generated metadata first and falls back to the committed metadata templates here.
There is only one output schema; these adapters describe how each source reaches it.
Canonical output is metric world coordinates at 2.5 Hz. OpenTraj analysis smoothing
and leakage-safe forecasting preprocessing are derived after conversion and never
replace canonical `x, y`.
