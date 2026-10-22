from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentSpec:
    """One transfer-experiment configuration — and the hashed identity of its run.

    An ``ExperimentSpec`` instance is not just a record of settings: it is fed directly to
    ``stable_config_hash`` (see ``config.py``) to produce ``run_id``. ``stable_config_hash``
    calls ``dataclasses.asdict()``, which serializes fields **by name**, not by declaration
    position. So changing the *order* in which fields are declared here changes nothing about
    how existing (positional) call sites construct an instance, but it silently changes which
    value each field name is bound to when there are same-typed fields — and therefore changes
    ``run_id`` for what is otherwise the identical experiment. All eight constructor arguments
    are positional at the single call site (``src/paper/generate_tables.py``), and this
    dataclass has two adjacent floats (``steps_fraction``, ``target_fraction``) and two adjacent
    ints (``data_seed``, ``model_seed``), so a transposition there is both easy to make and
    silent to a positional caller — only ``run_id`` would change.

    Fields:
        model: Model name for this run (e.g. ``"constant_velocity"``).
        target_dataset: Name of the target dataset being evaluated/trained on.
        train_datasets: Dataset name(s) actually used to build the training set.
        pretrain_source: Name of the source dataset used for pretraining, or ``None`` if this
            run has no pretraining stage. ``None`` is rendered as the string ``"none"`` in the
            published results row (see ``src/paper/generate_tables.py:103``).
        steps_fraction: Reserved and currently unexercised. It is threaded through the results
            schema (``results.py``) and this dataclass, but has no consumer anywhere in the
            training or aggregation code (no training script reads it as a fraction argument,
            and ``aggregate_transfer.py`` groups results by ``target_fraction`` only). Today it
            is always constructed as ``0.0`` from the single call site. Its precise intended
            semantics — e.g. whether it would eventually gate a fraction of STEPS-source data —
            are not established in code; do not assume a meaning from the name alone.
        target_fraction: The fraction of the **target** dataset used for downstream/finetune
            training. Constrained to ``(0, 1]`` at the argument-parsing call sites
            (``src/baselines/train_moflow.py:76`` and ``src/baselines/train_social_vae.py:284``).
        data_seed: Seed controlling data split/sampling.
        model_seed: Seed controlling model initialization/training stochasticity.
    """

    model: str
    target_dataset: str
    train_datasets: tuple[str, ...]
    pretrain_source: str | None
    steps_fraction: float
    target_fraction: float
    data_seed: int
    model_seed: int
