from __future__ import annotations

from pathlib import Path
from config import DatasetSpec

ROOT=Path(__file__).resolve().parents[2]
FOLDERS={
    "ptdpes":"PTDPES",
    "atc":"atc",
    "steps":"steps",
}


def canonical_pretrain_name(name:str)->str:
    return name.lower()


def pretrain_spec(name:str,fraction:float=1.0)->DatasetSpec:
    key=canonical_pretrain_name(name)
    if key not in FOLDERS:raise ValueError(f"Unsupported pretraining dataset {name!r}; choose {sorted(FOLDERS)}")
    path=ROOT/"datasets"/FOLDERS[key]
    if not path.exists():raise FileNotFoundError(f"Pretraining dataset {key!r} is not available yet: {path}")
    if not 0.0<fraction<=1.0:raise ValueError("pretraining fraction must be in (0, 1]")
    return DatasetSpec(key,path,fraction=fraction)


def parse_pretrain_names(single: str | None, multiple: list[str] | None) -> tuple[str, ...]:
    """Resolve the legacy singular option and the new multi-source option."""
    if multiple:
        values = [item for value in multiple for item in value.split(",")]
    elif single:
        values = [single]
    else:
        values = ["ptdpes"]
    names = tuple(dict.fromkeys(canonical_pretrain_name(value.strip()) for value in values if value.strip()))
    if not names:
        raise ValueError("At least one pretraining dataset is required")
    return names


def pretrain_specs(names: tuple[str, ...] | list[str], fraction: float = 1.0) -> tuple[DatasetSpec, ...]:
    return tuple(pretrain_spec(name, fraction) for name in names)
