"""Guard against divergence between forecasting_config's mode tuples and
preprocessing_views' independent Literal declarations of the same members.

forecasting_config.INPUT_MODES/TARGET_MODES are the single source of truth for
_validate and CLI choices=, but preprocessing_views.py separately declares
ForecastInputPreprocessing/ForecastTargetPreprocessing as Literal types with the
same members. A Literal cannot be statically derived from a tuple[str, ...], so
the two definitions are kept in sync by hand; this test is what makes that durable.
"""
from typing import get_args

import pytest

import preprocessing_views
from forecasting_config import INPUT_MODES, TARGET_MODES


@pytest.mark.parametrize(
    "modes, literal",
    [
        (INPUT_MODES, preprocessing_views.ForecastInputPreprocessing),
        (TARGET_MODES, preprocessing_views.ForecastTargetPreprocessing),
    ],
    ids=["input", "target"],
)
def test_modes_match_preprocessing_views_literal(modes, literal):
    assert set(modes) == set(get_args(literal))
