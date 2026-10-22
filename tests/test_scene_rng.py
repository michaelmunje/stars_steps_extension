"""Coverage for ``categorization.scene_rng``, the per-scene RNG seeding primitive.

``scene_rng`` seeds from ``hashlib.sha256(f"{data_seed}:{scene_key}".encode()).digest()``
and is used for both the per-scene Kalman RNG and the per-tag acceptance draw
(``src/preprocess.py``), so it decides which scenes enter the published training sets.
It previously had zero test coverage: see ``.claude/revise/issues/P-004.md``.
"""

import numpy as np

from categorization import scene_rng


def test_scene_rng_is_deterministic():
    """Two calls with the same (data_seed, scene_key) must yield identical draws."""
    first = scene_rng(42, "example_scene").uniform()
    second = scene_rng(42, "example_scene").uniform()
    assert first == second


def test_scene_rng_pinned_literal():
    """Pin the exact first draw for a fixed (data_seed, scene_key) pair.

    This value *is* the reproducibility contract: published datasets depend on
    exactly this seeding path (``hashlib.sha256(...).digest()``, first 8 bytes
    little-endian -> seed -> ``np.random.default_rng(seed).uniform()``). It was
    computed independently by executing the production code path, not derived
    from it after the fact:

        digest = hashlib.sha256(b"42:example_scene").digest()
        seed = int.from_bytes(digest[:8], "little")  # == 16004830596320769433
        np.random.default_rng(seed).uniform()         # == 0.8929978832292822

    If this assertion ever fails, that means the seeding algorithm changed
    (e.g. a different hash function, a different digest slice, or a different
    byte order) -- do NOT "fix" this test by updating the literal. A failure
    here means the accepted-scene set for published datasets silently changed,
    which is exactly the regression this test exists to catch. Notably,
    substituting Python's builtin (per-process-randomised) ``hash()`` for
    ``hashlib.sha256`` fails this exact assertion while leaving every other
    collectible test in the suite green.
    """
    value = scene_rng(42, "example_scene").uniform()
    assert value == 0.8929978832292822


def test_scene_rng_independence():
    """A silently-ignored seed or key component must change the draw."""
    base = scene_rng(42, "example_scene").uniform()
    with_suffix = scene_rng(42, "example_scene:accept").uniform()
    other_seed = scene_rng(43, "example_scene").uniform()

    assert base != with_suffix
    assert base != other_seed
    assert with_suffix != other_seed
