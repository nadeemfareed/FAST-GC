import numpy as np

from fastgc.hydrology import (
    condition_dem,
    depression_depth,
    priority_flood_fill,
)


def test_flat_surface_unchanged():
    z = np.full((9, 9), 100.0)

    filled = priority_flood_fill(z)

    np.testing.assert_allclose(
        filled,
        z,
    )


def test_single_cell_depression_filled():
    z = np.full((7, 7), 10.0)
    z[3, 3] = 2.0

    filled = priority_flood_fill(z)

    assert filled[3, 3] == 10.0


def test_nested_depression_fills_to_spill_elevation():
    z = np.array(
        [
            [10,10,10,10,10],
            [10, 8, 8, 8,10],
            [10, 8, 1, 8, 7],
            [10, 8, 8, 8,10],
            [10,10,10,10,10],
        ],
        dtype=float,
    )

    filled = priority_flood_fill(z)

    assert filled[2, 2] == 8.0
    assert filled[1, 1] == 8.0

    # Boundary outlet itself is never raised.
    assert filled[2, 4] == 7.0


def test_open_valley_is_not_filled():
    z = np.array(
        [
            [10,10, 1,10,10],
            [10, 8, 2, 8,10],
            [10, 7, 3, 7,10],
            [10, 8, 4, 8,10],
            [10,10, 5,10,10],
        ],
        dtype=float,
    )

    filled = priority_flood_fill(z)

    np.testing.assert_allclose(
        filled,
        z,
    )


def test_nodata_preserved_and_is_domain_boundary():
    z = np.full((7, 7), 10.0)

    z[3, 3] = 2.0
    z[3, 4] = np.nan

    filled = priority_flood_fill(z)

    assert np.isnan(filled[3, 4])

    # The low valid cell touches the outside-domain NaN,
    # therefore it already has an outlet and must not be raised.
    assert filled[3, 3] == 2.0


def test_depression_depth_exact():
    z = np.full((7, 7), 10.0)
    z[3, 3] = 2.0

    conditioned, depth = condition_dem(z)

    assert conditioned[3, 3] == 10.0
    assert depth[3, 3] == 8.0

    mask = np.ones_like(z, dtype=bool)
    mask[3, 3] = False

    assert np.all(depth[mask] == 0.0)


def test_conditioning_never_lowers_valid_terrain():
    rng = np.random.default_rng(42)

    z = rng.normal(
        100.0,
        10.0,
        size=(50, 60),
    )

    conditioned = priority_flood_fill(z)

    assert np.all(
        conditioned >= z - 1.0e-12
    )


def test_input_is_not_modified():
    z = np.full((7, 7), 10.0)
    z[3, 3] = 1.0

    original = z.copy()

    priority_flood_fill(z)

    np.testing.assert_array_equal(
        z,
        original,
    )


def test_depression_depth_nonnegative():
    rng = np.random.default_rng(7)

    z = rng.normal(
        size=(30, 30)
    )

    depth = depression_depth(z)

    valid = np.isfinite(depth)

    assert np.all(
        depth[valid] >= 0.0
    )
