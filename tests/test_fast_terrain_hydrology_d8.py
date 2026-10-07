import numpy as np

from fastgc.hydrology import (
    condition_dem,
    d8_contributing_area,
    d8_flow_accumulation,
    d8_flow_direction,
    d8_hydrology,
    d8_specific_catchment_area,
)


def test_d8_simple_east_flow():
    z = np.tile(
        np.array([5, 4, 3, 2, 1], dtype=float),
        (3, 1),
    )

    direction, receiver = d8_flow_direction(
        z, 1.0, 1.0
    )

    # Interior cells flow east.
    assert direction[1, 0] == 1
    assert direction[1, 1] == 1
    assert direction[1, 2] == 1
    assert direction[1, 3] == 1

    # Eastern edge is outlet.
    assert direction[1, 4] == 0
    assert receiver[1 * 5 + 4] == -1


def test_d8_uses_distance_weighted_steepest_descent():
    z = np.array(
        [
            [20, 20, 20],
            [20, 10,  8],
            [20,  7,  6],
        ],
        dtype=float,
    )

    direction, _ = d8_flow_direction(
        z, 1.0, 1.0
    )

    # From center:
    # E drop=2 / 1
    # S drop=3 / 1
    # SE drop=4 / sqrt(2) ~=2.828
    # Therefore south is steepest.
    assert direction[1, 1] == 4


def test_accumulation_single_channel():
    z = np.array(
        [[5, 4, 3, 2, 1]],
        dtype=float,
    )

    acc = d8_flow_accumulation(
        z, 1.0, 1.0
    )

    np.testing.assert_allclose(
        acc,
        [[1, 2, 3, 4, 5]],
    )


def test_accumulation_conserves_source_cells_at_outlets():
    z = np.array(
        [
            [9, 8, 7],
            [8, 5, 4],
            [7, 4, 1],
        ],
        dtype=float,
    )

    direction, receiver = d8_flow_direction(
        z, 1.0, 1.0
    )

    acc = d8_flow_accumulation(
        z,
        1.0,
        1.0,
        receiver=receiver,
    )

    outlets = (
        np.isfinite(z)
        & (direction == 0)
    )

    assert np.isclose(
        np.nansum(acc[outlets]),
        np.count_nonzero(np.isfinite(z)),
    )


def test_contributing_area_uses_physical_cell_area():
    z = np.array(
        [[5, 4, 3, 2, 1]],
        dtype=float,
    )

    acc = d8_flow_accumulation(
        z, 2.0, 3.0
    )

    area = d8_contributing_area(
        z,
        2.0,
        3.0,
        accumulation=acc,
    )

    np.testing.assert_allclose(
        area,
        acc * 6.0,
    )


def test_specific_catchment_area_units():
    z = np.array(
        [[5, 4, 3]],
        dtype=float,
    )

    acc = d8_flow_accumulation(
        z, 2.0, 2.0
    )

    sca = d8_specific_catchment_area(
        z,
        2.0,
        2.0,
        accumulation=acc,
    )

    # cell area=4 m2, representative width=2 m.
    np.testing.assert_allclose(
        sca,
        acc * 2.0,
    )


def test_nodata_preserved():
    z = np.array(
        [
            [5, 4, 3],
            [5, np.nan, 2],
            [5, 4, 1],
        ],
        dtype=float,
    )

    products = d8_hydrology(
        z, 1.0, 1.0
    )

    assert np.isnan(
        products["flow_accumulation"][1, 1]
    )

    assert np.isnan(
        products["contributing_area"][1, 1]
    )

    assert np.isnan(
        products["specific_catchment_area"][1, 1]
    )


def test_conditioned_dem_can_feed_d8_pipeline():
    z = np.full(
        (7, 7),
        10.0,
        dtype=float,
    )

    z[3, 3] = 1.0

    conditioned, depth = condition_dem(z)

    products = d8_hydrology(
        conditioned,
        1.0,
        1.0,
    )

    assert np.nanmin(depth) >= 0.0
    assert products["flow_direction"].shape == z.shape
    assert products["flow_accumulation"].shape == z.shape
    assert products["contributing_area"].shape == z.shape
    assert products["specific_catchment_area"].shape == z.shape
