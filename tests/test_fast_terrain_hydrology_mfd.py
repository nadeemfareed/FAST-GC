import numpy as np

from fastgc.hydrology import (
    mfd_contributing_area,
    mfd_flow_accumulation,
    mfd_flow_weights,
    mfd_hydrology,
    mfd_specific_catchment_area,
)


def test_mfd_weights_sum_to_one_when_downslope_exists():
    z = np.array(
        [
            [9, 8, 7],
            [8, 5, 4],
            [7, 4, 1],
        ],
        dtype=float,
    )

    w = mfd_flow_weights(
        z, 1.0, 1.0
    )

    assert np.isclose(
        np.sum(w[1, 1]),
        1.0,
    )


def test_mfd_outlet_has_zero_weights():
    z = np.array(
        [
            [3, 2],
            [2, 1],
        ],
        dtype=float,
    )

    w = mfd_flow_weights(
        z, 1.0, 1.0
    )

    assert np.sum(w[1, 1]) == 0.0


def test_single_channel_mfd_equals_deterministic_accumulation():
    z = np.array(
        [[5, 4, 3, 2, 1]],
        dtype=float,
    )

    acc = mfd_flow_accumulation(
        z, 1.0, 1.0
    )

    np.testing.assert_allclose(
        acc,
        [[1, 2, 3, 4, 5]],
    )


def test_mfd_partition_is_mass_conserving_per_cell():
    z = np.array(
        [
            [10, 9, 8],
            [ 9, 7, 5],
            [ 8, 5, 1],
        ],
        dtype=float,
    )

    w = mfd_flow_weights(
        z, 1.0, 1.0
    )

    sums = np.sum(w, axis=2)

    has_flow = sums > 0.0

    np.testing.assert_allclose(
        sums[has_flow],
        1.0,
    )


def test_mfd_area_uses_cell_area():
    z = np.array(
        [[3, 2, 1]],
        dtype=float,
    )

    acc = mfd_flow_accumulation(
        z, 2.0, 3.0
    )

    area = mfd_contributing_area(
        z,
        2.0,
        3.0,
        accumulation=acc,
    )

    np.testing.assert_allclose(
        area,
        acc * 6.0,
    )


def test_mfd_specific_area_units():
    z = np.array(
        [[3, 2, 1]],
        dtype=float,
    )

    acc = mfd_flow_accumulation(
        z, 2.0, 2.0
    )

    sca = mfd_specific_catchment_area(
        z,
        2.0,
        2.0,
        accumulation=acc,
    )

    np.testing.assert_allclose(
        sca,
        acc * 2.0,
    )


def test_mfd_nodata_preserved():
    z = np.array(
        [
            [3, 2, 1],
            [3, np.nan, 1],
        ],
        dtype=float,
    )

    products = mfd_hydrology(
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


def test_mfd_exponent_changes_partition_not_total_weight():
    z = np.array(
        [
            [10, 9, 8],
            [ 9, 7, 5],
            [ 8, 5, 1],
        ],
        dtype=float,
    )

    a = mfd_flow_weights(
        z, 1.0, 1.0, exponent=1.0
    )

    b = mfd_flow_weights(
        z, 1.0, 1.0, exponent=2.0
    )

    assert not np.allclose(
        a[1, 1],
        b[1, 1],
    )

    assert np.isclose(
        np.sum(a[1, 1]),
        1.0,
    )

    assert np.isclose(
        np.sum(b[1, 1]),
        1.0,
    )
