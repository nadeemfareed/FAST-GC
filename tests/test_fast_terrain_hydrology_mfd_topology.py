import numpy as np

from fastgc.hydrology import (
    d8_flow_direction_resolved,
    mfd_flow_accumulation,
    mfd_flow_weights,
)


def test_flat_domain_mass_reaches_boundary_outlets():
    dem = np.full((9, 9), 10.0)

    weights = mfd_flow_weights(
        dem,
        1.0,
        1.0,
    )

    acc = mfd_flow_accumulation(
        dem,
        1.0,
        1.0,
        weights=weights,
    )

    _, receiver = d8_flow_direction_resolved(
        dem,
        1.0,
        1.0,
    )

    outlets = receiver < 0

    # A flat domain legitimately retains multiple genuine
    # boundary outlets. Conservation is across the complete outlet set.
    assert np.count_nonzero(outlets) > 1

    assert np.isclose(
        np.sum(acc.ravel()[outlets]),
        float(dem.size),
        rtol=0.0,
        atol=1e-10,
    )


def test_flat_accumulation_conserves_mass_after_rotation():
    dem = np.full((9, 9), 10.0)

    for candidate in (
        dem,
        np.rot90(dem),
        np.flipud(dem),
        np.fliplr(dem),
    ):
        weights = mfd_flow_weights(
            candidate,
            1.0,
            1.0,
        )

        acc = mfd_flow_accumulation(
            candidate,
            1.0,
            1.0,
            weights=weights,
        )

        outgoing = np.sum(weights, axis=2)
        outlets = np.isclose(outgoing, 0.0)

        assert np.isclose(
            np.sum(acc[outlets]),
            float(candidate.size),
            rtol=0.0,
            atol=1e-10,
        )


def test_mfd_accumulation_conserves_mass_at_all_outlets():
    dem = np.array(
        [
            [12., 12., 12., 12., 11.],
            [12., 10., 10., 10., 11.],
            [12., 10., 10., 10., 10.],
            [12., 10., 10., 10., 11.],
            [12., 12., 12., 12., 11.],
        ]
    )

    weights = mfd_flow_weights(
        dem,
        1.0,
        1.0,
    )

    acc = mfd_flow_accumulation(
        dem,
        1.0,
        1.0,
        weights=weights,
    )

    outgoing = np.sum(weights, axis=2)
    outlets = np.isclose(outgoing, 0.0)

    assert np.isclose(
        np.sum(acc[outlets]),
        float(np.count_nonzero(np.isfinite(dem))),
        rtol=0.0,
        atol=1e-10,
    )


def test_mfd_accumulation_is_finite_and_never_below_self():
    dem = np.array(
        [
            [9., 9., 9., 8., 7.],
            [9., 5., 5., 5., 7.],
            [9., 5., 5., 5., 6.],
            [9., 5., 5., 5., 6.],
            [9., 9., 9., 8., 5.],
        ]
    )

    acc = mfd_flow_accumulation(
        dem,
        1.0,
        1.0,
    )

    valid = np.isfinite(dem)

    assert np.all(np.isfinite(acc[valid]))
    assert np.all(acc[valid] >= 1.0)


def test_mfd_mass_conservation_with_nodata_domain():
    dem = np.array(
        [
            [np.nan, 9., 9., 8., np.nan],
            [9., 7., 7., 7., 6.],
            [9., 7., 7., 7., 6.],
            [9., 7., 7., 7., 6.],
            [np.nan, 9., 9., 8., np.nan],
        ]
    )

    weights = mfd_flow_weights(
        dem,
        1.0,
        1.0,
    )

    acc = mfd_flow_accumulation(
        dem,
        1.0,
        1.0,
        weights=weights,
    )

    valid = np.isfinite(dem)
    outgoing = np.sum(weights, axis=2)
    outlets = valid & np.isclose(outgoing, 0.0)

    assert np.isclose(
        np.sum(acc[outlets]),
        float(np.count_nonzero(valid)),
        rtol=0.0,
        atol=1e-10,
    )

    assert np.all(np.isnan(acc[~valid]))
