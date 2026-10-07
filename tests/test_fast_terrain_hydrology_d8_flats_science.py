import numpy as np

from fastgc.hydrology import (
    condition_dem,
    d8_flow_direction,
    d8_flow_direction_resolved,
    resolve_d8_flats,
)


def assert_acyclic(receiver, valid):
    rec = np.asarray(receiver).reshape(-1)
    valid = np.asarray(valid).reshape(-1)

    for start in np.flatnonzero(valid):
        seen = set()
        i = int(start)

        while i >= 0:
            assert i not in seen
            seen.add(i)

            j = int(rec[i])

            if j < 0:
                break

            assert valid[j]
            i = j


def test_priority_flood_plateau_routes_without_cycles():
    raw = np.array(
        [
            [9., 9., 9., 9., 9.],
            [9., 4., 4., 4., 9.],
            [9., 4., 1., 4., 5.],
            [9., 4., 4., 4., 9.],
            [9., 9., 9., 9., 9.],
        ]
    )

    conditioned, depth = condition_dem(raw)

    direction, receiver = (
        d8_flow_direction_resolved(
            conditioned,
            1.0,
            1.0,
        )
    )

    valid = np.isfinite(conditioned)

    assert_acyclic(receiver, valid)
    assert np.all(direction[valid] <= 128)
    assert np.nanmax(depth) > 0.0


def test_existing_strict_downslope_receivers_are_preserved():
    dem = np.array(
        [
            [5., 4., 3.],
            [5., 4., 3.],
            [5., 4., 3.],
        ]
    )

    _, original = d8_flow_direction(
        dem,
        1.0,
        1.0,
    )

    resolved = resolve_d8_flats(
        dem,
        original,
    )

    mask = original >= 0

    assert np.array_equal(
        resolved[mask],
        original[mask],
    )


def test_boundary_flat_retains_multiple_real_domain_outlets():
    dem = np.full((7, 7), 10.0)

    _, receiver = d8_flow_direction_resolved(
        dem,
        1.0,
        1.0,
    )

    receiver = receiver.reshape(dem.shape)

    boundary = np.zeros(
        dem.shape,
        dtype=bool,
    )

    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True

    # Every boundary cell is a legitimate outlet under the
    # current open-domain convention.
    assert np.all(receiver[boundary] < 0)

    assert_acyclic(
        receiver,
        np.isfinite(dem),
    )


def test_nodata_boundary_is_hydrologic_domain_edge():
    dem = np.array(
        [
            [9., 9., 9., 9., 9.],
            [9., 5., 5., 5., 9.],
            [9., 5., np.nan, 5., 9.],
            [9., 5., 5., 5., 9.],
            [9., 9., 9., 9., 9.],
        ]
    )

    conditioned, _ = condition_dem(dem)

    assert np.isnan(conditioned[2, 2])

    # Cells bordering NoData must not be raised merely to close
    # that boundary opening.
    assert np.isclose(conditioned[1, 2], 5.0)
    assert np.isclose(conditioned[2, 1], 5.0)
    assert np.isclose(conditioned[2, 3], 5.0)
    assert np.isclose(conditioned[3, 2], 5.0)


def test_flat_resolution_is_deterministic():
    dem = np.array(
        [
            [12., 12., 12., 12., 11.],
            [12., 10., 10., 10., 11.],
            [12., 10., 10., 10., 10.],
            [12., 10., 10., 10., 11.],
            [12., 12., 12., 12., 11.],
        ]
    )

    results = []

    for _ in range(5):
        direction, receiver = (
            d8_flow_direction_resolved(
                dem,
                2.0,
                3.0,
            )
        )

        results.append(
            (
                direction.copy(),
                receiver.copy(),
            )
        )

    for direction, receiver in results[1:]:
        assert np.array_equal(
            direction,
            results[0][0],
        )

        assert np.array_equal(
            receiver,
            results[0][1],
        )


def test_resolved_receiver_is_always_d8_neighbor():
    dem = np.array(
        [
            [10., 10., 10., 10., 9.],
            [10.,  8.,  8.,  8., 9.],
            [10.,  8.,  8.,  8., 8.],
            [10.,  8.,  8.,  8., 9.],
            [10., 10., 10., 10., 9.],
        ]
    )

    _, receiver = d8_flow_direction_resolved(
        dem,
        2.0,
        5.0,
    )

    rows, cols = dem.shape

    for i, j in enumerate(receiver):
        if j < 0:
            continue

        r, c = divmod(i, cols)
        rr, cc = divmod(int(j), cols)

        assert abs(rr - r) <= 1
        assert abs(cc - c) <= 1
        assert (rr, cc) != (r, c)


def test_flat_routes_reach_outlet_or_existing_downslope_path():
    dem = np.array(
        [
            [20., 20., 20., 20., 20., 20., 20.],
            [20., 10., 10., 10., 10., 10., 20.],
            [20., 10., 10., 10., 10., 10., 20.],
            [20., 10., 10., 10., 10.,  9.,  8.],
            [20., 10., 10., 10., 10., 10., 20.],
            [20., 10., 10., 10., 10., 10., 20.],
            [20., 20., 20., 20., 20., 20., 20.],
        ]
    )

    _, receiver = d8_flow_direction_resolved(
        dem,
        1.0,
        1.0,
    )

    assert_acyclic(
        receiver,
        np.isfinite(dem),
    )

    # Every valid cell must eventually terminate.
    for start in range(dem.size):
        i = start

        for _ in range(dem.size + 1):
            j = int(receiver[i])

            if j < 0:
                break

            i = j
        else:
            raise AssertionError(
                "Resolved flat path did not terminate."
            )


def test_rectangular_pixels_do_not_break_flat_topology():
    dem = np.array(
        [
            [15., 15., 15., 15., 14.],
            [15., 10., 10., 10., 14.],
            [15., 10., 10., 10., 10.],
            [15., 10., 10., 10., 14.],
            [15., 15., 15., 15., 14.],
        ]
    )

    _, receiver = d8_flow_direction_resolved(
        dem,
        dx=2.0,
        dy=7.0,
    )

    assert_acyclic(
        receiver,
        np.isfinite(dem),
    )
