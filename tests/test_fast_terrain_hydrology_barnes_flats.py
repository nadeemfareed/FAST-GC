import numpy as np

from fastgc.hydrology import (
    d8_flow_direction,
    d8_flow_direction_resolved,
    resolve_d8_flats,
)


def _assert_acyclic(receiver, valid):
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


def test_barnes_flat_drains_toward_real_lower_exit():
    dem = np.array(
        [
            [20.,20.,20.,20.,20.,20.,20.],
            [20.,10.,10.,10.,10.,10.,20.],
            [20.,10.,10.,10.,10.,10.,20.],
            [20.,10.,10.,10.,10., 9., 8.],
            [20.,10.,10.,10.,10.,10.,20.],
            [20.,10.,10.,10.,10.,10.,20.],
            [20.,20.,20.,20.,20.,20.,20.],
        ]
    )

    _, rec = d8_flow_direction_resolved(
        dem,
        1.0,
        1.0,
    )

    _assert_acyclic(
        rec,
        np.isfinite(dem),
    )

    # Centre of flat must obtain drainage.
    assert rec[3 * 7 + 3] >= 0


def test_barnes_preserves_existing_downslope_receivers():
    dem = np.array(
        [
            [5.,4.,3.],
            [5.,4.,3.],
            [5.,4.,3.],
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

    np.testing.assert_array_equal(
        resolved[mask],
        original[mask],
    )


def test_closed_interior_flat_does_not_manufacture_outlet():
    # Directly exercise resolve_d8_flats on an artificial closed
    # interior flat. This state should normally be removed by
    # hydrologic conditioning first.
    dem = np.array(
        [
            [20.,20.,20.,20.,20.],
            [20.,10.,10.,10.,20.],
            [20.,10.,10.,10.,20.],
            [20.,10.,10.,10.,20.],
            [20.,20.,20.,20.,20.],
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

    interior = [
        r * 5 + c
        for r in range(1, 4)
        for c in range(1, 4)
    ]

    assert all(
        resolved[i] < 0
        for i in interior
    )


def test_open_flat_domain_keeps_real_boundary_outlets():
    dem = np.full((7, 7), 10.0)

    _, rec = d8_flow_direction_resolved(
        dem,
        1.0,
        1.0,
    )

    rr = rec.reshape(dem.shape)

    boundary = np.zeros(
        dem.shape,
        dtype=bool,
    )

    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True

    assert np.all(rr[boundary] < 0)

    _assert_acyclic(
        rec,
        np.isfinite(dem),
    )


def test_barnes_flat_resolution_is_deterministic():
    dem = np.array(
        [
            [15.,15.,15.,15.,15.,14.],
            [15.,10.,10.,10.,10.,14.],
            [15.,10.,10.,10.,10.,10.],
            [15.,10.,10.,10.,10.,14.],
            [15.,15.,15.,15.,15.,14.],
        ]
    )

    outputs = []

    for _ in range(4):
        outputs.append(
            d8_flow_direction_resolved(
                dem,
                1.0,
                1.0,
            )[1]
        )

    for rec in outputs[1:]:
        np.testing.assert_array_equal(
            rec,
            outputs[0],
        )


def test_barnes_resolved_network_is_acyclic():
    dem = np.full((31, 31), 100.0)

    # Genuine lower boundary outlet.
    dem[15, -1] = 99.0

    _, rec = d8_flow_direction_resolved(
        dem,
        1.0,
        1.0,
    )

    _assert_acyclic(
        rec,
        np.isfinite(dem),
    )
