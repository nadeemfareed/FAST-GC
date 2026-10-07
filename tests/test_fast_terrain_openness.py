import numpy as np
import pytest

from fastgc.terrain import (
    _openness_direction_offsets,
    compute_openness,
)


def test_openness_flat_is_90_degrees():
    z = np.zeros((101, 101), dtype=np.float64)

    p, n = compute_openness(
        z, 1.0, 1.0, 25.0
    )

    assert abs(float(p[50, 50]) - 90.0) < 1e-6
    assert abs(float(n[50, 50]) - 90.0) < 1e-6


def test_openness_hill_and_pit():
    yy, xx = np.mgrid[-50:51, -50:51]

    hill = -0.01 * (xx*xx + yy*yy)
    pit = +0.01 * (xx*xx + yy*yy)

    hp, hn = compute_openness(
        hill, 1.0, 1.0, 25.0
    )

    pp, pn = compute_openness(
        pit, 1.0, 1.0, 25.0
    )

    c = 50

    assert hp[c, c] > 90.0
    assert hn[c, c] < 90.0

    assert pp[c, c] < 90.0
    assert pn[c, c] > 90.0

    assert abs(
        float(hp[c, c]) - float(pn[c, c])
    ) < 1e-5

    assert abs(
        float(hn[c, c]) - float(pp[c, c])
    ) < 1e-5


def test_openness_exact_physical_radius_clip():
    offsets = _openness_direction_offsets(
        50.0,
        0.5,
        0.5,
        directions=8,
    )

    for direction in offsets:
        for dr, dc, distance in direction:
            assert distance <= 50.0 + 1e-12

    # 71 diagonal cells at 0.5 m are ~50.2046 m:
    # they must never enter a 50 m search radius.
    forbidden = {
        (-71, 71),
        (71, 71),
        (71, -71),
        (-71, -71),
    }

    all_cells = {
        (dr, dc)
        for direction in offsets
        for dr, dc, _ in direction
    }

    assert forbidden.isdisjoint(all_cells)


def test_openness_nodata_center_preserved():
    z = np.zeros((51, 51), dtype=np.float64)
    z[25, 25] = np.nan

    p, n = compute_openness(
        z, 1.0, 1.0, 10.0
    )

    assert np.isnan(p[25, 25])
    assert np.isnan(n[25, 25])


def test_openness_invalid_radius():
    z = np.zeros((11, 11), dtype=np.float64)

    with pytest.raises(ValueError):
        compute_openness(
            z, 1.0, 1.0, 0.0
        )
