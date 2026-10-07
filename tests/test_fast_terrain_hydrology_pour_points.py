import numpy as np
import pytest

from fastgc.hydrology import (
    delineate_watershed_snapped,
    snap_pour_point,
)


def test_snap_selects_maximum_contributing_area():
    area = np.array(
        [
            [1., 2., 3.],
            [2., 5., 9.],
            [1., 4., 7.],
        ]
    )

    got = snap_pour_point(
        area,
        row=1,
        col=0,
        dx=1.0,
        dy=1.0,
        radius_m=2.0,
    )

    assert got == (1, 2)


def test_snap_uses_true_physical_radius():
    area = np.zeros((5, 5), dtype=float)
    area[2, 2] = 10.0
    area[3, 3] = 100.0

    # dx=3, dy=4:
    # diagonal distance = 5 m.
    got = snap_pour_point(
        area,
        row=2,
        col=2,
        dx=3.0,
        dy=4.0,
        radius_m=4.9,
    )

    assert got == (2, 2)


def test_snap_includes_candidate_exactly_on_radius():
    area = np.ones((3, 3), dtype=float)
    area[2, 2] = 50.0

    got = snap_pour_point(
        area,
        row=1,
        col=1,
        dx=3.0,
        dy=4.0,
        radius_m=5.0,
    )

    assert got == (2, 2)


def test_stream_mask_restricts_candidates():
    area = np.array(
        [[1., 100., 20.]]
    )

    stream = np.array(
        [[True, False, True]]
    )

    got = snap_pour_point(
        area,
        row=0,
        col=1,
        dx=1.0,
        dy=1.0,
        radius_m=2.0,
        stream_mask=stream,
    )

    assert got == (0, 2)


def test_tie_breaks_by_distance():
    area = np.array(
        [[20., 1., 20.]]
    )

    got = snap_pour_point(
        area,
        row=0,
        col=1,
        dx=2.0,
        dy=1.0,
        radius_m=3.0,
    )

    # Equal area/equal distance -> deterministic raster order.
    assert got == (0, 0)


def test_zero_radius_keeps_original_cell():
    area = np.array(
        [[1., 2., 100.]]
    )

    got = snap_pour_point(
        area,
        row=0,
        col=1,
        dx=1.0,
        dy=1.0,
        radius_m=0.0,
    )

    assert got == (0, 1)


def test_no_candidate_raises():
    area = np.full(
        (3, 3),
        np.nan,
    )

    with pytest.raises(
        ValueError,
        match="No valid",
    ):
        snap_pour_point(
            area,
            row=1,
            col=1,
            dx=1.0,
            dy=1.0,
            radius_m=2.0,
        )


def test_snapped_watershed_uses_snapped_cell():
    dem = np.ones((1, 5), dtype=float)

    # 0 -> 2 <- 1
    #      |
    #      v
    #      3 -> 4
    receiver = np.array(
        [2, 2, 3, 4, -1],
        dtype=np.int64,
    )

    area = np.array(
        [[1., 1., 3., 4., 5.]]
    )

    mask, snapped = delineate_watershed_snapped(
        dem,
        receiver,
        area,
        pour_row=0,
        pour_col=3,
        dx=1.0,
        dy=1.0,
        snap_radius_m=1.0,
    )

    assert snapped == (0, 4)
    assert np.all(mask)
