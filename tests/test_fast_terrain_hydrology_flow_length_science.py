import numpy as np
import pytest

from fastgc.hydrology import (
    d8_downslope_flow_length,
    d8_longest_upslope_flow_length,
)


def test_cardinal_downslope_length():
    dem = np.array([[3., 2., 1.]])

    # 0 -> 1 -> 2 -> outlet
    receiver = np.array([1, 2, -1])

    got = d8_downslope_flow_length(
        dem,
        receiver,
        dx=2.0,
        dy=5.0,
    )

    assert np.allclose(
        got,
        [[4.0, 2.0, 0.0]],
    )


def test_vertical_downslope_uses_dy():
    dem = np.array(
        [[3.],
         [2.],
         [1.]]
    )

    receiver = np.array([1, 2, -1])

    got = d8_downslope_flow_length(
        dem,
        receiver,
        dx=2.0,
        dy=5.0,
    )

    assert np.allclose(
        got[:, 0],
        [10.0, 5.0, 0.0],
    )


def test_diagonal_step_uses_rectangular_pixel_geometry():
    dem = np.array(
        [[2., 9.],
         [9., 1.]]
    )

    # cell 0 -> cell 3 diagonally
    receiver = np.array([3, -1, -1, -1])

    got = d8_downslope_flow_length(
        dem,
        receiver,
        dx=3.0,
        dy=4.0,
    )

    # sqrt(3^2 + 4^2) = 5
    assert np.isclose(got[0, 0], 5.0)
    assert np.isclose(got[1, 1], 0.0)


def test_mixed_downslope_path_length():
    dem = np.array(
        [[4., 3., 9.],
         [9., 2., 1.]]
    )

    # 0 -> 1 horizontal (3 m)
    # 1 -> 4 vertical   (4 m)
    # 4 -> 5 horizontal (3 m)
    receiver = np.array(
        [1, 4, -1,
         -1, 5, -1]
    )

    got = d8_downslope_flow_length(
        dem,
        receiver,
        dx=3.0,
        dy=4.0,
    )

    assert np.isclose(got[0, 0], 10.0)
    assert np.isclose(got[0, 1], 7.0)
    assert np.isclose(got[1, 1], 3.0)
    assert np.isclose(got[1, 2], 0.0)


def test_longest_upslope_selects_longest_branch():
    dem = np.ones((3, 3))

    # Paths converging at cell 4:
    #
    # 0 -> 1 -> 4
    # 3 ------> 4
    #
    # dx=2, dy=3
    receiver = np.array(
        [1, 4, -1,
         4, -1, -1,
         -1, -1, -1]
    )

    got = d8_longest_upslope_flow_length(
        dem,
        receiver,
        dx=2.0,
        dy=3.0,
    )

    # 0 -> 1 = 2
    # 1 -> 4 = 3
    # total longest = 5
    assert np.isclose(got[1, 1], 5.0)


def test_longest_upslope_diagonal_geometry():
    dem = np.ones((2, 2))

    receiver = np.array(
        [3, -1,
         -1, -1]
    )

    got = d8_longest_upslope_flow_length(
        dem,
        receiver,
        dx=3.0,
        dy=4.0,
    )

    assert np.isclose(got[1, 1], 5.0)


def test_headwater_longest_length_is_zero():
    dem = np.ones((1, 3))
    receiver = np.array([1, 2, -1])

    got = d8_longest_upslope_flow_length(
        dem,
        receiver,
        dx=2.0,
        dy=2.0,
    )

    assert np.isclose(got[0, 0], 0.0)
    assert np.isclose(got[0, 1], 2.0)
    assert np.isclose(got[0, 2], 4.0)


def test_downslope_outlet_length_is_zero():
    dem = np.ones((1, 1))
    receiver = np.array([-1])

    got = d8_downslope_flow_length(
        dem,
        receiver,
        dx=2.0,
        dy=3.0,
    )

    assert np.isclose(got[0, 0], 0.0)


def test_flow_lengths_preserve_nodata():
    dem = np.array(
        [[3., 2., 1.],
         [np.nan, np.nan, np.nan]]
    )

    receiver = np.array(
        [1, 2, -1,
         -1, -1, -1]
    )

    down = d8_downslope_flow_length(
        dem,
        receiver,
        1.0,
        1.0,
    )

    up = d8_longest_upslope_flow_length(
        dem,
        receiver,
        1.0,
        1.0,
    )

    assert np.all(np.isnan(down[1]))
    assert np.all(np.isnan(up[1]))


@pytest.mark.parametrize(
    "func",
    [
        d8_downslope_flow_length,
        d8_longest_upslope_flow_length,
    ],
)
def test_flow_length_detects_cycle(func):
    dem = np.ones((1, 2))

    # Deliberately invalid routing cycle.
    receiver = np.array([1, 0])

    with pytest.raises(
        RuntimeError,
        match="cycle",
    ):
        func(
            dem,
            receiver,
            1.0,
            1.0,
        )


@pytest.mark.parametrize(
    "func",
    [
        d8_downslope_flow_length,
        d8_longest_upslope_flow_length,
    ],
)
def test_receiver_size_must_match_dem(func):
    dem = np.ones((2, 2))

    with pytest.raises(ValueError):
        func(
            dem,
            np.array([-1, -1]),
            1.0,
            1.0,
        )
