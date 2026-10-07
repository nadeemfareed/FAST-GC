import numpy as np

from fastgc.hydrology import (
    d8_basin_labels,
    delineate_watershed,
    subcatchment_labels,
    watershed_boundary_mask,
)


def test_two_terminal_basins():
    # Two independent chains:
    # 0 -> 1 -> outlet
    # 2 -> 3 -> outlet
    dem = np.ones((1, 4), dtype=float)

    receiver = np.array(
        [1, -1, 3, -1],
        dtype=np.int64,
    )

    labels = d8_basin_labels(
        dem,
        receiver,
    )

    assert labels[0, 0] == labels[0, 1]
    assert labels[0, 2] == labels[0, 3]
    assert labels[0, 0] != labels[0, 2]


def test_basin_ids_are_deterministic():
    dem = np.ones((1, 4), dtype=float)

    receiver = np.array(
        [1, -1, 3, -1],
        dtype=np.int64,
    )

    a = d8_basin_labels(dem, receiver)
    b = d8_basin_labels(dem, receiver)

    np.testing.assert_array_equal(a, b)


def test_watershed_boundary_between_basins():
    labels = np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ],
        dtype=np.int32,
    )

    boundary = watershed_boundary_mask(
        labels
    )

    assert np.all(boundary[:, 1])
    assert np.all(boundary[:, 2])
    assert not np.any(boundary[:, 0])
    assert not np.any(boundary[:, 3])


def test_pour_point_watershed():
    # 0 -> 2 <- 1
    #      |
    #      v
    #      3 -> 4
    dem = np.ones((1, 5), dtype=float)

    receiver = np.array(
        [2, 2, 3, 4, -1],
        dtype=np.int64,
    )

    mask = delineate_watershed(
        dem,
        receiver,
        pour_row=0,
        pour_col=3,
    )

    np.testing.assert_array_equal(
        mask,
        [[True, True, True, True, False]],
    )


def test_outlet_watershed_contains_entire_network():
    dem = np.ones((1, 5), dtype=float)

    receiver = np.array(
        [2, 2, 3, 4, -1],
        dtype=np.int64,
    )

    mask = delineate_watershed(
        dem,
        receiver,
        pour_row=0,
        pour_col=4,
    )

    assert np.all(mask)


def test_subcatchments_follow_first_stream_cell():
    stream = np.array(
        [[False, False, True, True, True]],
        dtype=bool,
    )

    receiver = np.array(
        [2, 2, 3, 4, -1],
        dtype=np.int64,
    )

    labels = subcatchment_labels(
        stream,
        receiver,
    )

    assert labels[0, 0] == labels[0, 2]
    assert labels[0, 1] == labels[0, 2]

    assert labels[0, 3] != labels[0, 2]
    assert labels[0, 4] != labels[0, 3]


def test_nodata_gets_zero_basin():
    dem = np.array(
        [[1.0, np.nan, 1.0]],
        dtype=float,
    )

    receiver = np.array(
        [-1, -1, -1],
        dtype=np.int64,
    )

    labels = d8_basin_labels(
        dem,
        receiver,
    )

    assert labels[0, 1] == 0
