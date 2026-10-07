import numpy as np

from fastgc.hydrology import (
    d8_downslope_flow_length,
    d8_longest_upslope_flow_length,
    stream_power_index,
    topographic_wetness_index,
)


def test_twi_matches_definition():
    a = np.array([[10.0, 20.0]])
    beta = np.deg2rad(
        np.array([[10.0, 20.0]])
    )

    got = topographic_wetness_index(
        a,
        beta,
    )

    expected = np.log(
        a / np.tan(beta)
    )

    np.testing.assert_allclose(
        got,
        expected,
    )


def test_twi_flat_is_finite_with_explicit_floor():
    a = np.array([[10.0]])
    beta = np.array([[0.0]])

    got = topographic_wetness_index(
        a,
        beta,
        min_slope_radians=1.0e-6,
    )

    assert np.isfinite(got[0, 0])


def test_spi_matches_definition():
    a = np.array([[5.0, 10.0]])
    beta = np.deg2rad(
        np.array([[5.0, 15.0]])
    )

    got = stream_power_index(
        a,
        beta,
    )

    np.testing.assert_allclose(
        got,
        a * np.tan(beta),
    )


def test_downslope_flow_length_cardinal_chain():
    dem = np.array(
        [[4.0, 3.0, 2.0, 1.0]]
    )

    receiver = np.array(
        [1, 2, 3, -1],
        dtype=np.int64,
    )

    got = d8_downslope_flow_length(
        dem,
        receiver,
        2.0,
        2.0,
    )

    np.testing.assert_allclose(
        got,
        [[6.0, 4.0, 2.0, 0.0]],
    )


def test_downslope_flow_length_diagonal():
    dem = np.array(
        [
            [2.0, 9.0],
            [9.0, 1.0],
        ]
    )

    receiver = np.array(
        [3, -1, -1, -1],
        dtype=np.int64,
    )

    got = d8_downslope_flow_length(
        dem,
        receiver,
        3.0,
        4.0,
    )

    assert np.isclose(
        got[0, 0],
        5.0,
    )


def test_longest_upslope_flow_length_chain():
    dem = np.array(
        [[4.0, 3.0, 2.0, 1.0]]
    )

    receiver = np.array(
        [1, 2, 3, -1],
        dtype=np.int64,
    )

    got = d8_longest_upslope_flow_length(
        dem,
        receiver,
        2.0,
        2.0,
    )

    np.testing.assert_allclose(
        got,
        [[0.0, 2.0, 4.0, 6.0]],
    )


def test_longest_path_chooses_longer_branch():
    dem = np.ones((2, 3))

    # Valid D8 topology:
    #
    # 0 -> 1 -> 2
    #      ^
    #      |
    #      4
    #
    # Path 0->1->2 is 2 cells long.
    # Path 4->1->2 is also valid D8 and shorter/equal;
    # outlet cell 2 must retain the longest upstream path.
    receiver = np.array(
        [1, 2, -1, -1, 1, -1],
        dtype=np.int64,
    )

    got = d8_longest_upslope_flow_length(
        dem,
        receiver,
        1.0,
        1.0,
    )

    assert np.isclose(
        got.ravel()[2],
        2.0,
    )


def test_indices_preserve_nodata():
    a = np.array(
        [[10.0, np.nan]]
    )

    beta = np.array(
        [[0.1, 0.2]]
    )

    twi = topographic_wetness_index(
        a,
        beta,
    )

    spi = stream_power_index(
        a,
        beta,
    )

    assert np.isnan(twi[0, 1])
    assert np.isnan(spi[0, 1])
