import numpy as np
import pytest

from fastgc.hydrology import (
    stream_power_index,
    topographic_wetness_index,
)


def test_twi_matches_definition():
    a = np.array([2.0, 5.0, 20.0])
    beta = np.deg2rad(
        np.array([5.0, 15.0, 30.0])
    )

    got = topographic_wetness_index(a, beta)
    expected = np.log(a / np.tan(beta))

    assert np.allclose(got, expected)


def test_spi_matches_definition():
    a = np.array([2.0, 5.0, 20.0])
    beta = np.deg2rad(
        np.array([5.0, 15.0, 30.0])
    )

    got = stream_power_index(a, beta)
    expected = a * np.tan(beta)

    assert np.allclose(got, expected)


def test_twi_zero_slope_uses_explicit_floor():
    a = np.array([10.0])
    beta = np.array([0.0])
    floor = 1.0e-6

    got = topographic_wetness_index(
        a,
        beta,
        min_slope_radians=floor,
    )

    expected = np.log(
        10.0 / np.tan(floor)
    )

    assert np.allclose(got, expected)


def test_spi_zero_slope_is_zero():
    got = stream_power_index(
        np.array([10.0]),
        np.array([0.0]),
    )

    assert np.allclose(got, 0.0)


def test_twi_increases_with_catchment_area():
    beta = np.full(3, np.deg2rad(10.0))
    a = np.array([1.0, 10.0, 100.0])

    twi = topographic_wetness_index(a, beta)

    assert np.all(np.diff(twi) > 0.0)


def test_twi_decreases_with_slope():
    a = np.full(3, 10.0)
    beta = np.deg2rad(
        np.array([2.0, 10.0, 30.0])
    )

    twi = topographic_wetness_index(a, beta)

    assert np.all(np.diff(twi) < 0.0)


def test_spi_increases_with_area_and_slope():
    a = np.array([1.0, 2.0, 4.0])
    beta = np.deg2rad(
        np.array([5.0, 10.0, 20.0])
    )

    spi = stream_power_index(a, beta)

    assert np.all(np.diff(spi) > 0.0)


def test_twi_nonpositive_area_is_nodata():
    a = np.array([-1.0, 0.0, 1.0])
    beta = np.full(3, np.deg2rad(10.0))

    twi = topographic_wetness_index(a, beta)

    assert np.isnan(twi[0])
    assert np.isnan(twi[1])
    assert np.isfinite(twi[2])


def test_spi_zero_area_is_zero():
    spi = stream_power_index(
        np.array([0.0]),
        np.array([np.deg2rad(20.0)]),
    )

    assert np.allclose(spi, 0.0)


def test_twi_and_spi_preserve_nodata():
    a = np.array([10.0, np.nan, 10.0])
    beta = np.array(
        [0.1, 0.1, np.nan]
    )

    twi = topographic_wetness_index(a, beta)
    spi = stream_power_index(a, beta)

    assert np.isfinite(twi[0])
    assert np.isfinite(spi[0])

    assert np.isnan(twi[1])
    assert np.isnan(twi[2])
    assert np.isnan(spi[1])
    assert np.isnan(spi[2])


@pytest.mark.parametrize(
    "func",
    [
        topographic_wetness_index,
        stream_power_index,
    ],
)
def test_negative_slope_is_rejected(func):
    with pytest.raises(
        ValueError,
        match="Slope angle",
    ):
        func(
            np.array([10.0]),
            np.array([-0.01]),
        )


def test_twi_rejects_invalid_slope_floor():
    with pytest.raises(ValueError):
        topographic_wetness_index(
            np.array([10.0]),
            np.array([0.1]),
            min_slope_radians=0.0,
        )


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        topographic_wetness_index(
            np.ones((2, 2)),
            np.ones((3, 3)),
        )

    with pytest.raises(ValueError):
        stream_power_index(
            np.ones((2, 2)),
            np.ones((3, 3)),
        )
