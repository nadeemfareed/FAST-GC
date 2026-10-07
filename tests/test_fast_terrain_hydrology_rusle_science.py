import numpy as np
import pytest

from fastgc.hydrology import (
    contributing_area_ls_factor,
    rusle_s_factor,
)


def test_rusle_s_below_nine_percent():
    slope_fraction = 0.05
    beta = np.arctan(slope_fraction)

    got = rusle_s_factor(np.array([beta]))[0]

    expected = 10.8 * np.sin(beta) + 0.03

    assert np.isclose(got, expected)


def test_rusle_s_at_nine_percent_uses_upper_branch():
    beta = np.arctan(0.09)

    got = rusle_s_factor(np.array([beta]))[0]

    expected = 16.8 * np.sin(beta) - 0.50

    assert np.isclose(got, expected)


def test_rusle_s_above_nine_percent():
    beta = np.arctan(0.20)

    got = rusle_s_factor(np.array([beta]))[0]

    expected = 16.8 * np.sin(beta) - 0.50

    assert np.isclose(got, expected)


def test_rusle_s_horizontal_surface():
    got = rusle_s_factor(
        np.array([0.0])
    )[0]

    assert np.isclose(got, 0.03)


def test_rusle_s_rejects_negative_slope():
    with pytest.raises(
        ValueError,
        match="Slope angle",
    ):
        rusle_s_factor(
            np.array([-0.01])
        )


def test_ls_matches_moore_burch_style_equation():
    a = np.array([1.0, 10.0, 50.0])
    beta = np.deg2rad(
        np.array([5.0, 10.0, 20.0])
    )

    m = 0.4
    n = 1.3

    got = contributing_area_ls_factor(
        a,
        beta,
        m=m,
        n=n,
    )

    expected = (
        (a / 22.13) ** m
        * (
            np.sin(beta) / 0.0896
        ) ** n
    )

    assert np.allclose(got, expected)


def test_ls_standard_reference_point():
    a = np.array([22.13])
    beta = np.array(
        [np.arcsin(0.0896)]
    )

    got = contributing_area_ls_factor(
        a,
        beta,
    )[0]

    assert np.isclose(
        got,
        1.0,
        rtol=0.0,
        atol=1e-12,
    )


def test_ls_zero_area_is_zero():
    got = contributing_area_ls_factor(
        np.array([0.0]),
        np.array([0.2]),
    )[0]

    assert np.isclose(got, 0.0)


def test_ls_zero_slope_is_zero():
    got = contributing_area_ls_factor(
        np.array([10.0]),
        np.array([0.0]),
    )[0]

    assert np.isclose(got, 0.0)


def test_ls_increases_with_contributing_area():
    a = np.array([1.0, 10.0, 100.0])
    beta = np.full(
        3,
        np.deg2rad(10.0),
    )

    ls = contributing_area_ls_factor(
        a,
        beta,
    )

    assert np.all(np.diff(ls) > 0.0)


def test_ls_increases_with_slope():
    a = np.full(3, 20.0)
    beta = np.deg2rad(
        np.array([2.0, 10.0, 30.0])
    )

    ls = contributing_area_ls_factor(
        a,
        beta,
    )

    assert np.all(np.diff(ls) > 0.0)


def test_ls_rejects_negative_slope():
    with pytest.raises(
        ValueError,
        match="Slope angle",
    ):
        contributing_area_ls_factor(
            np.array([10.0]),
            np.array([-0.01]),
        )


def test_ls_preserves_nodata():
    a = np.array([10.0, np.nan, 20.0])
    beta = np.array([0.1, 0.2, np.nan])

    got = contributing_area_ls_factor(
        a,
        beta,
    )

    assert np.isfinite(got[0])
    assert np.isnan(got[1])
    assert np.isnan(got[2])


@pytest.mark.parametrize(
    "name,value",
    [
        ("m", 0.0),
        ("n", 0.0),
        ("reference_length", 0.0),
        ("reference_sine", 0.0),
    ],
)
def test_ls_rejects_invalid_parameters(name, value):
    kwargs = {name: value}

    with pytest.raises(ValueError):
        contributing_area_ls_factor(
            np.array([10.0]),
            np.array([0.1]),
            **kwargs,
        )
