import numpy as np

from fastgc.hydrology import (
    contributing_area_ls_factor,
    rusle_s_factor,
    topographic_erosion_factors,
)


def test_rusle_s_low_slope_branch():
    beta = np.array(
        [[np.arctan(0.05)]],
        dtype=float,
    )

    got = rusle_s_factor(beta)

    expected = (
        10.8 * np.sin(beta) + 0.03
    )

    np.testing.assert_allclose(
        got,
        expected,
    )


def test_rusle_s_high_slope_branch():
    beta = np.array(
        [[np.arctan(0.20)]],
        dtype=float,
    )

    got = rusle_s_factor(beta)

    expected = (
        16.8 * np.sin(beta) - 0.50
    )

    np.testing.assert_allclose(
        got,
        expected,
    )


def test_rusle_s_is_nonnegative_on_flat():
    beta = np.array([[0.0]])

    got = rusle_s_factor(beta)

    assert got[0, 0] >= 0.0


def test_ls_reference_condition():
    # At a=22.13 and sin(beta)=0.0896,
    # both normalized terms equal one.
    beta = np.array(
        [[np.arcsin(0.0896)]]
    )

    area = np.array([[22.13]])

    got = contributing_area_ls_factor(
        area,
        beta,
    )

    np.testing.assert_allclose(
        got,
        [[1.0]],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_ls_matches_explicit_equation():
    area = np.array(
        [[10.0, 50.0]],
        dtype=float,
    )

    beta = np.deg2rad(
        np.array([[5.0, 15.0]])
    )

    got = contributing_area_ls_factor(
        area,
        beta,
        m=0.4,
        n=1.3,
    )

    expected = (
        (area / 22.13) ** 0.4
        * (
            np.sin(beta) / 0.0896
        ) ** 1.3
    )

    np.testing.assert_allclose(
        got,
        expected,
    )


def test_ls_increases_with_contributing_area():
    area = np.array(
        [[5.0, 20.0, 100.0]]
    )

    beta = np.full(
        area.shape,
        np.deg2rad(10.0),
    )

    ls = contributing_area_ls_factor(
        area,
        beta,
    )

    assert (
        ls[0, 0]
        < ls[0, 1]
        < ls[0, 2]
    )


def test_ls_increases_with_slope():
    area = np.full(
        (1, 3),
        50.0,
    )

    beta = np.deg2rad(
        [[2.0, 10.0, 25.0]]
    )

    ls = contributing_area_ls_factor(
        area,
        beta,
    )

    assert (
        ls[0, 0]
        < ls[0, 1]
        < ls[0, 2]
    )


def test_erosion_products_preserve_nodata():
    area = np.array(
        [[10.0, np.nan]]
    )

    beta = np.array(
        [[0.1, np.nan]]
    )

    products = topographic_erosion_factors(
        area,
        beta,
    )

    assert np.isnan(
        products[
            "rusle_s_factor"
        ][0, 1]
    )

    assert np.isnan(
        products[
            "contributing_area_ls_factor"
        ][0, 1]
    )


def test_product_names_are_explicit():
    area = np.array([[10.0]])
    beta = np.array([[0.1]])

    products = topographic_erosion_factors(
        area,
        beta,
    )

    assert set(products) == {
        "rusle_s_factor",
        "contributing_area_ls_factor",
    }
