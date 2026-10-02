import numpy as np

from fastgc.terrain import (
    _circular_metric_kernel,
    compute_multiscale_tpi,
)


def test_multiscale_tpi_plane_zero():
    y, x = np.mgrid[-100:101, -100:101]
    z = 100.0 + 0.2*x + 0.1*y

    out = compute_multiscale_tpi(
        z,
        1.0,
        1.0,
        20.0,
    )

    assert abs(float(out[100, 100])) < 1.0e-10


def test_multiscale_tpi_quadratic_resolution_convergence():
    vals = []

    for res in (2.0, 1.0, 0.5):
        coords = np.arange(-50.0, 50.0 + res, res)
        yy, xx = np.meshgrid(
            coords,
            coords,
            indexing="ij",
        )

        z = 0.0005 * (xx*xx + yy*yy)

        out = compute_multiscale_tpi(
            z,
            res,
            res,
            20.0,
        )

        c = len(coords) // 2
        vals.append(float(out[c, c]))

    assert abs(vals[2] + 0.1) < 2.0e-4
    assert abs(vals[1] - vals[2]) < 2.0e-4
    assert abs(vals[0] - vals[2]) < 2.0e-3


def test_multiscale_tpi_focal_cell_excluded():
    z = np.zeros((21, 21), dtype=np.float64)
    z[10, 10] = 100.0

    out = compute_multiscale_tpi(
        z,
        1.0,
        1.0,
        5.0,
    )

    assert np.isclose(
        out[10, 10],
        100.0,
    )


def test_multiscale_tpi_preserves_nodata_center():
    z = np.arange(
        441,
        dtype=np.float64,
    ).reshape(21, 21)

    z[10, 10] = np.nan

    out = compute_multiscale_tpi(
        z,
        1.0,
        1.0,
        5.0,
    )

    assert np.isnan(out[10, 10])


def test_multiscale_tpi_kernel_is_circular():
    k = _circular_metric_kernel(
        5.0,
        1.0,
        1.0,
    )

    cy = k.shape[0] // 2
    cx = k.shape[1] // 2

    assert k[cy, cx] == 0.0
    assert k[cy, cx + 5] == 1.0
    assert k[cy + 5, cx] == 1.0
    assert k[cy + 5, cx + 5] == 0.0


def test_multiscale_tpi_kernel_respects_anisotropic_pixels():
    k = _circular_metric_kernel(
        10.0,
        2.0,
        1.0,
    )

    assert k.shape == (21, 11)
