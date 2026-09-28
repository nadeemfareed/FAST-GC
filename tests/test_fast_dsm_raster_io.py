"""FAST_DSM GeoTIFF and spatial-metadata contract tests."""

from __future__ import annotations

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from fastgc.io_las import _write_tif


def test_dsm_geotiff_transform_and_resolution(tmp_path):
    arr = np.array(
        [
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ],
        dtype=np.float32,
    )

    out = tmp_path / "dsm_transform.tif"

    _write_tif(
        arr,
        str(out),
        xmin=500000.0,
        ymax=3200100.0,
        grid_res=2.0,
        crs="EPSG:32617",
    )

    with rasterio.open(out) as ds:
        assert ds.width == 3
        assert ds.height == 2

        assert ds.transform.a == pytest.approx(2.0)
        assert ds.transform.e == pytest.approx(-2.0)

        assert ds.transform.c == pytest.approx(500000.0)
        assert ds.transform.f == pytest.approx(3200100.0)

        assert ds.bounds.left == pytest.approx(500000.0)
        assert ds.bounds.right == pytest.approx(500006.0)
        assert ds.bounds.top == pytest.approx(3200100.0)
        assert ds.bounds.bottom == pytest.approx(3200096.0)


def test_dsm_geotiff_preserves_projected_crs(tmp_path):
    arr = np.ones((2, 2), dtype=np.float32)

    out = tmp_path / "dsm_crs.tif"

    _write_tif(
        arr,
        str(out),
        xmin=100.0,
        ymax=200.0,
        grid_res=1.0,
        crs="EPSG:32756",
    )

    with rasterio.open(out) as ds:
        assert ds.crs is not None
        assert ds.crs.to_epsg() == 32756


def test_dsm_geotiff_allows_local_coordinates_without_crs(tmp_path):
    arr = np.array(
        [
            [5.0, 6.0],
            [7.0, np.nan],
        ],
        dtype=np.float32,
    )

    out = tmp_path / "dsm_local.tif"

    _write_tif(
        arr,
        str(out),
        xmin=0.0,
        ymax=20.0,
        grid_res=1.0,
        crs=None,
    )

    with rasterio.open(out) as ds:
        assert ds.crs is None

        data = ds.read(1)

        assert data.shape == (2, 2)
        assert np.isfinite(data[0, 0])
        assert np.isnan(data[1, 1])


def test_dsm_geotiff_preserves_float_nodata_contract(tmp_path):
    arr = np.array(
        [
            [10.0, np.nan],
            [12.0, 13.0],
        ],
        dtype=np.float32,
    )

    out = tmp_path / "dsm_nodata.tif"

    _write_tif(
        arr,
        str(out),
        xmin=0.0,
        ymax=2.0,
        grid_res=1.0,
        crs=None,
    )

    with rasterio.open(out) as ds:
        assert ds.dtypes[0] == "float32"
        assert np.isnan(ds.nodata)

        data = ds.read(1)
        assert np.isnan(data[0, 1])


def test_dsm_geotiff_pixel_centers_match_surface_grid(tmp_path):
    arr = np.ones((2, 3), dtype=np.float32)

    out = tmp_path / "dsm_centers.tif"

    _write_tif(
        arr,
        str(out),
        xmin=10.0,
        ymax=30.0,
        grid_res=2.0,
        crs=None,
    )

    with rasterio.open(out) as ds:
        x0, y0 = ds.xy(0, 0)
        x1, y1 = ds.xy(1, 2)

        assert x0 == pytest.approx(11.0)
        assert y0 == pytest.approx(29.0)

        assert x1 == pytest.approx(15.0)
        assert y1 == pytest.approx(27.0)


def test_dsm_geotiff_rejects_nonpositive_resolution(tmp_path):
    arr = np.ones((2, 2), dtype=np.float32)

    out = tmp_path / "bad.tif"

    with pytest.raises(ValueError, match="grid_res must be > 0"):
        _write_tif(
            arr,
            str(out),
            xmin=0.0,
            ymax=2.0,
            grid_res=0.0,
            crs=None,
        )
