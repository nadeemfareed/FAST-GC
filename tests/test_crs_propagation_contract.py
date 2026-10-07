from pathlib import Path

import laspy
import numpy as np
import rasterio
from pyproj import CRS

from fastgc.io_las import _safe_parse_crs, _write_tif


def _make_las(path: Path, with_crs: bool) -> None:
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([500000.0, 5400000.0, 0.0])

    if with_crs:
        header.add_crs(CRS.from_epsg(25832))

    las = laspy.LasData(header)
    las.x = np.array([500000.0, 500001.0, 500002.0])
    las.y = np.array([5400000.0, 5400001.0, 5400002.0])
    las.z = np.array([100.0, 101.0, 102.0])
    las.write(path)


def test_las_crs_is_read_when_present(tmp_path):
    src = tmp_path / "projected.las"
    _make_las(src, True)

    with laspy.open(src) as reader:
        crs = _safe_parse_crs(reader)

    assert crs is not None
    assert crs.to_epsg() == 25832


def test_las_without_crs_remains_unknown(tmp_path):
    src = tmp_path / "unknown.las"
    _make_las(src, False)

    with laspy.open(src) as reader:
        crs = _safe_parse_crs(reader)

    assert crs is None


def test_geotiff_writer_preserves_projected_crs(tmp_path):
    out = tmp_path / "dem.tif"
    arr = np.arange(25, dtype=np.float32).reshape(5, 5)

    _write_tif(
        arr,
        str(out),
        500000.0,
        5400000.0,
        0.5,
        crs=CRS.from_epsg(25832),
    )

    with rasterio.open(out) as ds:
        assert ds.crs == rasterio.crs.CRS.from_epsg(25832)
        assert ds.transform.a == 0.5
        assert ds.transform.e == -0.5
        assert ds.transform.c == 500000.0
        assert ds.transform.f == 5400000.0


def test_geotiff_writer_does_not_invent_crs(tmp_path):
    out = tmp_path / "dem_unknown.tif"
    arr = np.ones((5, 5), dtype=np.float32)

    _write_tif(
        arr,
        str(out),
        500000.0,
        5400000.0,
        0.5,
        crs=None,
    )

    with rasterio.open(out) as ds:
        assert ds.crs is None
