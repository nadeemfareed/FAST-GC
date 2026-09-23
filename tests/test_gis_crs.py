"""Regression tests for FAST-GIS CRS inspection."""

import numpy as np
import laspy
import pyogrio
import pytest
import rasterio

from pyproj import CRS
from rasterio.transform import from_origin
from shapely.geometry import Point

from fastgc.gis.crs import inspect_crs


def test_las_projected_crs(tmp_path):
    path = tmp_path / "sample.las"

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_crs(CRS.from_epsg(32617))

    las = laspy.LasData(header)
    las.x = np.array([371776.0])
    las.y = np.array([3280914.0])
    las.z = np.array([50.0])
    las.write(path)

    result = inspect_crs(path)

    assert result.status == "identified"
    assert result.data_type == "point_cloud"
    assert result.epsg == 32617
    assert result.is_projected is True


def test_las_missing_crs(tmp_path):
    path = tmp_path / "missing.las"

    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.x = np.array([100.0])
    las.y = np.array([200.0])
    las.z = np.array([10.0])
    las.write(path)

    result = inspect_crs(path)

    assert result.status == "missing"
    assert result.crs is None
    assert result.epsg is None


def test_geotiff_geographic_crs(tmp_path):
    path = tmp_path / "sample.tif"

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-82.4, 29.7, 0.01, 0.01),
    ) as dst:
        dst.write(np.ones((2, 2), dtype="float32"), 1)

    result = inspect_crs(path)

    assert result.status == "identified"
    assert result.data_type == "raster"
    assert result.epsg == 4326
    assert result.is_geographic is True


def test_geojson_geographic_crs(tmp_path):
    path = tmp_path / "sample.geojson"

    pyogrio.write_dataframe(
        __import__("geopandas").GeoDataFrame(
            {"plot_id": ["P001"]},
            geometry=[Point(-82.3248, 29.6516)],
            crs="EPSG:4326",
        ),
        path,
        driver="GeoJSON",
    )

    result = inspect_crs(path)

    assert result.status == "identified"
    assert result.data_type == "vector"
    assert result.epsg == 4326
    assert result.is_geographic is True


def test_unsupported_format(tmp_path):
    path = tmp_path / "sample.xyz"
    path.write_text("1 2 3", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported"):
        inspect_crs(path)



# ============================================================
# Coordinate transformation tests
# ============================================================

from shapely.geometry import Point, box

from fastgc.gis.crs import (
    require_horizontal_crs,
    spatially_overlaps,
    transform_geometry,
)


def test_transform_geographic_to_utm():
    point = Point(-82.3248, 29.6516)

    transformed = transform_geometry(
        point,
        "EPSG:4326",
        "EPSG:32617",
    )

    assert transformed.x == pytest.approx(371776.274, abs=1.0)
    assert transformed.y == pytest.approx(3280914.294, abs=1.0)


def test_transform_utm_to_geographic():
    original = Point(-82.3248, 29.6516)

    projected = transform_geometry(
        original,
        "EPSG:4326",
        "EPSG:32617",
    )

    recovered = transform_geometry(
        projected,
        "EPSG:32617",
        "EPSG:4326",
    )

    assert recovered.x == pytest.approx(original.x, abs=1e-7)
    assert recovered.y == pytest.approx(original.y, abs=1e-7)


def test_spatial_overlap_across_crs():
    geographic_bounds = (
        -82.33,
        29.65,
        -82.32,
        29.66,
    )

    projected = transform_geometry(
        box(*geographic_bounds),
        "EPSG:4326",
        "EPSG:32617",
    )

    assert spatially_overlaps(
        geographic_bounds,
        "EPSG:4326",
        projected.bounds,
        "EPSG:32617",
    )


def test_spatial_nonoverlap():
    assert not spatially_overlaps(
        (-82.33, 29.65, -82.32, 29.66),
        "EPSG:4326",
        (-80.0, 25.0, -79.0, 26.0),
        "EPSG:4326",
    )


def test_missing_horizontal_crs_rejected():
    with pytest.raises(ValueError, match="missing"):
        require_horizontal_crs(None)


def test_vertical_only_crs_rejected():
    with pytest.raises(ValueError, match="vertical-only"):
        require_horizontal_crs("EPSG:5703")



# ============================================================
# LAS CRS record validation tests
# ============================================================

from laspy.vlrs.known import WktCoordinateSystemVlr

from fastgc.gis.crs import inspect_las_crs_records


def test_las_single_valid_crs_record(tmp_path):
    path = tmp_path / "single_crs.las"

    header = laspy.LasHeader(
        point_format=3,
        version="1.2",
    )
    header.add_crs(CRS.from_epsg(32617))

    laspy.LasData(header).write(path)

    result = inspect_las_crs_records(path)

    assert result is not None
    assert result.to_epsg() == 32617


def test_las_matching_crs_records(tmp_path):
    path = tmp_path / "matching_crs.las"

    header = laspy.LasHeader(
        point_format=3,
        version="1.4",
    )

    header.add_crs(CRS.from_epsg(32617))

    header.vlrs.append(
        WktCoordinateSystemVlr(
            CRS.from_epsg(32617).to_wkt()
        )
    )

    laspy.LasData(header).write(path)

    result = inspect_las_crs_records(path)

    assert result is not None
    assert result.to_epsg() == 32617


def test_las_conflicting_crs_records(tmp_path):
    path = tmp_path / "conflicting_crs.las"

    header = laspy.LasHeader(
        point_format=3,
        version="1.4",
    )

    header.add_crs(CRS.from_epsg(32617))

    header.vlrs.append(
        WktCoordinateSystemVlr(
            CRS.from_epsg(32618).to_wkt()
        )
    )

    laspy.LasData(header).write(path)

    with pytest.raises(
        ValueError,
        match="Conflicting LAS CRS metadata",
    ):
        inspect_las_crs_records(path)


def test_las_no_crs_records(tmp_path):
    path = tmp_path / "no_crs.las"

    header = laspy.LasHeader(
        point_format=3,
        version="1.2",
    )

    laspy.LasData(header).write(path)

    assert inspect_las_crs_records(path) is None



def test_inspect_crs_rejects_conflicting_las_metadata(tmp_path):
    """The public inspection interface must reject conflicting LAS CRS."""

    path = tmp_path / "conflicting_metadata.las"

    header = laspy.LasHeader(
        point_format=3,
        version="1.4",
    )

    header.add_crs(CRS.from_epsg(32617))

    header.vlrs.append(
        WktCoordinateSystemVlr(
            CRS.from_epsg(32618).to_wkt()
        )
    )

    laspy.LasData(header).write(path)

    with pytest.raises(
        ValueError,
        match="Conflicting LAS CRS metadata",
    ):
        inspect_crs(path)
