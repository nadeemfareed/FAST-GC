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



# ============================================================
# Compound CRS and elevation-preservation regression tests
# ============================================================

from fastgc.gis.crs import extract_horizontal_crs


def test_extract_compound_horizontal_crs():
    horizontal = extract_horizontal_crs("EPSG:6348+5703")

    assert horizontal.to_epsg() == 6348
    assert horizontal.is_projected
    assert len(horizontal.axis_info) == 2


def test_extract_geographic_3d_horizontal_crs():
    horizontal = extract_horizontal_crs("EPSG:4979")

    assert horizontal.to_epsg() == 4326
    assert horizontal.is_geographic
    assert len(horizontal.axis_info) == 2


def test_extract_vertical_only_crs_rejected():
    with pytest.raises(ValueError, match="vertical-only"):
        extract_horizontal_crs("EPSG:5703")


def test_compound_crs_geometry_preserves_z():
    original = Point(-69.0, 42.0, 125.5)

    transformed = transform_geometry(
        original,
        "EPSG:4979",
        "EPSG:6348+5703",
    )

    assert transformed.has_z
    assert transformed.z == pytest.approx(125.5)

    recovered = transform_geometry(
        transformed,
        "EPSG:6348+5703",
        "EPSG:4979",
    )

    assert recovered.x == pytest.approx(original.x, abs=1e-7)
    assert recovered.y == pytest.approx(original.y, abs=1e-7)
    assert recovered.z == pytest.approx(original.z)


def test_compound_crs_horizontal_requirement():
    horizontal = require_horizontal_crs("EPSG:6348+5703")

    assert horizontal.to_epsg() == 6348
    assert not horizontal.is_compound
    assert len(horizontal.axis_info) == 2


# Controlled horizontal transformation regression tests


def test_selector_compound_crs():
    from fastgc.gis.crs import select_horizontal_transformer

    transformer = select_horizontal_transformer(
        "EPSG:4979",
        "EPSG:6348+5703",
    )

    x, y = transformer.transform(-69.0, 42.0)

    assert abs(x) > 1000
    assert abs(y) > 1000


def test_selector_rejects_vertical_only_crs():
    from fastgc.gis.crs import select_horizontal_transformer

    with pytest.raises(ValueError, match="vertical-only"):
        select_horizontal_transformer(
            "EPSG:5703",
            "EPSG:32617",
        )


def test_selector_rejects_unknown_accuracy_when_required():
    from fastgc.gis.crs import select_horizontal_transformer

    with pytest.raises(
        ValueError,
        match="unknown numerical accuracy",
    ):
        select_horizontal_transformer(
            "EPSG:4326",
            "EPSG:32617",
            max_accuracy_m=1.0,
        )


def test_selector_rejects_invalid_accuracy_threshold():
    from fastgc.gis.crs import select_horizontal_transformer

    for threshold in (-1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite"):
            select_horizontal_transformer(
                "EPSG:4326",
                "EPSG:32617",
                max_accuracy_m=threshold,
            )


def test_selector_restores_proj_network_setting():
    from pyproj import network
    from fastgc.gis.crs import select_horizontal_transformer

    original = network.is_network_enabled()

    try:
        network.set_network_enabled(True)

        select_horizontal_transformer(
            "EPSG:4326",
            "EPSG:32617",
        )

        assert network.is_network_enabled()

        network.set_network_enabled(False)

        select_horizontal_transformer(
            "EPSG:4326",
            "EPSG:32617",
        )

        assert not network.is_network_enabled()

    finally:
        network.set_network_enabled(original)


def test_gis_coverage_florida():
    from pyproj.aoi import AreaOfInterest
    from fastgc.gis.crs import (
        select_horizontal_transformer,
        validate_transformer_coverage,
    )

    area = AreaOfInterest(-83, 29, -82, 30)

    transformer = select_horizontal_transformer(
        "EPSG:4326",
        "EPSG:32617",
        area_of_interest=area,
    )

    validate_transformer_coverage(transformer, area)


def test_gis_coverage_rejects_outside_area():
    from pyproj.aoi import AreaOfInterest
    from fastgc.gis.crs import (
        select_horizontal_transformer,
        validate_transformer_coverage,
    )

    transformer = select_horizontal_transformer(
        "EPSG:4326",
        "EPSG:32617",
    )

    outside = AreaOfInterest(10, 40, 11, 41)

    with pytest.raises(ValueError, match="does not cover"):
        validate_transformer_coverage(
            transformer,
            outside,
        )


def test_gis_coverage_rejects_antimeridian():
    from pyproj.aoi import AreaOfInterest
    from fastgc.gis.crs import (
        select_horizontal_transformer,
        validate_transformer_coverage,
    )

    transformer = select_horizontal_transformer(
        "EPSG:4326",
        "EPSG:32617",
    )

    crossing = AreaOfInterest(
        179,
        -10,
        -179,
        10,
    )

    with pytest.raises(
        ValueError,
        match="antimeridian",
    ):
        validate_transformer_coverage(
            transformer,
            crossing,
        )


def test_gis_coverage_rejects_invalid_bounds():
    from pyproj.aoi import AreaOfInterest
    from fastgc.gis.crs import (
        select_horizontal_transformer,
        validate_transformer_coverage,
    )

    transformer = select_horizontal_transformer(
        "EPSG:4326",
        "EPSG:32617",
    )

    invalid = AreaOfInterest(
        -83,
        30,
        -82,
        29,
    )

    with pytest.raises(
        ValueError,
        match="southern boundary",
    ):
        validate_transformer_coverage(
            transformer,
            invalid,
        )


def test_gis_selector_rejects_missing_grid(monkeypatch):
    from types import SimpleNamespace
    import fastgc.gis.crs as crs

    missing_grid = SimpleNamespace(
        short_name="missing_test_grid.tif",
        available=False,
    )

    unavailable_operation = SimpleNamespace(
        grids=[missing_grid],
    )

    class FakeTransformerGroup:
        def __init__(self, *args, **kwargs):
            self.best_available = False
            self.transformers = []
            self.unavailable_operations = [
                unavailable_operation,
            ]

    monkeypatch.setattr(
        crs,
        "TransformerGroup",
        FakeTransformerGroup,
    )

    with pytest.raises(
        RuntimeError,
        match="missing_test_grid.tif",
    ):
        crs.select_horizontal_transformer(
            "EPSG:4326",
            "EPSG:32617",
        )


def test_gis_concurrent_selector_isolation():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from pyproj import network
    from fastgc.gis.crs import select_horizontal_transformer

    def worker(enabled, barrier):
        previous = network.is_network_enabled()

        try:
            network.set_network_enabled(enabled)
            barrier.wait(timeout=15)

            transformer = select_horizontal_transformer(
                "EPSG:4326",
                "EPSG:32617",
            )

            x, y = transformer.transform(
                -82.3248,
                29.6516,
                errcheck=True,
            )

            return (
                network.is_network_enabled() == enabled
                and abs(x - 371776.274) < 1
                and abs(y - 3280914.294) < 1
            )

        finally:
            network.set_network_enabled(previous)

    with ThreadPoolExecutor(max_workers=2) as executor:
        for _ in range(10):
            barrier = Barrier(2)

            offline = executor.submit(
                worker,
                False,
                barrier,
            )

            online = executor.submit(
                worker,
                True,
                barrier,
            )

            assert offline.result()
            assert online.result()


def test_gis_selector_restores_network_after_failure(
    monkeypatch,
):
    from pyproj import network
    import fastgc.gis.crs as crs

    original = network.is_network_enabled()

    class FailingTransformerGroup:
        def __init__(self, *args, **kwargs):
            assert not network.is_network_enabled()
            raise RuntimeError("Simulated construction failure")

    monkeypatch.setattr(
        crs,
        "TransformerGroup",
        FailingTransformerGroup,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated construction failure",
    ):
        crs.select_horizontal_transformer(
            "EPSG:4326",
            "EPSG:32617",
        )

    assert network.is_network_enabled() == original


def test_gis_selector_preserves_caller_network_setting():
    from pyproj import network
    from fastgc.gis.crs import select_horizontal_transformer

    original = network.is_network_enabled()

    try:
        for enabled in (False, True):
            network.set_network_enabled(enabled)

            transformer = select_horizontal_transformer(
                "EPSG:4326",
                "EPSG:32617",
            )

            assert transformer is not None
            assert network.is_network_enabled() == enabled

    finally:
        network.set_network_enabled(original)


def test_gis_geometry_execution_stays_offline(monkeypatch):
    from pyproj import network
    from shapely.geometry import Point
    import fastgc.gis.crs as crs

    original_transform = crs.shapely_transform
    original_setting = network.is_network_enabled()
    observed = []

    def inspect_execution(function, geometry):
        observed.append(network.is_network_enabled())
        return original_transform(function, geometry)

    monkeypatch.setattr(
        crs,
        "shapely_transform",
        inspect_execution,
    )

    try:
        network.set_network_enabled(True)

        result = crs.transform_geometry(
            Point(-82.3248, 29.6516),
            "EPSG:4326",
            "EPSG:32617",
        )

        assert abs(result.x - 371776.274) < 1
        assert abs(result.y - 3280914.294) < 1
        assert observed == [False]
        assert network.is_network_enabled()

    finally:
        network.set_network_enabled(original_setting)


def test_gis_geometry_restores_network_after_execution_failure(
    monkeypatch,
):
    from pyproj import network
    from shapely.geometry import Point
    import fastgc.gis.crs as crs

    original_setting = network.is_network_enabled()

    def fail_during_execution(function, geometry):
        assert not network.is_network_enabled()
        raise RuntimeError("Simulated geometry execution failure")

    monkeypatch.setattr(
        crs,
        "shapely_transform",
        fail_during_execution,
    )

    try:
        network.set_network_enabled(True)

        with pytest.raises(
            RuntimeError,
            match="Simulated geometry execution failure",
        ):
            crs.transform_geometry(
                Point(-82.3248, 29.6516),
                "EPSG:4326",
                "EPSG:32617",
            )

        assert network.is_network_enabled()

    finally:
        network.set_network_enabled(original_setting)


def test_gis_geometry_passes_accuracy_controls():
    from shapely.geometry import Point
    from fastgc.gis.crs import transform_geometry

    with pytest.raises(
        ValueError,
        match="unknown numerical accuracy",
    ):
        transform_geometry(
            Point(-82.3248, 29.6516),
            "EPSG:4326",
            "EPSG:32617",
            require_known_accuracy=True,
        )
