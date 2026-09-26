from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from fastgc.gis.core_mask import (
    load_core_mask_metadata,
    mask_plot_rasters_to_core,
    mask_raster_to_core,
)


def _core_geometry():
    return {
        "type": "Polygon",
        "coordinates": [[(2, 2), (8, 2), (8, 8), (2, 8), (2, 2)]],
    }


def _write_raster(path, data, *, nodata):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:32618",
        transform=from_origin(0, 10, 1, 1),
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def test_load_core_mask_metadata(tmp_path):
    metadata = tmp_path / "Clip_Plot_001.json"
    metadata.write_text(
        json.dumps(
            {
                "schema": "fastgc.gis.plot",
                "point_extent": "core_plus_buffer",
                "processing_crs": "EPSG:32618",
                "core_geometry": _core_geometry(),
            }
        ),
        encoding="utf-8",
    )

    geometry, crs = load_core_mask_metadata(metadata)
    assert geometry["type"] == "Polygon"
    assert crs == "EPSG:32618"


def test_mask_float_raster_to_core(tmp_path):
    path = tmp_path / "float.tif"
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    _write_raster(path, data, nodata=np.nan)

    mask_raster_to_core(
        path,
        core_geometry=_core_geometry(),
        processing_crs="EPSG:32618",
    )

    with rasterio.open(path) as src:
        result = src.read(1)
        assert src.crs == rasterio.crs.CRS.from_epsg(32618)
        assert src.width == 6
        assert src.height == 6
        assert result.dtype == np.float32
        assert np.isnan(src.nodata)


def test_mask_integer_raster_preserves_integer_nodata(tmp_path):
    path = tmp_path / "integer.tif"
    data = np.arange(100, dtype=np.int32).reshape(10, 10)
    _write_raster(path, data, nodata=-9999)

    mask_raster_to_core(
        path,
        core_geometry=_core_geometry(),
        processing_crs="EPSG:32618",
    )

    with rasterio.open(path) as src:
        result = src.read(1)
        assert src.width == 6
        assert src.height == 6
        assert result.dtype == np.int32
        assert src.nodata == -9999


def test_mask_rejects_crs_mismatch(tmp_path):
    path = tmp_path / "wrong_crs.tif"
    data = np.ones((10, 10), dtype=np.float32)
    _write_raster(path, data, nodata=np.nan)

    with pytest.raises(ValueError, match="does not match"):
        mask_raster_to_core(
            path,
            core_geometry=_core_geometry(),
            processing_crs="EPSG:32617",
        )


def test_mask_plot_rasters_recursive_and_leaves_las_untouched(tmp_path):
    work = tmp_path / "Plot_001"
    chm = work / "FAST_CHM" / "p2r" / "Plot_001.tif"
    structure = work / "FAST_STRUCTURE" / "Plot_001_z_mean.tif"
    laz = work / "FAST_NORMALIZED" / "Plot_001.laz"

    chm.parent.mkdir(parents=True)
    structure.parent.mkdir(parents=True)
    laz.parent.mkdir(parents=True)

    data = np.ones((10, 10), dtype=np.float32)
    _write_raster(chm, data, nodata=np.nan)
    _write_raster(structure, data, nodata=np.nan)
    laz.write_bytes(b"unchanged-point-cloud")

    metadata = tmp_path / "Clip_Plot_001.json"
    metadata.write_text(
        json.dumps(
            {
                "schema": "fastgc.gis.plot",
                "point_extent": "core_plus_buffer",
                "processing_crs": "EPSG:32618",
                "core_geometry": _core_geometry(),
            }
        ),
        encoding="utf-8",
    )

    masked = mask_plot_rasters_to_core(
        work_root=work,
        metadata_path=metadata,
    )

    assert set(masked) == {chm, structure}
    assert laz.read_bytes() == b"unchanged-point-cloud"

    with rasterio.open(chm) as src:
        assert (src.width, src.height) == (6, 6)

    with rasterio.open(structure) as src:
        assert (src.width, src.height) == (6, 6)



def test_raster_without_crs_inherits_processing_crs(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from fastgc.gis.core_mask import mask_raster_to_core

    path = tmp_path / "no_crs.tif"
    data = np.arange(100, dtype=np.float32).reshape(10, 10)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="float32",
        transform=from_origin(0, 10, 1, 1),
        nodata=np.nan,
    ) as dst:
        dst.write(data, 1)

    with rasterio.open(path) as src:
        assert src.crs is None

    core = {
        "type": "Polygon",
        "coordinates": [[
            [2.0, 2.0],
            [8.0, 2.0],
            [8.0, 8.0],
            [2.0, 8.0],
            [2.0, 2.0],
        ]],
    }

    mask_raster_to_core(
        path,
        core_geometry=core,
        processing_crs="EPSG:32618",
    )

    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 32618
        assert src.width == 6
        assert src.height == 6
