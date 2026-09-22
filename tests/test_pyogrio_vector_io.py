from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, mapping

from fastgc.itd_algorithms.common import write_shapefile
from fastgc.treeclouds import _load_polygons
from fastgc.raster_post import run_clip


def _payload():
    poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"crown_id": 7, "max_h_m": 12.5},
            "geometry": mapping(poly),
        }],
    }


def test_pyogrio_shapefile_roundtrip(tmp_path: Path):
    shp = tmp_path / "crowns.shp"
    write_shapefile(shp, _payload(), crs=rasterio.crs.CRS.from_epsg(32611))
    records = _load_polygons(shp)
    assert len(records) == 1
    assert records[0]["crown_id"] == 7
    assert records[0]["area"] > 0


def test_raster_clip_uses_pyogrio_geometry(tmp_path: Path):
    shp = tmp_path / "clip.shp"
    write_shapefile(shp, _payload(), crs=rasterio.crs.CRS.from_epsg(32611))

    src = tmp_path / "src.tif"
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    profile = {
        "driver": "GTiff",
        "height": 4,
        "width": 4,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32611",
        "transform": from_origin(0, 4, 1, 1),
    }
    with rasterio.open(src, "w", **profile) as dst:
        dst.write(data, 1)

    out = tmp_path / "clip.tif"
    run_clip(src, shp, out)
    assert out.is_file()
    with rasterio.open(out) as ds:
        assert ds.width > 0 and ds.height > 0
