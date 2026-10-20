from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from fastgc import cli, core


def _dem(tmp_path):
    path = tmp_path / "input.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=8, width=8, count=1,
        dtype="float32", transform=from_origin(0, 8, 1, 1),
        crs="EPSG:32617", nodata=-9999.0,
    ) as dst:
        dst.write(np.arange(64, dtype="float32").reshape(8, 8), 1)
    return path


def test_cli_accepts_dem_without_sensor_mode(tmp_path):
    dem = _dem(tmp_path)
    with patch.object(cli, "run_fastgc") as runner:
        cli.main(["--in_path", str(dem), "--workflow", "derive-only",
                  "--products", "FAST_TERRAIN", "--terrain_products", "hillshade"])
    assert runner.call_args.kwargs["sensor_mode"] is None


def test_direct_dem_raster_and_vector_routing(tmp_path):
    dem = _dem(tmp_path)
    out = tmp_path / "outputs"
    with patch.object(core, "run_terrain_from_dem") as raster, \
         patch.object(core, "run_hydrology_vector_from_dem") as vector:
        result = core.run_fastgc(
            in_path=str(dem), out_dir=str(out), sensor_mode=None,
            workflow="derive-only", products=["FAST_TERRAIN"],
            terrain_products=["slope_degrees"], terrain_output="both",
            stream_threshold_area_m2=1250.0, stream_min_order=2,
        )
    assert result == str(out / "FAST_TERRAIN")
    assert raster.call_args.kwargs["dem_fp"] == dem
    assert raster.call_args.kwargs["terrain_products"] == ["slope_degrees"]
    assert raster.call_args.kwargs["stream_threshold_area_m2"] == 1250.0
    assert vector.call_args.kwargs["stream_min_order"] == 2
    assert vector.call_args.kwargs["output_root"] == out / "FAST_TERRAIN"


@pytest.mark.parametrize("output,rasters,vectors", [
    ("raster", 1, 0), ("vector", 0, 1), ("both", 1, 1),
])
def test_output_modes(tmp_path, output, rasters, vectors):
    dem = _dem(tmp_path)
    with patch.object(core, "run_terrain_from_dem") as raster, \
         patch.object(core, "run_hydrology_vector_from_dem") as vector:
        core.run_fastgc(in_path=str(dem), out_dir=str(tmp_path / "outputs"),
                        sensor_mode=None, workflow="derive-only",
                        products=["FAST_TERRAIN"], terrain_output=output)
    assert raster.call_count == rasters
    assert vector.call_count == vectors


def test_direct_dem_rejects_unrelated_products(tmp_path):
    dem = _dem(tmp_path)
    with pytest.raises(ValueError, match="only --products FAST_TERRAIN"):
        core.run_fastgc(in_path=str(dem), out_dir=str(tmp_path / "outputs"),
                        sensor_mode=None, workflow="derive-only",
                        products=["FAST_DEM", "FAST_TERRAIN"])


def test_direct_dem_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Input DEM not found"):
        core.run_fastgc(in_path=str(tmp_path / "missing.tif"),
                        out_dir=str(tmp_path / "outputs"), sensor_mode=None,
                        workflow="derive-only", products=["FAST_TERRAIN"])
