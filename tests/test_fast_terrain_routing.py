from pathlib import Path


def test_tile_run_merge_terrain_uses_merged_dem():
    """
    Scientific routing contract:

    tile-run-merge must preserve tile-level FAST_TERRAIN products but must not
    construct the authoritative merged FAST_TERRAIN by mosaicking those
    products. Instead it must:

      1. merge FAST_DEM;
      2. derive FAST_TERRAIN from Merged_<SENSOR>/FAST_DEM.

    This protects continuous terrain derivatives and neighborhood metrics from
    artificial internal LiDAR tile boundaries.
    """
    import fastgc.core as core

    source = Path(core.__file__).read_text(encoding="utf-8")

    # FAST_TERRAIN must be removed from the ordinary merge-product branch.
    assert (
        "if p not in {PRODUCT_CHM, PRODUCT_TERRAIN}"
        in source
    )

    # A DEM must be made available in the merged domain whenever terrain is
    # requested, including the case where FAST_TERRAIN was the only explicit
    # product requested.
    assert (
        "terrain_requested_for_merge and PRODUCT_DEM not in other_products"
        in source
    )

    # The continuous terrain stage must operate on merge_root. The terrain
    # runner resolves FAST_DEM beneath this root and writes FAST_TERRAIN beneath
    # the same root.
    terrain_block = source[
        source.index("if terrain_requested_for_merge:")
        :
        source.index("if cleanup_tiles:")
    ]

    assert "run_terrain_from_dem(" in terrain_block
    assert "dem_fp=merged_dem_fp" in terrain_block
    assert "output_root=merged_terrain_root" in terrain_block

    # The old authoritative route -- passing FAST_TERRAIN into the ordinary
    # merge call -- must not occur inside the new post-merge block.
    assert "products=[PRODUCT_TERRAIN]" not in terrain_block

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from fastgc.merge import merge_processed_tiles
from fastgc.terrain import run_terrain_from_dem


def _write_test_raster(path, array, transform):
    path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32617",
        "transform": transform,
        "nodata": -9999.0,
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.float32), 1)


def test_continuous_terrain_is_derived_from_merged_dem(tmp_path):
    """
    Behavioral contract for tile-run-merge terrain.

    Tile DEMs are merged first. Authoritative merged FAST_TERRAIN must then
    be derived from that continuous DEM, not mosaicked from tile-level
    FAST_TERRAIN rasters.
    """

    processed = tmp_path / "Processed_ALS"
    merged = tmp_path / "Merged_ALS"

    dem_root = processed / "FAST_DEM"

    # Two adjacent 4 x 4 DEM cores at 1 m resolution.
    #
    # Together they form one continuous east-rising plane:
    #
    #   z = x
    #
    # The correct merged slope is therefore 100 percent.
    left = np.tile(
        np.arange(4, dtype=np.float32),
        (4, 1),
    )

    right = np.tile(
        np.arange(4, 8, dtype=np.float32),
        (4, 1),
    )

    left_fp = dem_root / "tile_000_000.tif"
    right_fp = dem_root / "tile_001_000.tif"

    _write_test_raster(
        left_fp,
        left,
        from_origin(0.0, 4.0, 1.0, 1.0),
    )

    _write_test_raster(
        right_fp,
        right,
        from_origin(4.0, 4.0, 1.0, 1.0),
    )

    manifest = {
        "dataset_label": "synthetic",
        "union_bounds": [0.0, 0.0, 8.0, 4.0],
        "tiles": [
            {
                "tile_name": "tile_000_000.laz",
                "core_bounds": [0.0, 0.0, 4.0, 4.0],
            },
            {
                "tile_name": "tile_001_000.laz",
                "core_bounds": [4.0, 0.0, 8.0, 4.0],
            },
        ],
    }

    # Merge ONLY FAST_DEM, exactly as the new continuous-domain branch does.
    outputs = merge_processed_tiles(
        manifest=manifest,
        processed_root=processed,
        merged_root=merged,
        products=["FAST_DEM"],
    )

    assert "FAST_DEM" in outputs

    merged_dem_fp = Path(outputs["FAST_DEM"])
    assert merged_dem_fp.exists()

    # Now derive terrain from the merged DEM.
    run_terrain_from_dem(
        dem_fp=merged_dem_fp,
        output_root=merged / "FAST_TERRAIN",
        terrain_products=["slope_percent"],
        overwrite=True,
        n_jobs=1,
    )

    terrain_files = list(
        (merged / "FAST_TERRAIN" / "slope_percent").glob("*.tif")
    )

    assert len(terrain_files) == 1

    # Public merged-terrain naming contract.
    assert (
        terrain_files[0].name
        == "synthetic_FAST_TERRAIN_slope_percent.tif"
    )

    with rasterio.open(terrain_files[0]) as ds:
        slope = ds.read(1)
        assert ds.crs is not None
        assert ds.crs.to_epsg() == 32617

    # z=x at 1 m horizontal spacing => dz/dx=1, dz/dy=0,
    # therefore slope_percent = 100 everywhere.
    valid = np.isfinite(slope) & (slope != -9999.0)

    assert valid.any()

    np.testing.assert_allclose(
        slope[valid],
        100.0,
        rtol=0.0,
        atol=1.0e-5,
    )

    # Most importantly, the internal former tile boundary must not create
    # a terrain seam.
    np.testing.assert_allclose(
        slope[:, 3:5],
        100.0,
        rtol=0.0,
        atol=1.0e-5,
    )
