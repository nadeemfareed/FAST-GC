"""Workflow-level FAST_TERRAIN regression tests without expensive LiDAR processing."""
from pathlib import Path

import pytest

import fastgc.core as core


@pytest.mark.parametrize("terrain_output", ["raster", "vector", "both"])
def test_tile_run_merge_derives_from_continuous_dem(monkeypatch, tmp_path, terrain_output):
    input_las = tmp_path / "input.las"
    input_las.touch()
    workspace = tmp_path / "workspace"
    tiles = workspace / "tiles"
    tiles.mkdir(parents=True)
    processed = workspace / "Processed_ALS"
    processed.mkdir()
    merged_dem = workspace / "Merged_ALS" / "FAST_DEM" / "merged.tif"
    events = []

    def fake_tile(**kwargs):
        events.append(("tile", kwargs))
        return {"workspace_root": str(workspace), "tiles_dir": str(tiles), "tiles": []}

    def fake_process(**kwargs):
        events.append(("process", kwargs))
        return str(processed)

    def fake_merge(**kwargs):
        events.append(("merge", kwargs))
        assert "FAST_TERRAIN" not in kwargs["products"]
        if "FAST_DEM" in kwargs["products"]:
            merged_dem.parent.mkdir(parents=True, exist_ok=True)
            merged_dem.touch()
            return {"FAST_DEM": str(merged_dem)}
        return {}

    def fake_raster(**kwargs):
        events.append(("raster", kwargs))
        assert Path(kwargs["dem_fp"]) == merged_dem
        output = Path(kwargs["output_root"]) / "slope_degrees"
        output.mkdir(parents=True, exist_ok=True)
        (output / "synthetic_FAST_TERRAIN_slope_degrees.tif").touch()

    def fake_vector(**kwargs):
        events.append(("vector", kwargs))
        assert Path(kwargs["dem_fp"]) == merged_dem
        return Path(kwargs["output_root"]) / "streams.gpkg"

    monkeypatch.setattr(core, "tile_las_dataset", fake_tile)
    monkeypatch.setattr(core, "_run_processing_with_optional_fpfix", fake_process)
    monkeypatch.setattr(core, "merge_processed_tiles", fake_merge)
    monkeypatch.setattr(core, "run_terrain_from_dem", fake_raster)
    monkeypatch.setattr(core, "run_hydrology_vector_from_dem", fake_vector)
    monkeypatch.setattr(core, "_write_view_registry", lambda *args, **kwargs: None)

    result = core.run_fastgc(
        in_path=str(input_las),
        out_dir=str(tmp_path / "out"),
        sensor_mode="ALS",
        workflow="tile-run-merge",
        products=["FAST_TERRAIN"],
        terrain_products=["slope_degrees"],
        terrain_output=terrain_output,
        stream_threshold_area_m2=321.0,
        stream_min_order=3,
        n_jobs=1,
        apply_fp_fix=False,
    )

    assert Path(result) == workspace / "Merged_ALS"
    assert [name for name, _ in events].count("merge") == 1
    merge_kwargs = next(kwargs for name, kwargs in events if name == "merge")
    assert merge_kwargs["products"] == ["FAST_DEM"]
    names = [name for name, _ in events]
    assert names.index("merge") < min(
        names.index(name) for name in ("raster", "vector") if name in names
    )
    assert ("raster" in names) == (terrain_output in {"raster", "both"})
    assert ("vector" in names) == (terrain_output in {"vector", "both"})
    for name, kwargs in events:
        if name in {"raster", "vector"}:
            assert kwargs["stream_threshold_area_m2"] == 321.0
        if name == "vector":
            assert kwargs["stream_min_order"] == 3


def test_tile_run_merge_rejects_missing_merged_dem(monkeypatch, tmp_path):
    input_las = tmp_path / "input.las"
    input_las.touch()
    workspace = tmp_path / "workspace"
    tiles = workspace / "tiles"
    tiles.mkdir(parents=True)
    processed = workspace / "Processed_ALS"
    processed.mkdir()

    monkeypatch.setattr(core, "tile_las_dataset", lambda **kw: {
        "workspace_root": str(workspace), "tiles_dir": str(tiles), "tiles": []
    })
    monkeypatch.setattr(core, "_run_processing_with_optional_fpfix", lambda **kw: str(processed))
    monkeypatch.setattr(core, "merge_processed_tiles", lambda **kw: {})

    with pytest.raises(RuntimeError, match="merged FAST_DEM"):
        core.run_fastgc(
            in_path=str(input_las), out_dir=str(tmp_path / "out"),
            sensor_mode="ALS", workflow="tile-run-merge",
            products=["FAST_TERRAIN"], n_jobs=1, apply_fp_fix=False,
        )
