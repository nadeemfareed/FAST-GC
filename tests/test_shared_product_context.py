from __future__ import annotations

from fastgc import io_las


def test_shared_dem_normalized_loads_and_builds_once(monkeypatch):
    calls = {"load": 0, "dem": 0, "write_dem": 0, "write_norm": 0}
    ctx = {"classified_fp": "tile.las", "base": "tile"}
    dem_pack = {"dem": object(), "ground_mask": object(), "xmin": 0.0, "ymax": 1.0}

    def fake_load(fp):
        assert fp == "tile.las"
        calls["load"] += 1
        return ctx

    def fake_build(got_ctx, grid_res, dem_method="nearest"):
        assert got_ctx is ctx
        assert grid_res == 0.5
        assert dem_method == "nearest"
        calls["dem"] += 1
        return dem_pack

    def fake_write_dem(got_ctx, got_pack, product_dirs, grid_res):
        assert got_ctx is ctx and got_pack is dem_pack
        calls["write_dem"] += 1
        return {"status": "ok", "output": "dem.tif"}

    def fake_write_norm(got_ctx, got_pack, product_dirs, grid_res):
        assert got_ctx is ctx and got_pack is dem_pack
        calls["write_norm"] += 1
        return {"status": "ok", "output": "normalized.las"}

    monkeypatch.setattr(io_las, "_load_product_context", fake_load)
    monkeypatch.setattr(io_las, "_build_dem_bundle", fake_build)
    monkeypatch.setattr(io_las, "_write_dem_from_context", fake_write_dem)
    monkeypatch.setattr(io_las, "_write_normalized_from_context", fake_write_norm)

    result = io_las._write_dem_and_normalized_shared(
        "tile.las",
        {io_las.PRODUCT_DEM: "d", io_las.PRODUCT_NORMALIZED: "n"},
        0.5,
        dem_method="nearest",
    )

    assert result["status"] == "ok"
    assert result["outputs"][io_las.PRODUCT_DEM] == "dem.tif"
    assert result["outputs"][io_las.PRODUCT_NORMALIZED] == "normalized.las"
    assert calls == {"load": 1, "dem": 1, "write_dem": 1, "write_norm": 1}


def test_independent_writers_keep_existing_public_behavior(monkeypatch):
    ctx = {"classified_fp": "tile.las", "base": "tile"}
    pack = {"dem": object(), "ground_mask": object(), "xmin": 0.0, "ymax": 1.0}

    monkeypatch.setattr(io_las, "_load_and_build_dem", lambda *a, **k: (ctx, pack, None))
    monkeypatch.setattr(
        io_las,
        "_write_dem_from_context",
        lambda *a, **k: {"status": "ok", "product": io_las.PRODUCT_DEM, "output": "dem.tif"},
    )
    monkeypatch.setattr(
        io_las,
        "_write_normalized_from_context",
        lambda *a, **k: {"status": "ok", "product": io_las.PRODUCT_NORMALIZED, "output": "norm.las"},
    )

    dem = io_las._write_dem("tile.las", {}, 0.5, "nearest")
    norm = io_las._write_normalized("tile.las", {}, 0.5, "nearest")
    assert dem["product"] == io_las.PRODUCT_DEM
    assert norm["product"] == io_las.PRODUCT_NORMALIZED
