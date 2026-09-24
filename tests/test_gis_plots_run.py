from __future__ import annotations

import json
from pathlib import Path

import pytest

import fastgc.core as core
from fastgc.gis.plots_run import (
    load_plots_manifest,
    publish_plot_products,
)


def _make_collection(tmp_path: Path, *, sensor_mode: str = "ALS") -> Path:
    root = tmp_path / "ALS_plots"
    root.mkdir()

    plots = []

    for i in (1, 2):
        name = f"Plot_{i:03d}"
        plot_dir = root / name
        plot_dir.mkdir()

        point_file = plot_dir / f"{name}.laz"
        point_file.write_bytes(b"test")

        metadata = plot_dir / f"Clip_{name}.json"
        metadata.write_text(
            json.dumps(
                {
                    "schema": "fastgc.gis.plot",
                    "schema_version": 1,
                    "plot_id": f"plot_{i}",
                    "plot_name": name,
                    "sensor_mode": sensor_mode,
                    "point_file": f"{name}/{name}.laz",
                    "point_extent": "core_plus_buffer",
                    "contains_buffer": True,
                }
            ),
            encoding="utf-8",
        )

        plots.append(
            {
                "plot_id": f"plot_{i}",
                "plot_name": name,
                "point_file": f"{name}/{name}.laz",
                "metadata": f"{name}/Clip_{name}.json",
            }
        )

    (root / "plots_manifest.json").write_text(
        json.dumps(
            {
                "schema": "fastgc.gis.plots",
                "schema_version": 1,
                "sensor_mode": sensor_mode,
                "plot_count": 2,
                "point_extent": "core_plus_buffer",
                "buffer_m": 5.0,
                "plots": plots,
            }
        ),
        encoding="utf-8",
    )

    return root


def test_load_plots_manifest_uses_manifest_entries_only(tmp_path):
    root = _make_collection(tmp_path)

    # This LAS exists under the collection but is deliberately NOT in the
    # manifest. plots-run must never discover it recursively.
    unrelated = root / "UNLISTED.laz"
    unrelated.write_bytes(b"must-not-run")

    manifest = load_plots_manifest(root)

    assert manifest.sensor_mode == "ALS"
    assert len(manifest.jobs) == 2
    assert [j.plot_name for j in manifest.jobs] == [
        "Plot_001",
        "Plot_002",
    ]
    assert unrelated.resolve() not in {
        j.point_file for j in manifest.jobs
    }


def test_load_plots_manifest_rejects_escape(tmp_path):
    root = _make_collection(tmp_path)

    data = json.loads(
        (root / "plots_manifest.json").read_text(encoding="utf-8")
    )
    data["plots"][0]["point_file"] = "../outside.laz"

    (root / "plots_manifest.json").write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes"):
        load_plots_manifest(root)


def test_plots_run_dispatches_each_plot_to_existing_run_workflow(
    tmp_path, monkeypatch
):
    root = _make_collection(tmp_path)

    calls = []

    def fake_run_fastgc(**kwargs):
        calls.append(kwargs)
        return str(kwargs["out_dir"])

    monkeypatch.setattr(core, "run_fastgc", fake_run_fastgc)

    # Keep a reference to the real function because monkeypatching the module
    # symbol above is what the plots-run branch recursively calls.
    real_run = test_plots_run_dispatches_each_plot_to_existing_run_workflow._real
    out = real_run(
        in_path=str(root),
        out_dir=None,
        sensor_mode="ALS",
        products=["FAST_GC", "FAST_DEM"],
        workflow="plots-run",
    )

    assert Path(out) == root.resolve()
    assert len(calls) == 2

    for i, call in enumerate(calls, start=1):
        name = f"Plot_{i:03d}"

        assert Path(call["in_path"]) == (
            root / name / f"{name}.laz"
        ).resolve()

        assert Path(call["out_dir"]) == (
            root
            / ".fastgc_plots_run_work"
            / name
        ).resolve()

        assert call["sensor_mode"] == "ALS"
        assert call["workflow"] == "run"
        assert call["products"] == ["FAST_GC", "FAST_DEM"]
        assert call["recursive"] is False


test_plots_run_dispatches_each_plot_to_existing_run_workflow._real = (
    core.run_fastgc
)


def test_plots_run_rejects_sensor_mismatch(tmp_path):
    root = _make_collection(tmp_path, sensor_mode="ALS")

    with pytest.raises(ValueError, match="conflicts"):
        core.run_fastgc(
            in_path=str(root),
            out_dir=None,
            sensor_mode="ULS",
            products=["FAST_GC"],
            workflow="plots-run",
        )


def test_plots_run_rejects_different_out_dir(tmp_path):
    root = _make_collection(tmp_path)

    with pytest.raises(ValueError, match="--out_dir"):
        core.run_fastgc(
            in_path=str(root),
            out_dir=str(tmp_path / "somewhere_else"),
            sensor_mode="ALS",
            products=["FAST_GC"],
            workflow="plots-run",
        )


def test_load_plots_manifest_accepts_utf8_bom(tmp_path):
    root = _make_collection(tmp_path)

    manifest_path = root / "plots_manifest.json"
    content = manifest_path.read_text(encoding="utf-8")

    manifest_path.write_text(
        content,
        encoding="utf-8-sig",
    )

    manifest = load_plots_manifest(root)

    assert manifest.sensor_mode == "ALS"
    assert len(manifest.jobs) == 2

def test_publish_plot_products_groups_outputs_by_product(
    tmp_path,
):
    root = _make_collection(tmp_path)

    work = (
        root
        / ".fastgc_plots_run_work"
        / "Plot_001"
    )

    (work / "FAST_GC").mkdir(parents=True)
    (work / "FAST_DEM").mkdir(parents=True)
    (work / "FAST_NORMALIZED").mkdir(parents=True)
    (work / "FAST_CHM" / "p2r").mkdir(
        parents=True
    )

    (
        work
        / "FAST_GC"
        / "Plot_001.las"
    ).write_bytes(b"gc")

    (
        work
        / "FAST_DEM"
        / "Plot_001.tif"
    ).write_bytes(b"dem")

    (
        work
        / "FAST_NORMALIZED"
        / "Plot_001.las"
    ).write_bytes(b"normalized")

    (
        work
        / "FAST_CHM"
        / "p2r"
        / "Plot_001.tif"
    ).write_bytes(b"chm")

    published = publish_plot_products(
        collection_root=root,
        work_root=work,
        overwrite=False,
    )

    assert len(published) == 4

    assert (
        root
        / "FAST_GC"
        / "Plot_001.las"
    ).read_bytes() == b"gc"

    assert (
        root
        / "FAST_DEM"
        / "Plot_001.tif"
    ).read_bytes() == b"dem"

    assert (
        root
        / "FAST_NORMALIZED"
        / "Plot_001.las"
    ).read_bytes() == b"normalized"

    assert (
        root
        / "FAST_CHM"
        / "p2r"
        / "Plot_001.tif"
    ).read_bytes() == b"chm"

    # Original FAST-GIS source package remains untouched.
    assert (
        root
        / "Plot_001"
        / "Plot_001.laz"
    ).is_file()

    assert (
        root
        / "Plot_001"
        / "Clip_Plot_001.json"
    ).is_file()


def test_publish_plot_products_respects_overwrite(
    tmp_path,
):
    root = _make_collection(tmp_path)

    work = (
        root
        / ".fastgc_plots_run_work"
        / "Plot_001"
    )

    source = (
        work
        / "FAST_GC"
        / "Plot_001.las"
    )

    source.parent.mkdir(parents=True)
    source.write_bytes(b"new")

    destination = (
        root
        / "FAST_GC"
        / "Plot_001.las"
    )

    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old")

    publish_plot_products(
        collection_root=root,
        work_root=work,
        overwrite=False,
    )

    assert destination.read_bytes() == b"old"

    publish_plot_products(
        collection_root=root,
        work_root=work,
        overwrite=True,
    )

    assert destination.read_bytes() == b"new"
