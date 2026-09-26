from __future__ import annotations

import json
from pathlib import Path

import pytest

import fastgc.core as core
from fastgc.gis.plots_run import (
    load_external_plots,
    load_plots_manifest,
    publish_plot_products,
    resolve_plots_run_input,
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


def test_plots_run_masks_before_publish_and_cleanup(tmp_path, monkeypatch):
    """plots-run must mask each plot before publication and cleanup."""
    import fastgc.gis.core_mask as core_mask_module
    import fastgc.gis.plots_run as plots_run_module

    root = _make_collection(tmp_path)
    events = []

    def fake_run_fastgc(**kwargs):
        events.append(("run", Path(kwargs["out_dir"]).name))
        return str(kwargs["out_dir"])

    def fake_mask(*, work_root, metadata_path):
        events.append(
            ("mask", Path(work_root).name, Path(metadata_path).name)
        )
        return []

    def fake_publish(*, collection_root, work_root, overwrite):
        events.append(("publish", Path(work_root).name))
        return []

    def fake_cleanup(work_root):
        events.append(("cleanup", Path(work_root).name))

    monkeypatch.setattr(core, "run_fastgc", fake_run_fastgc)
    monkeypatch.setattr(
        core_mask_module, "mask_plot_rasters_to_core", fake_mask
    )
    monkeypatch.setattr(
        plots_run_module, "publish_plot_products", fake_publish
    )
    monkeypatch.setattr(
        plots_run_module, "cleanup_plot_work_root", fake_cleanup
    )

    real_run = (
        test_plots_run_dispatches_each_plot_to_existing_run_workflow._real
    )

    real_run(
        in_path=str(root),
        out_dir=None,
        sensor_mode="ALS",
        products=["FAST_GC", "FAST_DEM"],
        workflow="plots-run",
    )

    assert events == [
        ("run", "Plot_001"),
        ("mask", "Plot_001", "Clip_Plot_001.json"),
        ("publish", "Plot_001"),
        ("cleanup", "Plot_001"),
        ("run", "Plot_002"),
        ("mask", "Plot_002", "Clip_Plot_002.json"),
        ("publish", "Plot_002"),
        ("cleanup", "Plot_002"),
    ]


def test_plots_run_mask_failure_prevents_publish_and_cleanup(
    tmp_path, monkeypatch
):
    """A failed core mask must not publish or clean the private work."""
    import fastgc.gis.core_mask as core_mask_module
    import fastgc.gis.plots_run as plots_run_module

    root = _make_collection(tmp_path)
    events = []

    def fake_run_fastgc(**kwargs):
        events.append(("run", Path(kwargs["out_dir"]).name))
        return str(kwargs["out_dir"])

    def failing_mask(*, work_root, metadata_path):
        events.append(("mask", Path(work_root).name))
        raise ValueError("intentional core-mask failure")

    def forbidden_publish(**kwargs):
        events.append(("publish", "ERROR"))

    def forbidden_cleanup(*args, **kwargs):
        events.append(("cleanup", "ERROR"))

    monkeypatch.setattr(core, "run_fastgc", fake_run_fastgc)
    monkeypatch.setattr(
        core_mask_module, "mask_plot_rasters_to_core", failing_mask
    )
    monkeypatch.setattr(
        plots_run_module, "publish_plot_products", forbidden_publish
    )
    monkeypatch.setattr(
        plots_run_module, "cleanup_plot_work_root", forbidden_cleanup
    )

    real_run = (
        test_plots_run_dispatches_each_plot_to_existing_run_workflow._real
    )

    with pytest.raises(ValueError, match="intentional core-mask failure"):
        real_run(
            in_path=str(root),
            out_dir=None,
            sensor_mode="ALS",
            products=["FAST_GC", "FAST_DEM"],
            workflow="plots-run",
        )

    assert events == [
        ("run", "Plot_001"),
        ("mask", "Plot_001"),
    ]


def test_plots_run_completion_requires_all_requested_products(tmp_path):
    from fastgc.core import _plots_run_requested_products_complete

    plot_name = "Plot_001"

    (tmp_path / "FAST_GC").mkdir()
    (tmp_path / "FAST_DEM").mkdir()
    (tmp_path / "FAST_CHM" / "p2r").mkdir(parents=True)

    (tmp_path / "FAST_GC" / f"{plot_name}.las").write_bytes(b"x")
    (tmp_path / "FAST_DEM" / f"{plot_name}.tif").write_bytes(b"x")
    (tmp_path / "FAST_CHM" / "p2r" / f"{plot_name}.tif").write_bytes(b"x")

    assert _plots_run_requested_products_complete(
        collection_root=tmp_path,
        plot_name=plot_name,
        requested_products=["FAST_GC", "FAST_DEM", "FAST_CHM"],
        resolved_products=[
            "FAST_GC",
            "FAST_DEM",
            "FAST_NORMALIZED",
            "FAST_CHM",
        ],
        chm_method="p2r",
        chm_methods=None,
        chm_surface_method="p2r",
    )


def test_plots_run_completion_rejects_partial_products(tmp_path):
    from fastgc.core import _plots_run_requested_products_complete

    plot_name = "Plot_001"

    (tmp_path / "FAST_GC").mkdir()
    (tmp_path / "FAST_DEM").mkdir()

    (tmp_path / "FAST_GC" / f"{plot_name}.las").write_bytes(b"x")
    (tmp_path / "FAST_DEM" / f"{plot_name}.tif").write_bytes(b"x")

    assert not _plots_run_requested_products_complete(
        collection_root=tmp_path,
        plot_name=plot_name,
        requested_products=["FAST_GC", "FAST_DEM", "FAST_CHM"],
        resolved_products=[
            "FAST_GC",
            "FAST_DEM",
            "FAST_NORMALIZED",
            "FAST_CHM",
        ],
        chm_method="p2r",
        chm_methods=None,
        chm_surface_method="p2r",
    )


def test_external_plots_directory_preserves_original_names(tmp_path):
    source = tmp_path / "qualified"
    source.mkdir()

    a = source / "Tree_A.las"
    b = source / "tree_complex_02.laz"
    a.write_bytes(b"a")
    b.write_bytes(b"b")

    collection = tmp_path / "out" / "TLS_plots"

    manifest = load_external_plots(
        source,
        collection_root=collection,
        sensor_mode="TLS",
    )

    assert manifest.source_type == "external"
    assert manifest.sensor_mode == "TLS"
    assert [j.plot_name for j in manifest.jobs] == [
        "Tree_A",
        "tree_complex_02",
    ]
    assert [j.point_file for j in manifest.jobs] == [
        a.resolve(),
        b.resolve(),
    ]
    assert all(j.metadata_file is None for j in manifest.jobs)


@pytest.mark.parametrize("sensor", ["ALS", "ULS", "TLS"])
def test_external_plots_routes_to_sensor_collection(tmp_path, sensor):
    source = tmp_path / "qualified"
    source.mkdir()
    (source / "site_01.las").write_bytes(b"x")

    out = tmp_path / "products"

    manifest = resolve_plots_run_input(
        source,
        out_dir=out,
        sensor_mode=sensor,
    )

    assert manifest.source_type == "external"
    assert manifest.root == (out / f"{sensor}_plots").resolve()
    assert manifest.sensor_mode == sensor


def test_external_plots_requires_sensor_mode(tmp_path):
    source = tmp_path / "qualified"
    source.mkdir()
    (source / "site_01.las").write_bytes(b"x")

    with pytest.raises(ValueError, match="requires --sensor_mode"):
        resolve_plots_run_input(
            source,
            out_dir=tmp_path / "out",
            sensor_mode=None,
        )


def test_external_plots_nonrecursive_discovery(tmp_path):
    source = tmp_path / "qualified"
    source.mkdir()
    nested = source / "nested"
    nested.mkdir()

    (source / "top.las").write_bytes(b"x")
    (nested / "hidden.las").write_bytes(b"x")

    manifest = load_external_plots(
        source,
        collection_root=tmp_path / "out" / "TLS_plots",
        sensor_mode="TLS",
    )

    assert [j.plot_name for j in manifest.jobs] == ["top"]


def test_manifest_collection_stays_strict_when_extra_las_exists(tmp_path):
    root = _make_collection(tmp_path)
    extra = root / "NOT_IN_MANIFEST.las"
    extra.write_bytes(b"x")

    manifest = resolve_plots_run_input(
        root,
        out_dir=None,
        sensor_mode="ALS",
    )

    assert manifest.source_type == "fast_gis"
    assert "NOT_IN_MANIFEST" not in [j.plot_name for j in manifest.jobs]


def test_external_plots_default_output_is_sibling_collection(tmp_path):
    source = tmp_path / "qualified"
    source.mkdir()
    original = source / "Tree_A.las"
    original.write_bytes(b"original")

    manifest = resolve_plots_run_input(
        source,
        out_dir=None,
        sensor_mode="TLS",
    )

    expected = tmp_path / "qualified_FAST_GC" / "TLS_plots"
    assert manifest.root == expected.resolve()
    assert original.read_bytes() == b"original"
    assert manifest.jobs[0].point_file == original.resolve()


def test_external_plots_duplicate_stems_case_insensitive_rejected(tmp_path):
    source = tmp_path / "qualified"
    source.mkdir()

    (source / "Tree_A.las").write_bytes(b"a")
    (source / "tree_a.laz").write_bytes(b"b")

    with pytest.raises(ValueError, match="duplicate plot name"):
        load_external_plots(
            source,
            collection_root=tmp_path / "out" / "TLS_plots",
            sensor_mode="TLS",
        )


def test_external_plots_run_does_not_require_core_metadata(
    tmp_path,
    monkeypatch,
):
    """External LAS/LAZ plots must run without FAST-GIS core metadata."""
    source = tmp_path / "external"
    source.mkdir()
    point_file = source / "Existing_Plot_17.las"
    point_file.write_bytes(b"external")

    out = tmp_path / "products"

    calls = []

    def fake_run_fastgc(**kwargs):
        calls.append(kwargs)

        work = Path(kwargs["out_dir"])
        product = work / "FAST_GC"
        product.mkdir(parents=True, exist_ok=True)
        (product / "Existing_Plot_17.las").write_bytes(b"gc")

        return str(work)

    monkeypatch.setattr(core, "run_fastgc", fake_run_fastgc)

    real_run = (
        test_external_plots_run_does_not_require_core_metadata._real
    )

    result = real_run(
        in_path=str(source),
        out_dir=str(out),
        sensor_mode="TLS",
        products=["FAST_GC"],
        workflow="plots-run",
    )

    collection = out / "TLS_plots"

    assert Path(result) == collection.resolve()
    assert len(calls) == 1
    assert Path(calls[0]["in_path"]) == point_file.resolve()
    assert calls[0]["sensor_mode"] == "TLS"
    assert calls[0]["workflow"] == "run"

    published = (
        collection
        / "FAST_GC"
        / "Existing_Plot_17.las"
    )
    assert published.is_file()

    assert not (
        collection / ".fastgc_plots_run_work"
    ).exists()


test_external_plots_run_does_not_require_core_metadata._real = (
    core.run_fastgc
)


def test_fast_gis_plots_run_still_invokes_core_mask(
    tmp_path,
    monkeypatch,
):
    """Managed FAST-GIS plots must retain metadata-driven core masking."""
    root = _make_collection(tmp_path)

    calls = []
    mask_calls = []

    def fake_run_fastgc(**kwargs):
        calls.append(kwargs)

        plot_name = Path(kwargs["in_path"]).stem
        work = Path(kwargs["out_dir"])
        product = work / "FAST_GC"
        product.mkdir(parents=True, exist_ok=True)
        (product / f"{plot_name}.laz").write_bytes(b"gc")

        return str(work)

    def fake_mask(*, work_root, metadata_path):
        mask_calls.append(
            (Path(work_root), Path(metadata_path))
        )
        return []

    monkeypatch.setattr(core, "run_fastgc", fake_run_fastgc)

    import fastgc.gis.core_mask as core_mask

    monkeypatch.setattr(
        core_mask,
        "mask_plot_rasters_to_core",
        fake_mask,
    )

    real_run = (
        test_fast_gis_plots_run_still_invokes_core_mask._real
    )

    result = real_run(
        in_path=str(root),
        out_dir=None,
        sensor_mode="ALS",
        products=["FAST_GC"],
        workflow="plots-run",
    )

    assert Path(result) == root.resolve()
    assert len(calls) == 2
    assert len(mask_calls) == 2

    assert mask_calls[0][1].name == "Clip_Plot_001.json"
    assert mask_calls[1][1].name == "Clip_Plot_002.json"


test_fast_gis_plots_run_still_invokes_core_mask._real = (
    core.run_fastgc
)

def test_external_manifest_refreshes_after_publish_before_cleanup(
    tmp_path,
    monkeypatch,
):
    """External state must be persisted after publication and before cleanup."""
    source = tmp_path / "external"
    source.mkdir()

    point_file = source / "Tree_01.las"
    point_file.write_bytes(b"external")

    out = tmp_path / "products"

    events = []

    def fake_run_fastgc(**kwargs):
        work = Path(kwargs["out_dir"])
        product = work / "FAST_GC"
        product.mkdir(parents=True, exist_ok=True)
        (product / "Tree_01.las").write_bytes(b"gc")
        return str(work)

    real_publish = core.publish_plot_products if hasattr(
        core, "publish_plot_products"
    ) else None

    from fastgc.gis import plots_run as plots_run_module
    from fastgc.gis import collection_manifest as manifest_module

    original_publish = plots_run_module.publish_plot_products
    original_cleanup = plots_run_module.cleanup_plot_work_root
    original_write = manifest_module.write_external_collection_manifest

    def tracked_publish(**kwargs):
        result = original_publish(**kwargs)
        events.append("publish")
        return result

    def tracked_write(**kwargs):
        path = original_write(**kwargs)

        data = json.loads(path.read_text(encoding="utf-8"))
        state = data["products"].get("FAST_GC")

        if state and state["plots_complete"] == 1:
            events.append("manifest")

        return path

    def tracked_cleanup(work_root):
        events.append("cleanup")
        return original_cleanup(work_root)

    monkeypatch.setattr(core, "run_fastgc", fake_run_fastgc)
    monkeypatch.setattr(
        plots_run_module,
        "publish_plot_products",
        tracked_publish,
    )
    monkeypatch.setattr(
        manifest_module,
        "write_external_collection_manifest",
        tracked_write,
    )
    monkeypatch.setattr(
        plots_run_module,
        "cleanup_plot_work_root",
        tracked_cleanup,
    )

    real_run = (
        test_external_plots_run_does_not_require_core_metadata._real
    )

    result = real_run(
        in_path=str(source),
        out_dir=str(out),
        sensor_mode="TLS",
        products=["FAST_GC"],
        workflow="plots-run",
    )

    collection = out / "TLS_plots"
    manifest_path = collection / "plots_manifest.json"

    assert result == str(collection.resolve())
    assert manifest_path.is_file()

    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert data["source_origin"] == "external"
    assert data["products"]["FAST_GC"]["status"] == "complete"
    assert data["products"]["FAST_GC"]["plots_complete"] == 1
    assert data["products"]["FAST_GC"]["plots_total"] == 1

    assert not (
        collection / ".fastgc_plots_run_work"
    ).exists()
