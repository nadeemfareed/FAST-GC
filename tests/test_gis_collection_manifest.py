import json
from pathlib import Path

import pytest

from fastgc.gis.collection_manifest import (
    build_external_collection_manifest,
    load_external_collection_manifest,
    write_external_collection_manifest,
)
from fastgc.gis.plots_run import (
    load_external_plots,
    resolve_plots_run_input,
)


def _external_collection(tmp_path: Path):
    source = tmp_path / "qualified"
    source.mkdir()

    (source / "Tree_A.las").write_bytes(b"a")
    (source / "Tree_B.laz").write_bytes(b"b")

    collection = tmp_path / "products" / "TLS_plots"

    manifest = load_external_plots(
        source,
        collection_root=collection,
        sensor_mode="TLS",
    )

    return source, collection, manifest


def test_external_manifest_records_provenance(tmp_path):
    source, collection, manifest = _external_collection(tmp_path)

    payload = build_external_collection_manifest(
        source=source,
        collection_root=collection,
        sensor_mode="TLS",
        jobs=manifest.jobs,
    )

    assert payload["schema"] == "fastgc.plots.collection"
    assert payload["source_origin"] == "external"
    assert payload["collection_type"] == "external_plots"
    assert payload["processing_workflow"] == "plots-run"
    assert payload["sensor_mode"] == "TLS"

    support = payload["spatial_support"]
    assert support["type"] == "whole_input_file"
    assert support["core_geometry_authoritative"] is False
    assert support["buffer_known"] is False

    assert payload["plot_count"] == 2

    for plot in payload["plots"]:
        assert plot["source_origin"] == "external"
        assert plot["fast_gis_generated"] is False


def test_external_manifest_reconciles_existing_products(tmp_path):
    source, collection, manifest = _external_collection(tmp_path)

    gc = collection / "FAST_GC"
    gc.mkdir(parents=True)

    (gc / "Tree_A.las").write_bytes(b"gc-a")
    (gc / "Tree_B.laz").write_bytes(b"gc-b")

    chm = collection / "FAST_CHM" / "p2r"
    chm.mkdir(parents=True)

    (chm / "Tree_A.tif").write_bytes(b"chm-a")

    payload = build_external_collection_manifest(
        source=source,
        collection_root=collection,
        sensor_mode="TLS",
        jobs=manifest.jobs,
    )

    assert payload["products"]["FAST_GC"] == {
        "status": "complete",
        "plots_complete": 2,
        "plots_total": 2,
    }

    assert payload["products"]["FAST_CHM"] == {
        "status": "partial",
        "plots_complete": 1,
        "plots_total": 2,
    }


def test_external_manifest_write_and_load(tmp_path):
    source, collection, manifest = _external_collection(tmp_path)

    path = write_external_collection_manifest(
        source=source,
        collection_root=collection,
        sensor_mode="TLS",
        jobs=manifest.jobs,
    )

    assert path == collection / "plots_manifest.json"
    assert path.is_file()
    assert not (collection / ".plots_manifest.json.tmp").exists()

    data = load_external_collection_manifest(collection)

    assert data["source_origin"] == "external"
    assert data["plot_count"] == 2


def test_resolver_understands_persistent_external_collection(tmp_path):
    source, collection, manifest = _external_collection(tmp_path)

    write_external_collection_manifest(
        source=source,
        collection_root=collection,
        sensor_mode="TLS",
        jobs=manifest.jobs,
    )

    resolved = resolve_plots_run_input(
        collection,
        out_dir=None,
        sensor_mode=None,
    )

    assert resolved.source_type == "external"
    assert resolved.sensor_mode == "TLS"
    assert [job.plot_name for job in resolved.jobs] == [
        "Tree_A",
        "Tree_B",
    ]
    assert all(job.metadata_file is None for job in resolved.jobs)


def test_external_collection_rejects_sensor_conflict(tmp_path):
    source, collection, manifest = _external_collection(tmp_path)

    write_external_collection_manifest(
        source=source,
        collection_root=collection,
        sensor_mode="TLS",
        jobs=manifest.jobs,
    )

    with pytest.raises(ValueError, match="sensor_mode conflicts"):
        resolve_plots_run_input(
            collection,
            out_dir=None,
            sensor_mode="ALS",
        )


def test_unknown_plots_manifest_schema_is_not_treated_as_fast_gis(tmp_path):
    root = tmp_path / "collection"
    root.mkdir()

    (root / "plots_manifest.json").write_text(
        json.dumps(
            {
                "schema": "some.other.schema",
                "schema_version": 1,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported plots_manifest"):
        resolve_plots_run_input(
            root,
            out_dir=None,
            sensor_mode="TLS",
        )
