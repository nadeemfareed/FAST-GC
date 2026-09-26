from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


COLLECTION_SCHEMA = "fastgc.plots.collection"
COLLECTION_SCHEMA_VERSION = 1


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Manifest must contain a JSON object: {path}")

    return data


def read_manifest_schema(path: str | Path) -> str | None:
    data = read_json(path)
    schema = data.get("schema")
    return str(schema).strip() if schema is not None else None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    temp = path.with_name(f".{path.name}.tmp")

    try:
        with temp.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()

    return path


def _relative_output(collection_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(collection_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _product_outputs_for_plot(
    collection_root: Path,
    plot_name: str,
) -> dict[str, list[str]]:
    outputs: dict[str, list[str]] = {}

    if not collection_root.is_dir():
        return outputs

    plot_key = plot_name.casefold()

    for product_dir in sorted(collection_root.glob("FAST_*")):
        if not product_dir.is_dir():
            continue

        matches = sorted(
            p
            for p in product_dir.rglob("*")
            if p.is_file() and p.stem.casefold() == plot_key
        )

        if not matches:
            continue

        outputs[product_dir.name] = [
            _relative_output(collection_root, p)
            for p in matches
        ]

    return outputs


def build_external_collection_manifest(
    *,
    source: str | Path,
    collection_root: str | Path,
    sensor_mode: str,
    jobs: Iterable[Any],
) -> dict[str, Any]:
    source_path = Path(source).resolve()
    collection = Path(collection_root).resolve()
    sensor = str(sensor_mode).strip().upper()

    if sensor not in {"ALS", "ULS", "TLS"}:
        raise ValueError(
            "External collection manifest requires sensor_mode ALS, ULS, or TLS"
        )

    records: list[dict[str, Any]] = []
    product_counts: dict[str, int] = {}

    jobs_tuple = tuple(jobs)

    for job in jobs_tuple:
        discovered = _product_outputs_for_plot(
            collection,
            job.plot_name,
        )

        products: dict[str, Any] = {}

        for product_name, output_files in discovered.items():
            products[product_name] = {
                "status": "complete",
                "outputs": output_files,
            }
            product_counts[product_name] = (
                product_counts.get(product_name, 0) + 1
            )

        records.append(
            {
                "plot_id": str(job.plot_id),
                "plot_name": str(job.plot_name),
                "source_origin": "external",
                "source_file": str(Path(job.point_file).resolve()),
                "fast_gis_generated": False,
                "spatial_support": {
                    "type": "whole_input_file",
                    "core_geometry_authoritative": False,
                    "buffer_known": False,
                },
                "products": products,
            }
        )

    total = len(records)

    products_summary: dict[str, Any] = {}

    for product_name in sorted(product_counts):
        complete = product_counts[product_name]

        products_summary[product_name] = {
            "status": "complete" if complete == total else "partial",
            "plots_complete": complete,
            "plots_total": total,
        }

    return {
        "schema": COLLECTION_SCHEMA,
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "collection_type": "external_plots",
        "source_origin": "external",
        "source_description": (
            "Independent plot point clouds supplied externally. "
            "These plots were not generated or clipped by FAST-GIS."
        ),
        "processing_workflow": "plots-run",
        "sensor_mode": sensor,
        "source_root": str(source_path),
        "collection_root": str(collection),
        "plot_count": total,
        "spatial_support": {
            "type": "whole_input_file",
            "core_geometry_authoritative": False,
            "buffer_known": False,
        },
        "products": products_summary,
        "plots": records,
    }


def write_external_collection_manifest(
    *,
    source: str | Path,
    collection_root: str | Path,
    sensor_mode: str,
    jobs: Iterable[Any],
) -> Path:
    collection = Path(collection_root).resolve()

    payload = build_external_collection_manifest(
        source=source,
        collection_root=collection,
        sensor_mode=sensor_mode,
        jobs=jobs,
    )

    return _atomic_write_json(
        collection / "plots_manifest.json",
        payload,
    )


def load_external_collection_manifest(
    collection_root: str | Path,
) -> dict[str, Any]:
    root = Path(collection_root).resolve()
    path = root / "plots_manifest.json"

    data = read_json(path)

    if data.get("schema") != COLLECTION_SCHEMA:
        raise ValueError(
            f"Invalid FAST-GC plot collection schema: {data.get('schema')!r}"
        )

    if data.get("schema_version") != COLLECTION_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported FAST-GC plot collection schema_version: "
            f"{data.get('schema_version')!r}"
        )

    if data.get("source_origin") != "external":
        raise ValueError(
            "FAST-GC plot collection loader currently expects "
            "source_origin='external'"
        )

    if data.get("processing_workflow") != "plots-run":
        raise ValueError(
            "FAST-GC external collection must use "
            "processing_workflow='plots-run'"
        )

    return data
