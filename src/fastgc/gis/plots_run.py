from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PlotRunJob:
    plot_id: str
    plot_name: str
    point_file: Path
    metadata_file: Path | None
    output_root: Path


@dataclass(frozen=True)
class PlotRunManifest:
    root: Path
    sensor_mode: str
    jobs: tuple[PlotRunJob, ...]
    raw: dict[str, Any]
    source_type: str = "fast_gis"


def _safe_resolve(root: Path, relative: str, *, field: str) -> Path:
    rel = Path(str(relative))

    if rel.is_absolute():
        raise ValueError(
            f"{field} must be relative to the plots collection root: {relative}"
        )

    root_resolved = root.resolve()
    resolved = (root / rel).resolve()

    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"{field} escapes the plots collection root: {relative}"
        ) from exc

    return resolved


def load_plots_manifest(collection_root: str | Path) -> PlotRunManifest:
    root = Path(collection_root)

    if not root.is_dir():
        raise FileNotFoundError(
            f"FAST-GIS plots collection not found: {root}"
        )

    manifest_path = root / "plots_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"FAST-GIS plots manifest not found: {manifest_path}"
        )

    # utf-8-sig accepts both ordinary UTF-8 and UTF-8 files carrying a BOM.
    # This is important for manifests written by common Windows tooling.
    with manifest_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if data.get("schema") != "fastgc.gis.plots":
        raise ValueError(
            "Invalid FAST-GIS plots manifest schema: "
            f"{data.get('schema')!r}"
        )

    version = data.get("schema_version")
    if version != 1:
        raise ValueError(
            f"Unsupported FAST-GIS plots manifest schema_version: {version!r}"
        )

    sensor_mode = str(data.get("sensor_mode", "")).strip().upper()
    if sensor_mode not in {"ALS", "ULS", "TLS"}:
        raise ValueError(
            f"Invalid sensor_mode in plots manifest: {sensor_mode!r}"
        )

    records = data.get("plots")
    if not isinstance(records, list):
        raise ValueError("plots_manifest.json field 'plots' must be a list")

    declared_count = data.get("plot_count")
    if declared_count != len(records):
        raise ValueError(
            "plots_manifest.json plot_count does not match plots list: "
            f"{declared_count!r} != {len(records)}"
        )

    seen_names: set[str] = set()
    seen_points: set[Path] = set()
    jobs: list[PlotRunJob] = []

    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Plot record {index} must be an object")

        plot_name = str(record.get("plot_name", "")).strip()
        plot_id = str(record.get("plot_id", "")).strip()

        if not plot_name:
            raise ValueError(f"Plot record {index} has no plot_name")
        if not plot_id:
            raise ValueError(f"Plot record {index} has no plot_id")

        if plot_name in seen_names:
            raise ValueError(f"Duplicate plot_name in manifest: {plot_name}")
        seen_names.add(plot_name)

        point_rel = record.get("point_file")
        metadata_rel = record.get("metadata")

        if not isinstance(point_rel, str) or not point_rel.strip():
            raise ValueError(f"{plot_name}: missing point_file")
        if not isinstance(metadata_rel, str) or not metadata_rel.strip():
            raise ValueError(f"{plot_name}: missing metadata")

        point_file = _safe_resolve(
            root, point_rel, field=f"{plot_name}.point_file"
        )
        metadata_file = _safe_resolve(
            root, metadata_rel, field=f"{plot_name}.metadata"
        )

        if point_file.suffix.lower() not in {".las", ".laz"}:
            raise ValueError(
                f"{plot_name}: point_file must be LAS/LAZ: {point_file}"
            )

        if not point_file.is_file():
            raise FileNotFoundError(
                f"{plot_name}: point file not found: {point_file}"
            )

        if not metadata_file.is_file():
            raise FileNotFoundError(
                f"{plot_name}: metadata file not found: {metadata_file}"
            )

        if point_file in seen_points:
            raise ValueError(
                f"Duplicate point_file in plots manifest: {point_file}"
            )
        seen_points.add(point_file)

        # The point file must belong to the plot directory. This prevents a
        # malformed manifest from sending one plot's data into another plot's
        # output hierarchy.
        output_root = point_file.parent

        if output_root.name != plot_name:
            raise ValueError(
                f"{plot_name}: point_file must be inside its matching plot "
                f"directory; got {point_file}"
            )

        jobs.append(
            PlotRunJob(
                plot_id=plot_id,
                plot_name=plot_name,
                point_file=point_file,
                metadata_file=metadata_file,
                output_root=output_root,
            )
        )

    return PlotRunManifest(
        root=root.resolve(),
        sensor_mode=sensor_mode,
        jobs=tuple(jobs),
        raw=data,
    )

def load_external_plots(
    input_path: str | Path,
    *,
    collection_root: str | Path,
    sensor_mode: str,
) -> PlotRunManifest:
    """Build a plots-run collection from ordinary LAS/LAZ inputs.

    External source files are referenced in place and are never moved,
    renamed, or modified. Unlike FAST-GIS plots, external plots have no
    authoritative core geometry metadata and therefore are not core-masked.
    """
    source = Path(input_path).resolve()
    collection = Path(collection_root).resolve()

    sensor = str(sensor_mode).strip().upper()
    if sensor not in {"ALS", "ULS", "TLS"}:
        raise ValueError(
            "External plots-run input requires --sensor_mode ALS, ULS, or TLS"
        )

    if source.is_file():
        if source.suffix.lower() not in {".las", ".laz"}:
            raise ValueError(
                f"External plots-run input must be LAS/LAZ: {source}"
            )
        files = [source]
    elif source.is_dir():
        files = sorted(
            p.resolve()
            for p in source.iterdir()
            if p.is_file() and p.suffix.lower() in {".las", ".laz"}
        )
    else:
        raise FileNotFoundError(
            f"External plots-run input not found: {source}"
        )

    if not files:
        raise ValueError(
            f"No LAS/LAZ files found for external plots-run input: {source}"
        )

    seen_names: set[str] = set()
    jobs: list[PlotRunJob] = []

    for index, point_file in enumerate(files, start=1):
        plot_name = point_file.stem

        key = plot_name.casefold()
        if key in seen_names:
            raise ValueError(
                "External plots-run requires unique LAS/LAZ filename stems; "
                f"duplicate plot name: {plot_name}"
            )
        seen_names.add(key)

        jobs.append(
            PlotRunJob(
                plot_id=f"external_{index}",
                plot_name=plot_name,
                point_file=point_file,
                metadata_file=None,
                output_root=collection,
            )
        )

    raw = {
        "schema": "fastgc.plots.collection",
        "schema_version": 1,
        "collection_type": "external_plots",
        "source_origin": "external",
        "source_description": (
            "Independent plot point clouds supplied externally. "
            "These plots were not generated or clipped by FAST-GIS."
        ),
        "processing_workflow": "plots-run",
        "sensor_mode": sensor,
        "source_root": str(source),
        "collection_root": str(collection),
        "plot_count": len(jobs),
        "spatial_support": {
            "type": "whole_input_file",
            "core_geometry_authoritative": False,
            "buffer_known": False,
        },
    }

    return PlotRunManifest(
        root=collection,
        sensor_mode=sensor,
        jobs=tuple(jobs),
        raw=raw,
        source_type="external",
    )


def resolve_plots_run_input(
    input_path: str | Path,
    *,
    out_dir: str | Path | None,
    sensor_mode: str | None,
) -> PlotRunManifest:
    """Resolve FAST-GIS manifest collections or ordinary LAS/LAZ inputs."""
    source = Path(input_path).resolve()

    if source.is_dir() and (source / "plots_manifest.json").is_file():
        from .collection_manifest import (
            COLLECTION_SCHEMA,
            load_external_collection_manifest,
            read_manifest_schema,
        )

        manifest_path = source / "plots_manifest.json"
        schema = read_manifest_schema(manifest_path)

        if schema == "fastgc.gis.plots":
            manifest = load_plots_manifest(source)

            if out_dir is not None:
                requested_out = Path(out_dir).resolve()
                if requested_out != manifest.root:
                    raise ValueError(
                        "FAST-GIS workflow=plots-run publishes products at the "
                        "plots collection root; --out_dir must be omitted or equal "
                        f"to the plots collection root: {manifest.root}"
                    )

            return manifest

        if schema == COLLECTION_SCHEMA:
            data = load_external_collection_manifest(source)

            stored_source = data.get("source_root")
            stored_sensor = str(data.get("sensor_mode", "")).strip().upper()

            if stored_sensor not in {"ALS", "ULS", "TLS"}:
                raise ValueError(
                    "Invalid sensor_mode in external plots collection: "
                    f"{stored_sensor!r}"
                )

            if (
                sensor_mode is not None
                and str(sensor_mode).strip().upper() != stored_sensor
            ):
                raise ValueError(
                    "sensor_mode conflicts with external plots_manifest.json: "
                    f"CLI={str(sensor_mode).strip().upper()} "
                    f"manifest={stored_sensor}"
                )

            if not stored_source:
                raise ValueError(
                    "External plots_manifest.json is missing source_root"
                )

            if out_dir is not None:
                requested_out = Path(out_dir).resolve()
                sensor_dir = f"{stored_sensor}_plots"
                expected = (
                    requested_out
                    if requested_out.name.casefold() == sensor_dir.casefold()
                    else requested_out / sensor_dir
                ).resolve()

                if expected != source:
                    raise ValueError(
                        "External plots collection --out_dir does not resolve "
                        f"to the existing collection root: {source}"
                    )

            return load_external_plots(
                stored_source,
                collection_root=source,
                sensor_mode=stored_sensor,
            )

        raise ValueError(
            "Unsupported plots_manifest.json schema: "
            f"{schema!r}. Expected 'fastgc.gis.plots' or "
            f"{COLLECTION_SCHEMA!r}."
        )

    if sensor_mode is None:
        raise ValueError(
            "External LAS/LAZ workflow=plots-run requires --sensor_mode "
            "ALS, ULS, or TLS"
        )

    sensor = str(sensor_mode).strip().upper()
    if sensor not in {"ALS", "ULS", "TLS"}:
        raise ValueError(
            "External LAS/LAZ workflow=plots-run requires --sensor_mode "
            "ALS, ULS, or TLS"
        )

    if out_dir is None:
        base = source.parent if source.is_file() else source.parent
        name = source.stem if source.is_file() else source.name
        destination = base / f"{name}_FAST_GC" / f"{sensor}_plots"
    else:
        requested = Path(out_dir).resolve()
        sensor_dir = f"{sensor}_plots"
        destination = (
            requested
            if requested.name.casefold() == sensor_dir.casefold()
            else requested / sensor_dir
        )

    return load_external_plots(
        source,
        collection_root=destination,
        sensor_mode=sensor,
    )


def plot_work_root(
    manifest: PlotRunManifest,
    job: PlotRunJob,
) -> Path:
    """Private processing workspace used only by workflow=plots-run."""
    return (
        manifest.root
        / ".fastgc_plots_run_work"
        / job.plot_name
    )


def publish_plot_products(
    *,
    collection_root: str | Path,
    work_root: str | Path,
    overwrite: bool = False,
) -> list[Path]:
    """Publish one plot's FAST_* outputs into collection product folders.

    Normal FAST-GC workflow=run output organization is not changed.
    This function is used only by workflow=plots-run.

    Example:
        private/Plot_001/FAST_GC/Plot_001.las
            ->
        ALS_plots/FAST_GC/Plot_001.las

        private/Plot_001/FAST_CHM/p2r/Plot_001.tif
            ->
        ALS_plots/FAST_CHM/p2r/Plot_001.tif
    """
    collection = Path(collection_root).resolve()
    work = Path(work_root).resolve()

    try:
        work.relative_to(collection)
    except ValueError as exc:
        raise ValueError(
            "plots-run work_root must be inside the "
            f"collection root: {work}"
        ) from exc

    published: list[Path] = []

    if not work.is_dir():
        return published

    for product_dir in sorted(work.glob("FAST_*")):
        if not product_dir.is_dir():
            continue

        sources = sorted(
            p
            for p in product_dir.rglob("*")
            if p.is_file()
        )

        for source in sources:
            relative = source.relative_to(product_dir)

            destination = (
                collection
                / product_dir.name
                / relative
            )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if destination.exists():
                if not overwrite:
                    continue

                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()

            shutil.copy2(source, destination)
            published.append(destination)

    return published


def cleanup_plot_work_root(
    work_root: str | Path,
) -> None:
    """Remove a successfully published plots-run private workspace."""
    work = Path(work_root)

    if work.exists():
        shutil.rmtree(work)

    parent = work.parent

    if (
        parent.name == ".fastgc_plots_run_work"
        and parent.exists()
    ):
        try:
            parent.rmdir()
        except OSError:
            pass
