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
    metadata_file: Path
    output_root: Path


@dataclass(frozen=True)
class PlotRunManifest:
    root: Path
    sensor_mode: str
    jobs: tuple[PlotRunJob, ...]
    raw: dict[str, Any]


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
