"""Standalone, transactional multi-plot LAS/LAZ extraction for FAST-GIS.

Does not invoke or modify FAST-GC classification or tiling pipelines.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import time
from pathlib import Path

import laspy
from pyproj import CRS

from .clip import clip_las
from .plot_io import import_plots
from .progress import gis_progress
from .survey_catalog import build_survey_catalog
from .survey_regions import discover_survey_regions
from .survey_index import SurveyTileIndex
from .multi_clip import clip_las_sources


def _write_json(path: Path, value: dict) -> None:
    """Use a strict JSON document; non-JSON source attributes become strings."""
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")


def _metre_crs(crs: CRS) -> bool:
    """Require projected horizontal axes expressed in metres."""
    return bool(
        crs.is_projected
        and len(crs.axis_info) >= 2
        and all(
            math.isclose(float(axis.unit_conversion_factor), 1.0, rel_tol=0, abs_tol=1e-9)
            for axis in crs.axis_info[:2]
        )
    )


def extract_plots(
    input_las,
    plot_file,
    output_dir,
    *,
    buffer=0.0,
    chunk_size=500_000,
    source_crs=None,
    layer=None,
):
    """Extract plots from one LAS/LAZ or a catalogued multi-file survey.

    For folder input, ready source tiles are indexed by bounding box and only
    tiles intersecting each core/buffer geometry are scanned. Exact duplicate
    XYZ locations caused by source-tile overlap are removed during assembly.
    """
    source = Path(input_las).resolve()
    definitions = Path(plot_file).resolve()
    target = Path(output_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if not definitions.is_file():
        raise FileNotFoundError(definitions)
    if target.exists():
        raise FileExistsError(target)
    if isinstance(buffer, bool) or not isinstance(buffer, (int, float)) or not math.isfinite(buffer) or buffer < 0:
        raise ValueError("buffer must be a finite nonnegative number of metres")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    catalog = build_survey_catalog(source)
    ready = [t for t in catalog["tiles"] if t["status"] == "ready"]
    if not ready:
        raise ValueError("Survey contains no CRS-resolved LAS/LAZ sources")
    index = SurveyTileIndex(catalog)
    crs = index.crs
    if not _metre_crs(crs):
        raise ValueError("Plot extraction requires a projected CRS with metre horizontal units")
    regions = discover_survey_regions(catalog)
    source_points = sum(int(t.get("point_count", 0)) for t in ready)

    plots = import_plots(definitions, crs, source_crs=source_crs, layer=layer)
    if not plots:
        raise ValueError("No nonempty polygon plots found")
    names = [p.plot_name for p in plots]
    if len({n.casefold() for n in names}) != len(names):
        raise ValueError("Plot directory names are not unique")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}_staging_", dir=target.parent))
    started = time.perf_counter()
    entries = []
    try:
        bar = gis_progress("CLIPPING", len(plots), unit="plot")
        try:
            for plot in plots:
                name = plot.plot_name
                if plot.geometry.is_empty or not plot.geometry.is_valid:
                    raise ValueError(f"Invalid transformed plot geometry: {name}")
                if not CRS.from_user_input(plot.processing_crs).equals(crs):
                    raise ValueError(f"Plot processing CRS differs from survey CRS: {name}")
                folder = staging / name
                folder.mkdir()
                suffix = Path(ready[0]["path"]).suffix.lower()
                core_rel = f"{name}/{name}{suffix}"
                core_sources = index.paths(plot.geometry)
                core = clip_las_sources(core_sources, staging / core_rel, geometry=plot.geometry,
                                        chunk_size=chunk_size, deduplicate_xyz=True)
                buffer_rel = None
                buffered = None
                if buffer > 0:
                    buffer_geom = plot.geometry.buffer(buffer)
                    buffer_rel = f"{name}/{name}_buffer{buffer:g}{suffix}"
                    buffered = clip_las_sources(index.paths(buffer_geom), staging / buffer_rel,
                                                geometry=buffer_geom, chunk_size=chunk_size,
                                                deduplicate_xyz=True)
                entry = {
                    "plot_id": plot.plot_id, "plot_name": name,
                    "source_feature_id": plot.source_feature_id, "source_file": plot.source_file,
                    "source_crs": plot.source_crs, "processing_crs": plot.processing_crs,
                    "geometry_wkt": plot.geometry.wkt, "attributes": plot.attributes,
                    "core_file": core_rel, "core_points": core["selected_points"],
                    "core_source_count": core["source_count"],
                    "core_duplicates_removed": core["duplicates_removed"],
                    "core_sources": [x["path"] for x in core["sources"]],
                    "buffer_m": float(buffer), "buffer_file": buffer_rel,
                    "buffer_points": buffered["selected_points"] if buffered else None,
                    "buffer_source_count": buffered["source_count"] if buffered else None,
                    "buffer_duplicates_removed": buffered["duplicates_removed"] if buffered else None,
                    "buffer_sources": [x["path"] for x in buffered["sources"]] if buffered else [],
                }
                _write_json(folder / f"Clip_{name}.json", entry)
                entries.append(entry)
                bar.update(1)
        finally:
            bar.close()
        manifest = {
            "schema_version": 2, "input_path": str(source), "source_points": source_points,
            "source_crs": crs.to_wkt(), "survey_tile_count": catalog["tile_count"],
            "survey_ready_count": catalog["ready_count"], "survey_regions": regions["regions"],
            "plot_file": str(definitions), "buffer_m": float(buffer), "chunk_size": chunk_size,
            "deduplication": "output_grid_exact_xyz", "plot_count": len(entries), "plots": entries,
        }
        _write_json(staging / "survey_catalog.json", catalog)
        _write_json(staging / "survey_regions.json", regions)
        _write_json(staging / "workspace_manifest.json", manifest)
        if target.exists():
            raise FileExistsError(target)
        staging.rename(target)
        print(f"Published {len(entries)} plots in {time.perf_counter()-started:.2f} s: {target}", flush=True)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-las", required=True)
    parser.add_argument("--plots", required=True, help="KML, KMZ, SHP, GPKG, GeoJSON, or CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--buffer", type=float, default=0.0, help="Metres")
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--source-crs", help="Required for CSV plot centers")
    parser.add_argument("--layer", help="Optional GPKG/vector layer")
    args = parser.parse_args(argv)
    extract_plots(
        args.input_las, args.plots, args.output_dir,
        buffer=args.buffer, chunk_size=args.chunk_size,
        source_crs=args.source_crs, layer=args.layer,
    )


if __name__ == "__main__":
    main()
