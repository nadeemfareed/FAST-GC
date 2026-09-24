"""Transactional survey-aware plot extraction for FAST-GIS.

One point cloud is written per plot.  That file contains the authoritative
core plus its processing buffer; core and buffered geometries are retained in
JSON for downstream FAST-GC processing and final core masking.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import time
from pathlib import Path

from pyproj import CRS
from shapely.geometry import mapping

from .multi_clip import clip_las_sources
from .plot_io import import_plots
from .progress import gis_progress
from .publish import publish_directory
from .survey_catalog import build_survey_catalog
from .survey_regions import discover_survey_regions
from .survey_index import SurveyTileIndex


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")


def _metre_crs(crs: CRS) -> bool:
    return bool(crs.is_projected and len(crs.axis_info) >= 2 and all(
        math.isclose(float(a.unit_conversion_factor), 1.0, rel_tol=0, abs_tol=1e-9)
        for a in crs.axis_info[:2]
    ))


def extract_plots(input_las, plot_file, output_dir, *, buffer=5.0, chunk_size=500_000,
                  source_crs=None, layer=None, sensor_mode="ALS",
                  sampling_method="imported", shape=None, radius=None):
    source, definitions, target = map(lambda p: Path(p).resolve(), (input_las, plot_file, output_dir))
    sensor_mode = str(sensor_mode).upper()
    if sensor_mode not in {"ALS", "ULS", "TLS"}:
        raise ValueError("sensor_mode must be ALS, ULS, or TLS")
    if not source.exists(): raise FileNotFoundError(source)
    if not definitions.is_file(): raise FileNotFoundError(definitions)
    if target.exists(): raise FileExistsError(target)
    if isinstance(buffer, bool) or not isinstance(buffer, (int, float)) or not math.isfinite(buffer) or buffer < 0:
        raise ValueError("buffer must be a finite nonnegative number of metres")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    catalog = build_survey_catalog(source)
    ready = [t for t in catalog["tiles"] if t["status"] == "ready"]
    if not ready: raise ValueError("Survey contains no CRS-resolved LAS/LAZ sources")
    index = SurveyTileIndex(catalog)
    crs = index.crs
    if not _metre_crs(crs):
        raise ValueError("Plot extraction requires a projected CRS with metre horizontal units")
    regions = discover_survey_regions(catalog)
    plots = import_plots(definitions, crs, source_crs=source_crs, layer=layer)
    if not plots: raise ValueError("No nonempty polygon plots found")
    if len({p.plot_name.casefold() for p in plots}) != len(plots):
        raise ValueError("Plot directory names are not unique")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}_staging_", dir=target.parent))
    entries, started = [], time.perf_counter()
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
                rel = f"{name}/{name}{suffix}"
                core_geom = plot.geometry
                buffered_geom = core_geom.buffer(float(buffer)) if buffer > 0 else core_geom
                result = clip_las_sources(
                    index.paths(buffered_geom), staging / rel, geometry=buffered_geom,
                    core_geometry=core_geom, chunk_size=chunk_size, deduplicate_xyz=True,
                )
                entry = {
                    "schema": "fastgc.gis.plot", "schema_version": 1,
                    "plot_id": plot.plot_id, "plot_name": name,
                    "sensor_mode": sensor_mode, "sampling_method": sampling_method,
                    "point_file": rel, "point_extent": "core_plus_buffer" if buffer > 0 else "core",
                    "contains_buffer": bool(buffer > 0), "buffer_m": float(buffer),
                    "shape": shape or plot.attributes.get("shape"),
                    "core_radius_m": float(radius) if radius is not None else plot.attributes.get("radius_m"),
                    "radius_definition": "circumradius" if (shape or plot.attributes.get("shape")) == "hexagon" else None,
                    "processing_crs": plot.processing_crs,
                    "core_geometry": mapping(core_geom),
                    "buffered_geometry": mapping(buffered_geom),
                    "core_points": result["core_points"],
                    "points_written": result["selected_points"],
                    "source_count": result["source_count"],
                    "duplicates_removed": result["duplicates_removed"],
                    "source_files": [x["path"] for x in result["sources"]],
                    "source_feature_id": plot.source_feature_id,
                    "source_definition_file": Path(plot.source_file).name,
                    "attributes": plot.attributes,
                }
                _write_json(folder / f"Clip_{name}.json", entry)
                entries.append(entry)
                bar.update(1)
        finally:
            bar.close()

        plots_manifest = {
            "schema": "fastgc.gis.plots", "schema_version": 1,
            "sensor_mode": sensor_mode, "plot_count": len(entries),
            "point_extent": "core_plus_buffer" if buffer > 0 else "core", "buffer_m": float(buffer),
            "plots": [{"plot_id": e["plot_id"], "plot_name": e["plot_name"],
                       "point_file": e["point_file"],
                       "metadata": f"{e['plot_name']}/Clip_{e['plot_name']}.json"}
                      for e in entries],
        }
        _write_json(staging / "plots_manifest.json", plots_manifest)
        _write_json(staging / "survey_catalog.json", catalog)
        _write_json(staging / "survey_regions.json", regions)
        _write_json(staging / "workspace_manifest.json", {
            "schema": "fastgc.gis.workspace", "schema_version": 3,
            "sensor_mode": sensor_mode, "input_path": str(source),
            "source_crs": crs.to_wkt(), "survey_tile_count": catalog["tile_count"],
            "survey_ready_count": catalog["ready_count"], "sampling_method": sampling_method,
            "shape": shape, "core_radius_m": float(radius) if radius is not None else None,
            "buffer_m": float(buffer), "plot_count": len(entries),
            "plots_manifest": "plots_manifest.json",
            "deduplication": "output_grid_exact_xyz",
        })
        publish_directory(staging, target)
        print(f"Published {len(entries)} plots in {time.perf_counter()-started:.2f} s: {target}", flush=True)
        return json.loads((target / "workspace_manifest.json").read_text(encoding="utf-8"))
    finally:
        if staging.exists(): shutil.rmtree(staging, ignore_errors=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-las", required=True); p.add_argument("--plots", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--buffer", type=float, default=5.0)
    p.add_argument("--chunk-size", type=int, default=500_000); p.add_argument("--source-crs")
    p.add_argument("--layer"); p.add_argument("--sensor-mode", choices=["ALS","ULS","TLS"], required=True)
    a = p.parse_args(argv)
    extract_plots(a.input_las, a.plots, a.output_dir, buffer=a.buffer, chunk_size=a.chunk_size,
                  source_crs=a.source_crs, layer=a.layer, sensor_mode=a.sensor_mode)

if __name__ == "__main__": main()
