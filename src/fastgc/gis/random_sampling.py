"""FAST-GIS random circular sampling from observed raw LiDAR coverage.

This mode requires no FAST-GC raster/classification product. Candidate centres
are derived from an XY occupancy grid built directly from LAS/LAZ returns.
Disconnected survey regions receive deterministic quotas before plot selection.
"""
from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path

import laspy
import numpy as np
from pyproj import CRS
from scipy.ndimage import distance_transform_edt
from shapely.geometry import mapping

from .plot_runner import extract_plots, _metre_crs
from .geometry import plot_geometry
from .publish import publish_directory
from .progress import gis_progress
from .survey_catalog import build_survey_catalog
from .survey_regions import discover_survey_regions


def _ready_catalog(source):
    catalog = build_survey_catalog(source)
    ready = [t for t in catalog["tiles"] if t["status"] == "ready"]
    if not ready:
        raise ValueError("No survey LAS/LAZ tile has a resolved CRS")
    crs = CRS.from_wkt(ready[0]["crs_wkt"])
    if not _metre_crs(crs):
        raise ValueError("Random sampling requires a projected CRS with metre units")
    if any(not crs.equals(CRS.from_wkt(t["crs_wkt"])) for t in ready[1:]):
        raise ValueError("Ready survey tiles must share one CRS before sampling")
    return catalog, ready, crs


def _allocate_equal(count, region_count):
    if count < 1:
        raise ValueError("plots must be positive")
    if region_count < 1:
        raise ValueError("No survey regions available")
    base, remainder = divmod(int(count), int(region_count))
    return [base + (1 if i < remainder else 0) for i in range(region_count)]


def _region_occupancy(region, tile_by_id, *, cell_size=5.0, chunk_size=500_000):
    if cell_size <= 0:
        raise ValueError("coverage_cell_m must be positive")
    minx, miny, maxx, maxy = map(float, region["bounds"])
    width = max(1, int(math.ceil((maxx - minx) / cell_size)))
    height = max(1, int(math.ceil((maxy - miny) / cell_size)))
    occupied = np.zeros((height, width), dtype=bool)
    tiles = [tile_by_id[tile_id] for tile_id in region["tile_ids"]]
    total = sum(int(t["point_count"]) for t in tiles)
    bar = gis_progress("RAW COVERAGE", total, unit="point")
    try:
        for tile in tiles:
            with laspy.open(tile["path"]) as reader:
                for chunk in reader.chunk_iterator(chunk_size):
                    x = np.asarray(chunk.x, dtype=np.float64)
                    y = np.asarray(chunk.y, dtype=np.float64)
                    good = np.isfinite(x) & np.isfinite(y)
                    if good.any():
                        cols = np.floor((x[good] - minx) / cell_size).astype(np.int64)
                        rows = np.floor((maxy - y[good]) / cell_size).astype(np.int64)
                        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
                        occupied[rows[inside], cols[inside]] = True
                    bar.update(len(chunk))
    finally:
        bar.close()
    return occupied, (minx, miny, maxx, maxy)


def choose_random_plots(catalog, regions, *, count=30, radius=15.0, buffer=5.0,
                        seed=42, cell_size=5.0, chunk_size=500_000):
    """Choose deterministic non-overlapping plot footprints from observed XY coverage."""
    if radius <= 0 or buffer < 0:
        raise ValueError("radius must be positive and buffer nonnegative")
    region_list = list(regions.get("regions", []))
    quotas = _allocate_equal(count, len(region_list))
    tile_by_id = {t["tile_id"]: t for t in catalog["tiles"] if t["status"] == "ready"}
    rng = np.random.default_rng(seed)
    chosen = []
    footprint = radius + buffer

    for region, quota in zip(region_list, quotas):
        if quota == 0:
            continue
        occupied, bounds = _region_occupancy(
            region, tile_by_id, cell_size=cell_size, chunk_size=chunk_size
        )
        # Distance to the nearest unobserved cell or array edge. Padding prevents
        # boundary cells from being treated as having infinite outside support.
        padded = np.pad(occupied, 1, constant_values=False)
        clearance = distance_transform_edt(padded, sampling=cell_size)[1:-1, 1:-1]
        eligible = np.argwhere(clearance >= footprint + cell_size)
        if not len(eligible):
            raise ValueError(
                f"{region['region_id']} has no fully observed plot+buffer footprint"
            )
        order = rng.permutation(len(eligible))
        minx, _, _, maxy = bounds
        selected_here = 0
        for ix in order:
            row, col = eligible[int(ix)]
            x = minx + (float(col) + 0.5) * cell_size
            y = maxy - (float(row) + 0.5) * cell_size
            if any((x-p["x"])**2 + (y-p["y"])**2 < (2.0*footprint)**2 for p in chosen):
                continue
            chosen.append({
                "x": x, "y": y, "region_id": region["region_id"],
                "sampling_method": "random_observed_coverage",
            })
            selected_here += 1
            if selected_here == quota:
                break
        if selected_here != quota:
            raise ValueError(
                f"Insufficient eligible area in {region['region_id']}: "
                f"selected {selected_here} of {quota}; no partial plot set published"
            )
    return chosen, quotas


def sample_random_and_clip(source, output_dir, *, count=30, radius=15.0, buffer=5.0,
                           seed=42, cell_size=5.0, chunk_size=500_000,
                           sensor_mode="ALS", shape="hexagon"):
    """Build random plots from raw observed coverage and transactionally clip them."""
    source = Path(source).resolve()
    target = Path(output_dir).resolve()
    sensor_mode = str(sensor_mode).upper()
    if sensor_mode not in {"ALS", "ULS", "TLS"}:
        raise ValueError("sensor_mode must be ALS, ULS, or TLS")
    if not source.exists():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(target)

    catalog, _, crs = _ready_catalog(source)
    regions = discover_survey_regions(catalog)
    chosen, quotas = choose_random_plots(
        catalog, regions, count=count, radius=radius, buffer=buffer, seed=seed,
        cell_size=cell_size, chunk_size=chunk_size,
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}_stage_", dir=target.parent))
    try:
        features = []
        for i, item in enumerate(chosen, 1):
            name = f"Plot_{i:03d}"
            geom = plot_geometry(item["x"], item["y"], shape=shape, radius=radius)
            features.append({
                "type": "Feature", "id": name,
                "properties": {"plot_name": name, "shape": shape, "radius_m": radius, **item},
                "geometry": mapping(geom),
            })
        plots = stage / "sample_plots.geojson"
        plots.write_text(json.dumps({
            "type": "FeatureCollection", "name": "FAST_GIS_random",
            "crs": {"type": "name", "properties": {"name": crs.to_string()}},
            "features": features,
        }, indent=2), encoding="utf-8")
        (stage / "sampling_manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "method": "random_observed_coverage",
            "source": str(source), "crs": crs.to_string(),
            "sensor_mode": sensor_mode, "seed": seed, "plot_count": count,
            "shape": shape, "radius_m": radius, "radius_definition": "circumradius" if shape == "hexagon" else "circle_radius", "buffer_m": buffer,
            "coverage_cell_m": cell_size,
            "region_allocation": "equal", "region_quotas": quotas,
            "plots": chosen,
        }, indent=2), encoding="utf-8")
        clip_root = stage / f"{sensor_mode}_plots"
        extract_plots(source, plots, clip_root, buffer=buffer, chunk_size=chunk_size,
                      sensor_mode=sensor_mode, sampling_method="random_observed_coverage",
                      shape=shape, radius=radius)
        # Promote the collection manifest to a workspace-level hierarchy.
        collection_manifest = clip_root / "plots_manifest.json"
        workspace = {
            "schema": "fastgc.gis.workspace", "schema_version": 3,
            "sensor_mode": sensor_mode, "sampling_method": "random_observed_coverage",
            "shape": shape, "core_radius_m": float(radius), "buffer_m": float(buffer),
            "plot_count": int(count), "seed": int(seed),
            "collection": f"{sensor_mode}_plots",
            "plots_manifest": f"{sensor_mode}_plots/plots_manifest.json",
            "sample_plots": "sample_plots.geojson", "sampling_manifest": "sampling_manifest.json",
        }
        (stage / "workspace_manifest.json").write_text(json.dumps(workspace, indent=2), encoding="utf-8")
        publish_directory(stage, target)
        return target
    finally:
        if stage.exists():
            shutil.rmtree(stage)
