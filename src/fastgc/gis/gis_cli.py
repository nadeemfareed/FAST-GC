"""Integrated FAST-GIS command-line entry point."""
from __future__ import annotations

import argparse
from pathlib import Path

from .plot_runner import extract_plots
from .random_sampling import sample_random_and_clip
from .rough_sampling import sample_and_clip


def main(argv=None):
    p = argparse.ArgumentParser(prog="fastgc gis", description="FAST-GIS survey sampling and clipping")
    p.add_argument("--in_path", "--in-path", dest="in_path", required=True)
    p.add_argument("--out_dir", "--out-dir", dest="out_dir", required=True)
    p.add_argument("--sensor_mode", "--sensor-mode", dest="sensor_mode", choices=["ALS", "ULS", "TLS"], required=True)
    p.add_argument("--sampling", choices=["random", "rough_height_stratified", "imported"], default="random")
    p.add_argument("--dsm_path", "--dsm-path", dest="dsm_path",
                   help="Existing DSM GeoTIFF; required only for rough_height_stratified")
    p.add_argument("--plot_file", "--plot-file", dest="plot_file",
                   help="KML/KMZ/SHP/GPKG/GeoJSON/CSV; required only for imported")
    p.add_argument("--source_crs", "--source-crs", dest="source_crs",
                   help="Source CRS override for LAS/LAZ without embedded CRS, or for CSV plot centers")
    p.add_argument("--layer", help="Optional vector layer for imported GPKG/vector data")
    p.add_argument("--plots", type=int, default=30)
    p.add_argument("--shape", choices=["circle", "hexagon"], default="hexagon")
    p.add_argument("--radius", type=float, default=15.0)
    p.add_argument("--buffer", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--coverage_cell_m", "--coverage-cell-m", dest="coverage_cell_m", type=float, default=5.0,
                   help="Raw XY occupancy grid used by random sampling (metres)")
    p.add_argument("--low_max", "--low-max", dest="low_max", type=float, default=5.0)
    p.add_argument("--medium_max", "--medium-max", dest="medium_max", type=float, default=15.0)
    p.add_argument("--chunk_size", "--chunk-size", dest="chunk_size", type=int, default=500_000)
    a = p.parse_args(argv)

    source = Path(a.in_path)
    if not source.exists():
        p.error(f"Input LAS/LAZ or survey directory not found: {a.in_path}")
    if a.plots < 1:
        p.error("--plots must be positive")
    if a.radius <= 0 or a.buffer < 0:
        p.error("Require --radius > 0 and --buffer >= 0")
    if a.chunk_size < 1:
        p.error("--chunk_size must be positive")

    if a.sampling == "random":
        result = sample_random_and_clip(
            a.in_path, a.out_dir, count=a.plots, radius=a.radius, buffer=a.buffer,
            seed=a.seed, cell_size=a.coverage_cell_m, chunk_size=a.chunk_size,
            sensor_mode=a.sensor_mode, shape=a.shape,
            source_crs=a.source_crs,
        )
    elif a.sampling == "rough_height_stratified":
        if a.shape != "circle":
            p.error("rough_height_stratified currently supports --shape circle; random supports hexagon")
        if not a.dsm_path:
            p.error("--dsm_path is required for --sampling rough_height_stratified")
        if not Path(a.dsm_path).is_file():
            p.error(f"DSM not found: {a.dsm_path}")
        if not source.is_file():
            p.error("rough_height_stratified currently requires a single LAS/LAZ input")
        if not 0 < a.low_max < a.medium_max:
            p.error("Require 0 < low_max < medium_max")
        result = sample_and_clip(
            a.in_path, a.dsm_path, a.out_dir, count=a.plots,
            radius=a.radius, buffer=a.buffer, thresholds=(a.low_max, a.medium_max),
            seed=a.seed, chunk_size=a.chunk_size, sensor_mode=a.sensor_mode,
        )
    else:
        if not a.plot_file:
            p.error("--plot_file is required for --sampling imported")
        if not Path(a.plot_file).is_file():
            p.error(f"Plot definition file not found: {a.plot_file}")
        result = extract_plots(
            a.in_path, a.plot_file, a.out_dir, buffer=a.buffer,
            chunk_size=a.chunk_size, source_crs=a.source_crs, layer=a.layer,
            sensor_mode=a.sensor_mode, sampling_method="imported",
        )
    print(result)
    return result
