"""Integrated FAST-GIS command-line entry point."""
from __future__ import annotations
import argparse
from pathlib import Path
from .rough_sampling import sample_and_clip


def main(argv=None):
    p = argparse.ArgumentParser(prog="fastgc gis", description="FAST-GIS survey sampling and clipping")
    p.add_argument("--in_path", "--in-path", dest="in_path", required=True)
    p.add_argument("--out_dir", "--out-dir", dest="out_dir", required=True)
    p.add_argument("--sensor_mode", "--sensor-mode", dest="sensor_mode", choices=["ALS","ULS","TLS"], required=True)
    p.add_argument("--sampling", choices=["rough_height_stratified"], default="rough_height_stratified")
    p.add_argument("--dsm_path", "--dsm-path", dest="dsm_path", required=True,
                   help="Merged FAST_DSM GeoTIFF used for DSM-minus-local-low rough height sampling")
    p.add_argument("--plots", type=int, default=30)
    p.add_argument("--shape", choices=["circle"], default="circle")
    p.add_argument("--radius", type=float, default=10.0)
    p.add_argument("--buffer", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--low_max", "--low-max", dest="low_max", type=float, default=5.0)
    p.add_argument("--medium_max", "--medium-max", dest="medium_max", type=float, default=15.0)
    p.add_argument("--chunk_size", "--chunk-size", dest="chunk_size", type=int, default=500_000)
    a = p.parse_args(argv)
    if not 0 < a.low_max < a.medium_max:
        p.error("Require 0 < low_max < medium_max")
    print("FAST-GIS | [1/4] validating inputs", flush=True)
    if not Path(a.in_path).is_file(): p.error(f"Input LAS/LAZ not found: {a.in_path}")
    if not Path(a.dsm_path).is_file(): p.error(f"DSM not found: {a.dsm_path}")
    print("FAST-GIS | [2/4] building rough-height reference", flush=True)
    result = sample_and_clip(a.in_path, a.dsm_path, a.out_dir, count=a.plots,
        radius=a.radius, buffer=a.buffer, thresholds=(a.low_max,a.medium_max),
        seed=a.seed, chunk_size=a.chunk_size, sensor_mode=a.sensor_mode)
    print("FAST-GIS | [4/4] workspace published", flush=True)
    print(result)
    return result
