from __future__ import annotations

import argparse

from .core import DEFAULT_WORKFLOW, WORKFLOW_CHOICES, run_fastgc

PRODUCT_CHOICES = [
    "all",
    "FAST_GC",
    "FAST_DEM",
    "FAST_NORMALIZED",
    "FAST_DSM",
    "FAST_CHM",
]
RASTER_METHOD_CHOICES = ["min", "max", "mean", "nearest", "idw"]


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="fastgc",
        description="FAST-GC: sensor-aware ground classification and LiDAR derivative products",
    )
    parser.add_argument(
        "--in_path",
        "--in-path",
        dest="in_path",
        required=True,
        help="Input LAS/LAZ file or folder. For merge workflow, this should be the tiling workspace root containing tile_manifest.json.",
    )
    parser.add_argument(
        "--out_dir",
        "--out-dir",
        dest="out_dir",
        default=None,
        help=(
            "Output root folder. For folder input, FAST-GC creates Processed_<SENSOR> or workflow-specific "
            "folders inside this root. If omitted, outputs are created beside the input."
        ),
    )
    parser.add_argument(
        "--sensor_mode",
        "--sensor-mode",
        dest="sensor_mode",
        required=True,
        choices=["ALS", "ULS", "TLS"],
        help="Sensor mode.",
    )
    parser.add_argument(
        "--workflow",
        default=DEFAULT_WORKFLOW,
        choices=WORKFLOW_CHOICES,
        help="Workflow: direct run, tile-only, tile-run, tile-run-merge, or merge.",
    )
    parser.add_argument(
        "--products",
        nargs="+",
        default=["FAST_GC"],
        choices=PRODUCT_CHOICES,
        help=(
            "Products to write. Default is FAST_GC only. Use 'all' for the full pipeline. "
            "Dependencies are resolved automatically."
        ),
    )
    parser.add_argument(
        "--grid_res",
        "--grid-res",
        dest="grid_res",
        type=float,
        default=0.5,
        help="Raster grid resolution. Default: 0.5",
    )
    parser.add_argument(
        "--dem_method",
        "--dem-method",
        dest="dem_method",
        type=str,
        default="min",
        choices=RASTER_METHOD_CHOICES,
        help="DEM raster creation method. Default: min",
    )
    parser.add_argument(
        "--dsm_method",
        "--dsm-method",
        dest="dsm_method",
        type=str,
        default="max",
        choices=RASTER_METHOD_CHOICES,
        help="DSM raster creation method. Default: max",
    )
    parser.add_argument(
        "--chm_median_size",
        "--chm-median-size",
        dest="chm_median_size",
        type=int,
        default=0,
        help="Optional median-filter kernel size for CHM post-smoothing. Use 0 to disable. Default: 0",
    )
    parser.add_argument(
        "--chm_min_height",
        "--chm-min-height",
        dest="chm_min_height",
        type=float,
        default=0.0,
        help="Optional minimum CHM height threshold; smaller values are set to 0. Default: 0.0",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan input folders for LAS/LAZ files.",
    )
    parser.add_argument(
        "--tile_size_m",
        "--tile-size-m",
        dest="tile_size_m",
        type=float,
        default=30.0,
        help="Tile core size in meters for tiling workflows.",
    )
    parser.add_argument(
        "--buffer_m",
        "--buffer-m",
        dest="buffer_m",
        type=float,
        default=5.0,
        help="Tile buffer size in meters for tiling workflows.",
    )
    parser.add_argument(
        "--small_tile_merge_frac",
        "--small-tile-merge-frac",
        dest="small_tile_merge_frac",
        type=float,
        default=0.25,
        help="If a planned tile core area is smaller than this fraction of the nominal tile area, merge it into an immediate neighbor. Default: 0.25",
    )
    parser.add_argument(
        "--cleanup_tiles",
        action="store_true",
        help="Remove tiling workspace after successful tile-run-merge or merge.",
    )
    parser.add_argument(
        "--overwrite_tiles",
        action="store_true",
        help="Rebuild tiles even if a tile manifest already exists.",
    )

    # FP-fix ON by default; allow explicit disable.
    parser.set_defaults(apply_fp_fix=True)
    parser.add_argument(
        "--apply_fp_fix",
        dest="apply_fp_fix",
        action="store_true",
        help="Explicitly enable the built-in FP-fix stage (default behavior).",
    )
    parser.add_argument(
        "--no_fp_fix",
        dest="apply_fp_fix",
        action="store_false",
        help="Disable the built-in FP-fix stage.",
    )
    parser.add_argument(
        "--fp_fix_dem_res",
        type=float,
        default=0.25,
        help="Temporary DEM resolution for FP-fix residual checks. Default: 0.25",
    )
    parser.add_argument(
        "--fp_fix_nonground_to_ground_max_z",
        type=float,
        default=0.0,
        help="Promote non-ground to ground when residual z <= this threshold. Default: 0.0",
    )
    parser.add_argument(
        "--fp_fix_ground_to_nonground_min_z",
        type=float,
        default=0.06,
        help="Demote ground to non-ground when residual z > this threshold. Default: 0.06",
    )
    parser.add_argument(
        "--keep_fp_fix_temp",
        action="store_true",
        help="Keep provisional _temp_fp_fix workspace for inspection.",
    )

    args = parser.parse_args(argv)

    out = run_fastgc(
        in_path=args.in_path,
        out_dir=args.out_dir,
        sensor_mode=args.sensor_mode,
        products=args.products,
        grid_res=args.grid_res,
        dem_method=args.dem_method,
        dsm_method=args.dsm_method,
        chm_median_size=args.chm_median_size,
        chm_min_height=args.chm_min_height,
        recursive=args.recursive,
        workflow=args.workflow,
        tile_size_m=args.tile_size_m,
        buffer_m=args.buffer_m,
        cleanup_tiles=args.cleanup_tiles,
        overwrite_tiles=args.overwrite_tiles,
        small_tile_merge_frac=args.small_tile_merge_frac,
        apply_fp_fix=args.apply_fp_fix,
        fp_fix_dem_res=args.fp_fix_dem_res,
        fp_fix_nonground_to_ground_max_z=args.fp_fix_nonground_to_ground_max_z,
        fp_fix_ground_to_nonground_min_z=args.fp_fix_ground_to_nonground_min_z,
        keep_fp_fix_temp=args.keep_fp_fix_temp,
    )
    print(f"[OK] Output root: {out}")


if __name__ == "__main__":
    main()
