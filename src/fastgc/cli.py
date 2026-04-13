from __future__ import annotations

import argparse

from .core import DEFAULT_WORKFLOW, WORKFLOW_CHOICES, run_fastgc
from .monster import BACKEND_CHOICES, DEFAULT_BACKEND

PRODUCT_CHOICES = [
    "all",
    "FAST_GC",
    "FAST_DEM",
    "FAST_NORMALIZED",
    "FAST_DSM",
    "FAST_CHM",
    "FAST_TERRAIN",
    "FAST_CHANGE",
    "FAST_ITD",
    "FAST_STRUCTURE",
    "FAST_TREECLOUDS",
]

RASTER_METHOD_CHOICES = ["min", "max", "mean", "nearest", "idw"]
DSM_METHOD_CHOICES = ["min", "max", "mean", "nearest", "idw", "spikefree"]

CHM_METHOD_CHOICES = [
    "p2r",
    "p99",
    "tin",
    "pitfree",
    "csf_chm",
    "spikefree",
    "percentile",
    "percentile_top",
    "percentile_band",
]

CHM_SURFACE_METHOD_CHOICES = [
    "p2r",
    "p99",
    "tin",
    "pitfree",
    "csf_chm",
]

CHM_MULTI_METHOD_CHOICES = [
    "p2r",
    "p99",
    "tin",
    "pitfree",
    "csf_chm",
    "spikefree",
]

CHM_SMOOTH_CHOICES = ["none", "median", "gaussian"]

TERRAIN_PRODUCT_CHOICES = [
    "all",
    "slope_percent",
    "slope_degrees",
    "aspect",
    "hillshade",
    "curvature",
    "tpi",
    "twi",
    "dtw",
    "tci",
]

CHANGE_INPUT_CHOICES = [
    "FAST_DEM",
    "FAST_DSM",
    "FAST_CHM",
    "FAST_TERRAIN",
]

CHANGE_MODE_CHOICES = [
    "pairwise",
    "sequential",
    "baseline",
]

CHANGE_LOD_MODE_CHOICES = [
    "threshold_only",
    "rss",
    "max",
]

STRUCTURE_PRODUCT_CHOICES = [
    "all",
    "canopy_cover",
    "z_mean",
    "z_max",
    "z_sd",
    "FHD",
    "VCI",
    "n_points",
]

ITD_METHOD_CHOICES = [
    "placeholder",
    "lmf",
    "watershed",
    "yun2021",
    "dalponte2016",
    "li2012",
    "csp",
]

TREECLOUDS_LAS_SOURCE_CHOICES = [
    "FAST_NORMALIZED",
    "FAST_GC",
]


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
        help="Input LAS/LAZ file, folder, or processed product root depending on workflow.",
    )
    parser.add_argument(
        "--out_dir",
        "--out-dir",
        dest="out_dir",
        default=None,
        help="Output root folder. If omitted, outputs are created beside the input.",
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
        help="Workflow to execute.",
    )

    parser.add_argument(
        "--jobs",
        dest="jobs",
        type=int,
        default=1,
        help="Number of workers for tile-parallel stages. Use 1 for sequential, 0 for auto, negative values for cpu_count+n+1.",
    )
    parser.add_argument(
        "--joblib_backend",
        "--joblib-backend",
        dest="joblib_backend",
        type=str,
        default=DEFAULT_BACKEND,
        choices=sorted(BACKEND_CHOICES),
        help="Execution backend for tile-parallel stages.",
    )
    parser.add_argument(
        "--joblib_batch_size",
        "--joblib-batch-size",
        dest="joblib_batch_size",
        default="auto",
        help="Joblib batch size for tile-parallel stages. Use 'auto' or an integer.",
    )
    parser.add_argument(
        "--joblib_pre_dispatch",
        "--joblib-pre-dispatch",
        dest="joblib_pre_dispatch",
        default="2*n_jobs",
        help="Joblib pre_dispatch setting for tile-parallel stages.",
    )

    parser.add_argument(
        "--products",
        nargs="+",
        default=["FAST_GC"],
        choices=PRODUCT_CHOICES,
        help="Products to write. Default is FAST_GC only. Use 'all' for the full pipeline.",
    )

    parser.add_argument("--recursive", action="store_true", help="Recursively scan input folders.")
    parser.add_argument(
        "--skip_existing",
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        help="Skip steps/products that appear already completed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing downstream outputs.",
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
        help="DEM raster method. Default: min",
    )
    parser.add_argument(
        "--dsm_method",
        "--dsm-method",
        dest="dsm_method",
        type=str,
        default="max",
        choices=DSM_METHOD_CHOICES,
        help="DSM raster method. Includes 'spikefree' for DSM-first spikefree routing.",
    )

    parser.add_argument(
        "--chm_method",
        "--chm-method",
        dest="chm_method",
        type=str,
        default="p2r",
        choices=CHM_METHOD_CHOICES,
        help="Primary CHM mode. 'spikefree' is routed DSM-first; 'csf_chm' is cloth-simulation CHM.",
    )
    parser.add_argument(
        "--chm_methods",
        "--chm-methods",
        dest="chm_methods",
        nargs="+",
        default=None,
        help=(
            "Optional list of CHM targets to generate in one run. Supports plain methods "
            "(e.g. p2r p99 pitfree csf_chm spikefree) and custom selector specs, e.g. "
            "'percentile_top:pitfree:low=5' or 'percentile:pitfree:pct=5'."
        ),
    )
    parser.add_argument(
        "--chm_surface_method",
        "--chm-surface-method",
        dest="chm_surface_method",
        type=str,
        default="p2r",
        choices=CHM_SURFACE_METHOD_CHOICES,
        help="Surface method used when chm_method is a selector family.",
    )
    parser.add_argument(
        "--chm_smooth_method",
        "--chm-smooth-method",
        dest="chm_smooth_method",
        type=str,
        default="none",
        choices=CHM_SMOOTH_CHOICES,
        help="CHM post-smoothing method.",
    )
    parser.add_argument(
        "--chm_percentile",
        "--chm-percentile",
        dest="chm_percentile",
        type=float,
        default=99.0,
        help="Percentile used for p99 or percentile selector. Default: 99",
    )
    parser.add_argument(
        "--chm_percentile_low",
        "--chm-percentile-low",
        dest="chm_percentile_low",
        type=float,
        default=None,
        help="Lower percentile bound for percentile_top or percentile_band.",
    )
    parser.add_argument(
        "--chm_percentile_high",
        "--chm-percentile-high",
        dest="chm_percentile_high",
        type=float,
        default=None,
        help="Upper percentile bound for percentile_band.",
    )
    parser.add_argument(
        "--chm_pitfree_thresholds",
        "--chm-pitfree-thresholds",
        dest="chm_pitfree_thresholds",
        nargs="*",
        type=float,
        default=None,
        help="Pit-free thresholds in normalized-height units.",
    )
    parser.add_argument(
        "--chm_use_first_returns",
        "--chm-use-first-returns",
        dest="chm_use_first_returns",
        action="store_true",
        default=False,
        help="Prefer first returns for eligible CHM methods.",
    )
    parser.add_argument(
        "--chm_no_first_returns",
        "--chm-no-first-returns",
        dest="chm_use_first_returns",
        action="store_false",
        help="Disable first-return preference.",
    )
    parser.add_argument(
        "--chm_spikefree_freeze_distance",
        "--chm-spikefree-freeze-distance",
        dest="chm_spikefree_freeze_distance",
        type=float,
        default=None,
        help="Spikefree freeze distance in map units. Used for DSM spikefree and derived CHM spikefree.",
    )
    parser.add_argument(
        "--chm_spikefree_insertion_buffer",
        "--chm-spikefree-insertion-buffer",
        dest="chm_spikefree_insertion_buffer",
        type=float,
        default=None,
        help="Spikefree insertion buffer in height units. Used for DSM spikefree and derived CHM spikefree.",
    )
    parser.add_argument(
        "--chm_median_size",
        "--chm-median-size",
        dest="chm_median_size",
        type=int,
        default=0,
        help="Median filter kernel size for CHM post-smoothing. Use 0 to disable.",
    )
    parser.add_argument(
        "--chm_gaussian_sigma",
        "--chm-gaussian-sigma",
        dest="chm_gaussian_sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma for CHM smoothing. Default: 1.0",
    )
    parser.add_argument(
        "--chm_min_height",
        "--chm-min-height",
        dest="chm_min_height",
        type=float,
        default=0.0,
        help="Minimum CHM height threshold; smaller values are set to 0.",
    )
    parser.add_argument(
        "--chm_fill_ground_voids_zero",
        "--chm-fill-ground-voids-zero",
        dest="chm_fill_ground_voids_zero",
        action="store_true",
        default=True,
        help="Fill CHM voids with zero where support indicates ground/open cells. Default: on.",
    )
    parser.add_argument(
        "--chm_no_fill_ground_voids_zero",
        "--chm-no-fill-ground-voids-zero",
        dest="chm_fill_ground_voids_zero",
        action="store_false",
        help="Disable CHM zero-filling for true ground/open voids.",
    )
    parser.add_argument(
        "--chm_void_ground_threshold",
        "--chm-void-ground-threshold",
        dest="chm_void_ground_threshold",
        type=float,
        default=0.15,
        help="Threshold used to classify normalized support as true ground/open for CHM void filling.",
    )

    parser.add_argument(
        "--terrain_products",
        "--terrain-products",
        dest="terrain_products",
        nargs="+",
        default=["all"],
        choices=TERRAIN_PRODUCT_CHOICES,
        help="Terrain products to generate when FAST_TERRAIN is requested.",
    )
    parser.add_argument(
        "--hillshade_azimuth",
        "--hillshade-azimuth",
        dest="hillshade_azimuth",
        type=float,
        default=315.0,
        help="Hillshade illumination azimuth in degrees. Default: 315",
    )
    parser.add_argument(
        "--hillshade_altitude",
        "--hillshade-altitude",
        dest="hillshade_altitude",
        type=float,
        default=45.0,
        help="Hillshade illumination altitude in degrees. Default: 45",
    )
    parser.add_argument(
        "--hillshade_z_factor",
        "--hillshade-z-factor",
        dest="hillshade_z_factor",
        type=float,
        default=1.0,
        help="Hillshade z-factor. Default: 1.0",
    )
    parser.add_argument(
        "--tpi_radius",
        "--tpi-radius",
        dest="tpi_radius",
        type=int,
        default=3,
        help="Neighborhood radius in cells for TPI. Default: 3",
    )
    parser.add_argument(
        "--twi_eps",
        "--twi-eps",
        dest="twi_eps",
        type=float,
        default=1e-6,
        help="Small epsilon used in TWI computation for numerical stability.",
    )
    parser.add_argument(
        "--dtw_max_distance",
        "--dtw-max-distance",
        dest="dtw_max_distance",
        type=float,
        default=None,
        help="Optional cap for DTW search distance.",
    )

    parser.add_argument(
        "--change_input_type",
        "--change-input-type",
        dest="change_input_type",
        type=str,
        default="FAST_DEM",
        choices=CHANGE_INPUT_CHOICES,
        help="Source raster family for FAST_CHANGE.",
    )
    parser.add_argument(
        "--change_mode",
        "--change-mode",
        dest="change_mode",
        type=str,
        default="pairwise",
        choices=CHANGE_MODE_CHOICES,
        help="Change detection mode.",
    )
    parser.add_argument(
        "--change_threshold",
        "--change-threshold",
        dest="change_threshold",
        type=float,
        default=0.0,
        help="Minimum threshold for gain/loss masks.",
    )
    parser.add_argument(
        "--change_baseline_index",
        "--change-baseline-index",
        dest="change_baseline_index",
        type=int,
        default=0,
        help="Baseline raster index for baseline change mode.",
    )
    parser.add_argument(
        "--change_source_subdir",
        "--change-source-subdir",
        dest="change_source_subdir",
        type=str,
        default=None,
        help="Optional specific subfolder for FAST_CHM or FAST_TERRAIN change detection.",
    )
    parser.add_argument(
        "--change_sigma1",
        "--change-sigma1",
        dest="change_sigma1",
        type=float,
        default=0.0,
        help="Uncertainty term for the first raster/source in LOD computation.",
    )
    parser.add_argument(
        "--change_sigma2",
        "--change-sigma2",
        dest="change_sigma2",
        type=float,
        default=0.0,
        help="Uncertainty term for the second raster/source in LOD computation.",
    )
    parser.add_argument(
        "--change_lod_mode",
        "--change-lod-mode",
        dest="change_lod_mode",
        type=str,
        default="rss",
        choices=CHANGE_LOD_MODE_CHOICES,
        help="LOD strategy for FAST_CHANGE.",
    )

    parser.add_argument(
        "--itd_method",
        "--itd-method",
        dest="itd_method",
        type=str,
        default="placeholder",
        choices=ITD_METHOD_CHOICES,
        help="ITD method for tree detection workflows.",
    )
    parser.add_argument(
        "--itd_source_chm",
        "--itd-source-chm",
        dest="itd_source_chm",
        type=str,
        default=None,
        help="Optional CHM variant folder to use for ITD, e.g. pitfree or percentile_band.",
    )

    parser.add_argument(
        "--itd_min_height",
        "--itd-min-height",
        dest="itd_min_height",
        type=float,
        default=2.0,
        help="Minimum canopy/surface height used for ITD eligibility.",
    )
    parser.add_argument(
        "--itd_crown_window_m",
        "--itd-crown-window-m",
        dest="itd_crown_window_m",
        type=float,
        default=3.0,
        help="Approximate crown-detection window in map units used for local maxima search.",
    )
    parser.add_argument(
        "--itd_min_peak_separation_m",
        "--itd-min-peak-separation-m",
        dest="itd_min_peak_separation_m",
        type=float,
        default=1.5,
        help="Minimum separation between retained treetop candidates in map units.",
    )
    parser.add_argument(
        "--itd_angle_threshold_deg",
        "--itd-angle-threshold-deg",
        dest="itd_angle_threshold_deg",
        type=float,
        default=110.0,
        help="Valley-angle threshold used to prune likely false adjacent treetops.",
    )
    parser.add_argument(
        "--itd_screen_max_pair_distance_m",
        "--itd-screen-max-pair-distance-m",
        dest="itd_screen_max_pair_distance_m",
        type=float,
        default=6.0,
        help="Maximum treetop-pair distance checked during false-peak screening.",
    )
    parser.add_argument(
        "--itd_banded_neighborhood_px",
        "--itd-banded-neighborhood-px",
        dest="itd_banded_neighborhood_px",
        type=int,
        default=1,
        help="Half-width in pixels of the banded neighborhood sampled between adjacent treetop pairs.",
    )
    parser.add_argument(
        "--itd_min_crown_area_m2",
        "--itd-min-crown-area-m2",
        dest="itd_min_crown_area_m2",
        type=float,
        default=0.75,
        help="Minimum crown area retained after raster crown delineation.",
    )

    parser.add_argument(
        "--structure_products",
        "--structure-products",
        dest="structure_products",
        nargs="+",
        default=["all"],
        choices=STRUCTURE_PRODUCT_CHOICES,
        help="FAST_STRUCTURE products to generate from normalized point clouds.",
    )
    parser.add_argument(
        "--structure_res",
        "--structure-res",
        dest="structure_res",
        type=float,
        default=1.0,
        help="Output raster resolution for FAST_STRUCTURE products.",
    )
    parser.add_argument(
        "--structure_min_h",
        "--structure-min-h",
        dest="structure_min_h",
        type=float,
        default=2.0,
        help="Minimum normalized height considered vegetation for FAST_STRUCTURE metrics.",
    )
    parser.add_argument(
        "--structure_bin_size",
        "--structure-bin-size",
        dest="structure_bin_size",
        type=float,
        default=1.0,
        help="Vertical bin size for FHD and VCI metrics.",
    )
    parser.add_argument(
        "--canopy_thr",
        "--canopy-thr",
        dest="canopy_thr",
        type=float,
        default=2.0,
        help="Height threshold used for canopy cover.",
    )
    parser.add_argument(
        "--canopy_mode",
        "--canopy-mode",
        dest="canopy_mode",
        type=str,
        default="all_points",
        choices=["all_points"],
        help="Canopy-cover mode.",
    )
    parser.add_argument(
        "--structure_na_fill",
        "--structure-na-fill",
        dest="structure_na_fill",
        type=str,
        default="none",
        choices=["none", "3x3_mean"],
        help="Optional NA fill mode for FAST_STRUCTURE outputs.",
    )

    parser.add_argument(
        "--treeclouds_las_source",
        "--treeclouds-las-source",
        dest="treeclouds_las_source",
        type=str,
        default="FAST_NORMALIZED",
        choices=TREECLOUDS_LAS_SOURCE_CHOICES,
        help="LAS source used by FAST_TREECLOUDS. Default: FAST_NORMALIZED",
    )
    parser.add_argument(
        "--treeclouds_min_height",
        "--treeclouds-min-height",
        dest="treeclouds_min_height",
        type=float,
        default=0.5,
        help="Minimum point height retained during FAST_TREECLOUDS segmentation.",
    )
    parser.add_argument(
        "--treeclouds_write_individual",
        "--treeclouds-write-individual",
        dest="treeclouds_write_individual",
        action="store_true",
        default=False,
        help="Also write individual per-tree/per-segment LAS files.",
    )
    parser.add_argument(
        "--treeclouds_no_write_individual",
        "--treeclouds-no-write-individual",
        dest="treeclouds_write_individual",
        action="store_false",
        help="Disable individual per-tree/per-segment LAS writing.",
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
        help="Merge planned tiles smaller than this fraction into a neighbor.",
    )
    parser.add_argument(
        "--use_existing_tiles",
        "--use-existing-tiles",
        dest="use_existing_tiles",
        action="store_true",
        default=False,
        help="Treat input folder as already tiled data, build a manifest from existing tiles, and reuse them downstream without retiling.",
    )
    parser.add_argument(
        "--cleanup_tiles",
        "--cleanup-tiles",
        dest="cleanup_tiles",
        action="store_true",
        help="Remove tiling workspace after successful merge.",
    )
    parser.add_argument(
        "--overwrite_tiles",
        "--overwrite-tiles",
        dest="overwrite_tiles",
        action="store_true",
        help="Rebuild tiles even if a tile manifest already exists.",
    )

    parser.add_argument(
        "--apply_fp_fix",
        "--apply-fp-fix",
        dest="apply_fp_fix",
        action="store_true",
        default=True,
        help="Enable built-in FP-fix stage (default behavior).",
    )
    parser.add_argument(
        "--no_fp_fix",
        "--no-fp-fix",
        dest="apply_fp_fix",
        action="store_false",
        help="Disable built-in FP-fix stage.",
    )
    parser.add_argument(
        "--fp_fix_dem_res",
        "--fp-fix-dem-res",
        dest="fp_fix_dem_res",
        type=float,
        default=0.25,
        help="Temporary DEM resolution for FP-fix residual checks.",
    )
    parser.add_argument(
        "--fp_fix_nonground_to_ground_max_z",
        "--fp-fix-nonground-to-ground-max-z",
        dest="fp_fix_nonground_to_ground_max_z",
        type=float,
        default=0.0,
        help="Promote non-ground to ground when residual z <= this threshold.",
    )
    parser.add_argument(
        "--fp_fix_ground_to_nonground_min_z",
        "--fp-fix-ground-to-nonground-min-z",
        dest="fp_fix_ground_to_nonground_min_z",
        type=float,
        default=0.06,
        help="Demote ground to non-ground when residual z > this threshold.",
    )
    parser.add_argument(
        "--keep_fp_fix_temp",
        "--keep-fp-fix-temp",
        dest="keep_fp_fix_temp",
        action="store_true",
        help="Keep provisional _temp_fp_fix workspace for inspection.",
    )

    args = parser.parse_args(argv)

    if isinstance(args.joblib_batch_size, str) and args.joblib_batch_size.strip().lower() != "auto":
        try:
            args.joblib_batch_size = int(args.joblib_batch_size)
        except ValueError as exc:
            raise SystemExit(f"Invalid --joblib_batch_size: {args.joblib_batch_size}") from exc

    run_fastgc(
        in_path=args.in_path,
        out_dir=args.out_dir,
        sensor_mode=args.sensor_mode,
        products=args.products,
        grid_res=args.grid_res,
        recursive=args.recursive,
        n_jobs=args.jobs,
        joblib_backend=args.joblib_backend,
        joblib_batch_size=args.joblib_batch_size,
        joblib_pre_dispatch=args.joblib_pre_dispatch,
        skip_existing=args.skip_existing,
        overwrite=args.overwrite,
        dem_method=args.dem_method,
        dsm_method=args.dsm_method,
        chm_method=args.chm_method,
        chm_methods=args.chm_methods,
        chm_surface_method=args.chm_surface_method,
        chm_smooth_method=args.chm_smooth_method,
        chm_percentile=args.chm_percentile,
        chm_percentile_low=args.chm_percentile_low,
        chm_percentile_high=args.chm_percentile_high,
        chm_pitfree_thresholds=args.chm_pitfree_thresholds,
        chm_use_first_returns=args.chm_use_first_returns,
        chm_spikefree_freeze_distance=args.chm_spikefree_freeze_distance,
        chm_spikefree_insertion_buffer=args.chm_spikefree_insertion_buffer,
        chm_median_size=args.chm_median_size,
        chm_gaussian_sigma=args.chm_gaussian_sigma,
        chm_min_height=args.chm_min_height,
        chm_fill_ground_voids_zero=args.chm_fill_ground_voids_zero,
        chm_void_ground_threshold=args.chm_void_ground_threshold,
        terrain_products=args.terrain_products,
        hillshade_azimuth=args.hillshade_azimuth,
        hillshade_altitude=args.hillshade_altitude,
        hillshade_z_factor=args.hillshade_z_factor,
        tpi_radius=args.tpi_radius,
        twi_eps=args.twi_eps,
        dtw_max_distance=args.dtw_max_distance,
        change_input_type=args.change_input_type,
        change_mode=args.change_mode,
        change_threshold=args.change_threshold,
        change_baseline_index=args.change_baseline_index,
        change_source_subdir=args.change_source_subdir,
        change_sigma1=args.change_sigma1,
        change_sigma2=args.change_sigma2,
        change_lod_mode=args.change_lod_mode,
        itd_method=args.itd_method,
        itd_source_chm=args.itd_source_chm,
        itd_min_height=args.itd_min_height,
        itd_crown_window_m=args.itd_crown_window_m,
        itd_min_peak_separation_m=args.itd_min_peak_separation_m,
        itd_angle_threshold_deg=args.itd_angle_threshold_deg,
        itd_screen_max_pair_distance_m=args.itd_screen_max_pair_distance_m,
        itd_banded_neighborhood_px=args.itd_banded_neighborhood_px,
        itd_min_crown_area_m2=args.itd_min_crown_area_m2,
        structure_products=args.structure_products,
        structure_res=args.structure_res,
        structure_min_h=args.structure_min_h,
        structure_bin_size=args.structure_bin_size,
        canopy_thr=args.canopy_thr,
        canopy_mode=args.canopy_mode,
        structure_na_fill=args.structure_na_fill,
        treeclouds_las_source=args.treeclouds_las_source,
        treeclouds_min_height=args.treeclouds_min_height,
        treeclouds_write_individual=args.treeclouds_write_individual,
        workflow=args.workflow,
        tile_size_m=args.tile_size_m,
        buffer_m=args.buffer_m,
        cleanup_tiles=args.cleanup_tiles,
        overwrite_tiles=args.overwrite_tiles,
        small_tile_merge_frac=args.small_tile_merge_frac,
        use_existing_tiles=args.use_existing_tiles,
        apply_fp_fix=args.apply_fp_fix,
        fp_fix_dem_res=args.fp_fix_dem_res,
        fp_fix_nonground_to_ground_max_z=args.fp_fix_nonground_to_ground_max_z,
        fp_fix_ground_to_nonground_min_z=args.fp_fix_ground_to_nonground_min_z,
        keep_fp_fix_temp=args.keep_fp_fix_temp,
    )


if __name__ == "__main__":
    main()