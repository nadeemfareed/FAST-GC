from __future__ import annotations

import inspect
import json
from pathlib import Path
from time import perf_counter

from .chm import (
    build_chm_from_dem_and_dsm,
    build_chm_from_normalized_root,
    chm_method_output_dir,
    chm_output_label,
    resolve_normalized_root,
)
from .change import run_change_from_processed_root
from .io_las import (
    PRODUCT_CHM,
    PRODUCT_DEM,
    PRODUCT_DSM,
    PRODUCT_GC,
    PRODUCT_NORMALIZED,
    derive_products_from_classified_root,
    derive_products_from_raw_root,
    process_fastgc_path,
)
from .itd import run_itd_from_processed_root
from .merge import cleanup_tiling_workspace, merge_processed_tiles
from .monster import log_info, stage_banner
from .post_fp_fix import apply_fp_fix_to_output_root
from .preprocess import tile_las_dataset
from .structure import run_structure_from_root
from .terrain import run_terrain_from_processed_root
from .treeclouds import PRODUCT_TREECLOUDS, run_treeclouds_from_root

PRODUCT_TERRAIN = "FAST_TERRAIN"
PRODUCT_CHANGE = "FAST_CHANGE"
PRODUCT_ITD = "FAST_ITD"
PRODUCT_STRUCTURE = "FAST_STRUCTURE"

DEFAULT_WORKFLOW = "run"
WORKFLOW_CHOICES = ["run", "tile-only", "tile-run", "tile-run-merge", "merge", "derive-only"]

_CHM_ALGORITHMS = {"p2r", "p99", "tin", "pitfree", "csf_chm", "spikefree"}
_CHM_NATIVE = {"p2r", "p99", "tin", "pitfree", "csf_chm"}
_CHM_DSM_DERIVED = {"spikefree"}
_CHM_SELECTORS = {"percentile", "percentile_top", "percentile_band"}


def _requested_products(products: list[str] | None) -> list[str]:
    requested = list(products or [PRODUCT_GC])
    out: list[str] = []
    seen: set[str] = set()
    for p in requested:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _resolve_products(products: list[str] | None) -> list[str]:
    requested = _requested_products(products)

    if "all" in requested:
        return [
            PRODUCT_GC,
            PRODUCT_DEM,
            PRODUCT_NORMALIZED,
            PRODUCT_DSM,
            PRODUCT_CHM,
            PRODUCT_TERRAIN,
            PRODUCT_STRUCTURE,
        ]

    out = list(requested)
    seen = set(out)

    if PRODUCT_NORMALIZED in seen and PRODUCT_DEM not in seen:
        out.insert(out.index(PRODUCT_NORMALIZED), PRODUCT_DEM)
        seen.add(PRODUCT_DEM)

    if PRODUCT_CHM in seen:
        if PRODUCT_DEM not in seen:
            out.insert(out.index(PRODUCT_CHM), PRODUCT_DEM)
            seen.add(PRODUCT_DEM)
        if PRODUCT_NORMALIZED not in seen:
            out.insert(out.index(PRODUCT_CHM), PRODUCT_NORMALIZED)
            seen.add(PRODUCT_NORMALIZED)

    if PRODUCT_TERRAIN in seen and PRODUCT_DEM not in seen:
        out.insert(out.index(PRODUCT_TERRAIN), PRODUCT_DEM)
        seen.add(PRODUCT_DEM)

    if PRODUCT_STRUCTURE in seen and PRODUCT_NORMALIZED not in seen:
        insert_at = out.index(PRODUCT_STRUCTURE)
        if PRODUCT_DEM not in seen:
            out.insert(insert_at, PRODUCT_DEM)
            seen.add(PRODUCT_DEM)
            insert_at += 1
        out.insert(insert_at, PRODUCT_NORMALIZED)
        seen.add(PRODUCT_NORMALIZED)

    return out


def _needs_classified_source(products: list[str]) -> bool:
    needed = set(products)
    return bool(
        {
            PRODUCT_GC,
            PRODUCT_DEM,
            PRODUCT_NORMALIZED,
            PRODUCT_CHM,
            PRODUCT_TERRAIN,
            PRODUCT_STRUCTURE,
        }
        & needed
    )


def _needs_raw_dsm(products: list[str]) -> bool:
    return PRODUCT_DSM in set(products)


def _load_manifest_from_workspace(workspace_root: Path) -> dict:
    manifest_fp = workspace_root / "tile_manifest.json"
    if not manifest_fp.exists():
        raise FileNotFoundError(f"Tile manifest not found: {manifest_fp}")
    with manifest_fp.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_workspace_root(path: Path, sensor_mode: str) -> Path:
    """
    Accept any of:
      - workspace root containing tile_manifest.json
      - Processed_<SENSOR> folder
      - FAST_* product folder under Processed_<SENSOR>
    and resolve back to the tile workspace root that owns tile_manifest.json.
    """
    p = Path(path).resolve()

    if (p / "tile_manifest.json").exists():
        return p

    processed_name = f"Processed_{sensor_mode.upper()}"

    if p.name == processed_name:
        candidate = p.parent
        if (candidate / "tile_manifest.json").exists():
            return candidate

    if p.name.startswith("FAST_"):
        candidate = p.parent.parent
        if (candidate / "tile_manifest.json").exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve workspace root with tile_manifest.json from: {p}"
    )


def _resolve_processed_root(workspace_root: Path, sensor_mode: str) -> Path:
    candidates = [workspace_root / f"Processed_{sensor_mode.upper()}", workspace_root]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _existing_path(path: Path) -> bool:
    return path.exists() and any(path.iterdir()) if path.is_dir() else path.exists()


def _has_raster_outputs(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(p.is_file() and p.suffix.lower() in {".tif", ".tiff"} for p in path.rglob("*"))


def _contains_downstream_only_products(products: list[str]) -> bool:
    return bool({PRODUCT_CHANGE, PRODUCT_ITD, PRODUCT_TREECLOUDS} & set(products))


def _call_with_supported_kwargs(func, /, **kwargs):
    sig = inspect.signature(func)
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_var_kw:
        supported = dict(kwargs)
    else:
        supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(**supported)


def _is_las_like(path: str | Path) -> bool:
    path = Path(path)
    return path.is_file() and path.suffix.lower() in {".las", ".laz"}


def _ensure_tiles_for_raw_input(
    *,
    in_path: str,
    out_dir: str | None,
    sensor_mode: str,
    tile_size_m: float,
    buffer_m: float,
    recursive: bool,
    overwrite_tiles: bool,
    small_tile_merge_frac: float,
) -> dict:
    return tile_las_dataset(
        in_path=in_path,
        out_dir=out_dir,
        sensor_mode=sensor_mode,
        tile_size_m=tile_size_m,
        buffer_m=buffer_m,
        recursive=recursive,
        overwrite_tiles=overwrite_tiles,
        small_tile_merge_frac=small_tile_merge_frac,
    )


def _resolve_chm_targets(
    *,
    chm_method: str,
    chm_methods: list[str] | None,
    chm_surface_method: str,
) -> list[dict]:
    method = str(chm_method).strip().lower()
    surface_method = str(chm_surface_method).strip().lower()

    targets: list[dict] = []

    if chm_methods:
        for m in chm_methods:
            mm = str(m).strip().lower()
            if mm not in _CHM_ALGORITHMS:
                raise ValueError(
                    f"--chm_methods supports: {sorted(_CHM_ALGORITHMS | _CHM_SELECTORS)}"
                )
            targets.append(
                {
                    "method": mm,
                    "surface_method": None,
                    "label": chm_output_label(mm, None),
                    "is_dsm_derived": mm in _CHM_DSM_DERIVED,
                }
            )
        return targets

    if method in _CHM_ALGORITHMS:
        targets.append(
            {
                "method": method,
                "surface_method": None,
                "label": chm_output_label(method, None),
                "is_dsm_derived": method in _CHM_DSM_DERIVED,
            }
        )
        return targets

    if method in _CHM_SELECTORS:
        if surface_method not in _CHM_NATIVE:
            raise ValueError(f"Invalid --chm_surface_method: {surface_method}")
        targets.append(
            {
                "method": method,
                "surface_method": surface_method,
                "label": chm_output_label(method, surface_method),
                "is_dsm_derived": False,
            }
        )
        return targets

    raise ValueError(f"Unsupported CHM method: {method}")


def derive_chm_from_processed_root(
    processed_root: str | Path,
    sensor_mode: str,
    *,
    chm_method: str,
    chm_methods: list[str] | None,
    chm_surface_method: str,
    chm_smooth_method: str,
    grid_res: float,
    chm_percentile: float,
    chm_percentile_low: float | None,
    chm_percentile_high: float | None,
    chm_pitfree_thresholds: list[float] | None,
    chm_use_first_returns: bool,
    chm_spikefree_freeze_distance: float | None,
    chm_spikefree_insertion_buffer: float | None,
    chm_median_size: int,
    chm_gaussian_sigma: float,
    chm_min_height: float,
    chm_fill_ground_voids_zero: bool,
    chm_void_ground_threshold: float,
    n_jobs: int = 1,
    joblib_backend: str = "loky",
    joblib_batch_size: int | str = "auto",
    joblib_pre_dispatch: str | int = "2*n_jobs",
    skip_existing: bool = False,
    overwrite: bool = False,
) -> list[str]:
    processed_root = Path(processed_root)

    outputs: list[str] = []
    targets = _resolve_chm_targets(
        chm_method=chm_method,
        chm_methods=chm_methods,
        chm_surface_method=chm_surface_method,
    )

    normalized_root = None
    if any(not t["is_dsm_derived"] for t in targets):
        normalized_root = resolve_normalized_root(processed_root)
        if not Path(normalized_root).exists():
            raise FileNotFoundError(f"FAST_NORMALIZED folder not found: {normalized_root}")

    dem_root = processed_root / PRODUCT_DEM
    dsm_root = processed_root / PRODUCT_DSM

    for target in targets:
        out_root = Path(chm_method_output_dir(processed_root, target["method"], target["surface_method"]))
        if skip_existing and _existing_path(out_root) and not overwrite:
            print(f"[SKIP] Existing CHM output found: {out_root}")
            outputs.append(str(out_root))
            continue

        if target["is_dsm_derived"]:
            if target["method"] == "spikefree":
                if not _has_raster_outputs(dem_root):
                    raise FileNotFoundError(f"No DEM rasters found for CHM spikefree derivation: {dem_root}")

                if not _has_raster_outputs(dsm_root):
                    raise FileNotFoundError(f"No DSM rasters found for CHM spikefree derivation: {dsm_root}")

                out = _call_with_supported_kwargs(
                    build_chm_from_dem_and_dsm,
                    dem_root=dem_root,
                    dsm_root=dsm_root,
                    out_root=out_root,
                    method_label="spikefree",
                    min_height=chm_min_height,
                    smooth_method=chm_smooth_method,
                    median_size=chm_median_size,
                    gaussian_sigma=chm_gaussian_sigma,
                    overwrite=overwrite,
                    n_jobs=n_jobs,
                    joblib_backend=joblib_backend,
                    joblib_batch_size=joblib_batch_size,
                    joblib_pre_dispatch=joblib_pre_dispatch,
                )
                outputs.append(out)
                continue

            raise ValueError(f"Unsupported DSM-derived CHM target: {target['method']}")

        out = _call_with_supported_kwargs(
            build_chm_from_normalized_root,
            normalized_root=normalized_root,
            out_root=out_root,
            sensor_mode=sensor_mode,
            method=target["method"],
            surface_method=target["surface_method"],
            grid_res=grid_res,
            smooth_method=chm_smooth_method,
            percentile=chm_percentile,
            percentile_low=chm_percentile_low,
            percentile_high=chm_percentile_high,
            pitfree_thresholds=chm_pitfree_thresholds,
            use_first_returns=chm_use_first_returns,
            spikefree_freeze_distance=chm_spikefree_freeze_distance,
            spikefree_insertion_buffer=chm_spikefree_insertion_buffer,
            median_size=chm_median_size,
            gaussian_sigma=chm_gaussian_sigma,
            min_height=chm_min_height,
            fill_ground_voids_zero=chm_fill_ground_voids_zero,
            void_ground_threshold=chm_void_ground_threshold,
            n_jobs=n_jobs,
            joblib_backend=joblib_backend,
            joblib_batch_size=joblib_batch_size,
            joblib_pre_dispatch=joblib_pre_dispatch,
            overwrite=overwrite,
        )
        outputs.append(out)

    return outputs


def _run_processing_with_optional_fpfix(
    *,
    in_path: str,
    out_dir: str | None,
    sensor_mode: str,
    products: list[str] | None,
    grid_res: float,
    dem_method: str,
    dsm_method: str,
    chm_method: str,
    chm_methods: list[str] | None,
    chm_surface_method: str,
    chm_smooth_method: str,
    chm_percentile: float,
    chm_percentile_low: float | None,
    chm_percentile_high: float | None,
    chm_pitfree_thresholds: list[float] | None,
    chm_use_first_returns: bool,
    chm_spikefree_freeze_distance: float | None,
    chm_spikefree_insertion_buffer: float | None,
    chm_median_size: int,
    chm_gaussian_sigma: float,
    chm_min_height: float,
    chm_fill_ground_voids_zero: bool,
    chm_void_ground_threshold: float,
    terrain_products: list[str] | None,
    hillshade_azimuth: float,
    hillshade_altitude: float,
    hillshade_z_factor: float,
    tpi_radius: int,
    twi_eps: float,
    dtw_max_distance: float | None,
    structure_products: list[str] | None,
    structure_res: float | None,
    structure_min_h: float | None,
    structure_bin_size: float | None,
    canopy_thr: float | None,
    canopy_mode: str | None,
    structure_na_fill: str | None,
    recursive: bool,
    n_jobs: int = 1,
    joblib_backend: str = "loky",
    joblib_batch_size: int | str = "auto",
    joblib_pre_dispatch: str | int = "2*n_jobs",
    apply_fp_fix: bool = True,
    fp_fix_dem_res: float | None = None,
    fp_fix_nonground_to_ground_max_z: float = 0.0,
    fp_fix_ground_to_nonground_min_z: float = 0.06,
    keep_fp_fix_temp: bool = False,
    skip_existing: bool = False,
    overwrite: bool = False,
) -> str:
    resolved_products = _resolve_products(products)
    need_classified = _needs_classified_source(resolved_products)
    need_dsm = _needs_raw_dsm(resolved_products)

    if need_dsm and not need_classified:
        return derive_products_from_raw_root(
            raw_root=in_path,
            out_root=out_dir if out_dir is not None else in_path,
            sensor_mode=sensor_mode,
            products=[PRODUCT_DSM],
            grid_res=grid_res,
            dsm_method=dsm_method,
            recursive=recursive,
            n_jobs=n_jobs,
            joblib_backend=joblib_backend,
            joblib_batch_size=joblib_batch_size,
            joblib_pre_dispatch=joblib_pre_dispatch,
            spikefree_freeze_distance=chm_spikefree_freeze_distance,
            spikefree_insertion_buffer=chm_spikefree_insertion_buffer,
        )

    base_products = [p for p in resolved_products if p in {PRODUCT_GC, PRODUCT_DEM, PRODUCT_NORMALIZED, PRODUCT_DSM}]

    if not apply_fp_fix:
        out_root = process_fastgc_path(
            in_path=in_path,
            out_dir=out_dir,
            sensor_mode=sensor_mode,
            products=base_products,
            grid_res=grid_res,
            dem_method=dem_method,
            dsm_method=dsm_method,
            recursive=recursive,
            n_jobs=n_jobs,
            joblib_backend=joblib_backend,
            joblib_batch_size=joblib_batch_size,
            joblib_pre_dispatch=joblib_pre_dispatch,
            spikefree_freeze_distance=chm_spikefree_freeze_distance,
            spikefree_insertion_buffer=chm_spikefree_insertion_buffer,
        )
    else:
        out_root = process_fastgc_path(
            in_path=in_path,
            out_dir=out_dir,
            sensor_mode=sensor_mode,
            products=[PRODUCT_GC],
            grid_res=grid_res,
            dem_method=dem_method,
            dsm_method=dsm_method,
            recursive=recursive,
            n_jobs=n_jobs,
            joblib_backend=joblib_backend,
            joblib_batch_size=joblib_batch_size,
            joblib_pre_dispatch=joblib_pre_dispatch,
            spikefree_freeze_distance=chm_spikefree_freeze_distance,
            spikefree_insertion_buffer=chm_spikefree_insertion_buffer,
        )

        fix_summary = _call_with_supported_kwargs(
            apply_fp_fix_to_output_root,
            out_root=out_root,
            sensor_mode=sensor_mode,
            dem_res=grid_res if fp_fix_dem_res is None else fp_fix_dem_res,
            nonground_to_ground_max_z=fp_fix_nonground_to_ground_max_z,
            ground_to_nonground_min_z=fp_fix_ground_to_nonground_min_z,
            keep_temp=keep_fp_fix_temp,
            n_jobs=n_jobs,
            joblib_backend=joblib_backend,
            joblib_batch_size=joblib_batch_size,
            joblib_pre_dispatch=joblib_pre_dispatch,
        )
        print(f"[INFO] FP-FIX changed points: {int(fix_summary.get('total_changed_points', 0))}")

        downstream_from_gc = [p for p in resolved_products if p in {PRODUCT_DEM, PRODUCT_NORMALIZED}]
        if downstream_from_gc:
            derive_products_from_classified_root(
                classified_root=Path(out_root) / PRODUCT_GC,
                out_root=out_root,
                products=downstream_from_gc,
                grid_res=grid_res,
                dem_method=dem_method,
                n_jobs=n_jobs,
                joblib_backend=joblib_backend,
                joblib_batch_size=joblib_batch_size,
                joblib_pre_dispatch=joblib_pre_dispatch,
            )

        if need_dsm:
            derive_products_from_raw_root(
                raw_root=in_path,
                out_root=out_root,
                sensor_mode=sensor_mode,
                products=[PRODUCT_DSM],
                grid_res=grid_res,
                dsm_method=dsm_method,
                recursive=recursive,
                n_jobs=n_jobs,
                joblib_backend=joblib_backend,
                joblib_batch_size=joblib_batch_size,
                joblib_pre_dispatch=joblib_pre_dispatch,
                spikefree_freeze_distance=chm_spikefree_freeze_distance,
                spikefree_insertion_buffer=chm_spikefree_insertion_buffer,
            )

    if PRODUCT_CHM in resolved_products:
        derive_chm_from_processed_root(
            processed_root=out_root,
            sensor_mode=sensor_mode,
            chm_method=chm_method,
            chm_methods=chm_methods,
            chm_surface_method=chm_surface_method,
            chm_smooth_method=chm_smooth_method,
            grid_res=grid_res,
            chm_percentile=chm_percentile,
            chm_percentile_low=chm_percentile_low,
            chm_percentile_high=chm_percentile_high,
            chm_pitfree_thresholds=chm_pitfree_thresholds,
            chm_use_first_returns=chm_use_first_returns,
            chm_spikefree_freeze_distance=chm_spikefree_freeze_distance,
            chm_spikefree_insertion_buffer=chm_spikefree_insertion_buffer,
            chm_median_size=chm_median_size,
            chm_gaussian_sigma=chm_gaussian_sigma,
            chm_min_height=chm_min_height,
            chm_fill_ground_voids_zero=chm_fill_ground_voids_zero,
            chm_void_ground_threshold=chm_void_ground_threshold,
            n_jobs=n_jobs,
            joblib_backend=joblib_backend,
            joblib_batch_size=joblib_batch_size,
            joblib_pre_dispatch=joblib_pre_dispatch,
            skip_existing=skip_existing,
            overwrite=overwrite,
        )

    if PRODUCT_TERRAIN in resolved_products:
        _call_with_supported_kwargs(
            run_terrain_from_processed_root,
            processed_root=out_root,
            terrain_products=terrain_products,
            hillshade_azimuth=hillshade_azimuth,
            hillshade_altitude=hillshade_altitude,
            hillshade_z_factor=hillshade_z_factor,
            tpi_radius=tpi_radius,
            twi_eps=twi_eps,
            dtw_max_distance=dtw_max_distance,
            n_jobs=n_jobs,
            joblib_backend=joblib_backend,
            joblib_batch_size=joblib_batch_size,
            joblib_pre_dispatch=joblib_pre_dispatch,
            skip_existing=skip_existing,
            overwrite=overwrite,
        )

    if PRODUCT_STRUCTURE in resolved_products:
        _call_with_supported_kwargs(
            run_structure_from_root,
            source_root=out_root,
            sensor_mode=sensor_mode,
            structure_products=structure_products,
            structure_res=structure_res,
            structure_min_h=structure_min_h,
            structure_bin_size=structure_bin_size,
            canopy_thr=canopy_thr,
            canopy_mode=canopy_mode,
            structure_na_fill=structure_na_fill,
            n_jobs=n_jobs,
            joblib_backend=joblib_backend,
            joblib_batch_size=joblib_batch_size,
            joblib_pre_dispatch=joblib_pre_dispatch,
            skip_existing=skip_existing,
            overwrite=overwrite,
        )

    return out_root


def run_fastgc(
    in_path: str,
    out_dir: str | None,
    sensor_mode: str,
    products: list[str] | None = None,
    grid_res: float = 0.5,
    recursive: bool = False,
    n_jobs: int = 1,
    joblib_backend: str = "loky",
    joblib_batch_size: int | str = "auto",
    joblib_pre_dispatch: str | int = "2*n_jobs",
    *,
    skip_existing: bool = False,
    overwrite: bool = False,
    dem_method: str = "nearest",
    dsm_method: str = "max",
    chm_method: str = "p2r",
    chm_methods: list[str] | None = None,
    chm_surface_method: str = "p2r",
    chm_smooth_method: str = "none",
    chm_percentile: float = 99.0,
    chm_percentile_low: float | None = None,
    chm_percentile_high: float | None = None,
    chm_pitfree_thresholds: list[float] | None = None,
    chm_use_first_returns: bool = False,
    chm_spikefree_freeze_distance: float | None = None,
    chm_spikefree_insertion_buffer: float | None = None,
    chm_median_size: int = 0,
    chm_gaussian_sigma: float = 1.0,
    chm_min_height: float = 0.0,
    chm_fill_ground_voids_zero: bool = True,
    chm_void_ground_threshold: float = 0.15,
    terrain_products: list[str] | None = None,
    hillshade_azimuth: float = 315.0,
    hillshade_altitude: float = 45.0,
    hillshade_z_factor: float = 1.0,
    tpi_radius: int = 3,
    twi_eps: float = 1e-6,
    dtw_max_distance: float | None = None,
    structure_products: list[str] | None = None,
    structure_res: float | None = None,
    structure_min_h: float | None = None,
    structure_bin_size: float | None = None,
    canopy_thr: float | None = None,
    canopy_mode: str | None = None,
    structure_na_fill: str | None = None,
    change_input_type: str = "FAST_DEM",
    change_mode: str = "pairwise",
    change_threshold: float = 0.0,
    change_baseline_index: int = 0,
    change_source_subdir: str | None = None,
    change_sigma1: float = 0.0,
    change_sigma2: float = 0.0,
    change_lod_mode: str = "rss",
    itd_method: str = "placeholder",
    itd_source_chm: str | None = None,
    itd_min_height: float = 2.0,
    itd_crown_window_m: float = 3.0,
    itd_min_peak_separation_m: float = 1.5,
    itd_angle_threshold_deg: float = 110.0,
    itd_screen_max_pair_distance_m: float = 6.0,
    itd_banded_neighborhood_px: int = 1,
    itd_min_crown_area_m2: float = 0.75,
    treeclouds_las_source: str = "FAST_NORMALIZED",
    treeclouds_min_height: float = 0.5,
    treeclouds_write_individual: bool = False,
    workflow: str = DEFAULT_WORKFLOW,
    tile_size_m: float = 30.0,
    buffer_m: float = 5.0,
    cleanup_tiles: bool = False,
    overwrite_tiles: bool = False,
    small_tile_merge_frac: float = 0.25,
    apply_fp_fix: bool = True,
    fp_fix_dem_res: float | None = None,
    fp_fix_nonground_to_ground_max_z: float = 0.0,
    fp_fix_ground_to_nonground_min_z: float = 0.06,
    keep_fp_fix_temp: bool = False,
):
    workflow = str(workflow).strip().lower()
    if workflow not in WORKFLOW_CHOICES:
        raise ValueError(f"Unsupported workflow: {workflow}")

    if grid_res <= 0:
        raise ValueError("grid_res must be > 0")
    if fp_fix_dem_res is not None and fp_fix_dem_res <= 0:
        raise ValueError("fp_fix_dem_res must be > 0 when provided")

    requested_products = _requested_products(products)
    resolved_products = _resolve_products(products)
    total_t0 = perf_counter()

    stage_banner("WORKFLOW", source=str(in_path), total=len(resolved_products), unit="stage")
    log_info(f"Workflow: {workflow}")
    log_info(f"Sensor mode: {sensor_mode}")
    log_info(f"Requested products: {requested_products}")
    log_info(f"Resolved products: {resolved_products}")
    log_info(f"Joblib: jobs={n_jobs} | backend={joblib_backend} | batch_size={joblib_batch_size} | pre_dispatch={joblib_pre_dispatch}")
    log_info(f"Grid resolution: {grid_res}")
    if apply_fp_fix:
        log_info(f"FP-fix DEM resolution: {grid_res if fp_fix_dem_res is None else fp_fix_dem_res}")

    if workflow == "run":
        if _contains_downstream_only_products(resolved_products):
            raise ValueError(
                "FAST_CHANGE, FAST_ITD, and FAST_TREECLOUDS are downstream-only products. "
                "Use --workflow derive-only on an existing processed root."
            )

        out = _run_processing_with_optional_fpfix(
            in_path=in_path,
            out_dir=out_dir,
            sensor_mode=sensor_mode,
            products=resolved_products,
            grid_res=grid_res,
            dem_method=dem_method,
            dsm_method=dsm_method,
            chm_method=chm_method,
            chm_methods=chm_methods,
            chm_surface_method=chm_surface_method,
            chm_smooth_method=chm_smooth_method,
            chm_percentile=chm_percentile,
            chm_percentile_low=chm_percentile_low,
            chm_percentile_high=chm_percentile_high,
            chm_pitfree_thresholds=chm_pitfree_thresholds,
            chm_use_first_returns=chm_use_first_returns,
            chm_spikefree_freeze_distance=chm_spikefree_freeze_distance,
            chm_spikefree_insertion_buffer=chm_spikefree_insertion_buffer,
            chm_median_size=chm_median_size,
            chm_gaussian_sigma=chm_gaussian_sigma,
            chm_min_height=chm_min_height,
            chm_fill_ground_voids_zero=chm_fill_ground_voids_zero,
            chm_void_ground_threshold=chm_void_ground_threshold,
            terrain_products=terrain_products,
            hillshade_azimuth=hillshade_azimuth,
            hillshade_altitude=hillshade_altitude,
            hillshade_z_factor=hillshade_z_factor,
            tpi_radius=tpi_radius,
            twi_eps=twi_eps,
            dtw_max_distance=dtw_max_distance,
            structure_products=structure_products,
            structure_res=structure_res,
            structure_min_h=structure_min_h,
            structure_bin_size=structure_bin_size,
            canopy_thr=canopy_thr,
            canopy_mode=canopy_mode,
            structure_na_fill=structure_na_fill,
            recursive=recursive,
            n_jobs=n_jobs,
            joblib_backend=joblib_backend,
            joblib_batch_size=joblib_batch_size,
            joblib_pre_dispatch=joblib_pre_dispatch,
            apply_fp_fix=apply_fp_fix,
            fp_fix_dem_res=fp_fix_dem_res,
            fp_fix_nonground_to_ground_max_z=fp_fix_nonground_to_ground_max_z,
            fp_fix_ground_to_nonground_min_z=fp_fix_ground_to_nonground_min_z,
            keep_fp_fix_temp=keep_fp_fix_temp,
            skip_existing=skip_existing,
            overwrite=overwrite,
        )
        print(f"[TIME] WORKFLOW run     : {perf_counter() - total_t0:.2f}s")
        return out

    if workflow == "merge":
        workspace_root = _resolve_workspace_root(Path(in_path), sensor_mode)
        manifest = _load_manifest_from_workspace(workspace_root)
        processed_root = _resolve_processed_root(workspace_root, sensor_mode)
        log_info(f"merge workspace: {workspace_root}")
        log_info(f"merge processed root: {processed_root}")
        merge_root = workspace_root / f"Merged_{sensor_mode.upper()}"
        merge_root.mkdir(parents=True, exist_ok=True)

        merge_products = [p for p in requested_products if p != "all"]

        merged_outputs: dict[str, str] = {}
        if PRODUCT_CHM in merge_products:
            chm_targets = _resolve_chm_targets(
                chm_method=chm_method,
                chm_methods=chm_methods,
                chm_surface_method=chm_surface_method,
            )
            for target in chm_targets:
                out = merge_processed_tiles(
                    manifest=manifest,
                    processed_root=processed_root,
                    merged_root=merge_root,
                    products=[PRODUCT_CHM],
                    chm_method=target["label"],
                )
                merged_outputs.update(out)

        other_products = [p for p in merge_products if p != PRODUCT_CHM]
        if other_products:
            out = merge_processed_tiles(
                manifest=manifest,
                processed_root=processed_root,
                merged_root=merge_root,
                products=other_products,
                chm_method=None,
            )
            merged_outputs.update(out)

        if cleanup_tiles:
            tiles_dir = workspace_root / "tiles"
            if tiles_dir.exists():
                cleanup_tiling_workspace(tiles_dir)

        print(f"[TIME] WORKFLOW merge   : {perf_counter() - total_t0:.2f}s")
        return str(merge_root if merged_outputs else processed_root)

    if workflow == "derive-only":
        p = Path(in_path)
        workspace_manifest: dict | None = None
        direct_dsm_input: Path | None = None

        explicitly_requested = set(requested_products)
        explicitly_requested.discard("all")

        if _is_las_like(p) and bool({PRODUCT_DSM, PRODUCT_ITD, PRODUCT_TREECLOUDS} & explicitly_requested):
            workspace_manifest = _ensure_tiles_for_raw_input(
                in_path=in_path,
                out_dir=out_dir,
                sensor_mode=sensor_mode,
                tile_size_m=tile_size_m,
                buffer_m=buffer_m,
                recursive=recursive,
                overwrite_tiles=overwrite_tiles,
                small_tile_merge_frac=small_tile_merge_frac,
            )
            workspace_root = Path(workspace_manifest["workspace_root"])
            processed_root = _resolve_processed_root(workspace_root, sensor_mode)
            raw_tiles_root = Path(workspace_manifest["tiles_dir"])
        elif (p / "tile_manifest.json").exists():
            processed_root = _resolve_processed_root(p, sensor_mode)
            raw_tiles_root = p / "tiles"
        elif p.name == PRODUCT_DSM and p.exists():
            processed_root = p.parent
            raw_tiles_root = processed_root.parent / "tiles"
            direct_dsm_input = p
        elif (p / PRODUCT_NORMALIZED).exists() or (p / PRODUCT_GC).exists() or (p / PRODUCT_DEM).exists():
            processed_root = p
            raw_tiles_root = p.parent / "tiles"
        elif p.name == f"Processed_{sensor_mode.upper()}" and p.exists():
            processed_root = p
            raw_tiles_root = p.parent / "tiles"
        elif p.name == PRODUCT_GC and p.exists():
            processed_root = p.parent
            raw_tiles_root = processed_root.parent / "tiles"
        elif p.name == PRODUCT_NORMALIZED and p.exists():
            processed_root = p.parent
            raw_tiles_root = processed_root.parent / "tiles"
        else:
            processed_root = p
            raw_tiles_root = p.parent / "tiles"

        classified_root = processed_root / PRODUCT_GC
        normalized_root = processed_root / PRODUCT_NORMALIZED
        dem_root = processed_root / PRODUCT_DEM
        dsm_root = processed_root / PRODUCT_DSM

        print(f"[INFO] derive-only root: {processed_root}")
        if normalized_root.exists():
            print(f"[INFO] Existing FAST_NORMALIZED found: {normalized_root}")

        if PRODUCT_DEM in explicitly_requested and not (_has_raster_outputs(dem_root) and skip_existing and not overwrite):
            if not classified_root.exists():
                raise FileNotFoundError(f"FAST_GC folder not found for DEM derivation: {classified_root}")
            derive_products_from_classified_root(
                classified_root=classified_root,
                out_root=processed_root,
                products=[PRODUCT_DEM],
                grid_res=grid_res,
                dem_method=dem_method,
                n_jobs=n_jobs,
                joblib_backend=joblib_backend,
                joblib_batch_size=joblib_batch_size,
                joblib_pre_dispatch=joblib_pre_dispatch,
            )

        need_normalized_for_chm = PRODUCT_CHM in explicitly_requested
        need_normalized_for_structure = PRODUCT_STRUCTURE in explicitly_requested

        if (
            (PRODUCT_NORMALIZED in explicitly_requested)
            or (need_normalized_for_chm and not normalized_root.exists())
            or (need_normalized_for_structure and not normalized_root.exists())
        ) and not (_existing_path(normalized_root) and skip_existing and not overwrite):
            if not classified_root.exists():
                raise FileNotFoundError(f"FAST_GC folder not found for NORMALIZED derivation: {classified_root}")
            derive_products_from_classified_root(
                classified_root=classified_root,
                out_root=processed_root,
                products=[PRODUCT_NORMALIZED],
                grid_res=grid_res,
                dem_method=dem_method,
                n_jobs=n_jobs,
                joblib_backend=joblib_backend,
                joblib_batch_size=joblib_batch_size,
                joblib_pre_dispatch=joblib_pre_dispatch,
            )

        if PRODUCT_DSM in explicitly_requested:
            if not (_has_raster_outputs(dsm_root) and skip_existing and not overwrite):
                if not raw_tiles_root.exists():
                    if _is_las_like(p):
                        workspace_manifest = _ensure_tiles_for_raw_input(
                            in_path=in_path,
                            out_dir=out_dir,
                            sensor_mode=sensor_mode,
                            tile_size_m=tile_size_m,
                            buffer_m=buffer_m,
                            recursive=recursive,
                            overwrite_tiles=overwrite_tiles,
                            small_tile_merge_frac=small_tile_merge_frac,
                        )
                        workspace_root = Path(workspace_manifest["workspace_root"])
                        processed_root = _resolve_processed_root(workspace_root, sensor_mode)
                        raw_tiles_root = Path(workspace_manifest["tiles_dir"])
                        dsm_root = processed_root / PRODUCT_DSM
                    else:
                        raise FileNotFoundError(f"Raw tiles root not found for DSM derivation: {raw_tiles_root}")

                derive_products_from_raw_root(
                    raw_root=raw_tiles_root,
                    out_root=processed_root,
                    sensor_mode=sensor_mode,
                    products=[PRODUCT_DSM],
                    grid_res=grid_res,
                    dsm_method=dsm_method,
                    recursive=True,
                    n_jobs=n_jobs,
                    joblib_backend=joblib_backend,
                    joblib_batch_size=joblib_batch_size,
                    joblib_pre_dispatch=joblib_pre_dispatch,
                    spikefree_freeze_distance=chm_spikefree_freeze_distance,
                    spikefree_insertion_buffer=chm_spikefree_insertion_buffer,
                )

                if workspace_manifest is not None:
                    merge_root = Path(workspace_manifest["workspace_root"]) / f"Merged_{sensor_mode.upper()}"
                    merge_root.mkdir(parents=True, exist_ok=True)
                    merge_processed_tiles(
                        manifest=workspace_manifest,
                        processed_root=processed_root,
                        merged_root=merge_root,
                        products=[PRODUCT_DSM],
                        chm_method=None,
                    )

        if PRODUCT_CHM in explicitly_requested:
            targets = _resolve_chm_targets(
                chm_method=chm_method,
                chm_methods=chm_methods,
                chm_surface_method=chm_surface_method,
            )
            has_spikefree = any(t["method"] == "spikefree" for t in targets)
            has_native = any(not t["is_dsm_derived"] for t in targets)

            if has_native and not normalized_root.exists():
                raise FileNotFoundError(f"FAST_NORMALIZED folder not found for CHM derivation: {normalized_root}")

            if has_spikefree:
                if not _has_raster_outputs(dem_root):
                    if not classified_root.exists():
                        raise FileNotFoundError(f"FAST_GC folder not found for DEM derivation: {classified_root}")
                    derive_products_from_classified_root(
                        classified_root=classified_root,
                        out_root=processed_root,
                        products=[PRODUCT_DEM],
                        grid_res=grid_res,
                        dem_method=dem_method,
                        n_jobs=n_jobs,
                        joblib_backend=joblib_backend,
                        joblib_batch_size=joblib_batch_size,
                        joblib_pre_dispatch=joblib_pre_dispatch,
                    )

                if not _has_raster_outputs(dsm_root):
                    if not raw_tiles_root.exists():
                        if _is_las_like(p):
                            workspace_manifest = _ensure_tiles_for_raw_input(
                                in_path=in_path,
                                out_dir=out_dir,
                                sensor_mode=sensor_mode,
                                tile_size_m=tile_size_m,
                                buffer_m=buffer_m,
                                recursive=recursive,
                                overwrite_tiles=overwrite_tiles,
                                small_tile_merge_frac=small_tile_merge_frac,
                            )
                            workspace_root = Path(workspace_manifest["workspace_root"])
                            processed_root = _resolve_processed_root(workspace_root, sensor_mode)
                            raw_tiles_root = Path(workspace_manifest["tiles_dir"])
                            dsm_root = processed_root / PRODUCT_DSM
                        else:
                            raise FileNotFoundError(f"Raw tiles root not found for DSM derivation: {raw_tiles_root}")

                    derive_products_from_raw_root(
                        raw_root=raw_tiles_root,
                        out_root=processed_root,
                        sensor_mode=sensor_mode,
                        products=[PRODUCT_DSM],
                        grid_res=grid_res,
                        dsm_method="spikefree",
                        recursive=True,
                        n_jobs=n_jobs,
                        joblib_backend=joblib_backend,
                        joblib_batch_size=joblib_batch_size,
                        joblib_pre_dispatch=joblib_pre_dispatch,
                        spikefree_freeze_distance=chm_spikefree_freeze_distance,
                        spikefree_insertion_buffer=chm_spikefree_insertion_buffer,
                    )

                    if workspace_manifest is not None:
                        merge_root = Path(workspace_manifest["workspace_root"]) / f"Merged_{sensor_mode.upper()}"
                        merge_root.mkdir(parents=True, exist_ok=True)
                        merge_processed_tiles(
                            manifest=workspace_manifest,
                            processed_root=processed_root,
                            merged_root=merge_root,
                            products=[PRODUCT_DSM],
                            chm_method=None,
                        )

                if not _has_raster_outputs(dsm_root):
                    raise FileNotFoundError(f"No DSM rasters found even after spikefree DSM generation: {dsm_root}")

            derive_chm_from_processed_root(
                processed_root=processed_root,
                sensor_mode=sensor_mode,
                chm_method=chm_method,
                chm_methods=chm_methods,
                chm_surface_method=chm_surface_method,
                chm_smooth_method=chm_smooth_method,
                grid_res=grid_res,
                chm_percentile=chm_percentile,
                chm_percentile_low=chm_percentile_low,
                chm_percentile_high=chm_percentile_high,
                chm_pitfree_thresholds=chm_pitfree_thresholds,
                chm_use_first_returns=chm_use_first_returns,
                chm_spikefree_freeze_distance=chm_spikefree_freeze_distance,
                chm_spikefree_insertion_buffer=chm_spikefree_insertion_buffer,
                chm_median_size=chm_median_size,
                chm_gaussian_sigma=chm_gaussian_sigma,
                chm_min_height=chm_min_height,
                chm_fill_ground_voids_zero=chm_fill_ground_voids_zero,
                chm_void_ground_threshold=chm_void_ground_threshold,
                n_jobs=n_jobs,
                joblib_backend=joblib_backend,
                joblib_batch_size=joblib_batch_size,
                joblib_pre_dispatch=joblib_pre_dispatch,
                skip_existing=skip_existing,
                overwrite=overwrite,
            )

        if PRODUCT_TERRAIN in explicitly_requested:
            if not _has_raster_outputs(dem_root):
                if not classified_root.exists():
                    raise FileNotFoundError(f"FAST_GC folder not found for TERRAIN derivation: {classified_root}")
                derive_products_from_classified_root(
                    classified_root=classified_root,
                    out_root=processed_root,
                    products=[PRODUCT_DEM],
                    grid_res=grid_res,
                    dem_method=dem_method,
                    n_jobs=n_jobs,
                    joblib_backend=joblib_backend,
                    joblib_batch_size=joblib_batch_size,
                    joblib_pre_dispatch=joblib_pre_dispatch,
                )

            _call_with_supported_kwargs(
                run_terrain_from_processed_root,
                processed_root=processed_root,
                terrain_products=terrain_products,
                hillshade_azimuth=hillshade_azimuth,
                hillshade_altitude=hillshade_altitude,
                hillshade_z_factor=hillshade_z_factor,
                tpi_radius=tpi_radius,
                twi_eps=twi_eps,
                dtw_max_distance=dtw_max_distance,
                n_jobs=n_jobs,
                joblib_backend=joblib_backend,
                joblib_batch_size=joblib_batch_size,
                joblib_pre_dispatch=joblib_pre_dispatch,
                skip_existing=skip_existing,
                overwrite=overwrite,
            )

        if PRODUCT_STRUCTURE in explicitly_requested:
            structure_root = processed_root
            if p.is_dir() and p.name.startswith("Merged_"):
                structure_root = p
            elif _is_las_like(p) and "FAST_NORMALIZED" in p.stem:
                structure_root = p
            _call_with_supported_kwargs(
                run_structure_from_root,
                source_root=structure_root,
                sensor_mode=sensor_mode,
                structure_products=structure_products,
                structure_res=structure_res,
                structure_min_h=structure_min_h,
                structure_bin_size=structure_bin_size,
                canopy_thr=canopy_thr,
                canopy_mode=canopy_mode,
                structure_na_fill=structure_na_fill,
                n_jobs=n_jobs,
                joblib_backend=joblib_backend,
                joblib_batch_size=joblib_batch_size,
                joblib_pre_dispatch=joblib_pre_dispatch,
                skip_existing=skip_existing,
                overwrite=overwrite,
            )

        if PRODUCT_CHANGE in explicitly_requested:
            _call_with_supported_kwargs(
                run_change_from_processed_root,
                processed_root=processed_root,
                change_input_type=change_input_type,
                change_mode=change_mode,
                change_threshold=change_threshold,
                baseline_index=change_baseline_index,
                skip_existing=skip_existing,
                overwrite=overwrite,
                source_subdir=change_source_subdir,
                sigma1=change_sigma1,
                sigma2=change_sigma2,
                lod_mode=change_lod_mode,
                n_jobs=n_jobs,
                joblib_backend=joblib_backend,
                joblib_batch_size=joblib_batch_size,
                joblib_pre_dispatch=joblib_pre_dispatch,
            )

        if PRODUCT_ITD in explicitly_requested:
            itd_input_root = processed_root

            chm_root = processed_root / PRODUCT_CHM
            has_any_chm = chm_root.exists() and any(chm_root.iterdir())
            has_dsm = _has_raster_outputs(dsm_root)

            if direct_dsm_input is not None:
                itd_input_root = direct_dsm_input
            elif itd_source_chm is not None and str(itd_source_chm).strip():
                itd_input_root = processed_root
            elif has_any_chm:
                itd_input_root = processed_root
            else:
                if not has_dsm:
                    if not raw_tiles_root.exists():
                        if _is_las_like(p):
                            workspace_manifest = _ensure_tiles_for_raw_input(
                                in_path=in_path,
                                out_dir=out_dir,
                                sensor_mode=sensor_mode,
                                tile_size_m=tile_size_m,
                                buffer_m=buffer_m,
                                recursive=recursive,
                                overwrite_tiles=overwrite_tiles,
                                small_tile_merge_frac=small_tile_merge_frac,
                            )
                            workspace_root = Path(workspace_manifest["workspace_root"])
                            processed_root = _resolve_processed_root(workspace_root, sensor_mode)
                            raw_tiles_root = Path(workspace_manifest["tiles_dir"])
                            dsm_root = processed_root / PRODUCT_DSM
                        else:
                            raise FileNotFoundError(f"Raw tiles root not found for DSM derivation: {raw_tiles_root}")

                    derive_products_from_raw_root(
                        raw_root=raw_tiles_root,
                        out_root=processed_root,
                        sensor_mode=sensor_mode,
                        products=[PRODUCT_DSM],
                        grid_res=grid_res,
                        dsm_method=dsm_method,
                        recursive=True,
                        n_jobs=n_jobs,
                        joblib_backend=joblib_backend,
                        joblib_batch_size=joblib_batch_size,
                        joblib_pre_dispatch=joblib_pre_dispatch,
                        spikefree_freeze_distance=chm_spikefree_freeze_distance,
                        spikefree_insertion_buffer=chm_spikefree_insertion_buffer,
                    )

                    if workspace_manifest is not None:
                        merge_root = Path(workspace_manifest["workspace_root"]) / f"Merged_{sensor_mode.upper()}"
                        merge_root.mkdir(parents=True, exist_ok=True)
                        merge_processed_tiles(
                            manifest=workspace_manifest,
                            processed_root=processed_root,
                            merged_root=merge_root,
                            products=[PRODUCT_DSM],
                            chm_method=None,
                        )

                itd_input_root = dsm_root

            _call_with_supported_kwargs(
                run_itd_from_processed_root,
                processed_root=itd_input_root,
                method=itd_method,
                source_chm=itd_source_chm,
                itd_min_height=itd_min_height,
                itd_crown_window_m=itd_crown_window_m,
                itd_min_peak_separation_m=itd_min_peak_separation_m,
                itd_angle_threshold_deg=itd_angle_threshold_deg,
                itd_screen_max_pair_distance_m=itd_screen_max_pair_distance_m,
                itd_banded_neighborhood_px=itd_banded_neighborhood_px,
                itd_min_crown_area_m2=itd_min_crown_area_m2,
                n_jobs=n_jobs,
                joblib_backend=joblib_backend,
                joblib_batch_size=joblib_batch_size,
                joblib_pre_dispatch=joblib_pre_dispatch,
                skip_existing=skip_existing,
                overwrite=overwrite,
            )

        if PRODUCT_TREECLOUDS in explicitly_requested:
            treeclouds_root = processed_root
            if p.is_dir() and p.name.startswith("Merged_"):
                treeclouds_root = p
            elif _is_las_like(p) and "FAST_CHM_" in p.stem:
                treeclouds_root = p.parent
            elif _is_las_like(p) and ("FAST_NORMALIZED" in p.stem or "FAST_GC" in p.stem):
                treeclouds_root = p.parent

            _call_with_supported_kwargs(
                run_treeclouds_from_root,
                source_root=treeclouds_root,
                method=itd_method,
                source_chm=itd_source_chm,
                las_source=treeclouds_las_source,
                min_height=treeclouds_min_height,
                write_individual=treeclouds_write_individual,
                n_jobs=n_jobs,
                joblib_backend=joblib_backend,
                joblib_batch_size=joblib_batch_size,
                joblib_pre_dispatch=joblib_pre_dispatch,
                skip_existing=skip_existing,
                overwrite=overwrite,
            )

        print(f"[TIME] WORKFLOW derive-only: {perf_counter() - total_t0:.2f}s")
        return str(processed_root)

    if workflow in {"tile-only", "tile-run", "tile-run-merge"}:
        if _contains_downstream_only_products(resolved_products):
            raise ValueError(
                "FAST_CHANGE, FAST_ITD, and FAST_TREECLOUDS are downstream-only products. "
                "Use --workflow derive-only on an existing processed root."
            )

        manifest = tile_las_dataset(
            in_path=in_path,
            out_dir=out_dir,
            sensor_mode=sensor_mode,
            tile_size_m=tile_size_m,
            buffer_m=buffer_m,
            recursive=recursive,
            overwrite_tiles=overwrite_tiles,
            small_tile_merge_frac=small_tile_merge_frac,
        )
        workspace_root = Path(manifest["workspace_root"])
        tiles_dir = Path(manifest["tiles_dir"])

        if workflow == "tile-only":
            print(f"[TIME] WORKFLOW total  : {perf_counter() - total_t0:.2f}s")
            return str(workspace_root)

        processed_root = Path(
            _run_processing_with_optional_fpfix(
                in_path=str(tiles_dir),
                out_dir=str(workspace_root),
                sensor_mode=sensor_mode,
                products=resolved_products,
                grid_res=grid_res,
                dem_method=dem_method,
                dsm_method=dsm_method,
                chm_method=chm_method,
                chm_methods=chm_methods,
                chm_surface_method=chm_surface_method,
                chm_smooth_method=chm_smooth_method,
                chm_percentile=chm_percentile,
                chm_percentile_low=chm_percentile_low,
                chm_percentile_high=chm_percentile_high,
                chm_pitfree_thresholds=chm_pitfree_thresholds,
                chm_use_first_returns=chm_use_first_returns,
                chm_spikefree_freeze_distance=chm_spikefree_freeze_distance,
                chm_spikefree_insertion_buffer=chm_spikefree_insertion_buffer,
                chm_median_size=chm_median_size,
                chm_gaussian_sigma=chm_gaussian_sigma,
                chm_min_height=chm_min_height,
                chm_fill_ground_voids_zero=chm_fill_ground_voids_zero,
                chm_void_ground_threshold=chm_void_ground_threshold,
                terrain_products=terrain_products,
                hillshade_azimuth=hillshade_azimuth,
                hillshade_altitude=hillshade_altitude,
                hillshade_z_factor=hillshade_z_factor,
                tpi_radius=tpi_radius,
                twi_eps=twi_eps,
                dtw_max_distance=dtw_max_distance,
                structure_products=structure_products,
                structure_res=structure_res,
                structure_min_h=structure_min_h,
                structure_bin_size=structure_bin_size,
                canopy_thr=canopy_thr,
                canopy_mode=canopy_mode,
                structure_na_fill=structure_na_fill,
                recursive=False,
                n_jobs=n_jobs,
                joblib_backend=joblib_backend,
                joblib_batch_size=joblib_batch_size,
                joblib_pre_dispatch=joblib_pre_dispatch,
                apply_fp_fix=apply_fp_fix,
                fp_fix_dem_res=fp_fix_dem_res,
                fp_fix_nonground_to_ground_max_z=fp_fix_nonground_to_ground_max_z,
                fp_fix_ground_to_nonground_min_z=fp_fix_ground_to_nonground_min_z,
                keep_fp_fix_temp=keep_fp_fix_temp,
                skip_existing=skip_existing,
                overwrite=overwrite,
            )
        )

        if workflow == "tile-run":
            print(f"[TIME] WORKFLOW total  : {perf_counter() - total_t0:.2f}s")
            return str(processed_root)

        merge_root = workspace_root / f"Merged_{sensor_mode.upper()}"
        merge_root.mkdir(parents=True, exist_ok=True)

        merge_products = [p for p in requested_products if p != "all"]

        merged_outputs: dict[str, str] = {}
        if PRODUCT_CHM in merge_products:
            chm_targets = _resolve_chm_targets(
                chm_method=chm_method,
                chm_methods=chm_methods,
                chm_surface_method=chm_surface_method,
            )
            for target in chm_targets:
                out = merge_processed_tiles(
                    manifest=manifest,
                    processed_root=processed_root,
                    merged_root=merge_root,
                    products=[PRODUCT_CHM],
                    chm_method=target["label"],
                )
                merged_outputs.update(out)

        other_products = [p for p in merge_products if p != PRODUCT_CHM]
        if other_products:
            out = merge_processed_tiles(
                manifest=manifest,
                processed_root=processed_root,
                merged_root=merge_root,
                products=other_products,
                chm_method=None,
            )
            merged_outputs.update(out)

        if cleanup_tiles:
            tiles_dir = workspace_root / "tiles"
            if tiles_dir.exists():
                cleanup_tiling_workspace(tiles_dir)

        print(f"[TIME] WORKFLOW total   : {perf_counter() - total_t0:.2f}s")
        return str(merge_root if merged_outputs else processed_root)

    raise ValueError(f"Unsupported workflow: {workflow}")