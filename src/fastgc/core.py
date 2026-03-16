from __future__ import annotations

import inspect
import json
from pathlib import Path
from time import perf_counter

from .chm import (
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
from .terrain import run_terrain_from_processed_root

PRODUCT_TERRAIN = "FAST_TERRAIN"
PRODUCT_CHANGE = "FAST_CHANGE"
PRODUCT_ITD = "FAST_ITD"

DEFAULT_WORKFLOW = "run"
WORKFLOW_CHOICES = ["run", "tile-only", "tile-run", "tile-run-merge", "merge", "derive-only"]

_CHM_ALGORITHMS = {"p2r", "p99", "tin", "pitfree", "spikefree"}
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

    return out


def _needs_classified_source(products: list[str]) -> bool:
    needed = set(products)
    return bool({PRODUCT_GC, PRODUCT_DEM, PRODUCT_NORMALIZED, PRODUCT_CHM, PRODUCT_TERRAIN} & needed)


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


def _contains_downstream_only_products(products: list[str]) -> bool:
    return bool({PRODUCT_CHANGE, PRODUCT_ITD} & set(products))


def _call_with_supported_kwargs(func, /, **kwargs):
    """Call a function using only the kwargs it actually supports.

    This keeps core.py compatible with modules that have already been rewired
    for Joblib/monster.py as well as older modules that still expose the older
    sequential signatures.
    """
    sig = inspect.signature(func)
    supported = {}
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_var_kw:
        supported = dict(kwargs)
    else:
        supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(**supported)


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
                raise ValueError(f"--chm_methods only supports surface algorithms: {sorted(_CHM_ALGORITHMS)}")
            targets.append(
                {
                    "method": mm,
                    "surface_method": None,
                    "label": chm_output_label(mm, None),
                }
            )
        return targets

    if method in _CHM_ALGORITHMS:
        targets.append(
            {
                "method": method,
                "surface_method": None,
                "label": chm_output_label(method, None),
            }
        )
        return targets

    if method in _CHM_SELECTORS:
        if surface_method not in _CHM_ALGORITHMS:
            raise ValueError(f"Invalid --chm_surface_method: {surface_method}")
        targets.append(
            {
                "method": method,
                "surface_method": surface_method,
                "label": chm_output_label(method, surface_method),
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
    normalized_root = resolve_normalized_root(processed_root)
    if not Path(normalized_root).exists():
        raise FileNotFoundError(f"FAST_NORMALIZED folder not found: {normalized_root}")

    outputs: list[str] = []
    targets = _resolve_chm_targets(
        chm_method=chm_method,
        chm_methods=chm_methods,
        chm_surface_method=chm_surface_method,
    )

    for target in targets:
        out_root = Path(chm_method_output_dir(processed_root, target["method"], target["surface_method"]))
        if skip_existing and _existing_path(out_root) and not overwrite:
            print(f"[SKIP] Existing CHM output found: {out_root}")
            outputs.append(str(out_root))
            continue

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
    recursive: bool,
    n_jobs: int = 1,
    joblib_backend: str = "loky",
    joblib_batch_size: int | str = "auto",
    joblib_pre_dispatch: str | int = "2*n_jobs",
    apply_fp_fix: bool = True,
    fp_fix_dem_res: float = 0.25,
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
        )

        fix_summary = _call_with_supported_kwargs(
            apply_fp_fix_to_output_root,
            out_root=out_root,
            sensor_mode=sensor_mode,
            dem_res=fp_fix_dem_res,
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
    dem_method: str = "min",
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
    workflow: str = DEFAULT_WORKFLOW,
    tile_size_m: float = 30.0,
    buffer_m: float = 5.0,
    cleanup_tiles: bool = False,
    overwrite_tiles: bool = False,
    small_tile_merge_frac: float = 0.25,
    apply_fp_fix: bool = True,
    fp_fix_dem_res: float = 0.25,
    fp_fix_nonground_to_ground_max_z: float = 0.0,
    fp_fix_ground_to_nonground_min_z: float = 0.06,
    keep_fp_fix_temp: bool = False,
):
    workflow = str(workflow).strip().lower()
    if workflow not in WORKFLOW_CHOICES:
        raise ValueError(f"Unsupported workflow: {workflow}")

    requested_products = _requested_products(products)
    resolved_products = _resolve_products(products)
    total_t0 = perf_counter()

    stage_banner("WORKFLOW", source=str(in_path), total=len(resolved_products), unit="stage")
    log_info(f"Workflow: {workflow}")
    log_info(f"Sensor mode: {sensor_mode}")
    log_info(f"Requested products: {requested_products}")
    log_info(f"Resolved products: {resolved_products}")
    log_info(f"Joblib: jobs={n_jobs} | backend={joblib_backend} | batch_size={joblib_batch_size} | pre_dispatch={joblib_pre_dispatch}")

    if workflow == "run":
        if _contains_downstream_only_products(resolved_products):
            raise ValueError(
                "FAST_CHANGE and FAST_ITD are downstream-only products. "
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

        if (p / "tile_manifest.json").exists():
            processed_root = _resolve_processed_root(p, sensor_mode)
            raw_tiles_root = p / "tiles"
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

        print(f"[INFO] derive-only root: {processed_root}")
        if normalized_root.exists():
            print(f"[INFO] Existing FAST_NORMALIZED found: {normalized_root}")

        explicitly_requested = set(requested_products)
        explicitly_requested.discard("all")

        if PRODUCT_DEM in explicitly_requested and not (_existing_path(dem_root) and skip_existing and not overwrite):
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
        if (
            (PRODUCT_NORMALIZED in explicitly_requested) or
            (need_normalized_for_chm and not normalized_root.exists())
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
            dsm_root = processed_root / PRODUCT_DSM
            if not (_existing_path(dsm_root) and skip_existing and not overwrite):
                if not raw_tiles_root.exists():
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
                )

        if PRODUCT_CHM in explicitly_requested:
            if not normalized_root.exists():
                raise FileNotFoundError(f"FAST_NORMALIZED folder not found for CHM derivation: {normalized_root}")

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
            if not dem_root.exists():
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
            _call_with_supported_kwargs(
                run_itd_from_processed_root,
                processed_root=processed_root,
                method=itd_method,
                source_chm=itd_source_chm,
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
                "FAST_CHANGE and FAST_ITD are downstream-only products. "
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