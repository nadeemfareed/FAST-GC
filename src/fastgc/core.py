from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from .io_las import PRODUCT_GC, derive_products_from_classified_root, process_fastgc_path
from .merge import cleanup_tiling_workspace, merge_processed_tiles
from .post_fp_fix import apply_fp_fix_to_output_root
from .preprocess import tile_las_dataset

DEFAULT_WORKFLOW = "run"
WORKFLOW_CHOICES = ["run", "tile-only", "tile-run", "tile-run-merge", "merge"]


def _resolve_products(products: list[str] | None) -> list[str]:
    requested = list(products or [PRODUCT_GC])

    if "all" in requested:
        return [PRODUCT_GC, "FAST_DEM", "FAST_NORMALIZED", "FAST_DSM", "FAST_CHM"]

    out: list[str] = []
    seen: set[str] = set()

    for p in requested:
        if p not in seen:
            out.append(p)
            seen.add(p)

    if PRODUCT_GC not in seen:
        out.insert(0, PRODUCT_GC)
        seen.add(PRODUCT_GC)

    if "FAST_NORMALIZED" in seen and "FAST_DEM" not in seen:
        out.insert(out.index("FAST_NORMALIZED"), "FAST_DEM")
        seen.add("FAST_DEM")

    if "FAST_CHM" in seen:
        if "FAST_DEM" not in seen:
            out.insert(out.index("FAST_CHM"), "FAST_DEM")
            seen.add("FAST_DEM")
        if "FAST_DSM" not in seen:
            out.insert(out.index("FAST_CHM"), "FAST_DSM")
            seen.add("FAST_DSM")

    return out


def _load_manifest_from_workspace(workspace_root: Path) -> dict:
    manifest_fp = workspace_root / "tile_manifest.json"
    if not manifest_fp.exists():
        raise FileNotFoundError(f"Tile manifest not found: {manifest_fp}")
    with manifest_fp.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_processed_root(workspace_root: Path, sensor_mode: str) -> Path:
    candidates = [
        workspace_root / f"Processed_{sensor_mode.upper()}",
        workspace_root,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _run_processing_with_optional_fpfix(
    *,
    in_path: str,
    out_dir: str | None,
    sensor_mode: str,
    products: list[str] | None,
    grid_res: float,
    dem_method: str,
    dsm_method: str,
    chm_median_size: int,
    chm_min_height: float,
    recursive: bool,
    apply_fp_fix: bool,
    fp_fix_dem_res: float,
    fp_fix_nonground_to_ground_max_z: float,
    fp_fix_ground_to_nonground_min_z: float,
    keep_fp_fix_temp: bool,
) -> str:
    resolved_products = _resolve_products(products)

    if not apply_fp_fix:
        return process_fastgc_path(
            in_path=in_path,
            out_dir=out_dir,
            sensor_mode=sensor_mode,
            products=resolved_products,
            grid_res=grid_res,
            dem_method=dem_method,
            dsm_method=dsm_method,
            chm_median_size=chm_median_size,
            chm_min_height=chm_min_height,
            recursive=recursive,
        )

    # Stage 1: classification only
    out_root = process_fastgc_path(
        in_path=in_path,
        out_dir=out_dir,
        sensor_mode=sensor_mode,
        products=[PRODUCT_GC],
        grid_res=grid_res,
        dem_method=dem_method,
        dsm_method=dsm_method,
        chm_median_size=chm_median_size,
        chm_min_height=chm_min_height,
        recursive=recursive,
    )

    # Stage 2: residual-based post-fix
    fix_summary = apply_fp_fix_to_output_root(
        out_root=out_root,
        sensor_mode=sensor_mode,
        dem_res=fp_fix_dem_res,
        nonground_to_ground_max_z=fp_fix_nonground_to_ground_max_z,
        ground_to_nonground_min_z=fp_fix_ground_to_nonground_min_z,
        keep_temp=keep_fp_fix_temp,
    )
    changed = int(fix_summary.get("total_changed_points", 0))
    print(f"[INFO] FP-FIX changed points: {changed}")

    # Stage 3: regenerate requested downstream products from corrected FAST_GC tiles/files
    downstream = [p for p in resolved_products if p != PRODUCT_GC]
    if downstream:
        derive_products_from_classified_root(
            classified_root=Path(out_root) / PRODUCT_GC,
            out_root=out_root,
            products=downstream,
            grid_res=grid_res,
            dem_method=dem_method,
            dsm_method=dsm_method,
            chm_median_size=chm_median_size,
            chm_min_height=chm_min_height,
        )

    return out_root


def run_fastgc(
    in_path: str,
    out_dir: str | None,
    sensor_mode: str,
    products: list[str] | None = None,
    grid_res: float = 0.5,
    recursive: bool = False,
    *,
    dem_method: str = "min",
    dsm_method: str = "max",
    chm_median_size: int = 0,
    chm_min_height: float = 0.0,
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
    if workflow not in WORKFLOW_CHOICES:
        raise ValueError(f"Unsupported workflow: {workflow}")

    resolved_products = _resolve_products(products)
    total_t0 = perf_counter()

    if workflow == "run":
        print(f"[FLOW] FP-FIX           : {'ON' if apply_fp_fix else 'OFF'}")
        print(f"[FLOW] DEM method       : {dem_method}")
        print(f"[FLOW] DSM method       : {dsm_method}")
        print(f"[FLOW] CHM median size  : {chm_median_size}")
        print(f"[FLOW] CHM min height   : {chm_min_height}")
        out = _run_processing_with_optional_fpfix(
            in_path=in_path,
            out_dir=out_dir,
            sensor_mode=sensor_mode,
            products=products,
            grid_res=grid_res,
            dem_method=dem_method,
            dsm_method=dsm_method,
            chm_median_size=chm_median_size,
            chm_min_height=chm_min_height,
            recursive=recursive,
            apply_fp_fix=apply_fp_fix,
            fp_fix_dem_res=fp_fix_dem_res,
            fp_fix_nonground_to_ground_max_z=fp_fix_nonground_to_ground_max_z,
            fp_fix_ground_to_nonground_min_z=fp_fix_ground_to_nonground_min_z,
            keep_fp_fix_temp=keep_fp_fix_temp,
        )
        print(f"[TIME] WORKFLOW run: {perf_counter() - total_t0:.2f}s")
        return out

    if workflow == "merge":
        workspace_root = Path(in_path)
        manifest = _load_manifest_from_workspace(workspace_root)
        processed_root = _resolve_processed_root(workspace_root, sensor_mode)

        merge_root = workspace_root / f"Merged_{sensor_mode.upper()}"
        merge_root.mkdir(parents=True, exist_ok=True)

        merged_outputs = merge_processed_tiles(
            manifest=manifest,
            processed_root=processed_root,
            merged_root=merge_root,
            products=resolved_products,
        )

        if cleanup_tiles:
            cleanup_tiling_workspace(workspace_root)

        print(f"[TIME] WORKFLOW merge: {perf_counter() - total_t0:.2f}s")
        return str(merge_root if merged_outputs else processed_root)

    print(f"[FLOW] Workflow         : {workflow}")
    print(f"[FLOW] Sensor mode      : {sensor_mode.upper()}")
    print(f"[FLOW] Products         : {' '.join(resolved_products)}")
    print(f"[FLOW] Tile size (m)    : {tile_size_m}")
    print(f"[FLOW] Buffer (m)       : {buffer_m}")
    print(f"[FLOW] DEM method       : {dem_method}")
    print(f"[FLOW] DSM method       : {dsm_method}")
    print(f"[FLOW] CHM median size  : {chm_median_size}")
    print(f"[FLOW] CHM min height   : {chm_min_height}")
    print(f"[FLOW] FP-FIX           : {'ON' if apply_fp_fix else 'OFF'}")

    tile_t0 = perf_counter()
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
    print(f"[TIME] WORKFLOW tiling : {perf_counter() - tile_t0:.2f}s")

    if workflow == "tile-only":
        print(f"[TIME] WORKFLOW total  : {perf_counter() - total_t0:.2f}s")
        return str(workspace_root)

    run_t0 = perf_counter()
    processed_root = Path(
        _run_processing_with_optional_fpfix(
            in_path=str(tiles_dir),
            out_dir=str(workspace_root),
            sensor_mode=sensor_mode,
            products=products,
            grid_res=grid_res,
            dem_method=dem_method,
            dsm_method=dsm_method,
            chm_median_size=chm_median_size,
            chm_min_height=chm_min_height,
            recursive=False,
            apply_fp_fix=apply_fp_fix,
            fp_fix_dem_res=fp_fix_dem_res,
            fp_fix_nonground_to_ground_max_z=fp_fix_nonground_to_ground_max_z,
            fp_fix_ground_to_nonground_min_z=fp_fix_ground_to_nonground_min_z,
            keep_fp_fix_temp=keep_fp_fix_temp,
        )
    )
    print(f"[TIME] WORKFLOW run     : {perf_counter() - run_t0:.2f}s")

    if workflow == "tile-run":
        print(f"[TIME] WORKFLOW total  : {perf_counter() - total_t0:.2f}s")
        return str(processed_root)

    merge_t0 = perf_counter()
    merge_root = workspace_root / f"Merged_{sensor_mode.upper()}"
    merge_root.mkdir(parents=True, exist_ok=True)

    merged_outputs = merge_processed_tiles(
        manifest=manifest,
        processed_root=processed_root,
        merged_root=merge_root,
        products=resolved_products,
    )
    print(f"[TIME] WORKFLOW merge   : {perf_counter() - merge_t0:.2f}s")

    if cleanup_tiles:
        cleanup_tiling_workspace(workspace_root)

    print(f"[TIME] WORKFLOW total   : {perf_counter() - total_t0:.2f}s")
    return str(merge_root if merged_outputs else processed_root)
