from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

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
from .merge import cleanup_tiling_workspace, merge_processed_tiles
from .post_fp_fix import apply_fp_fix_to_output_root
from .preprocess import tile_las_dataset

DEFAULT_WORKFLOW = "run"
WORKFLOW_CHOICES = ["run", "tile-only", "tile-run", "tile-run-merge", "merge", "derive-only"]


def _resolve_products(products: list[str] | None) -> list[str]:
    requested = list(products or [PRODUCT_GC])

    if "all" in requested:
        return [PRODUCT_GC, PRODUCT_DEM, PRODUCT_NORMALIZED, PRODUCT_DSM, PRODUCT_CHM]

    out: list[str] = []
    seen: set[str] = set()

    for p in requested:
        if p not in seen:
            out.append(p)
            seen.add(p)

    # Only real dependencies
    if PRODUCT_NORMALIZED in seen and PRODUCT_DEM not in seen:
        out.insert(out.index(PRODUCT_NORMALIZED), PRODUCT_DEM)
        seen.add(PRODUCT_DEM)

    if PRODUCT_CHM in seen:
        if PRODUCT_DEM not in seen:
            out.insert(out.index(PRODUCT_CHM), PRODUCT_DEM)
            seen.add(PRODUCT_DEM)
        if PRODUCT_DSM not in seen:
            out.insert(out.index(PRODUCT_CHM), PRODUCT_DSM)
            seen.add(PRODUCT_DSM)

    return out


def _needs_classified_source(products: list[str]) -> bool:
    needed = set(products)
    return bool({PRODUCT_GC, PRODUCT_DEM, PRODUCT_NORMALIZED, PRODUCT_CHM} & needed)


def _needs_raw_dsm(products: list[str]) -> bool:
    return PRODUCT_DSM in set(products)


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

    need_classified = _needs_classified_source(resolved_products)
    need_dsm = _needs_raw_dsm(resolved_products)

    # DSM-only path
    if need_dsm and not need_classified:
        return derive_products_from_raw_root(
            raw_root=in_path,
            out_root=out_dir if out_dir is not None else in_path,
            sensor_mode=sensor_mode,
            products=[PRODUCT_DSM],
            grid_res=grid_res,
            dsm_method=dsm_method,
            recursive=recursive,
        )

    # No FP-FIX path
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

    # Stage 2: FP-FIX
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

    # Stage 3a: DEM / NORMALIZED from corrected FAST_GC
    downstream_from_gc = [
        p for p in resolved_products if p in {PRODUCT_DEM, PRODUCT_NORMALIZED}
    ]
    if downstream_from_gc:
        derive_products_from_classified_root(
            classified_root=Path(out_root) / PRODUCT_GC,
            out_root=out_root,
            products=downstream_from_gc,
            grid_res=grid_res,
            dem_method=dem_method,
            dsm_method=dsm_method,
            chm_median_size=chm_median_size,
            chm_min_height=chm_min_height,
        )

    # Stage 3b: DSM from raw tiles
    if need_dsm:
        derive_products_from_raw_root(
            raw_root=in_path,
            out_root=out_root,
            sensor_mode=sensor_mode,
            products=[PRODUCT_DSM],
            grid_res=grid_res,
            dsm_method=dsm_method,
            recursive=recursive,
        )

    # Stage 3c: CHM last
    if PRODUCT_CHM in resolved_products:
        derive_products_from_classified_root(
            classified_root=Path(out_root) / PRODUCT_GC,
            out_root=out_root,
            products=[PRODUCT_CHM],
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
        print(f"[FLOW] Workflow         : run")
        print(f"[FLOW] Sensor mode      : {sensor_mode.upper()}")
        print(f"[FLOW] Products         : {' '.join(resolved_products)}")
        print(f"[FLOW] DEM method       : {dem_method}")
        print(f"[FLOW] DSM method       : {dsm_method}")
        print(f"[FLOW] CHM median size  : {chm_median_size}")
        print(f"[FLOW] CHM min height   : {chm_min_height}")
        print(f"[FLOW] FP-FIX           : {'ON' if apply_fp_fix else 'OFF'}")

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
        print(f"[TIME] WORKFLOW run     : {perf_counter() - total_t0:.2f}s")
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
            tiles_dir = workspace_root / "tiles"
            if tiles_dir.exists():
                cleanup_tiling_workspace(tiles_dir)

        print(f"[TIME] WORKFLOW merge   : {perf_counter() - total_t0:.2f}s")
        return str(merge_root if merged_outputs else processed_root)

    if workflow == "derive-only":
        workspace_root = Path(in_path)

        if (workspace_root / "tile_manifest.json").exists():
            processed_root = _resolve_processed_root(workspace_root, sensor_mode)
            classified_root = processed_root / PRODUCT_GC
            raw_tiles_root = workspace_root / "tiles"
            out_root_final = processed_root
        elif workspace_root.name == PRODUCT_GC and workspace_root.exists():
            classified_root = workspace_root
            processed_root = workspace_root.parent
            raw_tiles_root = processed_root.parent / "tiles"
            out_root_final = processed_root
        elif workspace_root.exists():
            processed_root = workspace_root
            classified_root = processed_root / PRODUCT_GC
            raw_tiles_root = processed_root.parent / "tiles"
            out_root_final = processed_root
        else:
            raise FileNotFoundError(f"Input path not found: {workspace_root}")

        requested = _resolve_products(products)

        print(f"[FLOW] Workflow         : derive-only")
        print(f"[FLOW] Sensor mode      : {sensor_mode.upper()}")
        print(f"[FLOW] Products         : {' '.join(requested)}")
        print(f"[FLOW] DEM method       : {dem_method}")
        print(f"[FLOW] DSM method       : {dsm_method}")
        print(f"[FLOW] CHM median size  : {chm_median_size}")
        print(f"[FLOW] CHM min height   : {chm_min_height}")
        print(f"[FLOW] FP-FIX           : BYPASSED (existing products reused)")

        from_gc = [p for p in requested if p in {PRODUCT_DEM, PRODUCT_NORMALIZED}]
        if from_gc:
            if not classified_root.exists():
                raise FileNotFoundError(f"Classified root not found: {classified_root}")
            derive_products_from_classified_root(
                classified_root=classified_root,
                out_root=out_root_final,
                products=from_gc,
                grid_res=grid_res,
                dem_method=dem_method,
                dsm_method=dsm_method,
                chm_median_size=chm_median_size,
                chm_min_height=chm_min_height,
            )

        if PRODUCT_DSM in requested:
            if not raw_tiles_root.exists():
                raise FileNotFoundError(f"Raw tiles root not found for DSM derivation: {raw_tiles_root}")
            derive_products_from_raw_root(
                raw_root=raw_tiles_root,
                out_root=out_root_final,
                sensor_mode=sensor_mode,
                products=[PRODUCT_DSM],
                grid_res=grid_res,
                dsm_method=dsm_method,
                recursive=True,
            )

        if PRODUCT_CHM in requested:
            if not classified_root.exists():
                raise FileNotFoundError(f"Classified root not found: {classified_root}")
            derive_products_from_classified_root(
                classified_root=classified_root,
                out_root=out_root_final,
                products=[PRODUCT_CHM],
                grid_res=grid_res,
                dem_method=dem_method,
                dsm_method=dsm_method,
                chm_median_size=chm_median_size,
                chm_min_height=chm_min_height,
            )

        print(f"[TIME] WORKFLOW derive-only: {perf_counter() - total_t0:.2f}s")
        return str(out_root_final)

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
        print(f"[TIME] WORKFLOW total   : {perf_counter() - total_t0:.2f}s")
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
        print(f"[TIME] WORKFLOW total   : {perf_counter() - total_t0:.2f}s")
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
        tiles_dir = workspace_root / "tiles"
        if tiles_dir.exists():
            cleanup_tiling_workspace(tiles_dir)

    print(f"[TIME] WORKFLOW total   : {perf_counter() - total_t0:.2f}s")
    return str(merge_root if merged_outputs else processed_root)