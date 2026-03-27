from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any

import laspy
import numpy as np
from tqdm import tqdm

from .monster import log_info, stage_banner

try:
    import rasterio
    from rasterio.merge import merge as rio_merge
    from rasterio.windows import Window, from_bounds
except Exception:
    rasterio = None
    rio_merge = None
    from_bounds = None
    Window = None


POINT_PRODUCTS = {"FAST_GC", "FAST_NORMALIZED"}
RASTER_PRODUCTS = {"FAST_DEM", "FAST_DSM", "FAST_CHM"}
TERRAIN_PRODUCT = "FAST_TERRAIN"


def load_manifest(workspace_root: str | Path) -> dict[str, Any]:
    workspace_root = Path(workspace_root)
    manifest_fp = workspace_root / "tile_manifest.json"
    if not manifest_fp.exists():
        raise FileNotFoundError(f"Tile manifest not found: {manifest_fp}")
    with manifest_fp.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dataset_label(manifest: dict[str, Any]) -> str:
    return str(manifest.get("dataset_label", "dataset"))


def _expected_tile_output_path(
    processed_root: str | Path,
    product: str,
    tile_name: str,
    *,
    chm_method: str | None = None,
    terrain_subproduct: str | None = None,
) -> Path:
    processed_root = Path(processed_root)

    if product in POINT_PRODUCTS:
        return processed_root / product / tile_name

    if product == "FAST_CHM":
        method = str(chm_method or "p2r").strip().lower()
        return processed_root / "FAST_CHM" / method / f"{Path(tile_name).stem}.tif"

    if product == TERRAIN_PRODUCT:
        if not terrain_subproduct:
            raise ValueError("terrain_subproduct is required for FAST_TERRAIN tile paths")
        return processed_root / TERRAIN_PRODUCT / terrain_subproduct / f"{Path(tile_name).stem}.tif"

    if product in RASTER_PRODUCTS:
        return processed_root / product / f"{Path(tile_name).stem}.tif"

    return processed_root / product / tile_name


def _core_mask(x: np.ndarray, y: np.ndarray, core_bounds: list[float]) -> np.ndarray:
    xmin, ymin, xmax, ymax = [float(v) for v in core_bounds]
    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)


def _ensure_same_point_layout(first_header: laspy.LasHeader, next_header: laspy.LasHeader):
    if first_header.point_format.id != next_header.point_format.id:
        raise RuntimeError(
            f"Cannot merge point products with different point formats: "
            f"{first_header.point_format.id} vs {next_header.point_format.id}"
        )
    if tuple(first_header.scales) != tuple(next_header.scales):
        raise RuntimeError(
            f"Cannot merge point products with different scales: "
            f"{tuple(first_header.scales)} vs {tuple(next_header.scales)}"
        )
    if tuple(first_header.offsets) != tuple(next_header.offsets):
        raise RuntimeError(
            f"Cannot merge point products with different offsets: "
            f"{tuple(first_header.offsets)} vs {tuple(next_header.offsets)}"
        )


def merge_point_product(
    manifest: dict[str, Any],
    processed_root: str | Path,
    merged_root: str | Path,
    product: str,
) -> str | None:
    processed_root = Path(processed_root)
    merged_root = Path(merged_root)
    merged_root.mkdir(parents=True, exist_ok=True)
    out_fp = merged_root / f"{_dataset_label(manifest)}_{product}.las"

    available_tiles = [
        tile for tile in manifest["tiles"]
        if _expected_tile_output_path(processed_root, product, tile["tile_name"]).exists()
    ]
    if not available_tiles:
        log_info(f"No tiles available for merge: {product}")
        return None

    stage_banner(f"MERGE {product}", source=str(processed_root / product), total=len(available_tiles), unit="tile")

    writer = None
    kept_tiles = 0
    kept_points = 0
    first_header = None
    t0 = perf_counter()

    try:
        pbar = tqdm(available_tiles, desc=f"MERGE {product}", unit="tile", dynamic_ncols=True)
        for idx, tile in enumerate(pbar, start=1):
            tile_fp = _expected_tile_output_path(processed_root, product, tile["tile_name"])
            las = laspy.read(tile_fp)

            if first_header is None:
                first_header = las.header
                writer = laspy.open(out_fp, mode="w", header=first_header)
            else:
                _ensure_same_point_layout(first_header, las.header)

            x = np.asarray(las.x, dtype=np.float64)
            y = np.asarray(las.y, dtype=np.float64)
            mask = _core_mask(x, y, tile["core_bounds"])
            n_keep = int(np.count_nonzero(mask))
            if n_keep == 0:
                elapsed = perf_counter() - t0
                pbar.set_postfix_str(
                    f"{idx}/{len(available_tiles)} | {elapsed / max(idx, 1):.2f}s/tile | kept_pts={kept_points}"
                )
                continue

            writer.write_points(las.points[mask].copy())
            kept_tiles += 1
            kept_points += n_keep

            elapsed = perf_counter() - t0
            pbar.set_postfix_str(
                f"{idx}/{len(available_tiles)} | {elapsed / max(idx, 1):.2f}s/tile | kept_pts={kept_points}"
            )

        pbar.close()
    finally:
        if writer is not None:
            writer.close()

    if kept_tiles > 0:
        elapsed = perf_counter() - t0
        print(
            f"[TIME] MERGE {product}: {elapsed:.2f}s | "
            f"{kept_tiles} tiles kept | {elapsed / max(len(available_tiles), 1):.2f}s/tile | "
            f"points={kept_points}"
        )
        return str(out_fp)

    if out_fp.exists():
        out_fp.unlink()

    return None


def _normalize_bounds(bounds: list[float] | tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = [float(v) for v in bounds]
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin
    return xmin, ymin, xmax, ymax


def _intersect_bounds(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    axmin, aymin, axmax, aymax = a
    bxmin, bymin, bxmax, bymax = b

    xmin = max(axmin, bxmin)
    ymin = max(aymin, bymin)
    xmax = min(axmax, bxmax)
    ymax = min(aymax, bymax)

    if xmax <= xmin or ymax <= ymin:
        return None

    return xmin, ymin, xmax, ymax


def _src_bounds(src) -> tuple[float, float, float, float]:
    b = src.bounds
    return float(b.left), float(b.bottom), float(b.right), float(b.top)


def _safe_window_from_bounds(src, bounds: tuple[float, float, float, float]) -> Window | None:
    """
    Build a window safely:
    - intersect requested bounds with src bounds
    - snap to grid
    - clamp to raster extent
    - return None for empty/degenerate windows
    """
    if from_bounds is None or Window is None:
        raise RuntimeError("rasterio is required for raster merge workflows.")

    overlap = _intersect_bounds(_normalize_bounds(bounds), _src_bounds(src))
    if overlap is None:
        return None

    try:
        win_f = from_bounds(*overlap, transform=src.transform)
    except Exception:
        return None

    col_off = math.floor(float(win_f.col_off))
    row_off = math.floor(float(win_f.row_off))
    col_end = math.ceil(float(win_f.col_off + win_f.width))
    row_end = math.ceil(float(win_f.row_off + win_f.height))

    col_off = max(0, col_off)
    row_off = max(0, row_off)
    col_end = min(int(src.width), col_end)
    row_end = min(int(src.height), row_end)

    width = col_end - col_off
    height = row_end - row_off

    if width <= 0 or height <= 0:
        return None

    return Window(
        col_off=int(col_off),
        row_off=int(row_off),
        width=int(width),
        height=int(height),
    )


def _snap_window_to_grid(src, core_bounds: list[float]) -> Window | None:
    """
    Backward-compatible wrapper name kept intact.
    Now returns None for empty/non-overlapping windows instead of raising.
    """
    return _safe_window_from_bounds(src, _normalize_bounds(core_bounds))


def _crop_raster_to_core(tile_src: Path, tile_dst: Path, core_bounds: list[float]) -> bool:
    if rasterio is None or from_bounds is None or Window is None:
        raise RuntimeError("rasterio is required for raster merge workflows.")

    with rasterio.open(tile_src) as src:
        win = _snap_window_to_grid(src, core_bounds)

        if win is None:
            return False
        if win.width <= 0 or win.height <= 0:
            return False

        data = src.read(window=win)
        if data.size == 0:
            return False
        if data.shape[1] <= 0 or data.shape[2] <= 0:
            return False

        transform = src.window_transform(win)
        profile = src.profile.copy()
        profile.update(
            height=int(data.shape[1]),
            width=int(data.shape[2]),
            transform=transform,
            count=int(data.shape[0]),
        )

        if getattr(src, "nodata", None) is not None:
            profile["nodata"] = src.nodata

        tile_dst.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(tile_dst, "w", **profile) as dst:
            dst.write(data)

    return True


def _discover_terrain_subproducts(processed_root: str | Path) -> list[str]:
    processed_root = Path(processed_root)
    terrain_root = processed_root / TERRAIN_PRODUCT
    if not terrain_root.exists():
        return []

    out: list[str] = []
    for child in sorted(terrain_root.iterdir()):
        if child.is_dir() and any(child.glob("*.tif")):
            out.append(child.name)
    return out


def merge_raster_product(
    manifest: dict[str, Any],
    processed_root: str | Path,
    merged_root: str | Path,
    product: str,
    *,
    chm_method: str | None = None,
    terrain_subproduct: str | None = None,
) -> str | None:
    if rasterio is None or rio_merge is None:
        raise RuntimeError("rasterio is required for raster merge workflows.")

    processed_root = Path(processed_root)
    merged_root = Path(merged_root)
    merged_root.mkdir(parents=True, exist_ok=True)

    if product == TERRAIN_PRODUCT:
        if not terrain_subproduct:
            raise ValueError("terrain_subproduct is required when merging FAST_TERRAIN")
        trim_label = f"{product}_{terrain_subproduct}"
        out_dir = merged_root / TERRAIN_PRODUCT / terrain_subproduct
        out_dir.mkdir(parents=True, exist_ok=True)
    elif product == "FAST_CHM":
        trim_label = f"{product}_{str(chm_method or 'p2r').lower()}"
        out_dir = merged_root
    else:
        trim_label = product
        out_dir = merged_root

    temp_trim_dir = merged_root / "_trimmed_rasters" / trim_label
    temp_trim_dir.mkdir(parents=True, exist_ok=True)

    trimmed: list[Path] = []
    t0 = perf_counter()

    stage_banner(f"TRIM {trim_label}", source=str(processed_root), total=len(manifest["tiles"]), unit="tile")
    pbar_trim = tqdm(manifest["tiles"], desc=f"TRIM {trim_label}", unit="tile", dynamic_ncols=True)

    skipped_empty = 0
    missing_src = 0

    for idx, tile in enumerate(pbar_trim, start=1):
        tile_src = _expected_tile_output_path(
            processed_root,
            product,
            tile["tile_name"],
            chm_method=chm_method,
            terrain_subproduct=terrain_subproduct,
        )
        if not tile_src.exists():
            missing_src += 1
            elapsed = perf_counter() - t0
            pbar_trim.set_postfix_str(
                f"{idx}/{len(manifest['tiles'])} | {elapsed / max(idx, 1):.2f}s/tile | kept={len(trimmed)} | skip={skipped_empty}"
            )
            continue

        tile_dst = temp_trim_dir / tile_src.name
        kept = _crop_raster_to_core(tile_src, tile_dst, tile["core_bounds"])
        if kept:
            trimmed.append(tile_dst)
        else:
            skipped_empty += 1

        elapsed = perf_counter() - t0
        pbar_trim.set_postfix_str(
            f"{idx}/{len(manifest['tiles'])} | {elapsed / max(idx, 1):.2f}s/tile | kept={len(trimmed)} | skip={skipped_empty}"
        )

    pbar_trim.close()

    if missing_src:
        log_info(f"{trim_label}: missing source rasters skipped = {missing_src}")
    if skipped_empty:
        log_info(f"{trim_label}: empty/non-overlapping core trims skipped = {skipped_empty}")

    if not trimmed:
        log_info(f"No trimmed rasters available for merge: {trim_label}")
        return None

    stage_banner(f"MERGE {trim_label}", source=str(temp_trim_dir), total=len(trimmed), unit="tile")

    datasets = [rasterio.open(fp) for fp in trimmed]
    try:
        mosaic, transform = rio_merge(datasets, method="first")

        profile = datasets[0].profile.copy()
        profile.update(
            height=int(mosaic.shape[1]),
            width=int(mosaic.shape[2]),
            transform=transform,
            count=int(mosaic.shape[0]),
        )

        if getattr(datasets[0], "nodata", None) is not None:
            profile["nodata"] = datasets[0].nodata

        out_name = f"{_dataset_label(manifest)}_{product}"
        if product == "FAST_CHM":
            out_name += f"_{str(chm_method or 'p2r').lower()}"
        elif product == TERRAIN_PRODUCT:
            out_name += f"_{terrain_subproduct}"

        out_fp = out_dir / f"{out_name}.tif"

        with rasterio.open(out_fp, "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        for ds in datasets:
            ds.close()

    elapsed = perf_counter() - t0
    print(
        f"[TIME] MERGE {trim_label}: {elapsed:.2f}s | "
        f"{len(trimmed)} rasters | {elapsed / max(len(trimmed), 1):.2f}s/tile"
    )
    return str(out_fp)


def merge_processed_tiles(
    manifest: dict[str, Any],
    processed_root: str | Path,
    merged_root: str | Path,
    products: list[str],
    *,
    chm_method: str | None = None,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    merged_root = Path(merged_root)

    for product in products:
        if product in POINT_PRODUCTS:
            out = merge_point_product(manifest, processed_root, merged_root, product)
            if out:
                outputs[product] = out

        elif product in RASTER_PRODUCTS:
            out = merge_raster_product(
                manifest,
                processed_root,
                merged_root,
                product,
                chm_method=chm_method,
            )
            if out:
                key = product if product != "FAST_CHM" else f"{product}_{str(chm_method or 'p2r').lower()}"
                outputs[key] = out

        elif product == TERRAIN_PRODUCT:
            terrain_subproducts = _discover_terrain_subproducts(processed_root)
            if not terrain_subproducts:
                log_info(f"No terrain subproducts available for merge: {processed_root}/{TERRAIN_PRODUCT}")
                continue

            for sub in terrain_subproducts:
                out = merge_raster_product(
                    manifest,
                    processed_root,
                    merged_root,
                    TERRAIN_PRODUCT,
                    terrain_subproduct=sub,
                )
                if out:
                    outputs[f"{TERRAIN_PRODUCT}_{sub}"] = out

        else:
            continue

    trim_root = merged_root / "_trimmed_rasters"
    if trim_root.exists():
        shutil.rmtree(trim_root, ignore_errors=True)

    return outputs


def cleanup_tiling_workspace(workspace_root: str | Path):
    shutil.rmtree(workspace_root, ignore_errors=True)


__all__ = [
    "cleanup_tiling_workspace",
    "load_manifest",
    "merge_processed_tiles",
]