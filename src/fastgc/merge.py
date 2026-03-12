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
) -> Path:
    processed_root = Path(processed_root)
    if product in POINT_PRODUCTS:
        return processed_root / product / tile_name
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
        return None

    writer = None
    kept_tiles = 0
    kept_points = 0
    first_header = None
    t0 = perf_counter()

    try:
        pbar = tqdm(
            available_tiles,
            desc=f"MERGE {product}",
            unit="tile",
            dynamic_ncols=True,
        )
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


def _snap_window_to_grid(src, core_bounds: list[float]) -> Window:
    """
    Convert geographic core bounds to a raster window and snap it conservatively
    to pixel indices so adjacent tiles meet without leaving seams.

    Strategy:
    - compute floating window from bounds
    - floor row/col offsets
    - ceil width/height end positions
    - clip to source raster bounds
    """
    if from_bounds is None or Window is None:
        raise RuntimeError("rasterio is required for raster merge workflows.")

    win_f = from_bounds(*core_bounds, transform=src.transform)

    col_off = math.floor(win_f.col_off)
    row_off = math.floor(win_f.row_off)
    col_end = math.ceil(win_f.col_off + win_f.width)
    row_end = math.ceil(win_f.row_off + win_f.height)

    col_off = max(0, col_off)
    row_off = max(0, row_off)
    col_end = min(src.width, col_end)
    row_end = min(src.height, row_end)

    width = col_end - col_off
    height = row_end - row_off

    return Window(col_off=col_off, row_off=row_off, width=width, height=height)


def _crop_raster_to_core(tile_src: Path, tile_dst: Path, core_bounds: list[float]) -> bool:
    if rasterio is None or from_bounds is None or Window is None:
        raise RuntimeError("rasterio is required for raster merge workflows.")

    with rasterio.open(tile_src) as src:
        win = _snap_window_to_grid(src, core_bounds)

        if win.width <= 0 or win.height <= 0:
            return False

        data = src.read(window=win)
        if data.size == 0:
            return False

        transform = src.window_transform(win)
        profile = src.profile.copy()
        profile.update(
            height=int(data.shape[1]),
            width=int(data.shape[2]),
            transform=transform,
            count=int(data.shape[0]),
        )

        # Preserve nodata explicitly if present
        if getattr(src, "nodata", None) is not None:
            profile["nodata"] = src.nodata

        tile_dst.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(tile_dst, "w", **profile) as dst:
            dst.write(data)

    return True


def merge_raster_product(
    manifest: dict[str, Any],
    processed_root: str | Path,
    merged_root: str | Path,
    product: str,
) -> str | None:
    if rasterio is None or rio_merge is None:
        raise RuntimeError("rasterio is required for raster merge workflows.")

    processed_root = Path(processed_root)
    merged_root = Path(merged_root)
    merged_root.mkdir(parents=True, exist_ok=True)

    temp_trim_dir = merged_root / "_trimmed_rasters" / product
    temp_trim_dir.mkdir(parents=True, exist_ok=True)

    trimmed: list[Path] = []
    t0 = perf_counter()

    pbar_trim = tqdm(
        manifest["tiles"],
        desc=f"TRIM {product}",
        unit="tile",
        dynamic_ncols=True,
    )
    for idx, tile in enumerate(pbar_trim, start=1):
        tile_src = _expected_tile_output_path(processed_root, product, tile["tile_name"])
        if not tile_src.exists():
            elapsed = perf_counter() - t0
            pbar_trim.set_postfix_str(
                f"{idx}/{len(manifest['tiles'])} | {elapsed / max(idx, 1):.2f}s/tile"
            )
            continue

        tile_dst = temp_trim_dir / tile_src.name
        if _crop_raster_to_core(tile_src, tile_dst, tile["core_bounds"]):
            trimmed.append(tile_dst)

        elapsed = perf_counter() - t0
        pbar_trim.set_postfix_str(
            f"{idx}/{len(manifest['tiles'])} | {elapsed / max(idx, 1):.2f}s/tile"
        )
    pbar_trim.close()

    if not trimmed:
        return None

    datasets = [rasterio.open(fp) for fp in trimmed]
    try:
        # Let rasterio merge them after trimming. "first" preserves the first valid
        # pixel encountered and avoids unintended blending.
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

        out_fp = merged_root / f"{_dataset_label(manifest)}_{product}.tif"
        with rasterio.open(out_fp, "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        for ds in datasets:
            ds.close()

    elapsed = perf_counter() - t0
    print(
        f"[TIME] MERGE {product}: {elapsed:.2f}s | "
        f"{len(trimmed)} rasters | {elapsed / max(len(trimmed), 1):.2f}s/tile"
    )
    return str(out_fp)


def merge_processed_tiles(
    manifest: dict[str, Any],
    processed_root: str | Path,
    merged_root: str | Path,
    products: list[str],
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    merged_root = Path(merged_root)

    for product in products:
        if product in POINT_PRODUCTS:
            out = merge_point_product(manifest, processed_root, merged_root, product)
        elif product in RASTER_PRODUCTS:
            out = merge_raster_product(manifest, processed_root, merged_root, product)
        else:
            continue

        if out:
            outputs[product] = out

    return outputs


def cleanup_tiling_workspace(workspace_root: str | Path):
    shutil.rmtree(workspace_root, ignore_errors=True)


__all__ = [
    "cleanup_tiling_workspace",
    "load_manifest",
    "merge_processed_tiles",
]