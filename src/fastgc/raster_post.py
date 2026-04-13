from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    label,
    median_filter,
    binary_dilation,
    generate_binary_structure,
)

try:
    import rasterio
    from rasterio.mask import mask
    from rasterio.warp import Resampling
except Exception:
    rasterio = None


# ============================================================
# BASIC UTILS
# ============================================================

def _require_rasterio() -> None:
    if rasterio is None:
        raise RuntimeError("rasterio is required for raster post-processing.")


def _ensure_parent_dir(path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _to_float32(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.float32:
        return arr.astype(np.float32, copy=False)
    return arr


def _read_raster(fp: str | os.PathLike) -> Tuple[np.ndarray, dict, Optional[float]]:
    _require_rasterio()
    with rasterio.open(fp) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
    return arr, profile, nodata


def _write_raster(
    arr: np.ndarray,
    profile: dict,
    out_fp: str | os.PathLike,
    *,
    nodata: Optional[float] = None,
    compress: str = "lzw",
    overwrite: bool = True,
) -> None:
    _require_rasterio()

    out_fp = str(out_fp)
    if (not overwrite) and os.path.exists(out_fp):
        return

    _ensure_parent_dir(out_fp)

    out_profile = profile.copy()
    out_profile.update(
        dtype="float32",
        count=1,
        compress=compress,
    )
    if nodata is not None:
        out_profile["nodata"] = nodata

    with rasterio.open(out_fp, "w", **out_profile) as dst:
        dst.write(_to_float32(arr), 1)


def _valid_mask(arr: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata
    return valid


def _nodata_value_for_write(arr: np.ndarray, nodata: Optional[float]) -> float:
    if nodata is not None:
        return float(nodata)
    # Safe default for float rasters
    return -9999.0


def _apply_nodata(arr: np.ndarray, mask_invalid: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    out = arr.copy()
    nd = _nodata_value_for_write(out, nodata)
    out[mask_invalid] = nd
    return out


# ============================================================
# VOID / HOLE DETECTION
# ============================================================

def _connected_holes(
    valid: np.ndarray,
    *,
    connectivity: int = 8,
) -> Tuple[np.ndarray, int]:
    """
    Returns labeled void regions inside invalid cells.
    """
    if connectivity == 4:
        structure = generate_binary_structure(2, 1)
    else:
        structure = generate_binary_structure(2, 2)

    holes = ~valid
    labels, n = label(holes, structure=structure)
    return labels, int(n)


def _labels_touching_border(labels: np.ndarray) -> set[int]:
    touching = set()
    if labels.size == 0:
        return touching

    border_vals = np.concatenate(
        [
            labels[0, :],
            labels[-1, :],
            labels[:, 0],
            labels[:, -1],
        ]
    )
    for v in np.unique(border_vals):
        if v > 0:
            touching.add(int(v))
    return touching


def _small_hole_mask(
    arr: np.ndarray,
    nodata: Optional[float],
    *,
    max_hole_pixels: int = 25,
    exclude_edge_holes: bool = True,
    connectivity: int = 8,
) -> np.ndarray:
    """
    Select only small nodata/invalid regions for filling.
    """
    valid = _valid_mask(arr, nodata)
    labels, n = _connected_holes(valid, connectivity=connectivity)

    if n == 0:
        return np.zeros_like(valid, dtype=bool)

    counts = np.bincount(labels.ravel())
    touching = _labels_touching_border(labels) if exclude_edge_holes else set()

    fill_mask = np.zeros_like(valid, dtype=bool)
    for lab in range(1, len(counts)):
        if lab in touching:
            continue
        if counts[lab] <= max_hole_pixels:
            fill_mask[labels == lab] = True

    return fill_mask


# ============================================================
# LOCAL FILL HELPERS
# ============================================================

def _window_bounds(r: int, c: int, nrows: int, ncols: int, radius: int) -> Tuple[int, int, int, int]:
    r0 = max(0, r - radius)
    r1 = min(nrows, r + radius + 1)
    c0 = max(0, c - radius)
    c1 = min(ncols, c + radius + 1)
    return r0, r1, c0, c1


def _local_values(
    arr: np.ndarray,
    valid: np.ndarray,
    r: int,
    c: int,
    radius: int,
) -> np.ndarray:
    r0, r1, c0, c1 = _window_bounds(r, c, arr.shape[0], arr.shape[1], radius)
    vals = arr[r0:r1, c0:c1][valid[r0:r1, c0:c1]]
    return vals


def _local_mean_fill(
    arr: np.ndarray,
    valid: np.ndarray,
    target_mask: np.ndarray,
    *,
    radius: int = 2,
    min_neighbors: int = 3,
) -> np.ndarray:
    out = arr.copy()
    rows, cols = np.where(target_mask)

    for r, c in zip(rows, cols):
        vals = _local_values(arr, valid, r, c, radius)
        if vals.size >= min_neighbors:
            out[r, c] = float(np.mean(vals))
    return out


def _local_median_fill(
    arr: np.ndarray,
    valid: np.ndarray,
    target_mask: np.ndarray,
    *,
    radius: int = 2,
    min_neighbors: int = 3,
) -> np.ndarray:
    out = arr.copy()
    rows, cols = np.where(target_mask)

    for r, c in zip(rows, cols):
        vals = _local_values(arr, valid, r, c, radius)
        if vals.size >= min_neighbors:
            out[r, c] = float(np.median(vals))
    return out


def _idw_fill(
    arr: np.ndarray,
    valid: np.ndarray,
    target_mask: np.ndarray,
    *,
    radius: int = 3,
    power: float = 2.0,
    min_neighbors: int = 3,
    eps: float = 1e-6,
) -> np.ndarray:
    out = arr.copy()
    rows, cols = np.where(target_mask)

    for r, c in zip(rows, cols):
        r0, r1, c0, c1 = _window_bounds(r, c, arr.shape[0], arr.shape[1], radius)
        sub_valid = valid[r0:r1, c0:c1]
        if not np.any(sub_valid):
            continue

        rr, cc = np.where(sub_valid)
        rr = rr + r0
        cc = cc + c0
        vals = arr[rr, cc]

        if vals.size < min_neighbors:
            continue

        d = np.sqrt((rr - r) ** 2 + (cc - c) ** 2)
        d = np.maximum(d, eps)
        w = 1.0 / (d ** power)
        out[r, c] = float(np.sum(w * vals) / np.sum(w))

    return out


def _nearest_fill_selected(
    arr: np.ndarray,
    valid: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    """
    Fill only selected cells by nearest valid cell, without changing other invalid cells.
    """
    if np.all(valid):
        return arr.copy()

    _, inds = distance_transform_edt(~valid, return_indices=True)
    out = arr.copy()
    out[target_mask] = arr[inds[0][target_mask], inds[1][target_mask]]
    return out


# ============================================================
# MAIN VOID-FILL FUNCTION
# ============================================================

def fill_small_voids(
    arr: np.ndarray,
    nodata: Optional[float],
    *,
    method: str = "hybrid",
    max_hole_pixels: int = 25,
    exclude_edge_holes: bool = True,
    connectivity: int = 8,
    local_radius: int = 2,
    idw_radius: int = 3,
    min_neighbors: int = 3,
    preserve_negative_values: bool = True,
) -> np.ndarray:
    """
    Fill only small internal voids while preserving large or edge-touching gaps.

    Methods:
      - nearest
      - localmean
      - median
      - idw
      - hybrid

    hybrid:
      local mean for very small holes where enough neighbors exist,
      otherwise nearest fallback.
    """
    arr = _to_float32(arr)
    valid = _valid_mask(arr, nodata)

    if np.all(valid):
        return arr.copy()

    target_mask = _small_hole_mask(
        arr,
        nodata,
        max_hole_pixels=max_hole_pixels,
        exclude_edge_holes=exclude_edge_holes,
        connectivity=connectivity,
    )

    if not np.any(target_mask):
        return arr.copy()

    method = method.lower().strip()

    if method == "nearest":
        filled = _nearest_fill_selected(arr, valid, target_mask)

    elif method == "localmean":
        filled = _local_mean_fill(
            arr,
            valid,
            target_mask,
            radius=local_radius,
            min_neighbors=min_neighbors,
        )

    elif method == "median":
        filled = _local_median_fill(
            arr,
            valid,
            target_mask,
            radius=local_radius,
            min_neighbors=min_neighbors,
        )

    elif method == "idw":
        filled = _idw_fill(
            arr,
            valid,
            target_mask,
            radius=idw_radius,
            min_neighbors=min_neighbors,
        )

    elif method == "hybrid":
        # First try local mean, then use nearest as fallback for still-invalid selected cells.
        temp = _local_mean_fill(
            arr,
            valid,
            target_mask,
            radius=local_radius,
            min_neighbors=min_neighbors,
        )
        temp_valid = _valid_mask(temp, nodata)
        still_bad = target_mask & (~temp_valid)
        if np.any(still_bad):
            temp2 = _nearest_fill_selected(arr, valid, still_bad)
            temp[still_bad] = temp2[still_bad]
        filled = temp

    else:
        raise ValueError(
            f"Unknown fill method: {method}. "
            "Use one of nearest, localmean, median, idw, hybrid."
        )

    if not preserve_negative_values:
        filled = np.maximum(filled, 0.0)

    # Preserve all non-target invalid cells as nodata
    still_invalid = (~valid) & (~target_mask)
    filled = _apply_nodata(filled, still_invalid, nodata)

    return filled


def run_fill_voids(
    in_fp: str | os.PathLike,
    out_fp: str | os.PathLike,
    *,
    method: str = "hybrid",
    max_hole_pixels: int = 25,
    exclude_edge_holes: bool = True,
    connectivity: int = 8,
    local_radius: int = 2,
    idw_radius: int = 3,
    min_neighbors: int = 3,
    preserve_negative_values: bool = True,
    overwrite: bool = True,
) -> None:
    arr, profile, nodata = _read_raster(in_fp)
    filled = fill_small_voids(
        arr,
        nodata,
        method=method,
        max_hole_pixels=max_hole_pixels,
        exclude_edge_holes=exclude_edge_holes,
        connectivity=connectivity,
        local_radius=local_radius,
        idw_radius=idw_radius,
        min_neighbors=min_neighbors,
        preserve_negative_values=preserve_negative_values,
    )
    _write_raster(
        filled,
        profile,
        out_fp,
        nodata=_nodata_value_for_write(filled, nodata),
        overwrite=overwrite,
    )


# ============================================================
# RESAMPLING
# ============================================================

def resample_raster(
    in_fp: str | os.PathLike,
    out_fp: str | os.PathLike,
    new_res: float,
    *,
    method: str = "average",
    overwrite: bool = True,
) -> None:
    _require_rasterio()

    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }

    if method not in resampling_map:
        raise ValueError(f"Unsupported resampling method: {method}")

    out_fp = str(out_fp)
    if (not overwrite) and os.path.exists(out_fp):
        return

    _ensure_parent_dir(out_fp)

    with rasterio.open(in_fp) as src:
        scale = src.res[0] / float(new_res)
        new_height = int(round(src.height * scale))
        new_width = int(round(src.width * scale))

        data = src.read(
            out_shape=(1, new_height, new_width),
            resampling=resampling_map[method],
        )

        transform = src.transform * src.transform.scale(
            src.width / new_width,
            src.height / new_height,
        )

        profile = src.profile.copy()
        profile.update(
            height=new_height,
            width=new_width,
            transform=transform,
            compress="lzw",
        )

    with rasterio.open(out_fp, "w", **profile) as dst:
        dst.write(data)


def run_resample(
    in_fp: str | os.PathLike,
    out_dir: str | os.PathLike,
    resolutions: Sequence[float],
    *,
    method: str = "average",
    overwrite: bool = True,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for r in resolutions:
        out_fp = out_dir / f"{Path(in_fp).stem}_{str(r).replace('.', 'p')}m.tif"
        resample_raster(in_fp, out_fp, r, method=method, overwrite=overwrite)


# ============================================================
# CLIPPING
# ============================================================

def run_clip(
    in_fp: str | os.PathLike,
    shp_fp: str | os.PathLike,
    out_fp: str | os.PathLike,
    *,
    crop: bool = True,
    overwrite: bool = True,
) -> None:
    _require_rasterio()

    out_fp = str(out_fp)
    if (not overwrite) and os.path.exists(out_fp):
        return

    import fiona

    with fiona.open(shp_fp, "r") as shp:
        geoms = [feature["geometry"] for feature in shp]

    with rasterio.open(in_fp) as src:
        out_img, out_transform = mask(src, geoms, crop=crop)

        out_meta = src.meta.copy()
        out_meta.update(
            {
                "height": out_img.shape[1],
                "width": out_img.shape[2],
                "transform": out_transform,
                "compress": "lzw",
            }
        )

    _ensure_parent_dir(out_fp)
    with rasterio.open(out_fp, "w", **out_meta) as dest:
        dest.write(out_img)


# ============================================================
# BATCH PROCESSING
# ============================================================

def process_folder(
    input_path: str | os.PathLike,
    operation: str,
    out_dir: str | os.PathLike,
    *,
    method: str = "hybrid",
    resolutions: Optional[Sequence[float]] = None,
    shp: Optional[str | os.PathLike] = None,
    max_hole_pixels: int = 25,
    exclude_edge_holes: bool = True,
    connectivity: int = 8,
    local_radius: int = 2,
    idw_radius: int = 3,
    min_neighbors: int = 3,
    preserve_negative_values: bool = True,
    overwrite: bool = True,
) -> None:
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.rglob("*.tif"))

    for fp in files:
        name = fp.stem

        if operation == "fill":
            out_fp = out_dir / f"{name}_filled.tif"
            run_fill_voids(
                fp,
                out_fp,
                method=method,
                max_hole_pixels=max_hole_pixels,
                exclude_edge_holes=exclude_edge_holes,
                connectivity=connectivity,
                local_radius=local_radius,
                idw_radius=idw_radius,
                min_neighbors=min_neighbors,
                preserve_negative_values=preserve_negative_values,
                overwrite=overwrite,
            )

        elif operation == "resample":
            if resolutions is None or len(resolutions) == 0:
                raise ValueError("resolutions must be provided for operation='resample'")
            run_resample(
                fp,
                out_dir,
                resolutions,
                method=method,
                overwrite=overwrite,
            )

        elif operation == "clip":
            if shp is None:
                raise ValueError("shp must be provided for operation='clip'")
            out_fp = out_dir / f"{name}_clipped.tif"
            run_clip(
                fp,
                shp,
                out_fp,
                crop=True,
                overwrite=overwrite,
            )

        else:
            raise ValueError(f"Unknown operation: {operation}")


# ============================================================
# OPTIONAL SIMPLE CLI-LIKE ENTRY
# ============================================================

if __name__ == "__main__":
    print(
        "raster_post.py loaded. Use process_folder(...) or the individual "
        "functions run_fill_voids(), run_resample(), and run_clip()."
    )