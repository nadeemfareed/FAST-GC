from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import fiona
import numpy as np
import rasterio
from rasterio import features
from scipy import ndimage as ndi
from shapely.geometry import shape, mapping
from shapely.geometry.base import BaseGeometry


def read_surface_raster(
    path: str | Path,
) -> tuple[np.ndarray, dict, rasterio.Affine, rasterio.crs.CRS | None]:
    path = Path(path)
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        nodata = src.nodata
        crs = src.crs

    if nodata is not None:
        arr[arr == nodata] = np.nan

    return arr, profile, transform, crs


def write_label_raster(path: str | Path, labels: np.ndarray, profile: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out_profile = profile.copy()
    out_profile.update(dtype="int32", count=1, nodata=0, compress="deflate")

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(labels.astype(np.int32), 1)

    return str(path)


def write_float_raster(path: str | Path, arr: np.ndarray, profile: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=np.nan, compress="deflate")

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(arr.astype(np.float32), 1)

    return str(path)


def write_json(path: str | Path, payload: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return str(path)


def map_point(transform: rasterio.Affine, row: int, col: int) -> tuple[float, float]:
    x, y = rasterio.transform.xy(transform, row, col, offset="center")
    return float(x), float(y)


def pixel_size(transform: rasterio.Affine) -> tuple[float, float]:
    return abs(float(transform.a)), abs(float(transform.e))


def cell_area(transform: rasterio.Affine) -> float:
    dx, dy = pixel_size(transform)
    return dx * dy


def fill_nan_by_nearest(arr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    arr = arr.astype(np.float32, copy=True)
    valid = np.isfinite(arr)
    if mask is not None:
        valid = valid & mask.astype(bool)

    if not np.any(valid):
        out = np.zeros_like(arr, dtype=np.float32)
        if mask is not None:
            out[~mask.astype(bool)] = np.nan
        return out

    seed = np.where(valid, arr, np.nan).astype(np.float32, copy=False)
    inds = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = seed[tuple(inds)].astype(np.float32)

    if mask is not None:
        mask_bool = mask.astype(bool)
        filled[~mask_bool] = np.nan

    return filled


def nan_gaussian(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return arr.astype(np.float32, copy=True)

    arr = arr.astype(np.float32, copy=False)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.full_like(arr, np.nan, dtype=np.float32)

    vals = np.where(valid, arr, 0.0)
    w = valid.astype(np.float32)

    smooth_vals = ndi.gaussian_filter(vals, sigma=float(sigma), mode="nearest")
    smooth_w = ndi.gaussian_filter(w, sigma=float(sigma), mode="nearest")

    out = np.divide(
        smooth_vals,
        np.maximum(smooth_w, 1e-6),
        out=np.zeros_like(smooth_vals, dtype=np.float32),
        where=smooth_w > 1e-6,
    )
    out[~valid & (smooth_w <= 1e-6)] = np.nan
    return out.astype(np.float32)


def morphological_hole_fill(
    arr: np.ndarray,
    *,
    structure_size: int = 3,
    min_valid_value: float = 0.0,
) -> np.ndarray:
    """
    Lightweight crown-hole suppression before peak screening.
    """
    work = arr.astype(np.float32, copy=True)
    valid = np.isfinite(work) & (work >= min_valid_value)
    if not np.any(valid):
        return work

    mask = valid.astype(np.uint8)
    filled_mask = ndi.binary_fill_holes(mask).astype(bool)

    footprint = np.ones((max(3, structure_size), max(3, structure_size)), dtype=bool)
    local_max = ndi.maximum_filter(np.where(valid, work, -np.inf), footprint=footprint, mode="nearest")

    to_fill = filled_mask & ~valid
    work[to_fill] = local_max[to_fill]
    return work


def dual_gaussian_filter(
    arr: np.ndarray,
    *,
    sigma_dist: float = 0.9,
    sigma_height: float = 1.8,
    height_scale_quantile: float = 0.90,
) -> np.ndarray:
    """
    Paper-inspired dual Gaussian blend:
    - one smoother driven more by spatial distance
    - one smoother weighted by local height contrast
    """
    base = fill_nan_by_nearest(arr)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.full_like(arr, np.nan, dtype=np.float32)

    g1 = nan_gaussian(base, sigma_dist)
    g2 = nan_gaussian(base, sigma_height)

    positive = base[finite & (base > 0)]
    if positive.size == 0:
        out = g1
        out[~finite] = np.nan
        return out.astype(np.float32)

    hscale = float(np.nanquantile(positive, height_scale_quantile))
    hscale = max(hscale, 1e-6)

    gx = ndi.sobel(base, axis=1, mode="nearest")
    gy = ndi.sobel(base, axis=0, mode="nearest")
    grad = np.hypot(gx, gy)

    local_relief = np.clip(grad / (hscale + 1e-6), 0.0, 1.0)
    w = np.exp(-2.0 * local_relief).astype(np.float32)

    out = w * g2 + (1.0 - w) * g1
    out[~finite] = np.nan
    return out.astype(np.float32)


def local_maxima_mask(
    arr: np.ndarray,
    *,
    window_pixels: int,
    min_height: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    win = max(3, int(window_pixels))
    if win % 2 == 0:
        win += 1

    work = arr.copy()
    if valid_mask is None:
        valid_mask = np.isfinite(work)

    work[~valid_mask] = -np.inf
    mx = ndi.maximum_filter(work, size=win, mode="nearest")
    peaks = valid_mask & np.isfinite(arr) & (arr >= float(min_height)) & (work == mx)
    return peaks


def prune_peaks_by_distance(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    *,
    transform: rasterio.Affine,
    min_separation_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(rows) <= 1 or min_separation_m <= 0:
        return rows, cols, values

    order = np.argsort(values)[::-1]
    keep_idx: list[int] = []
    kept_xy: list[tuple[float, float]] = []

    for idx in order:
        x, y = map_point(transform, int(rows[idx]), int(cols[idx]))
        if all(math.hypot(x - px, y - py) >= min_separation_m for px, py in kept_xy):
            keep_idx.append(int(idx))
            kept_xy.append((x, y))

    keep_idx = np.asarray(keep_idx, dtype=int)
    return rows[keep_idx], cols[keep_idx], values[keep_idx]


def _line_and_band_samples(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    band_width_px: int,
    shape_: tuple[int, int],
) -> list[tuple[int, int]]:
    n = max(abs(r1 - r0), abs(c1 - c0)) + 1
    if n <= 0:
        return []

    rr = np.linspace(r0, r1, n)
    cc = np.linspace(c0, c1, n)

    dr = float(r1 - r0)
    dc = float(c1 - c0)
    norm = math.hypot(dr, dc)
    if norm == 0:
        return [(r0, c0)]

    pr = -dc / norm
    pc = dr / norm

    out: set[tuple[int, int]] = set()
    offsets = range(-max(0, band_width_px), max(0, band_width_px) + 1)

    for r, c in zip(rr, cc):
        for off in offsets:
            ry = int(round(r + pr * off))
            cx = int(round(c + pc * off))
            if 0 <= ry < shape_[0] and 0 <= cx < shape_[1]:
                out.add((ry, cx))

    return list(out)


def _angle_at_valley(
    peak_a: tuple[int, int],
    peak_b: tuple[int, int],
    valley: tuple[int, int],
    arr: np.ndarray,
    transform: rasterio.Affine,
) -> float:
    ax, ay = map_point(transform, peak_a[0], peak_a[1])
    bx, by = map_point(transform, peak_b[0], peak_b[1])
    vx, vy = map_point(transform, valley[0], valley[1])

    az = float(arr[peak_a])
    bz = float(arr[peak_b])
    vz = float(arr[valley])

    va = np.array([ax - vx, ay - vy, max(az - vz, 0.0)], dtype=np.float64)
    vb = np.array([bx - vx, by - vy, max(bz - vz, 0.0)], dtype=np.float64)

    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom <= 0:
        return 0.0

    cosang = float(np.clip(np.dot(va, vb) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def screen_false_peaks(
    arr: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    *,
    transform: rasterio.Affine,
    angle_threshold_deg: float = 110.0,
    band_width_px: int = 1,
    max_pair_distance_m: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(rows)
    if n <= 1:
        return rows, cols, values

    keep = np.ones(n, dtype=bool)
    xy = np.array([map_point(transform, int(r), int(c)) for r, c in zip(rows, cols)], dtype=np.float64)

    for i in range(n):
        if not keep[i]:
            continue

        for j in range(i + 1, n):
            if not keep[j]:
                continue

            d = float(math.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1]))
            if d <= 0 or d > max_pair_distance_m:
                continue

            samples = _line_and_band_samples(
                int(rows[i]), int(cols[i]),
                int(rows[j]), int(cols[j]),
                int(band_width_px),
                arr.shape,
            )
            if len(samples) < 3:
                continue

            vals = np.array([arr[r, c] for r, c in samples], dtype=np.float32)
            valid = np.isfinite(vals)
            if not np.any(valid):
                continue

            samples_valid = [samples[k] for k, ok in enumerate(valid) if ok]
            vals_valid = vals[valid]
            valley = samples_valid[int(np.argmin(vals_valid))]

            angle = _angle_at_valley(
                (int(rows[i]), int(cols[i])),
                (int(rows[j]), int(cols[j])),
                valley,
                arr,
                transform,
            )

            if angle >= angle_threshold_deg:
                if values[i] >= values[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    return rows[keep], cols[keep], values[keep]


def build_marker_raster(shape_: tuple[int, int], rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    markers = np.zeros(shape_, dtype=np.int32)
    for idx, (r, c) in enumerate(zip(rows, cols), start=1):
        markers[int(r), int(c)] = idx
    return markers


def watershed_labels(filtered: np.ndarray, markers: np.ndarray, canopy_mask: np.ndarray) -> np.ndarray:
    if np.max(markers) == 0:
        return np.zeros(filtered.shape, dtype=np.int32)

    canopy_mask = canopy_mask.astype(bool)
    work = fill_nan_by_nearest(filtered, mask=canopy_mask)
    work[~canopy_mask] = 0.0

    finite = canopy_mask & np.isfinite(work)
    grad = ndi.gaussian_gradient_magnitude(np.where(finite, work, 0.0), sigma=1.0)

    if np.any(finite):
        inv = np.nanmax(work[finite]) - work
        cost = grad + 0.35 * inv
        cmin = float(np.nanmin(cost[finite]))
        cmax = float(np.nanmax(cost[finite]))
        if cmax <= cmin:
            img = np.zeros_like(work, dtype=np.uint16)
        else:
            norm = (cost - cmin) / (cmax - cmin)
            img = np.zeros_like(work, dtype=np.uint16)
            img[finite] = np.clip(norm[finite] * 65535.0, 0, 65535).astype(np.uint16)
    else:
        img = np.zeros_like(work, dtype=np.uint16)

    labels = ndi.watershed_ift(img, markers.astype(np.int32))
    labels = labels.astype(np.int32)
    labels[~canopy_mask] = 0
    return labels


def relabel_sequential(labels: np.ndarray) -> np.ndarray:
    out = np.zeros_like(labels, dtype=np.int32)
    ids = np.unique(labels[labels > 0])
    for new_id, old_id in enumerate(ids, start=1):
        out[labels == old_id] = int(new_id)
    return out


def remove_small_segments(
    labels: np.ndarray,
    *,
    transform: rasterio.Affine,
    min_area_m2: float,
) -> np.ndarray:
    if min_area_m2 <= 0:
        return labels

    a = cell_area(transform)
    min_pixels = max(1, int(math.ceil(min_area_m2 / max(a, 1e-9))))

    out = labels.copy()
    seg_ids, counts = np.unique(out[out > 0], return_counts=True)
    for seg_id, count in zip(seg_ids, counts):
        if int(count) < min_pixels:
            out[out == int(seg_id)] = 0

    return relabel_sequential(out)


def label_peak_counts(labels: np.ndarray, peak_rows: np.ndarray, peak_cols: np.ndarray) -> dict[int, int]:
    counts: dict[int, int] = {}
    for r, c in zip(peak_rows, peak_cols):
        seg = int(labels[int(r), int(c)])
        if seg <= 0:
            continue
        counts[seg] = counts.get(seg, 0) + 1
    return counts


def smooth_polygon_geom(
    geom: BaseGeometry,
    *,
    smooth_distance: float = 0.20,
    simplify_tol: float = 0.10,
) -> BaseGeometry:
    g = geom
    try:
        if smooth_distance > 0:
            g = g.buffer(smooth_distance).buffer(-smooth_distance)
        if simplify_tol > 0:
            g = g.simplify(simplify_tol, preserve_topology=True)
        if not g.is_valid:
            g = g.buffer(0)
    except Exception:
        g = geom
    return g


def peaks_to_feature_collection(
    rows: Iterable[int],
    cols: Iterable[int],
    values: Iterable[float],
    *,
    transform: rasterio.Affine,
    labels: np.ndarray | None = None,
    peak_counts: dict[int, int] | None = None,
) -> dict:
    feats = []
    for idx, (r, c, z) in enumerate(zip(rows, cols, values), start=1):
        x, y = map_point(transform, int(r), int(c))
        seg_id = int(labels[int(r), int(c)]) if labels is not None else idx
        feats.append(
            {
                "type": "Feature",
                "properties": {
                    "tree_id": seg_id,
                    "peak_id": idx,
                    "h_m": float(z),
                    "multi_apx": int((peak_counts or {}).get(seg_id, 0) > 1),
                    "pk_count": int((peak_counts or {}).get(seg_id, 1)),
                },
                "geometry": {"type": "Point", "coordinates": [x, y]},
            }
        )
    return {"type": "FeatureCollection", "features": feats}


def labels_to_feature_collection(
    labels: np.ndarray,
    *,
    transform: rasterio.Affine,
    height_surface: np.ndarray,
    peak_counts: dict[int, int] | None = None,
    smooth_distance: float = 0.20,
    simplify_tol: float = 0.10,
) -> dict:
    mask = labels > 0
    a = cell_area(transform)

    feats = []
    for geom, value in features.shapes(labels.astype(np.int32), mask=mask, transform=transform):
        seg_id = int(value)
        if seg_id <= 0:
            continue

        seg_mask = labels == seg_id
        vals = height_surface[seg_mask]
        vals = vals[np.isfinite(vals)]

        area_px = int(np.count_nonzero(seg_mask))
        area_m2 = float(area_px * a)

        shp = shape(geom)
        shp = smooth_polygon_geom(
            shp,
            smooth_distance=smooth_distance,
            simplify_tol=simplify_tol,
        )
        if shp.is_empty:
            continue

        crown_radius_m = float(math.sqrt(max(shp.area, 0.0) / math.pi)) if shp.area > 0 else 0.0

        feats.append(
            {
                "type": "Feature",
                "properties": {
                    "tree_id": seg_id,
                    "area_px": area_px,
                    "area_m2": area_m2,
                    "rad_m": crown_radius_m,
                    "max_h_m": float(np.nanmax(vals)) if vals.size else None,
                    "mean_h_m": float(np.nanmean(vals)) if vals.size else None,
                    "multi_apx": int((peak_counts or {}).get(seg_id, 0) > 1),
                    "pk_count": int((peak_counts or {}).get(seg_id, 1)),
                },
                "geometry": mapping(shp),
            }
        )

    return {"type": "FeatureCollection", "features": feats}


def _normalize_shp_value(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_, bool)):
        return int(bool(v))
    if v is None:
        return None
    return v


def _schema_type_for_value(v) -> str:
    if isinstance(v, (np.integer, int, np.bool_, bool)):
        return "int"
    if isinstance(v, (np.floating, float)):
        return "float"
    return "str:80"


def write_shapefile(
    path: str | Path,
    payload: dict,
    *,
    crs: rasterio.crs.CRS | None = None,
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    feats = payload.get("features", [])
    if not feats:
        raise ValueError(f"No features available to write shapefile: {path}")

    geom_type = feats[0]["geometry"]["type"]
    first_props = feats[0].get("properties", {})

    schema = {
        "geometry": geom_type,
        "properties": {k: _schema_type_for_value(v) for k, v in first_props.items()},
    }

    crs_wkt = crs.to_wkt() if crs is not None else None

    with fiona.open(
        path,
        mode="w",
        driver="ESRI Shapefile",
        schema=schema,
        crs_wkt=crs_wkt,
        encoding="UTF-8",
    ) as dst:
        for feat in feats:
            props = {k: _normalize_shp_value(v) for k, v in feat.get("properties", {}).items()}
            dst.write(
                {
                    "geometry": feat["geometry"],
                    "properties": props,
                }
            )

    return str(path)

# --- Compatibility wrappers kept for existing ITD algorithm modules ---

def read_chm_raster(path: str | Path):
    return read_surface_raster(path)


def write_geojson(path: str | Path, payload: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return str(path)


def peaks_to_geojson(
    rows: Iterable[int],
    cols: Iterable[int],
    values: Iterable[float],
    *,
    transform: rasterio.Affine,
    labels: np.ndarray | None = None,
    peak_counts: dict[int, int] | None = None,
) -> dict:
    return peaks_to_feature_collection(
        rows,
        cols,
        values,
        transform=transform,
        labels=labels,
        peak_counts=peak_counts,
    )


def labels_to_geojson(
    labels: np.ndarray,
    *,
    transform: rasterio.Affine,
    height_surface: np.ndarray,
    peak_counts: dict[int, int] | None = None,
    smooth_distance: float = 0.20,
    simplify_tol: float = 0.10,
) -> dict:
    return labels_to_feature_collection(
        labels,
        transform=transform,
        height_surface=height_surface,
        peak_counts=peak_counts,
        smooth_distance=smooth_distance,
        simplify_tol=simplify_tol,
    )
