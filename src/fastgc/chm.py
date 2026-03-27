from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable

import laspy
import numpy as np
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
    maximum_filter,
    median_filter,
)
from scipy.spatial import Delaunay, QhullError

try:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
except Exception:  # pragma: no cover
    rasterio = None
    from_origin = None
    Resampling = None
    reproject = None

from .monster import run_stage
from .sensors import sensor_defaults

# ------------------------------------------------------------------------------
# Public method choices
# ------------------------------------------------------------------------------

CHM_METHOD_CHOICES = {
    "p2r",
    "p99",
    "tin",
    "pitfree",
    "csf_chm",
    "spikefree",       # derived later as DSM - DEM; kept here for pipeline continuity
    "percentile",
    "percentile_top",
    "percentile_band",
}

# Native CHM constructors from normalized LAS.
# NOTE: spikefree is intentionally removed from native CHM constructors.
CHM_SURFACE_METHOD_CHOICES = {"p2r", "p99", "tin", "pitfree", "csf_chm"}

CHM_SMOOTH_CHOICES = {"none", "median", "gaussian"}


# ------------------------------------------------------------------------------
# Generic I/O helpers
# ------------------------------------------------------------------------------

def _require_rasterio():
    if rasterio is None or from_origin is None:
        raise RuntimeError("rasterio is required for CHM GeoTIFF outputs.")


def _safe_parse_crs(las: laspy.LasData):
    try:
        crs = las.header.parse_crs()
    except Exception:
        crs = None

    if crs is None:
        return None

    if rasterio is None:
        return crs

    try:
        from rasterio.crs import CRS as RioCRS
        return RioCRS.from_user_input(crs)
    except Exception:
        try:
            if hasattr(crs, "to_wkt"):
                from rasterio.crs import CRS as RioCRS
                return RioCRS.from_wkt(crs.to_wkt())
        except Exception:
            pass
    return crs


def _read_las(fp: str):
    las = laspy.read(fp)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    crs = _safe_parse_crs(las)

    return_number = None
    number_of_returns = None

    try:
        return_number = np.asarray(las.return_number, dtype=np.int16)
    except Exception:
        pass

    try:
        number_of_returns = np.asarray(las.number_of_returns, dtype=np.int16)
    except Exception:
        pass

    return las, x, y, z, crs, return_number, number_of_returns


def _iter_las_files(in_path: str, recursive: bool = False) -> list[str]:
    p = Path(in_path)
    if p.is_file():
        return [str(p)]

    if recursive:
        files = [str(q) for q in p.rglob("*") if q.is_file() and q.suffix.lower() in {".las", ".laz"}]
    else:
        files = [str(q) for q in p.iterdir() if q.is_file() and q.suffix.lower() in {".las", ".laz"}]

    files.sort()
    return files


def _write_tif(
    arr: np.ndarray,
    out_fp: str,
    xmin: float,
    ymax: float,
    grid_res: float,
    crs=None,
    nodata=np.nan,
):
    _require_rasterio()

    os.makedirs(os.path.dirname(out_fp), exist_ok=True)
    transform = from_origin(float(xmin), float(ymax), float(grid_res), float(grid_res))

    profile = {
        "driver": "GTiff",
        "height": int(arr.shape[0]),
        "width": int(arr.shape[1]),
        "count": 1,
        "dtype": str(arr.dtype),
        "transform": transform,
        "compress": "lzw",
    }

    if np.issubdtype(arr.dtype, np.floating):
        profile["nodata"] = nodata
    elif nodata is not None and not np.isnan(nodata):
        profile["nodata"] = nodata

    if crs is not None:
        try:
            from rasterio.crs import CRS as RioCRS
            profile["crs"] = RioCRS.from_user_input(crs)
        except Exception:
            try:
                if hasattr(crs, "to_wkt"):
                    profile["crs"] = crs.to_wkt()
            except Exception:
                pass

    with rasterio.open(out_fp, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def _bounds_from_points(x: np.ndarray, y: np.ndarray):
    return float(np.min(x)), float(np.min(y)), float(np.max(x)), float(np.max(y))


def _grid_spec(bounds: tuple[float, float, float, float], res: float):
    xmin, ymin, xmax, ymax = bounds
    nx = int(np.floor((xmax - xmin) / res)) + 1
    ny = int(np.floor((ymax - ymin) / res)) + 1
    xs = xmin + (np.arange(nx, dtype=np.float64) + 0.5) * res
    ys = ymax - (np.arange(ny, dtype=np.float64) + 0.5) * res
    return xmin, ymin, xmax, ymax, nx, ny, xs, ys


def _grid_xy(bounds: tuple[float, float, float, float], res: float):
    xmin, ymin, xmax, ymax, nx, ny, xs, ys = _grid_spec(bounds, res)
    xx, yy = np.meshgrid(xs, ys)
    return xs, ys, xx, yy


def _cell_index_arrays(
    x: np.ndarray,
    y: np.ndarray,
    bounds: tuple[float, float, float, float],
    res: float,
):
    xmin, ymin, xmax, ymax, nx, ny, _xs, _ys = _grid_spec(bounds, res)
    ix = np.floor((x - xmin) / res).astype(np.int32)
    iy = np.floor((ymax - y) / res).astype(np.int32)
    np.clip(ix, 0, nx - 1, out=ix)
    np.clip(iy, 0, ny - 1, out=iy)
    flat = iy.astype(np.int64) * int(nx) + ix.astype(np.int64)
    return ix, iy, flat, nx, ny, xmin, ymax


# ------------------------------------------------------------------------------
# Raster support + post-processing helpers
# ------------------------------------------------------------------------------

def _rasterize_stat(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
    res: float,
    mode: str = "max",
    percentile: float = 99.0,
):
    _ix, _iy, flat, nx, ny, xmin, ymax = _cell_index_arrays(x, y, bounds, res)
    grid = np.full((ny, nx), np.nan, dtype=np.float32)

    if z.size == 0:
        return grid, xmin, ymax

    if mode == "max":
        out = np.full(nx * ny, -np.inf, dtype=np.float64)
        np.maximum.at(out, flat, z)
        valid = np.isfinite(out)
        grid.flat[valid] = out[valid].astype(np.float32)
        return grid, xmin, ymax

    if mode == "percentile":
        order = np.argsort(flat, kind="mergesort")
        flat_s = flat[order]
        z_s = z[order]
        starts = np.r_[0, 1 + np.flatnonzero(flat_s[1:] != flat_s[:-1])]
        ends = np.r_[starts[1:], flat_s.size]
        unique = flat_s[starts]
        vals = np.empty(unique.size, dtype=np.float32)
        pct = float(np.clip(percentile, 0.0, 100.0))
        for j, (a, b) in enumerate(zip(starts, ends)):
            vals[j] = np.float32(np.percentile(z_s[a:b], pct))
        grid.flat[unique] = vals
        return grid, xmin, ymax

    raise ValueError(f"Unsupported rasterize stat mode: {mode}")


def _support_masks(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
    res: float,
    ground_threshold: float,
):
    _ix, _iy, flat, nx, ny, _xmin, _ymax = _cell_index_arrays(x, y, bounds, res)
    support = np.zeros(nx * ny, dtype=bool)
    zero_support = np.zeros(nx * ny, dtype=bool)

    if z.size == 0:
        return support.reshape(ny, nx), zero_support.reshape(ny, nx)

    support[flat] = True

    order = np.argsort(flat, kind="mergesort")
    flat_s = flat[order]
    z_s = z[order]
    starts = np.r_[0, 1 + np.flatnonzero(flat_s[1:] != flat_s[:-1])]
    unique = flat_s[starts]
    zmax = np.maximum.reduceat(z_s, starts)

    thr = float(max(0.0, ground_threshold))
    zero_support[unique] = zmax <= thr

    return support.reshape(ny, nx), zero_support.reshape(ny, nx)


def _mask_outside_footprint(grid: np.ndarray, support_mask: np.ndarray, *, grow_cells: int = 1):
    if grid.shape != support_mask.shape:
        raise ValueError(
            f"CHM/support grid shape mismatch: grid={grid.shape}, support={support_mask.shape}"
        )
    out = np.array(grid, copy=True, dtype=np.float32)
    footprint = support_mask.copy()
    grow = max(0, int(grow_cells))
    if grow > 0 and np.any(footprint):
        footprint = binary_dilation(footprint, iterations=grow)
    out[~footprint] = np.nan
    return out, footprint


def _fill_missing_with_zero(
    grid: np.ndarray,
    support_mask: np.ndarray,
    zero_support_mask: np.ndarray,
):
    out = np.array(grid, copy=True, dtype=np.float32)
    fill_mask = (~np.isfinite(out)) & support_mask & zero_support_mask
    out[fill_mask] = 0.0
    return out


def _fill_internal_canopy_voids_nearest(
    grid: np.ndarray,
    footprint: np.ndarray,
) -> np.ndarray:
    out = np.array(grid, copy=True, dtype=np.float32)
    valid = np.isfinite(out)
    target = footprint & (~valid)

    if not np.any(target):
        return out

    donor_mask = valid.copy()
    if not np.any(donor_mask):
        return out

    zero_mask = footprint & np.isfinite(out) & (out == 0.0)
    _, indices = distance_transform_edt(~donor_mask, return_indices=True)
    nearest_vals = out[indices[0], indices[1]]
    out[target] = nearest_vals[target]

    out[~footprint] = np.nan
    out[zero_mask] = 0.0
    return out


def _apply_min_height(grid: np.ndarray, min_height: float):
    out = np.array(grid, copy=True, dtype=np.float32)
    mh = float(max(0.0, min_height))
    valid = np.isfinite(out)
    out[valid & (out < mh)] = 0.0
    return out


def _fix_pits_and_voids(
    grid: np.ndarray,
    *,
    footprint: np.ndarray | None = None,
    pit_threshold: float = 1.0,
    median_size: int = 3,
) -> np.ndarray:
    """Fix local pits and interior voids without globally smoothing the raster.

    Outside the footprint stays NaN. Interior NaNs are treated as ground/open
    locations (0.0). Only locally low pixels are replaced using the local median.
    """
    out = np.array(grid, copy=True, dtype=np.float32)
    if out.size == 0:
        return out

    if footprint is None:
        footprint = np.isfinite(out)
    else:
        footprint = np.asarray(footprint, dtype=bool)

    inside = footprint
    if not np.any(inside):
        return out

    out[inside & (~np.isfinite(out))] = 0.0

    size = int(max(1, median_size))
    if size % 2 == 0:
        size += 1

    filled = np.array(out, copy=True, dtype=np.float32)
    filled[~inside] = 0.0
    med = median_filter(filled, size=size, mode="nearest")

    thr = float(max(0.0, pit_threshold))
    valid_inside = inside & np.isfinite(out)
    pit_mask = valid_inside & (out < (med - thr))
    out[pit_mask] = med[pit_mask]

    out[~inside] = np.nan
    return out


def _apply_smoothing(
    grid: np.ndarray,
    *,
    method: str,
    median_size: int,
    gaussian_sigma: float,
):
    method = str(method).strip().lower()
    if method == "none":
        return grid.astype(np.float32)

    valid = np.isfinite(grid)
    if not np.any(valid):
        return grid.astype(np.float32)

    filled = np.array(grid, copy=True, dtype=np.float32)
    filled[~valid] = 0.0

    if method == "median":
        size = int(max(1, median_size))
        if size % 2 == 0:
            size += 1
        sm = median_filter(filled, size=size, mode="nearest")
    elif method == "gaussian":
        sigma = float(max(0.0, gaussian_sigma))
        if sigma <= 0:
            return grid.astype(np.float32)
        sm = gaussian_filter(filled, sigma=sigma, mode="nearest")
    else:
        raise ValueError(f"Unsupported CHM smooth method: {method}")

    out = np.array(grid, copy=True, dtype=np.float32)
    out[valid] = sm[valid]
    out[~valid] = np.nan
    return out


# ------------------------------------------------------------------------------
# Point pre-processing helpers
# ------------------------------------------------------------------------------

def _select_all_canopy_points(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    return_number: np.ndarray | None,
    use_first_returns: bool,
):
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z >= 0.0)
    if use_first_returns and return_number is not None:
        keep &= return_number == 1
    return x[keep], y[keep], z[keep]


def _selector_filter_points(
    selector: str | None,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    percentile: float,
    percentile_low: float | None,
    percentile_high: float | None,
):
    if selector is None:
        return x, y, z

    selector = str(selector).strip().lower()

    if selector == "percentile":
        thr = float(percentile)
        keep = z <= thr

    elif selector == "percentile_top":
        thr = float(percentile_low if percentile_low is not None else percentile)
        keep = z >= thr

    elif selector == "percentile_band":
        lo = float(percentile_low if percentile_low is not None else percentile)
        hi = float(percentile_high if percentile_high is not None else lo)
        if hi < lo:
            lo, hi = hi, lo
        keep = (z >= lo) & (z <= hi)

    else:
        raise ValueError(f"Unsupported CHM selector: {selector}")

    return x[keep], y[keep], z[keep]


def _densify_subcircle(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    radius: float | None,
    *,
    n_dirs: int = 8,
):
    if radius is None:
        return x, y, z

    r = float(radius)
    if r <= 0:
        return x, y, z

    angles = np.linspace(0.0, 2.0 * np.pi, n_dirs, endpoint=False)
    xs = [x]
    ys = [y]
    zs = [z]
    for a in angles:
        xs.append(x + r * np.cos(a))
        ys.append(y + r * np.sin(a))
        zs.append(z.copy())
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)


def _cellmax_support_points(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
    res: float,
):
    _ix, _iy, flat, nx, ny, xmin, ymax = _cell_index_arrays(x, y, bounds, res)

    if z.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    zmax = np.full(nx * ny, -np.inf, dtype=np.float64)
    np.maximum.at(zmax, flat, z)
    valid = np.isfinite(zmax)

    idx = np.flatnonzero(valid)
    iy_cells = idx // nx
    ix_cells = idx % nx

    xs = xmin + (ix_cells.astype(np.float64) + 0.5) * res
    ys = ymax - (iy_cells.astype(np.float64) + 0.5) * res
    zs = zmax[idx]
    return xs, ys, zs


def _triangle_barycentric_mask(
    px: np.ndarray,
    py: np.ndarray,
    tri_xy: np.ndarray,
):
    x1, y1 = tri_xy[0]
    x2, y2 = tri_xy[1]
    x3, y3 = tri_xy[2]

    det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(det) < 1e-12:
        return None, None, None, None

    l1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / det
    l2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / det
    l3 = 1.0 - l1 - l2
    inside = (l1 >= -1e-8) & (l2 >= -1e-8) & (l3 >= -1e-8)
    return inside, l1, l2, l3


# ------------------------------------------------------------------------------
# Surface builders
# ------------------------------------------------------------------------------

def _rasterize_constrained_tin(
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    bounds: tuple[float, float, float, float],
    res: float,
    *,
    max_edge: float | None,
):
    xs, ys, _xx, _yy = _grid_xy(bounds, res)
    xmin, ymin, xmax, ymax, nx, ny, _xs, _ys = _grid_spec(bounds, res)
    grid = np.full((ny, nx), np.nan, dtype=np.float32)

    if sx.size < 1:
        return grid, float(xs[0] - 0.5 * res), float(ys[0] + 0.5 * res)
    if sx.size < 3:
        return _rasterize_stat(sx, sy, sz, bounds, res, mode="max")

    pts = np.column_stack([sx, sy])

    try:
        tri = Delaunay(pts)
    except QhullError:
        return _rasterize_stat(sx, sy, sz, bounds, res, mode="max")

    edge_limit = float(max_edge) if max_edge is not None else math.inf
    if edge_limit <= 0:
        edge_limit = math.inf

    for simplex in tri.simplices:
        tri_xy = pts[simplex]
        tri_z = sz[simplex]

        e01 = np.hypot(*(tri_xy[0] - tri_xy[1]))
        e12 = np.hypot(*(tri_xy[1] - tri_xy[2]))
        e20 = np.hypot(*(tri_xy[2] - tri_xy[0]))
        longest = max(e01, e12, e20)

        if longest > edge_limit:
            continue

        tri_xmin = max(xmin, float(np.min(tri_xy[:, 0])))
        tri_xmax = min(xmax, float(np.max(tri_xy[:, 0])))
        tri_ymin = max(ymin, float(np.min(tri_xy[:, 1])))
        tri_ymax = min(ymax, float(np.max(tri_xy[:, 1])))

        c0 = max(0, int(np.floor((tri_xmin - xmin) / res)))
        c1 = min(nx - 1, int(np.floor((tri_xmax - xmin) / res)))
        r0 = max(0, int(np.floor((ymax - tri_ymax) / res)))
        r1 = min(ny - 1, int(np.floor((ymax - tri_ymin) / res)))

        if c1 < c0 or r1 < r0:
            continue

        sub_x = xs[c0:c1 + 1]
        sub_y = ys[r0:r1 + 1]
        sub_xx, sub_yy = np.meshgrid(sub_x, sub_y)

        inside, l1, l2, l3 = _triangle_barycentric_mask(sub_xx, sub_yy, tri_xy)
        if inside is None or not np.any(inside):
            continue

        zsurf = l1 * tri_z[0] + l2 * tri_z[1] + l3 * tri_z[2]

        sub = grid[r0:r1 + 1, c0:c1 + 1]
        write_mask = inside & (~np.isfinite(sub))
        sub[write_mask] = zsurf[write_mask].astype(np.float32)

        overlap_mask = inside & np.isfinite(sub)
        sub[overlap_mask] = np.maximum(sub[overlap_mask], zsurf[overlap_mask]).astype(np.float32)
        grid[r0:r1 + 1, c0:c1 + 1] = sub

    out_xmin = float(xs[0] - 0.5 * res)
    out_ymax = float(ys[0] + 0.5 * res)
    return grid, out_xmin, out_ymax


def _max_envelope(grids: list[np.ndarray]) -> np.ndarray:
    if not grids:
        raise ValueError("No grids supplied for max-envelope CHM.")
    out = np.array(grids[0], copy=True, dtype=np.float32)
    for g in grids[1:]:
        both = np.isfinite(out) & np.isfinite(g)
        out[both] = np.maximum(out[both], g[both])
        only_g = ~np.isfinite(out) & np.isfinite(g)
        out[only_g] = g[only_g]
    return out.astype(np.float32)


def _normalize_thresholds(thresholds: list[float] | None, sensor_mode: str):
    if thresholds:
        vals = [float(v) for v in thresholds]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            return sorted(set(vals))

    cfg = sensor_defaults(sensor_mode)
    vals = cfg.get("chm_pitfree_thresholds", [0.0, 2.0, 5.0, 10.0, 15.0])
    return sorted(set(float(v) for v in vals))


def _normalize_pitfree_max_edge(
    thresholds: list[float],
    max_edge: float | list[float] | None,
    grid_res: float,
):
    if max_edge is None:
        return [max(4.0 * grid_res, 1.0) for _ in thresholds]

    if isinstance(max_edge, (list, tuple)):
        vals = [float(v) for v in max_edge]
        if len(vals) == 1:
            return vals * len(thresholds)
        if len(vals) != len(thresholds):
            raise ValueError("pitfree_max_edge must have length 1 or match pitfree_thresholds.")
        return vals

    return [float(max_edge) for _ in thresholds]


def _cloth_sim_chm(
    base_grid: np.ndarray,
    footprint: np.ndarray,
    *,
    n_iter: int = 40,
    drop_step: float = 0.20,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    out = np.array(base_grid, copy=True, dtype=np.float32)
    if not np.any(np.isfinite(out)):
        return out

    support = np.where(np.isfinite(out), out, 0.0).astype(np.float32)
    start = float(np.nanmax(out)) + max(1.0, 2.0 * drop_step)
    cloth = np.full(out.shape, start, dtype=np.float32)
    cloth[~footprint] = np.nan

    for _ in range(max(1, int(n_iter))):
        cloth_fill = np.where(np.isfinite(cloth), cloth, 0.0)
        cloth_smooth = gaussian_filter(cloth_fill, sigma=max(0.1, float(smooth_sigma)), mode="nearest")
        cloth_next = cloth_smooth - float(drop_step)
        cloth_next = np.maximum(cloth_next, support)
        cloth_next[~footprint] = np.nan
        cloth = cloth_next.astype(np.float32)

    return cloth.astype(np.float32)


def _build_surface(
    surface_method: str,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
    res: float,
    *,
    sensor_mode: str,
    percentile: float,
    pitfree_thresholds: list[float] | None,
    use_first_returns: bool,
    pitfree_max_edge: float | list[float] | None,
    pitfree_subcircle: float | None,
    pitfree_highest: bool,
):
    surface_method = str(surface_method).strip().lower()

    if x.size < 1:
        raise RuntimeError("No points available for CHM surface construction.")

    if surface_method == "p2r":
        return _rasterize_stat(x, y, z, bounds, res, mode="max")

    if surface_method == "p99":
        return _rasterize_stat(x, y, z, bounds, res, mode="percentile", percentile=float(percentile))

    if surface_method == "tin":
        sx, sy, sz = _cellmax_support_points(x, y, z, bounds, res)
        return _rasterize_constrained_tin(sx, sy, sz, bounds, res, max_edge=None)

    if surface_method == "pitfree":
        thresholds = _normalize_thresholds(pitfree_thresholds, sensor_mode=sensor_mode)
        if 0.0 not in thresholds:
            thresholds = [0.0] + thresholds

        edge_limits = _normalize_pitfree_max_edge(thresholds, pitfree_max_edge, res)
        layers: list[np.ndarray] = []
        out_xmin = None
        out_ymax = None

        px, py, pz = _densify_subcircle(x, y, z, pitfree_subcircle)

        for thr, edge_lim in zip(thresholds, edge_limits):
            keep = pz >= float(thr)
            if np.count_nonzero(keep) < 1:
                continue

            lx = px[keep]
            ly = py[keep]
            lz = pz[keep]

            if pitfree_highest:
                sx, sy, sz = _cellmax_support_points(lx, ly, lz, bounds, res)
            else:
                sx, sy, sz = lx, ly, lz

            if sx.size < 1:
                continue

            grid_i, gxmin, gymax = _rasterize_constrained_tin(
                sx, sy, sz, bounds, res, max_edge=edge_lim
            )
            layers.append(grid_i)
            out_xmin = gxmin
            out_ymax = gymax

        if not layers:
            return _rasterize_stat(x, y, z, bounds, res, mode="max")

        return _max_envelope(layers), float(out_xmin), float(out_ymax)

    if surface_method == "csf_chm":
        base_grid, gxmin, gymax = _rasterize_stat(x, y, z, bounds, res, mode="max")
        support_mask, _zero_support = _support_masks(x, y, z, bounds, res, ground_threshold=0.15)
        base_grid, footprint = _mask_outside_footprint(base_grid, support_mask, grow_cells=1)
        cloth = _cloth_sim_chm(base_grid, footprint, n_iter=40, drop_step=0.20, smooth_sigma=1.0)
        cloth[~footprint] = np.nan
        return cloth.astype(np.float32), gxmin, gymax

    if surface_method == "spikefree":
        raise ValueError(
            "spikefree is now DSM-first and should not be constructed directly from CHM normalized surfaces."
        )

    raise ValueError(f"Unsupported CHM surface method: {surface_method}")


# ------------------------------------------------------------------------------
# Public labels / directories
# ------------------------------------------------------------------------------

def chm_output_label(method: str, surface_method: str | None = None) -> str:
    method = str(method).strip().lower()
    if surface_method is None:
        return method
    surface_method = str(surface_method).strip().lower()
    if method in CHM_SURFACE_METHOD_CHOICES:
        return method
    return f"{method}_{surface_method}"


def chm_method_output_dir(processed_root: str | os.PathLike[str], method: str, surface_method: str | None = None) -> str:
    return os.path.join(str(processed_root), "FAST_CHM", chm_output_label(method, surface_method))


def resolve_normalized_root(processed_root: str | os.PathLike[str]) -> str:
    processed_root = Path(processed_root)
    if processed_root.name == "FAST_NORMALIZED" and processed_root.exists():
        return str(processed_root)

    cand = processed_root / "FAST_NORMALIZED"
    if cand.exists():
        return str(cand)

    raise FileNotFoundError(f"FAST_NORMALIZED folder not found under: {processed_root}")


# ------------------------------------------------------------------------------
# Native CHM from normalized LAS
# ------------------------------------------------------------------------------

def _build_single_chm_tile(
    norm_fp: str,
    out_fp: str,
    *,
    sensor_mode: str,
    method: str,
    selector: str | None,
    algorithm: str,
    grid_res: float,
    smooth_method: str,
    percentile: float,
    percentile_low: float | None,
    percentile_high: float | None,
    pitfree_thresholds: list[float] | None,
    use_first_returns: bool,
    median_size: int,
    gaussian_sigma: float,
    min_height: float,
    fill_ground_voids_zero: bool,
    void_ground_threshold: float,
    overwrite: bool,
    pitfree_max_edge: float | list[float] | None = None,
    pitfree_subcircle: float | None = None,
    pitfree_highest: bool = True,
):
    if os.path.exists(out_fp) and not overwrite:
        return {
            "status": "skipped",
            "tile": norm_fp,
            "out_fp": out_fp,
            "reason": "output exists",
        }

    _las, x, y, z, crs, return_number, _number_of_returns = _read_las(norm_fp)

    if x.size == 0:
        arr = np.zeros((1, 1), dtype=np.float32)
        _write_tif(arr, out_fp, 0.0, 0.0, grid_res, crs=crs)
        return {"status": "ok", "tile": norm_fp, "out_fp": out_fp}

    x_all, y_all, z_all = _select_all_canopy_points(
        x, y, z,
        return_number=return_number,
        use_first_returns=use_first_returns,
    )

    if x_all.size < 1:
        arr = np.zeros((1, 1), dtype=np.float32)
        xmin = float(np.min(x))
        ymax = float(np.max(y))
        _write_tif(arr, out_fp, xmin, ymax, grid_res, crs=crs)
        return {"status": "ok", "tile": norm_fp, "out_fp": out_fp}

    bounds = _bounds_from_points(x_all, y_all)
    support_mask, zero_support_mask = _support_masks(
        x_all, y_all, z_all, bounds, grid_res, void_ground_threshold
    )

    x_use, y_use, z_use = _selector_filter_points(
        selector,
        x_all, y_all, z_all,
        percentile=percentile,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )

    if x_use.size < 1:
        xs, ys, xx, _yy = _grid_xy(bounds, grid_res)
        arr = np.full(xx.shape, np.nan, dtype=np.float32)
        xmin = float(xs[0] - 0.5 * grid_res)
        ymax = float(ys[0] + 0.5 * grid_res)
    else:
        arr, xmin, ymax = _build_surface(
            algorithm,
            x_use, y_use, z_use,
            bounds,
            grid_res,
            sensor_mode=sensor_mode,
            percentile=percentile,
            pitfree_thresholds=pitfree_thresholds,
            use_first_returns=use_first_returns,
            pitfree_max_edge=pitfree_max_edge,
            pitfree_subcircle=pitfree_subcircle,
            pitfree_highest=pitfree_highest,
        )

    arr, footprint = _mask_outside_footprint(arr, support_mask, grow_cells=1)

    if fill_ground_voids_zero:
        arr = _fill_missing_with_zero(
            arr,
            support_mask,
            zero_support_mask,
        )

    arr = _fill_internal_canopy_voids_nearest(arr, footprint)
    arr = _apply_min_height(arr, min_height)
    arr = _fix_pits_and_voids(
        arr,
        footprint=footprint,
        pit_threshold=max(0.25, float(void_ground_threshold)),
        median_size=max(3, int(median_size) if int(median_size) > 0 else 3),
    )
    arr = _apply_smoothing(
        arr,
        method=smooth_method,
        median_size=median_size,
        gaussian_sigma=gaussian_sigma,
    )
    arr[~footprint] = np.nan

    _write_tif(arr, out_fp, xmin, ymax, grid_res, crs=crs)
    return {"status": "ok", "tile": norm_fp, "out_fp": out_fp}


def _run_chm_phase(
    normalized_files: list[str],
    *,
    out_root: str | os.PathLike[str],
    stage_name: str,
    sensor_mode: str,
    method: str,
    selector: str | None,
    algorithm: str,
    grid_res: float,
    smooth_method: str,
    percentile: float,
    percentile_low: float | None,
    percentile_high: float | None,
    pitfree_thresholds: list[float] | None,
    use_first_returns: bool,
    median_size: int,
    gaussian_sigma: float,
    min_height: float,
    fill_ground_voids_zero: bool,
    void_ground_threshold: float,
    overwrite: bool,
    n_jobs: int,
    joblib_backend: str,
    joblib_batch_size: int | str,
    joblib_pre_dispatch: str | int,
    source: str | None,
    pitfree_max_edge: float | list[float] | None = None,
    pitfree_subcircle: float | None = None,
    pitfree_highest: bool = True,
) -> dict[str, list[object]]:
    items = []
    for norm_fp in normalized_files:
        items.append({
            "path": norm_fp,
            "out_fp": os.path.join(str(out_root), f"{Path(norm_fp).stem}.tif"),
        })

    def _worker(item: dict[str, str]):
        return _build_single_chm_tile(
            item["path"],
            item["out_fp"],
            sensor_mode=sensor_mode,
            method=method,
            selector=selector,
            algorithm=algorithm,
            grid_res=grid_res,
            smooth_method=smooth_method,
            percentile=percentile,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
            pitfree_thresholds=pitfree_thresholds,
            use_first_returns=use_first_returns,
            median_size=median_size,
            gaussian_sigma=gaussian_sigma,
            min_height=min_height,
            fill_ground_voids_zero=fill_ground_voids_zero,
            void_ground_threshold=void_ground_threshold,
            overwrite=overwrite,
            pitfree_max_edge=pitfree_max_edge,
            pitfree_subcircle=pitfree_subcircle,
            pitfree_highest=pitfree_highest,
        )

    summary = run_stage(
        stage_name=stage_name,
        items=items,
        func=_worker,
        n_jobs=n_jobs,
        backend=joblib_backend,
        batch_size=joblib_batch_size,
        pre_dispatch=joblib_pre_dispatch,
        source=source,
        unit="tile",
        show_banner=True,
        show_progress=True,
        item_name_fn=lambda item: Path(item["path"]).name,
    )

    out: dict[str, list[object]] = {"ok": [], "skipped": [], "failed": []}
    for rec in summary.records:
        if rec.status == "ok":
            out["ok"].append(rec.result)
        elif rec.status == "skipped":
            out["skipped"].append(rec.result if isinstance(rec.result, dict) else {"tile": rec.name, "reason": "skipped"})
        else:
            out["failed"].append(
                {
                    "tile": rec.name,
                    "reason": (rec.error or "worker failed").splitlines()[0],
                }
            )

    if out["failed"]:
        print(f"[ERROR] {stage_name} failed tiles:")
        for rec in out["failed"]:
            print(f"  - {Path(str(rec['tile'])).name}: {rec['reason']}")
        raise RuntimeError(f"{stage_name} failed for {len(out['failed'])} tile(s).")

    return out


def build_chm_from_normalized_root(
    normalized_root: str | os.PathLike[str],
    out_root: str | os.PathLike[str],
    *,
    sensor_mode: str,
    method: str = "p2r",
    surface_method: str | None = None,
    grid_res: float = 0.5,
    smooth_method: str = "none",
    percentile: float = 99.0,
    percentile_low: float | None = None,
    percentile_high: float | None = None,
    pitfree_thresholds: list[float] | None = None,
    use_first_returns: bool = False,
    spikefree_freeze_distance: float | None = None,
    spikefree_insertion_buffer: float | None = None,
    median_size: int = 0,
    gaussian_sigma: float = 1.0,
    min_height: float = 0.0,
    fill_ground_voids_zero: bool = True,
    void_ground_threshold: float = 0.15,
    overwrite: bool = False,
    n_jobs: int = 1,
    joblib_backend: str = "loky",
    joblib_batch_size: int | str = "auto",
    joblib_pre_dispatch: str | int = "2*n_jobs",
    pitfree_max_edge: float | list[float] | None = None,
    pitfree_subcircle: float | None = None,
    pitfree_highest: bool = True,
) -> str:
    method = str(method).strip().lower()
    if method not in CHM_METHOD_CHOICES:
        raise ValueError(f"Unsupported CHM method: {method}")

    if method == "spikefree":
        raise ValueError(
            "spikefree is now DSM-first. Build FAST_DSM spikefree first, then derive FAST_CHM spikefree as DSM - DEM."
        )

    if method in CHM_SURFACE_METHOD_CHOICES:
        selector = None
        algorithm = method
    else:
        selector = method
        algorithm = str(surface_method or "p2r").strip().lower()
        if algorithm not in CHM_SURFACE_METHOD_CHOICES:
            raise ValueError(f"Unsupported CHM surface method: {algorithm}")

    smooth_method = str(smooth_method).strip().lower()
    if smooth_method not in CHM_SMOOTH_CHOICES:
        raise ValueError(f"Unsupported CHM smooth method: {smooth_method}")

    normalized_files = _iter_las_files(str(normalized_root), recursive=False)
    if not normalized_files:
        raise FileNotFoundError(f"No normalized LAS/LAZ files found in: {normalized_root}")

    os.makedirs(out_root, exist_ok=True)
    stage_name = f"FAST-GC derive CHM [{chm_output_label(method, surface_method)}]"
    _run_chm_phase(
        normalized_files,
        out_root=str(out_root),
        stage_name=stage_name,
        sensor_mode=sensor_mode,
        method=method,
        selector=selector,
        algorithm=algorithm,
        grid_res=grid_res,
        smooth_method=smooth_method,
        percentile=percentile,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
        pitfree_thresholds=pitfree_thresholds,
        use_first_returns=use_first_returns,
        median_size=median_size,
        gaussian_sigma=gaussian_sigma,
        min_height=min_height,
        fill_ground_voids_zero=fill_ground_voids_zero,
        void_ground_threshold=void_ground_threshold,
        overwrite=overwrite,
        n_jobs=n_jobs,
        joblib_backend=joblib_backend,
        joblib_batch_size=joblib_batch_size,
        joblib_pre_dispatch=joblib_pre_dispatch,
        source=str(normalized_root),
        pitfree_max_edge=pitfree_max_edge,
        pitfree_subcircle=pitfree_subcircle,
        pitfree_highest=pitfree_highest,
    )
    return str(out_root)


# ------------------------------------------------------------------------------
# Derived CHM from DSM - DEM (used later for spikefree)
# ------------------------------------------------------------------------------

def _read_raster(fp: str):
    _require_rasterio()
    with rasterio.open(fp) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        return {
            "arr": arr,
            "transform": src.transform,
            "crs": src.crs,
            "height": src.height,
            "width": src.width,
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
        }


def _write_like(ref: dict, arr: np.ndarray, out_fp: str):
    _require_rasterio()
    os.makedirs(os.path.dirname(out_fp), exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": int(ref["height"]),
        "width": int(ref["width"]),
        "count": 1,
        "dtype": "float32",
        "transform": ref["transform"],
        "compress": "lzw",
        "nodata": np.nan,
    }
    if ref["crs"] is not None:
        profile["crs"] = ref["crs"]

    with rasterio.open(out_fp, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def _align_raster_to_reference(ref: dict, src: dict) -> np.ndarray:
    """
    Align src raster onto ref raster grid.
    Uses bilinear resampling for continuous surfaces.
    """
    if (
        ref["arr"].shape == src["arr"].shape
        and ref["transform"] == src["transform"]
        and ref["crs"] == src["crs"]
    ):
        return src["arr"].astype(np.float32, copy=False)

    if reproject is None or Resampling is None:
        raise RuntimeError("rasterio.warp is required to align DEM/DSM grids.")

    dst = np.full(ref["arr"].shape, np.nan, dtype=np.float32)
    reproject(
        source=src["arr"],
        destination=dst,
        src_transform=src["transform"],
        src_crs=src["crs"],
        src_nodata=np.nan,
        dst_transform=ref["transform"],
        dst_crs=ref["crs"],
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    return dst


def build_chm_from_dem_and_dsm(
    dem_root: str | os.PathLike[str],
    dsm_root: str | os.PathLike[str],
    out_root: str | os.PathLike[str],
    *,
    method_label: str = "spikefree",
    min_height: float = 0.0,
    smooth_method: str = "none",
    median_size: int = 0,
    gaussian_sigma: float = 1.0,
    overwrite: bool = False,
    n_jobs: int = 1,
    joblib_backend: str = "loky",
    joblib_batch_size: int | str = "auto",
    joblib_pre_dispatch: str | int = "2*n_jobs",
) -> str:
    dem_root = Path(dem_root)
    dsm_root = Path(dsm_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    dem_files = sorted([p for p in dem_root.glob("*.tif") if p.is_file()])
    dsm_files = sorted([p for p in dsm_root.glob("*.tif") if p.is_file()])

    if not dem_files:
        raise FileNotFoundError(f"No DEM rasters found in: {dem_root}")
    if not dsm_files:
        raise FileNotFoundError(f"No DSM rasters found in: {dsm_root}")

    dem_map = {p.stem: p for p in dem_files}
    dsm_map = {p.stem: p for p in dsm_files}
    common = sorted(set(dem_map) & set(dsm_map))
    if not common:
        raise FileNotFoundError("No matching DEM/DSM raster names were found for CHM derivation.")

    items = []
    for stem in common:
        items.append({
            "stem": stem,
            "dem_fp": str(dem_map[stem]),
            "dsm_fp": str(dsm_map[stem]),
            "out_fp": str(out_root / f"{stem}.tif"),
        })

    def _worker(item: dict[str, str]):
        if os.path.exists(item["out_fp"]) and not overwrite:
            return {
                "status": "skipped",
                "tile": item["stem"],
                "out_fp": item["out_fp"],
                "reason": "output exists",
            }

        try:
            dem = _read_raster(item["dem_fp"])
            dsm = _read_raster(item["dsm_fp"])
            dsm_aligned = _align_raster_to_reference(dem, dsm)
        except Exception as e:
            return {
                "status": "skipped",
                "tile": item["stem"],
                "out_fp": item["out_fp"],
                "reason": f"DEM/DSM alignment failed: {e}",
            }

        valid = np.isfinite(dem["arr"]) & np.isfinite(dsm_aligned)
        if not np.any(valid):
            return {
                "status": "skipped",
                "tile": item["stem"],
                "out_fp": item["out_fp"],
                "reason": "no overlapping valid DEM/DSM cells after alignment",
            }

        chm = np.full(dem["arr"].shape, np.nan, dtype=np.float32)
        chm[valid] = dsm_aligned[valid] - dem["arr"][valid]
        chm = np.where(np.isfinite(chm), chm, np.nan)
        chm = np.maximum(chm, 0.0).astype(np.float32)
        chm = _apply_min_height(chm, min_height)
        chm = _apply_smoothing(
            chm,
            method=smooth_method,
            median_size=median_size,
            gaussian_sigma=gaussian_sigma,
        )

        _write_like(dem, chm, item["out_fp"])
        return {"status": "ok", "tile": item["stem"], "out_fp": item["out_fp"]}

    stage_name = f"FAST-GC derive CHM from DSM-DEM [{method_label}]"
    summary = run_stage(
        stage_name=stage_name,
        items=items,
        func=_worker,
        n_jobs=n_jobs,
        backend=joblib_backend,
        batch_size=joblib_batch_size,
        pre_dispatch=joblib_pre_dispatch,
        source=str(dsm_root),
        unit="tile",
        show_banner=True,
        show_progress=True,
        item_name_fn=lambda item: item["stem"],
    )

    ok = []
    skipped = []
    failed = []

    for r in summary.records:
        if r.status == "ok":
            ok.append(r)
        elif r.status == "skipped":
            skipped.append(r)
        else:
            failed.append(r)

    if skipped:
        print(f"[INFO] {stage_name} skipped items:")
        for r in skipped:
            reason = "skipped"
            if isinstance(r.result, dict):
                reason = r.result.get("reason", reason)
            print(f"  - {r.name}: {reason}")

    if failed:
        print(f"[ERROR] {stage_name} failed for {len(failed)} item(s).")
        for r in failed:
            print(f"  - {r.name}: {(r.error or 'worker failed').splitlines()[0]}")
        raise RuntimeError(f"{stage_name} failed for {len(failed)} item(s).")

    if not ok and skipped:
        print(
            f"[INFO] {stage_name}: no tile-wise outputs were written. "
            f"You can still derive spikefree CHM after merging DEM and DSM."
        )

    return str(out_root)


__all__ = [
    "CHM_METHOD_CHOICES",
    "CHM_SURFACE_METHOD_CHOICES",
    "CHM_SMOOTH_CHOICES",
    "build_chm_from_normalized_root",
    "build_chm_from_dem_and_dsm",
    "resolve_normalized_root",
    "chm_output_label",
    "chm_method_output_dir",
]