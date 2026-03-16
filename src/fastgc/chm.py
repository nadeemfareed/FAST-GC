from __future__ import annotations

import os
from pathlib import Path

import laspy
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    maximum_filter,
    median_filter,
)
from scipy.spatial import QhullError

try:
    import rasterio
    from rasterio.transform import from_origin
except Exception:  # pragma: no cover
    rasterio = None
    from_origin = None

from .monster import log_info, run_stage
from .sensors import sensor_defaults

CHM_METHOD_CHOICES = {
    "p2r",
    "p99",
    "tin",
    "pitfree",
    "spikefree",
    "percentile",
    "percentile_top",
    "percentile_band",
}
CHM_SURFACE_METHOD_CHOICES = {"p2r", "p99", "tin", "pitfree", "spikefree"}
CHM_SMOOTH_CHOICES = {"none", "median", "gaussian"}


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


def _write_tif(arr: np.ndarray, out_fp: str, xmin: float, ymax: float, grid_res: float, crs=None, nodata=np.nan):
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


def _cell_index_arrays(x: np.ndarray, y: np.ndarray, bounds: tuple[float, float, float, float], res: float):
    xmin, ymin, xmax, ymax, nx, ny, _xs, _ys = _grid_spec(bounds, res)
    ix = np.floor((x - xmin) / res).astype(np.int32)
    iy = np.floor((ymax - y) / res).astype(np.int32)
    np.clip(ix, 0, nx - 1, out=ix)
    np.clip(iy, 0, ny - 1, out=iy)
    flat = iy.astype(np.int64) * int(nx) + ix.astype(np.int64)
    return ix, iy, flat, nx, ny, xmin, ymax


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


def _interpolate_tin_grid(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    bounds: tuple[float, float, float, float],
    res: float,
):
    xs, ys, xx, yy = _grid_xy(bounds, res)
    pts = np.column_stack([px, py])

    if pts.shape[0] < 3:
        raise ValueError("Need at least 3 support points for TIN interpolation.")

    try:
        lin = LinearNDInterpolator(pts, pz, fill_value=np.nan)
        grid = lin(xx, yy)
    except QhullError:
        grid = np.full(xx.shape, np.nan, dtype=np.float64)

    if np.any(~np.isfinite(grid)):
        nn = NearestNDInterpolator(pts, pz)
        fill = nn(xx, yy)
        grid = np.where(np.isfinite(grid), grid, fill)

    xmin = float(xs[0] - 0.5 * res)
    ymax = float(ys[0] + 0.5 * res)
    return grid.astype(np.float32), xmin, ymax


def _max_envelope(grids: list[np.ndarray]) -> np.ndarray:
    if not grids:
        raise ValueError("No grids supplied for max-envelope CHM.")
    out = grids[0].copy()
    for g in grids[1:]:
        both = np.isfinite(out) & np.isfinite(g)
        out[both] = np.maximum(out[both], g[both])
        only_g = ~np.isfinite(out) & np.isfinite(g)
        out[only_g] = g[only_g]
    return out.astype(np.float32)


def _normalize_thresholds(thresholds: list[float] | None, sensor_mode: str):
    if thresholds:
        return sorted(set(float(v) for v in thresholds))
    cfg = sensor_defaults(sensor_mode)
    vals = cfg.get("chm_pitfree_thresholds", [0.0, 2.0, 5.0, 10.0, 15.0])
    return sorted(set(float(v) for v in vals))


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


def _mask_outside_footprint(grid: np.ndarray, support_mask: np.ndarray):
    if grid.shape != support_mask.shape:
        raise ValueError(
            f"CHM/support grid shape mismatch: grid={grid.shape}, support={support_mask.shape}. "
            "Both surfaces must be built from the same raster spec."
        )
    out = grid.copy()
    footprint = maximum_filter(support_mask.astype(np.uint8), size=3, mode="nearest") > 0
    out[~footprint] = np.nan
    return out.astype(np.float32), footprint


def _fill_missing_with_zero(
    grid: np.ndarray,
    support_mask: np.ndarray,
    zero_support_mask: np.ndarray,
):
    out = grid.copy()
    fill_mask = (~np.isfinite(out)) & support_mask & zero_support_mask
    out[fill_mask] = 0.0
    return out.astype(np.float32)


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

    zero_mask = footprint & np.isfinite(out) & (out == 0)

    _, indices = distance_transform_edt(~donor_mask, return_indices=True)
    nearest_vals = out[indices[0], indices[1]]
    out[target] = nearest_vals[target]

    out[~footprint] = np.nan
    out[zero_mask] = 0.0
    return out.astype(np.float32)


def _apply_min_height(grid: np.ndarray, min_height: float):
    out = grid.copy()
    mh = float(max(0.0, min_height))
    valid = np.isfinite(out)
    out[valid & (out < mh)] = 0.0
    return out.astype(np.float32)


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

    filled = grid.copy()
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

    out = grid.copy()
    out[valid] = sm[valid]
    return out.astype(np.float32)


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
    bounds: tuple[float, float, float, float],
    res: float,
    *,
    percentile: float,
    percentile_low: float | None,
    percentile_high: float | None,
):
    """
    Selector logic based on absolute normalized-height thresholds, not local percentiles.

    Design used here:
    - percentile:      keep 0 <= z <= threshold
    - percentile_top:  keep z >= threshold
    - percentile_band: keep z_low <= z <= z_high
    """
    if selector is None:
        return x, y, z

    selector = str(selector).strip().lower()

    if selector == "percentile":
        z_thr = float(percentile)
        keep = (z >= 0.0) & (z <= z_thr)

    elif selector == "percentile_top":
        z_thr = float(percentile_low if percentile_low is not None else percentile)
        keep = z >= z_thr

    elif selector == "percentile_band":
        z_low = float(percentile_low if percentile_low is not None else percentile)
        z_high = float(percentile_high if percentile_high is not None else z_low)
        if z_high < z_low:
            z_low, z_high = z_high, z_low
        keep = (z >= z_low) & (z <= z_high)

    else:
        raise ValueError(f"Unsupported CHM selector: {selector}")

    return x[keep], y[keep], z[keep]


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
    spikefree_freeze_distance: float | None,
    spikefree_insertion_buffer: float | None,
):
    surface_method = str(surface_method).strip().lower()

    if x.size < 1:
        raise RuntimeError("No points available for CHM surface construction.")

    if surface_method == "p2r":
        return _rasterize_stat(x, y, z, bounds, res, mode="max")

    if surface_method == "p99":
        return _rasterize_stat(x, y, z, bounds, res, mode="percentile", percentile=float(percentile))

    if surface_method == "tin":
        if x.size < 3:
            return _rasterize_stat(x, y, z, bounds, res, mode="max")
        return _interpolate_tin_grid(x, y, z, bounds, res)

    if surface_method == "pitfree":
        thresholds = _normalize_thresholds(pitfree_thresholds, sensor_mode=sensor_mode)
        layers: list[np.ndarray] = []
        xmin = None
        ymax = None
        for thr in thresholds:
            m = z >= float(thr)
            if np.count_nonzero(m) < 3:
                continue
            try:
                g, gxmin, gymax = _interpolate_tin_grid(x[m], y[m], z[m], bounds, res)
            except Exception:
                g, gxmin, gymax = _rasterize_stat(x[m], y[m], z[m], bounds, res, mode="max")
            layers.append(g)
            xmin = gxmin
            ymax = gymax
        if not layers:
            return _rasterize_stat(x, y, z, bounds, res, mode="max")
        return _max_envelope(layers), float(xmin), float(ymax)

    if surface_method == "spikefree":
        grid, xmin, ymax = _build_surface(
            "pitfree",
            x,
            y,
            z,
            bounds,
            res,
            sensor_mode=sensor_mode,
            percentile=percentile,
            pitfree_thresholds=pitfree_thresholds,
            spikefree_freeze_distance=spikefree_freeze_distance,
            spikefree_insertion_buffer=spikefree_insertion_buffer,
        )
        fd = float(spikefree_freeze_distance) if spikefree_freeze_distance is not None else max(res * 2.0, 0.75)
        ib = float(spikefree_insertion_buffer) if spikefree_insertion_buffer is not None else max(0.35, res * 0.75)
        radius = max(1, int(np.ceil(fd / res)))

        valid = np.isfinite(grid)
        if np.any(valid):
            filled = grid.copy()
            filled[~valid] = -np.inf
            neigh = maximum_filter(filled, size=2 * radius + 1, mode="nearest")
            limit = maximum_filter(np.where(valid, grid - ib, -np.inf), size=2 * radius + 1, mode="nearest")
            spike_mask = valid & (grid > (limit + ib))
            grid[spike_mask] = neigh[spike_mask]

        return grid.astype(np.float32), xmin, ymax

    raise ValueError(f"Unsupported CHM surface method: {surface_method}")


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
    spikefree_freeze_distance: float | None,
    spikefree_insertion_buffer: float | None,
    median_size: int,
    gaussian_sigma: float,
    min_height: float,
    fill_ground_voids_zero: bool,
    void_ground_threshold: float,
    overwrite: bool,
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
        bounds,
        grid_res,
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
            spikefree_freeze_distance=spikefree_freeze_distance,
            spikefree_insertion_buffer=spikefree_insertion_buffer,
        )

    arr, footprint = _mask_outside_footprint(arr, support_mask)

    if fill_ground_voids_zero:
        arr = _fill_missing_with_zero(
            arr,
            support_mask,
            zero_support_mask,
        )

    arr = _fill_internal_canopy_voids_nearest(arr, footprint)
    arr = _apply_min_height(arr, min_height)
    arr = _apply_smoothing(
        arr,
        method=smooth_method,
        median_size=median_size,
        gaussian_sigma=gaussian_sigma,
    )

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
    spikefree_freeze_distance: float | None,
    spikefree_insertion_buffer: float | None,
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
            spikefree_freeze_distance=spikefree_freeze_distance,
            spikefree_insertion_buffer=spikefree_insertion_buffer,
            median_size=median_size,
            gaussian_sigma=gaussian_sigma,
            min_height=min_height,
            fill_ground_voids_zero=fill_ground_voids_zero,
            void_ground_threshold=void_ground_threshold,
            overwrite=overwrite,
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

    if out["skipped"]:
        log_info(f"{stage_name} skipped tiles:")
        for rec in out["skipped"]:
            tile_name = Path(str(rec.get("tile", ""))).name
            reason = rec.get("reason", "skipped")
            print(f"  - {tile_name}: {reason}")

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
) -> str:
    method = str(method).strip().lower()
    if method not in CHM_METHOD_CHOICES:
        raise ValueError(f"Unsupported CHM method: {method}")

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
        spikefree_freeze_distance=spikefree_freeze_distance,
        spikefree_insertion_buffer=spikefree_insertion_buffer,
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
    )
    return str(out_root)


__all__ = [
    "CHM_METHOD_CHOICES",
    "CHM_SURFACE_METHOD_CHOICES",
    "CHM_SMOOTH_CHOICES",
    "build_chm_from_normalized_root",
    "resolve_normalized_root",
    "chm_output_label",
    "chm_method_output_dir",
]