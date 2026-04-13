from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

try:
    import rasterio
except Exception:  # pragma: no cover
    rasterio = None

from .monster import log_info, run_stage, stage_banner

PRODUCT_TERRAIN = "FAST_TERRAIN"

TERRAIN_PRODUCT_CHOICES = {
    "all",
    "slope_percent",
    "slope_degrees",
    "aspect",
    "hillshade",
    "curvature",
    "tpi",
    "twi",
    "dtw",
    "tci",
}


def _require_rasterio():
    if rasterio is None:
        raise RuntimeError("rasterio is required for FAST_TERRAIN products.")


def _resolve_terrain_products(products: list[str] | None) -> list[str]:
    requested = list(products or ["all"])

    if "all" in requested:
        return [
            "slope_percent",
            "slope_degrees",
            "aspect",
            "hillshade",
            "curvature",
            "tpi",
            "twi",
            "dtw",
            "tci",
        ]

    out: list[str] = []
    seen: set[str] = set()
    for p in requested:
        if p not in TERRAIN_PRODUCT_CHOICES:
            raise ValueError(f"Unsupported terrain product: {p}")
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _read_dem(dem_fp: str):
    _require_rasterio()
    with rasterio.open(dem_fp) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        nodata = src.nodata
    return arr, profile, transform, nodata


def _write_raster(arr: np.ndarray, profile: dict, out_fp: str, nodata=None):
    _require_rasterio()
    os.makedirs(os.path.dirname(out_fp), exist_ok=True)

    profile_out = profile.copy()
    profile_out.update(
        dtype="float32",
        count=1,
        compress="lzw",
    )
    if nodata is not None:
        profile_out["nodata"] = nodata

    with rasterio.open(out_fp, "w", **profile_out) as dst:
        dst.write(arr.astype(np.float32), 1)


def _pixel_size(transform) -> tuple[float, float]:
    dx = float(transform.a)
    dy = float(abs(transform.e))
    return dx, dy


def _dem_valid_mask(dem: np.ndarray, nodata) -> np.ndarray:
    valid = np.isfinite(dem)
    if nodata is not None and np.isfinite(nodata):
        valid &= dem != np.float32(nodata)
    return valid


def _apply_valid_mask(arr: np.ndarray, valid_mask: np.ndarray, nodata) -> np.ndarray:
    out = np.array(arr, copy=True, dtype=np.float32)
    if nodata is None or (isinstance(nodata, float) and np.isnan(nodata)):
        out[~valid_mask] = np.nan
    else:
        out[~valid_mask] = np.float32(nodata)
    return out


def _nanmean_filter(arr: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return arr.astype(np.float32, copy=True)

    h, w = arr.shape
    out = np.full((h, w), np.nan, dtype=np.float32)

    for r in range(h):
        r0 = max(0, r - radius)
        r1 = min(h, r + radius + 1)
        for c in range(w):
            c0 = max(0, c - radius)
            c1 = min(w, c + radius + 1)
            win = arr[r0:r1, c0:c1]
            valid = np.isfinite(win)
            if np.any(valid):
                out[r, c] = float(np.mean(win[valid]))
    return out


def _fill_nan_with_nearest(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    valid = np.isfinite(arr)
    if np.all(valid):
        return arr.copy()
    if not np.any(valid):
        return np.zeros_like(arr, dtype=np.float32)

    _, inds = distance_transform_edt(~valid, return_indices=True)
    out = arr.copy()
    out[~valid] = arr[inds[0][~valid], inds[1][~valid]]
    return out.astype(np.float32, copy=False)


def _gradients(dem: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    dem_fill = _fill_nan_with_nearest(dem)
    dzdy, dzdx = np.gradient(dem_fill, dy, dx)
    return dzdx.astype(np.float32), dzdy.astype(np.float32)


def _cellsize_mean(dx: float, dy: float) -> float:
    return 0.5 * (float(dx) + float(dy))


def _d8_flow_receivers(dem: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    dem_fill = _fill_nan_with_nearest(dem).astype(np.float64, copy=False)
    h, w = dem_fill.shape
    pad = np.pad(dem_fill, 1, mode="edge")

    offsets = [
        (-1, -1, (dx * dx + dy * dy) ** 0.5),
        (-1,  0, dy),
        (-1,  1, (dx * dx + dy * dy) ** 0.5),
        ( 0, -1, dx),
        ( 0,  1, dx),
        ( 1, -1, (dx * dx + dy * dy) ** 0.5),
        ( 1,  0, dy),
        ( 1,  1, (dx * dx + dy * dy) ** 0.5),
    ]

    center = dem_fill
    best_slope = np.zeros((h, w), dtype=np.float64)
    best_dir = np.full((h, w), -1, dtype=np.int8)

    for k, (oy, ox, dist) in enumerate(offsets):
        neigh = pad[1 + oy:1 + oy + h, 1 + ox:1 + ox + w]
        slope = (center - neigh) / max(dist, 1e-9)
        better = slope > best_slope
        best_slope[better] = slope[better]
        best_dir[better] = k

    flat = np.arange(h * w, dtype=np.int64).reshape(h, w)
    recv = np.full((h, w), -1, dtype=np.int64)

    for k, (oy, ox, _dist) in enumerate(offsets):
        mask = best_dir == k
        if not np.any(mask):
            continue
        yy, xx = np.where(mask)
        ry = yy + oy
        rx = xx + ox
        inside = (ry >= 0) & (ry < h) & (rx >= 0) & (rx < w)
        if np.any(inside):
            recv[yy[inside], xx[inside]] = flat[ry[inside], rx[inside]]

    return recv, best_slope.astype(np.float32)


def _flow_accumulation_d8(dem: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    dem_fill = _fill_nan_with_nearest(dem).astype(np.float64, copy=False)
    recv, best_slope = _d8_flow_receivers(dem_fill, dx, dy)

    n = dem_fill.size
    recv_flat = recv.ravel()
    elev_flat = dem_fill.ravel()

    acc = np.ones(n, dtype=np.float64)
    order = np.argsort(elev_flat)[::-1]  # high to low

    for idx in order:
        r = recv_flat[idx]
        if r >= 0 and r != idx:
            acc[r] += acc[idx]

    return acc.reshape(dem_fill.shape).astype(np.float32), best_slope


def _specific_catchment_area(dem: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    acc_cells, best_slope = _flow_accumulation_d8(dem, dx, dy)
    contour_width = _cellsize_mean(dx, dy)
    sca = acc_cells * contour_width
    return sca.astype(np.float32), best_slope.astype(np.float32)


def _channel_mask_from_sca(sca: np.ndarray, dx: float, dy: float) -> np.ndarray:
    valid = np.isfinite(sca)
    if not np.any(valid):
        return np.zeros_like(sca, dtype=bool)

    cell_area = float(dx) * float(dy)
    # Approximate drainage initiation threshold of ~1000 m² upslope area,
    # but never lower than 50 cells for stability on small rasters.
    thr_cells = max(50.0, 1000.0 / max(cell_area, 1e-9))
    contour_width = _cellsize_mean(dx, dy)
    thr_sca = thr_cells * contour_width

    channels = valid & (sca >= thr_sca)
    if np.any(channels):
        return channels

    # Fallback: top 1% of contributing area if threshold is too strict.
    q = np.nanpercentile(sca[valid], 99.0)
    channels = valid & (sca >= q)
    return channels


def compute_slope_percent(dem: np.ndarray, dx: float, dy: float) -> np.ndarray:
    dzdx, dzdy = _gradients(dem, dx, dy)
    slope = np.sqrt(dzdx**2 + dzdy**2)
    # Percent rise is not capped at 100. A 45° slope is 100%, and steeper slopes exceed 100%.
    return (slope * 100.0).astype(np.float32)


def compute_slope_degrees(dem: np.ndarray, dx: float, dy: float) -> np.ndarray:
    dzdx, dzdy = _gradients(dem, dx, dy)
    slope = np.sqrt(dzdx**2 + dzdy**2)
    return np.degrees(np.arctan(slope)).astype(np.float32)


def compute_aspect(dem: np.ndarray, dx: float, dy: float) -> np.ndarray:
    dzdx, dzdy = _gradients(dem, dx, dy)
    aspect = np.degrees(np.arctan2(dzdy, -dzdx))
    aspect = 90.0 - aspect
    aspect = np.where(aspect < 0.0, aspect + 360.0, aspect)
    aspect = np.where(aspect >= 360.0, aspect - 360.0, aspect)

    slope_mag = np.sqrt(dzdx**2 + dzdy**2)
    # Flat cells should not be encoded as north (0°). Use NaN.
    aspect = np.where(slope_mag > 1e-8, aspect, np.nan)
    return aspect.astype(np.float32)


def compute_hillshade(
    dem: np.ndarray,
    dx: float,
    dy: float,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
) -> np.ndarray:
    dzdx, dzdy = _gradients(dem * float(z_factor), dx, dy)
    slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    aspect = np.arctan2(dzdy, -dzdx)
    az_rad = np.radians(360.0 - azimuth + 90.0)
    alt_rad = np.radians(altitude)
    hs = (
        np.sin(alt_rad) * np.cos(slope)
        + np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect)
    )
    hs = np.clip(hs, 0.0, 1.0) * 255.0
    return hs.astype(np.float32)


def compute_curvature(dem: np.ndarray, dx: float, dy: float) -> np.ndarray:
    dem_fill = _fill_nan_with_nearest(dem)
    dzdy, dzdx = np.gradient(dem_fill, dy, dx)
    d2zdx2 = np.gradient(dzdx, dx, axis=1)
    d2zdy2 = np.gradient(dzdy, dy, axis=0)
    # Simple generalized curvature (Laplacian-style proxy).
    curvature = d2zdx2 + d2zdy2
    return curvature.astype(np.float32)


def compute_tpi(dem: np.ndarray, radius: int = 3) -> np.ndarray:
    mean_local = _nanmean_filter(dem, radius=max(1, int(radius)))
    tpi = dem - mean_local
    return tpi.astype(np.float32)


def compute_twi(dem: np.ndarray, dx: float, dy: float, eps: float = 1e-6) -> np.ndarray:
    sca, _best_slope = _specific_catchment_area(dem, dx, dy)
    slope_deg = compute_slope_degrees(dem, dx, dy)
    slope_rad = np.radians(np.maximum(slope_deg, 0.001))
    tan_beta = np.tan(slope_rad)
    twi = np.log((sca + float(eps)) / np.maximum(tan_beta, float(eps)))
    return twi.astype(np.float32)


def compute_tci(dem: np.ndarray, dx: float, dy: float) -> np.ndarray:
    dzdx, dzdy = _gradients(dem, dx, dy)
    gx = -dzdx
    gy = -dzdy
    mag = np.sqrt(gx**2 + gy**2)

    ux = np.divide(gx, mag, out=np.zeros_like(gx, dtype=np.float32), where=mag > 1e-8)
    uy = np.divide(gy, mag, out=np.zeros_like(gy, dtype=np.float32), where=mag > 1e-8)

    duxdx = np.gradient(ux, dx, axis=1)
    duydy = np.gradient(uy, dy, axis=0)

    # Positive = convergent, negative = divergent.
    tci = -(duxdx + duydy)
    tci = np.where(mag > 1e-8, tci, np.nan)
    return tci.astype(np.float32)


def compute_dtw(dem: np.ndarray, dx: float, dy: float, max_distance: float | None = None) -> np.ndarray:
    dem_fill = _fill_nan_with_nearest(dem)
    sca, _best_slope = _specific_catchment_area(dem_fill, dx, dy)
    channels = _channel_mask_from_sca(sca, dx, dy)

    if not np.any(channels):
        return np.full_like(dem_fill, np.nan, dtype=np.float32)

    # Distance to nearest channel cell; also retrieve nearest-channel indices.
    dist, inds = distance_transform_edt(
        ~channels,
        sampling=(float(dy), float(dx)),
        return_indices=True,
    )

    ch_z = dem_fill[inds[0], inds[1]]
    dtw = dem_fill - ch_z
    dtw = np.maximum(dtw, 0.0)

    if max_distance is not None:
        dtw = np.where(dist <= float(max_distance), dtw, np.nan)

    return dtw.astype(np.float32)


def _terrain_output_path(processed_root: Path, product_name: str, dem_name: str) -> Path:
    return processed_root / PRODUCT_TERRAIN / product_name / dem_name


def _compute_terrain_array(
    product: str,
    dem: np.ndarray,
    dx: float,
    dy: float,
    *,
    hillshade_azimuth: float,
    hillshade_altitude: float,
    hillshade_z_factor: float,
    tpi_radius: int,
    twi_eps: float,
    dtw_max_distance: float | None,
) -> np.ndarray:
    if product == "slope_percent":
        return compute_slope_percent(dem, dx, dy)
    if product == "slope_degrees":
        return compute_slope_degrees(dem, dx, dy)
    if product == "aspect":
        return compute_aspect(dem, dx, dy)
    if product == "hillshade":
        return compute_hillshade(
            dem,
            dx,
            dy,
            azimuth=hillshade_azimuth,
            altitude=hillshade_altitude,
            z_factor=hillshade_z_factor,
        )
    if product == "curvature":
        return compute_curvature(dem, dx, dy)
    if product == "tpi":
        return compute_tpi(dem, radius=tpi_radius)
    if product == "twi":
        return compute_twi(dem, dx, dy, eps=twi_eps)
    if product == "dtw":
        return compute_dtw(dem, dx, dy, max_distance=dtw_max_distance)
    if product == "tci":
        return compute_tci(dem, dx, dy)
    raise ValueError(f"Unsupported terrain product: {product}")


def _process_dem_for_product(item: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    dem_fp = Path(item["dem_fp"])
    out_fp = Path(item["out_fp"])
    skip_existing = bool(item["skip_existing"])
    overwrite = bool(item["overwrite"])

    if out_fp.exists() and out_fp.is_file() and skip_existing and not overwrite and not force:
        return {"status": "skipped", "path": str(out_fp), "name": dem_fp.name}

    dem, profile, transform, nodata = _read_dem(str(dem_fp))
    valid_mask = _dem_valid_mask(dem, nodata)
    dx, dy = _pixel_size(transform)

    arr = _compute_terrain_array(
        item["product"],
        dem,
        dx,
        dy,
        hillshade_azimuth=float(item["hillshade_azimuth"]),
        hillshade_altitude=float(item["hillshade_altitude"]),
        hillshade_z_factor=float(item["hillshade_z_factor"]),
        tpi_radius=int(item["tpi_radius"]),
        twi_eps=float(item["twi_eps"]),
        dtw_max_distance=item["dtw_max_distance"],
    )

    arr = _apply_valid_mask(arr, valid_mask, nodata)
    _write_raster(arr, profile, str(out_fp), nodata=nodata)
    return {"status": "ok", "path": str(out_fp), "name": dem_fp.name}


def run_terrain_from_processed_root(
    processed_root: str | os.PathLike[str],
    *,
    terrain_products: list[str] | None = None,
    hillshade_azimuth: float = 315.0,
    hillshade_altitude: float = 45.0,
    hillshade_z_factor: float = 1.0,
    tpi_radius: int = 3,
    twi_eps: float = 1e-6,
    dtw_max_distance: float | None = None,
    skip_existing: bool = False,
    overwrite: bool = False,
    n_jobs: int | None = None,
    joblib_backend: str = "loky",
    joblib_batch_size: int | str = "auto",
    joblib_pre_dispatch: str = "2*n_jobs",
) -> str:
    _require_rasterio()

    processed_root = Path(processed_root)
    dem_root = processed_root / "FAST_DEM"
    if not dem_root.exists():
        raise FileNotFoundError(f"FAST_DEM folder not found: {dem_root}")

    requested = _resolve_terrain_products(terrain_products)
    dem_files = sorted([p for p in dem_root.glob("*.tif") if p.is_file()])
    if not dem_files:
        raise FileNotFoundError(f"No DEM rasters found in: {dem_root}")

    out_root = processed_root / PRODUCT_TERRAIN
    out_root.mkdir(parents=True, exist_ok=True)

    stage_banner(
        "FAST_TERRAIN",
        source=str(dem_root),
        total=len(dem_files),
        unit="tile",
        extra=f"products={', '.join(requested)}",
    )

    for product in requested:
        items: list[dict[str, Any]] = []
        for dem_fp in dem_files:
            out_fp = _terrain_output_path(processed_root, product, dem_fp.name)
            items.append(
                {
                    "dem_fp": str(dem_fp),
                    "out_fp": str(out_fp),
                    "product": product,
                    "skip_existing": skip_existing,
                    "overwrite": overwrite,
                    "hillshade_azimuth": hillshade_azimuth,
                    "hillshade_altitude": hillshade_altitude,
                    "hillshade_z_factor": hillshade_z_factor,
                    "tpi_radius": tpi_radius,
                    "twi_eps": twi_eps,
                    "dtw_max_distance": dtw_max_distance,
                }
            )

        log_info(f"Terrain product: {product} | DEM tiles={len(items)}")
        run_stage(
            stage_name=f"FAST-GC derive TERRAIN [{product}]",
            items=items,
            worker=_process_dem_for_product,
            item_name_fn=lambda d: Path(d["dem_fp"]).name,
            unit="tile",
            n_jobs=n_jobs,
            backend=joblib_backend,
            batch_size=joblib_batch_size,
            pre_dispatch=joblib_pre_dispatch,
        )

    return str(out_root)


__all__ = [
    "PRODUCT_TERRAIN",
    "TERRAIN_PRODUCT_CHOICES",
    "run_terrain_from_processed_root",
    "compute_slope_percent",
    "compute_slope_degrees",
    "compute_aspect",
    "compute_hillshade",
    "compute_curvature",
    "compute_tpi",
    "compute_twi",
    "compute_dtw",
    "compute_tci",
]