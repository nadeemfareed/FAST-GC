from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import (
    convolve,
    distance_transform_edt,
    maximum_filter,
    minimum_filter,
)

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
    "profile_curvature",
    "tangential_curvature",
    "planform_curvature",
    "mean_curvature",
    "gaussian_curvature",
    "tpi",
    "multiscale_tpi",
    "tri",
    "roughness",
    "local_relief",
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
            "profile_curvature",
            "tangential_curvature",
            "planform_curvature",
            "mean_curvature",
            "gaussian_curvature",
            "tpi",
            "tri",
            "roughness",
            "local_relief",
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


def _write_raster(
    arr: np.ndarray,
    profile: dict,
    out_fp: str,
    nodata=None,
    *,
    tags: dict[str, str] | None = None,
):
    """Write one FAST_TERRAIN raster and optional provenance metadata."""
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

        if tags:
            dst.update_tags(
                **{
                    str(key): str(value)
                    for key, value in tags.items()
                }
            )


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


def _surface_derivatives(
    dem: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return first and second DEM derivatives.

    Returns
    -------
    p, q, r, s, t
        p = dz/dx
        q = dz/dy
        r = d2z/dx2
        s = d2z/dxdy
        t = d2z/dy2

    Notes
    -----
    Derivatives are expressed in the coordinate units represented by dx,
    dy, and DEM elevation. FAST_TERRAIN assumes compatible horizontal and
    vertical metric units for physically interpretable metric derivatives.
    """
    # Curvature depends on second derivatives and is especially sensitive
    # to premature precision loss. Preserve float64 throughout the
    # derivative path rather than using _fill_nan_with_nearest(), whose
    # historical float32 behavior is retained for the existing hydrology
    # implementation until that subsystem is audited separately.
    dem64 = np.asarray(dem, dtype=np.float64)
    valid = np.isfinite(dem64)

    if np.all(valid):
        dem_fill = dem64.copy()
    elif not np.any(valid):
        dem_fill = np.zeros_like(dem64, dtype=np.float64)
    else:
        _, inds = distance_transform_edt(~valid, return_indices=True)
        dem_fill = dem64.copy()
        dem_fill[~valid] = dem64[inds[0][~valid], inds[1][~valid]]

    q, p = np.gradient(dem_fill, float(dy), float(dx))

    r = np.gradient(p, float(dx), axis=1)
    t = np.gradient(q, float(dy), axis=0)

    s_from_p = np.gradient(p, float(dy), axis=0)
    s_from_q = np.gradient(q, float(dx), axis=1)
    s = 0.5 * (s_from_p + s_from_q)

    return (
        p.astype(np.float64, copy=False),
        q.astype(np.float64, copy=False),
        r.astype(np.float64, copy=False),
        s.astype(np.float64, copy=False),
        t.astype(np.float64, copy=False),
    )


def compute_curvature(dem: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Legacy Laplacian curvature retained for backward compatibility.

    This product is z_xx + z_yy. It is not a substitute for profile,
    plan, mean, or Gaussian curvature.
    """
    _p, _q, r, _s, t = _surface_derivatives(dem, dx, dy)
    return (r + t).astype(np.float32)


def compute_profile_curvature(
    dem: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Curvature of the surface in the local gradient direction.

    Positive/negative sign follows the mathematical surface convention
    used by this implementation. Cells with effectively zero gradient
    have no defined profile direction and are returned as NaN.
    """
    p, q, r, s, t = _surface_derivatives(dem, dx, dy)

    g2 = p * p + q * q
    denom = g2 * np.power(1.0 + g2, 1.5)
    numer = r * p * p + 2.0 * s * p * q + t * q * q

    out = np.full(dem.shape, np.nan, dtype=np.float64)
    valid = g2 > 1e-16
    out[valid] = -numer[valid] / denom[valid]

    return out.astype(np.float32)


def compute_tangential_curvature(
    dem: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Geometric tangential curvature of the graph surface z=f(x,y).

    This is the curvature of the normal section tangential to the
    contour line. The convention follows the geometric curvature
    system summarized by Minar et al. (2020).

    Cells with effectively zero gradient have no defined contour
    direction and are returned as NaN.
    """
    p, q, r, s, t = _surface_derivatives(dem, dx, dy)

    g2 = p * p + q * q
    numer = r * q * q - 2.0 * s * p * q + t * p * p
    denom = g2 * np.sqrt(1.0 + g2)

    out = np.full(dem.shape, np.nan, dtype=np.float64)
    valid = g2 > 1e-16

    out[valid] = -numer[valid] / denom[valid]

    return out.astype(np.float32)


def compute_planform_curvature(
    dem: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Planform curvature of the graph surface z=f(x,y).

    Planform curvature is the curvature of the horizontal projection
    of the contour line.

    Cells with effectively zero gradient have no defined contour
    direction and are returned as NaN.
    """
    p, q, r, s, t = _surface_derivatives(dem, dx, dy)

    g2 = p * p + q * q
    numer = r * q * q - 2.0 * s * p * q + t * p * p
    denom = np.power(g2, 1.5)

    out = np.full(dem.shape, np.nan, dtype=np.float64)
    valid = g2 > 1e-16

    out[valid] = -numer[valid] / denom[valid]

    return out.astype(np.float32)

def compute_mean_curvature(
    dem: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Mean curvature using the geomorphometric sign convention.

    Positive values represent locally convex terrain and negative values
    locally concave terrain.  The magnitude is the geometric mean curvature
    of the graph surface z=f(x,y); the leading minus sign selects the
    geographical/geomorphometric surface-normal convention used by the
    directional curvature products in FAST_TERRAIN.
    """
    p, q, r, s, t = _surface_derivatives(dem, dx, dy)

    g2 = p * p + q * q
    numer = (
        (1.0 + q * q) * r
        - 2.0 * p * q * s
        + (1.0 + p * p) * t
    )
    denom = 2.0 * np.power(1.0 + g2, 1.5)

    return (-numer / denom).astype(np.float32)


def compute_gaussian_curvature(
    dem: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Gaussian curvature of the graph surface z=f(x,y)."""
    p, q, r, s, t = _surface_derivatives(dem, dx, dy)

    g2 = p * p + q * q
    numer = r * t - s * s
    denom = np.power(1.0 + g2, 2.0)

    return (numer / denom).astype(np.float32)


def _finite_neighborhood_sum(
    arr: np.ndarray,
    kernel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite-value neighborhood sum and finite sample count."""
    arr64 = np.asarray(arr, dtype=np.float64)
    valid = np.isfinite(arr64)

    values = np.where(valid, arr64, 0.0)

    total = convolve(
        values,
        kernel,
        mode="constant",
        cval=0.0,
    )

    count = convolve(
        valid.astype(np.float64),
        kernel,
        mode="constant",
        cval=0.0,
    )

    return total, count


def compute_tpi(dem: np.ndarray, radius: int = 3) -> np.ndarray:
    """Topographic Position Index.

    TPI is focal elevation minus the mean elevation of the surrounding
    finite cells. The focal cell itself is excluded.
    """
    arr = np.asarray(dem, dtype=np.float64)
    radius = max(1, int(radius))

    size = 2 * radius + 1
    kernel = np.ones((size, size), dtype=np.float64)
    kernel[radius, radius] = 0.0

    total, count = _finite_neighborhood_sum(arr, kernel)

    mean = np.divide(
        total,
        count,
        out=np.full(arr.shape, np.nan, dtype=np.float64),
        where=count > 0.0,
    )

    out = arr - mean
    out[~np.isfinite(arr)] = np.nan

    return out.astype(np.float32)



def _circular_metric_kernel(
    radius_m: float,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Circular Euclidean neighborhood in physical map units."""
    radius_m = float(radius_m)
    dx = abs(float(dx))
    dy = abs(float(dy))

    if not np.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("TPI physical radius must be finite and > 0.")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("DEM pixel sizes must be > 0.")

    rx = int(np.ceil(radius_m / dx))
    ry = int(np.ceil(radius_m / dy))

    x = np.arange(-rx, rx + 1, dtype=np.float64) * dx
    y = np.arange(-ry, ry + 1, dtype=np.float64) * dy
    xx, yy = np.meshgrid(x, y)

    kernel = (
        (xx * xx + yy * yy)
        <= (radius_m * radius_m + 1.0e-12)
    ).astype(np.float64)

    # Exclude focal cell from surrounding-terrain mean.
    kernel[ry, rx] = 0.0

    return kernel


def compute_multiscale_tpi(
    dem: np.ndarray,
    dx: float,
    dy: float,
    radius_m: float,
) -> np.ndarray:
    """Physical-radius circular Topographic Position Index."""
    arr = np.asarray(dem, dtype=np.float64)
    valid = np.isfinite(arr)

    kernel = _circular_metric_kernel(
        radius_m,
        dx,
        dy,
    )

    values = np.where(valid, arr, 0.0)

    total = fftconvolve(
        values,
        kernel,
        mode="same",
    )

    count = fftconvolve(
        valid.astype(np.float64),
        kernel,
        mode="same",
    )

    # Recover exact integer sample counts from FFT roundoff.
    count = np.rint(count)

    mean = np.divide(
        total,
        count,
        out=np.full(arr.shape, np.nan, dtype=np.float64),
        where=count > 0.0,
    )

    out = arr - mean
    out[~valid] = np.nan

    return out.astype(np.float32)


def compute_tri(dem: np.ndarray) -> np.ndarray:
    """Riley Terrain Ruggedness Index using a 3x3 neighborhood.

    TRI = sqrt(sum((z_neighbor - z_center)^2)) over finite surrounding
    cells. The focal cell itself is excluded.
    """
    arr = np.asarray(dem, dtype=np.float64)
    valid = np.isfinite(arr)

    kernel = np.ones((3, 3), dtype=np.float64)
    kernel[1, 1] = 0.0

    values = np.where(valid, arr, 0.0)
    values_sq = values * values

    sum_z, count = _finite_neighborhood_sum(arr, kernel)

    sum_z2 = convolve(
        values_sq,
        kernel,
        mode="constant",
        cval=0.0,
    )

    # sum((zi-z0)^2)
    # = sum(zi^2) - 2*z0*sum(zi) + n*z0^2
    center = np.where(valid, arr, 0.0)

    ss = (
        sum_z2
        - 2.0 * center * sum_z
        + count * center * center
    )

    # Protect against tiny negative roundoff.
    ss = np.maximum(ss, 0.0)

    out = np.sqrt(ss)
    out[(~valid) | (count <= 0.0)] = np.nan

    return out.astype(np.float32)


def compute_roughness(dem: np.ndarray) -> np.ndarray:
    """Terrain roughness as the elevation range of a 3x3 neighborhood.

    Roughness is the difference between the maximum and minimum finite
    elevations in the 3x3 window centered on each cell:

        roughness = max(z_3x3) - min(z_3x3)

    This is the Wilson et al./GDAL terrain-roughness definition.  With
    compatible metric horizontal and vertical coordinates, output elevation
    differences are expressed in the DEM vertical unit.

    Missing neighbors are ignored where finite neighbors remain; the original
    DEM nodata footprint is restored to NaN by the terrain processing path.
    """
    arr = np.asarray(dem, dtype=np.float64)
    valid = np.isfinite(arr)

    hi_source = np.where(valid, arr, -np.inf)
    lo_source = np.where(valid, arr, np.inf)

    local_max = maximum_filter(
        hi_source,
        size=3,
        mode="constant",
        cval=-np.inf,
    )

    local_min = minimum_filter(
        lo_source,
        size=3,
        mode="constant",
        cval=np.inf,
    )

    out = local_max - local_min
    out[~valid] = np.nan

    return out.astype(np.float32)


def compute_local_relief(
    dem: np.ndarray,
    radius: int = 3,
) -> np.ndarray:
    """Local elevation range within a configurable square neighborhood."""
    arr = np.asarray(dem, dtype=np.float64)
    valid = np.isfinite(arr)

    radius = max(1, int(radius))
    size = 2 * radius + 1

    hi_source = np.where(valid, arr, -np.inf)
    lo_source = np.where(valid, arr, np.inf)

    local_max = maximum_filter(
        hi_source,
        size=size,
        mode="constant",
        cval=-np.inf,
    )

    local_min = minimum_filter(
        lo_source,
        size=size,
        mode="constant",
        cval=np.inf,
    )

    out = local_max - local_min
    out[~valid] = np.nan

    return out.astype(np.float32)

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
    multiscale_tpi_radius_m: float | None,
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
    if product == "profile_curvature":
        return compute_profile_curvature(dem, dx, dy)
    if product == "tangential_curvature":
        return compute_tangential_curvature(dem, dx, dy)
    if product == "planform_curvature":
        return compute_planform_curvature(dem, dx, dy)
    if product == "mean_curvature":
        return compute_mean_curvature(dem, dx, dy)
    if product == "gaussian_curvature":
        return compute_gaussian_curvature(dem, dx, dy)
    if product == "tpi":
        return compute_tpi(dem, radius=tpi_radius)
    if product == "multiscale_tpi":
        if multiscale_tpi_radius_m is None:
            raise ValueError(
                "multiscale_tpi requires a physical radius in metres."
            )
        return compute_multiscale_tpi(
            dem,
            dx,
            dy,
            radius_m=multiscale_tpi_radius_m,
        )
    if product == "tri":
        return compute_tri(dem)
    if product == "roughness":
        return compute_roughness(dem)
    if product == "local_relief":
        return compute_local_relief(dem, radius=tpi_radius)
    if product == "twi":
        return compute_twi(dem, dx, dy, eps=twi_eps)
    if product == "dtw":
        return compute_dtw(dem, dx, dy, max_distance=dtw_max_distance)
    if product == "tci":
        return compute_tci(dem, dx, dy)
    raise ValueError(f"Unsupported terrain product: {product}")


FAST_TERRAIN_SCHEMA = "1"


def _terrain_provenance(item: dict[str, Any]) -> dict[str, str]:
    """Return deterministic provenance tags for one terrain computation."""
    dtw_max_distance = item.get("dtw_max_distance")

    return {
        "FASTGC_PRODUCT": "FAST_TERRAIN",
        "FASTGC_TERRAIN_PRODUCT": str(item["product"]),
        "FASTGC_ANALYTICAL_DOMAIN": str(
            item.get("analytical_domain", "unspecified")
        ),
        "FASTGC_SOURCE_DEM": Path(item["dem_fp"]).name,
        "FASTGC_TERRAIN_SCHEMA": FAST_TERRAIN_SCHEMA,
        "FASTGC_HILLSHADE_AZIMUTH": str(
            float(item["hillshade_azimuth"])
        ),
        "FASTGC_HILLSHADE_ALTITUDE": str(
            float(item["hillshade_altitude"])
        ),
        "FASTGC_HILLSHADE_Z_FACTOR": str(
            float(item["hillshade_z_factor"])
        ),
        "FASTGC_TPI_RADIUS_CELLS": str(
            int(item["tpi_radius"])
        ),
        "FASTGC_MULTISCALE_TPI_RADIUS_M": (
            "none"
            if item.get("multiscale_tpi_radius_m") is None
            else str(float(item["multiscale_tpi_radius_m"]))
        ),
        "FASTGC_MULTISCALE_TPI_GEOMETRY": (
            "euclidean_circle_focal_excluded"
        ),
        "FASTGC_TWI_EPS": str(
            float(item["twi_eps"])
        ),
        "FASTGC_DTW_MAX_DISTANCE": (
            "none"
            if dtw_max_distance is None
            else str(float(dtw_max_distance))
        ),
    }


def _terrain_provenance_matches(
    out_fp: Path,
    expected: dict[str, str],
) -> bool:
    """Return True only when an existing raster has matching provenance."""
    if not out_fp.exists() or not out_fp.is_file():
        return False

    try:
        with rasterio.open(out_fp) as src:
            actual = src.tags()
    except Exception:
        return False

    return all(
        actual.get(key) == str(value)
        for key, value in expected.items()
    )


def _process_dem_for_product(
    item: dict[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    dem_fp = Path(item["dem_fp"])
    out_fp = Path(item["out_fp"])
    skip_existing = bool(item["skip_existing"])
    overwrite = bool(item["overwrite"])

    provenance = _terrain_provenance(item)

    if (
        out_fp.exists()
        and out_fp.is_file()
        and skip_existing
        and not overwrite
        and not force
        and _terrain_provenance_matches(
            out_fp,
            provenance,
        )
    ):
        return {
            "status": "skipped",
            "path": str(out_fp),
            "name": dem_fp.name,
        }

    dem, profile, transform, nodata = _read_dem(
        str(dem_fp)
    )

    valid_mask = _dem_valid_mask(
        dem,
        nodata,
    )

    dx, dy = _pixel_size(transform)

    arr = _compute_terrain_array(
        item["product"],
        dem,
        dx,
        dy,
        hillshade_azimuth=float(
            item["hillshade_azimuth"]
        ),
        hillshade_altitude=float(
            item["hillshade_altitude"]
        ),
        hillshade_z_factor=float(
            item["hillshade_z_factor"]
        ),
        tpi_radius=int(
            item["tpi_radius"]
        ),
        multiscale_tpi_radius_m=item.get(
            "multiscale_tpi_radius_m"
        ),
        twi_eps=float(
            item["twi_eps"]
        ),
        dtw_max_distance=item[
            "dtw_max_distance"
        ],
    )

    arr = _apply_valid_mask(
        arr,
        valid_mask,
        nodata,
    )

    _write_raster(
        arr,
        profile,
        str(out_fp),
        nodata=nodata,
        tags=provenance,
    )

    return {
        "status": "ok",
        "path": str(out_fp),
        "name": dem_fp.name,
    }



def _terrain_product_jobs(
    requested: list[str],
    multiscale_tpi_radii_m: tuple[float, ...],
) -> list[tuple[str, float | None, str]]:
    """Expand terrain products into deterministic output jobs."""
    jobs: list[tuple[str, float | None, str]] = []

    for product in requested:
        if product != "multiscale_tpi":
            jobs.append((product, None, product))
            continue

        radii = []
        seen = set()

        for radius in multiscale_tpi_radii_m:
            radius = float(radius)

            if not np.isfinite(radius) or radius <= 0.0:
                raise ValueError(
                    "Multiscale TPI radii must be finite and > 0 metres."
                )

            if radius in seen:
                continue

            seen.add(radius)
            radii.append(radius)

        if not radii:
            raise ValueError(
                "At least one multiscale TPI radius is required."
            )

        for radius in radii:
            label = (
                f"{radius:g}".replace(".", "p")
            )
            jobs.append(
                (
                    product,
                    radius,
                    f"multiscale_tpi_{label}m",
                )
            )

    return jobs


def run_terrain_from_dem(
    dem_fp: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    terrain_products: list[str] | None = None,
    hillshade_azimuth: float = 315.0,
    hillshade_altitude: float = 45.0,
    hillshade_z_factor: float = 1.0,
    tpi_radius: int = 3,
    multiscale_tpi_radii_m: tuple[float, ...] = (5.0, 10.0, 25.0, 50.0, 100.0),
    twi_eps: float = 1e-6,
    dtw_max_distance: float | None = None,
    skip_existing: bool = False,
    overwrite: bool = False,
    n_jobs: int | None = None,
    joblib_backend: str = "loky",
    joblib_batch_size: int | str = "auto",
    joblib_pre_dispatch: str = "2*n_jobs",
) -> str:
    """Derive FAST_TERRAIN products from one explicit DEM raster.

    This entry point is intended for analytical domains that are already
    represented by a single authoritative DEM, including the continuous
    merged DEM produced by tile-run-merge.

    The terrain mathematics is identical to tile processing because all
    computation is delegated to _process_dem_for_product().
    """
    _require_rasterio()

    dem_fp = Path(dem_fp)
    if not dem_fp.exists() or not dem_fp.is_file():
        raise FileNotFoundError(f"DEM raster not found: {dem_fp}")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    requested = _resolve_terrain_products(terrain_products)

    stage_banner(
        "FAST_TERRAIN",
        source=str(dem_fp),
        total=1,
        unit="DEM",
        extra=f"products={', '.join(requested)}",
    )

    product_jobs = _terrain_product_jobs(
        requested,
        multiscale_tpi_radii_m,
    )

    for product, multiscale_radius_m, output_label in product_jobs:
        out_dir = output_root / output_label
        out_dir.mkdir(parents=True, exist_ok=True)

        # Continuous-domain terrain keeps the established public merged
        # naming convention:
        #
        #   <dataset>_FAST_TERRAIN_<product>.tif
        #
        # The source DEM is normally:
        #
        #   <dataset>_FAST_DEM.tif
        #
        # Only the output filename changes; computation still uses the
        # authoritative merged DEM through _process_dem_for_product().
        dem_stem = dem_fp.stem
        if dem_stem.endswith("_FAST_DEM"):
            dataset_stem = dem_stem[:-len("_FAST_DEM")]
        else:
            dataset_stem = dem_stem

        out_fp = (
            out_dir
            / f"{dataset_stem}_FAST_TERRAIN_{output_label}{dem_fp.suffix}"
        )

        item = {
            "dem_fp": str(dem_fp),
            "out_fp": str(out_fp),
            "product": product,
            "analytical_domain": "continuous_merged_dem",
            "skip_existing": skip_existing,
            "overwrite": overwrite,
            "hillshade_azimuth": hillshade_azimuth,
            "hillshade_altitude": hillshade_altitude,
            "hillshade_z_factor": hillshade_z_factor,
            "tpi_radius": tpi_radius,
            "multiscale_tpi_radius_m": multiscale_radius_m,
            "twi_eps": twi_eps,
            "dtw_max_distance": dtw_max_distance,
        }

        log_info(
            f"Terrain product: {product} | "
            f"DEM={dem_fp.name}"
        )

        run_stage(
            stage_name=f"FAST-GC derive TERRAIN [{product}]",
            items=[item],
            worker=_process_dem_for_product,
            item_name_fn=lambda d: Path(d["dem_fp"]).name,
            unit="DEM",
            n_jobs=n_jobs,
            backend=joblib_backend,
            batch_size=joblib_batch_size,
            pre_dispatch=joblib_pre_dispatch,
        )

    return str(output_root)


def run_terrain_from_processed_root(
    processed_root: str | os.PathLike[str],
    *,
    terrain_products: list[str] | None = None,
    hillshade_azimuth: float = 315.0,
    hillshade_altitude: float = 45.0,
    hillshade_z_factor: float = 1.0,
    tpi_radius: int = 3,
    multiscale_tpi_radii_m: tuple[float, ...] = (5.0, 10.0, 25.0, 50.0, 100.0),
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

    product_jobs = _terrain_product_jobs(
        requested,
        multiscale_tpi_radii_m,
    )

    for product, multiscale_radius_m, output_label in product_jobs:
        items: list[dict[str, Any]] = []
        for dem_fp in dem_files:
            out_fp = _terrain_output_path(
                processed_root,
                output_label,
                dem_fp.name,
            )
            items.append(
                {
                    "dem_fp": str(dem_fp),
                    "out_fp": str(out_fp),
                    "product": product,
                    "analytical_domain": "buffered_dem_tile",
                    "skip_existing": skip_existing,
                    "overwrite": overwrite,
                    "hillshade_azimuth": hillshade_azimuth,
                    "hillshade_altitude": hillshade_altitude,
                    "hillshade_z_factor": hillshade_z_factor,
                    "tpi_radius": tpi_radius,
                    "multiscale_tpi_radius_m": multiscale_radius_m,
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
    "run_terrain_from_dem",
    "compute_slope_percent",
    "compute_slope_degrees",
    "compute_aspect",
    "compute_hillshade",
    "compute_curvature",
    "compute_tpi",
    "compute_multiscale_tpi",
    "compute_twi",
    "compute_dtw",
    "compute_tci",
]