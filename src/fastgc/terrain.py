from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

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


def _existing_output(path: Path) -> bool:
    return path.exists() and path.is_file()


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


def _nanmean_filter(arr: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return arr.copy()

    h, w = arr.shape
    out = np.full_like(arr, np.nan, dtype=np.float32)

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
    out = arr.copy()
    if not np.any(~np.isfinite(out)):
        return out

    mask = ~np.isfinite(out)
    if np.all(mask):
        return np.zeros_like(out, dtype=np.float32)

    for _ in range(max(out.shape)):
        if not np.any(mask):
            break

        updated = out.copy()
        h, w = out.shape
        for r in range(h):
            r0 = max(0, r - 1)
            r1 = min(h, r + 2)
            for c in range(w):
                if not mask[r, c]:
                    continue
                c0 = max(0, c - 1)
                c1 = min(w, c + 2)
                win = out[r0:r1, c0:c1]
                valid = win[np.isfinite(win)]
                if valid.size:
                    updated[r, c] = float(valid[0])
        out = updated
        mask = ~np.isfinite(out)

    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def _gradients(dem: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    dem_fill = _fill_nan_with_nearest(dem)
    dzdy, dzdx = np.gradient(dem_fill, dy, dx)
    return dzdx.astype(np.float32), dzdy.astype(np.float32)


def compute_slope_percent(dem: np.ndarray, dx: float, dy: float) -> np.ndarray:
    dzdx, dzdy = _gradients(dem, dx, dy)
    slope = np.sqrt(dzdx**2 + dzdy**2)
    return (slope * 100.0).astype(np.float32)


def compute_slope_degrees(dem: np.ndarray, dx: float, dy: float) -> np.ndarray:
    dzdx, dzdy = _gradients(dem, dx, dy)
    slope = np.sqrt(dzdx**2 + dzdy**2)
    return np.degrees(np.arctan(slope)).astype(np.float32)


def compute_aspect(dem: np.ndarray, dx: float, dy: float) -> np.ndarray:
    dzdx, dzdy = _gradients(dem, dx, dy)
    aspect = np.degrees(np.arctan2(dzdy, -dzdx))
    aspect = 90.0 - aspect
    aspect = np.where(aspect < 0, aspect + 360.0, aspect)
    aspect = np.where(aspect >= 360.0, aspect - 360.0, aspect)
    slope = np.sqrt(dzdx**2 + dzdy**2)
    aspect = np.where(slope > 0, aspect, 0.0)
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
    curvature = d2zdx2 + d2zdy2
    return curvature.astype(np.float32)


def compute_tpi(dem: np.ndarray, radius: int = 3) -> np.ndarray:
    mean_local = _nanmean_filter(dem, radius=max(1, int(radius)))
    tpi = dem - mean_local
    return tpi.astype(np.float32)


def compute_twi(dem: np.ndarray, dx: float, dy: float, eps: float = 1e-6) -> np.ndarray:
    slope_deg = compute_slope_degrees(dem, dx, dy)
    slope_rad = np.radians(np.maximum(slope_deg, 0.001))
    tan_beta = np.tan(slope_rad)
    curv = compute_curvature(dem, dx, dy)
    acc_proxy = np.maximum(0.0, -curv) + 1.0
    twi = np.log((acc_proxy + float(eps)) / np.maximum(tan_beta, float(eps)))
    return twi.astype(np.float32)


def compute_tci(dem: np.ndarray, dx: float, dy: float) -> np.ndarray:
    curv = compute_curvature(dem, dx, dy)
    slope = compute_slope_percent(dem, dx, dy)
    tci = (-curv) / (1.0 + slope / 100.0)
    return tci.astype(np.float32)


def compute_dtw(dem: np.ndarray, dx: float, dy: float, max_distance: float | None = None) -> np.ndarray:
    dem_fill = _fill_nan_with_nearest(dem)
    local_min = _nanmean_filter(dem_fill, radius=3)
    dtw = np.maximum(0.0, dem_fill - np.minimum(dem_fill, local_min))
    if max_distance is not None:
        dtw = np.minimum(dtw, float(max_distance))
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
