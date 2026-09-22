
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


@dataclass(frozen=True)
class PostFinalSurfaceConfig:
    enabled: bool = True
    cell_m: float = 0.50
    p90_quantile: float = 0.90
    smooth_sigma_cells: float = 2.0

    # Point-to-cell-min tolerance, made looser only as slope increases.
    base_tol_m: float = 0.30
    slope_start_deg: float = 10.0
    slope_gain_m_per_deg: float = 0.020
    max_tol_m: float = 1.00

    # "Thin sheet" condition after subtracting slope-explained vertical span.
    max_sheet_excess_m: float = 0.25

    # Extremely tall columns are not trusted unless their vertical span is
    # already explained by the local terrain slope.
    hard_vertical_span_m: float = 8.0
    hard_span_explained_fraction: float = 0.80


def _cfg(cfg: dict[str, Any] | None) -> PostFinalSurfaceConfig:
    cfg = cfg or {}
    return PostFinalSurfaceConfig(
        enabled=bool(cfg.get("als_postfinal_surface_enabled", True)),
        cell_m=float(cfg.get("als_postfinal_surface_cell_m", 0.50)),
        p90_quantile=float(cfg.get("als_postfinal_surface_p90_quantile", 0.90)),
        smooth_sigma_cells=float(cfg.get("als_postfinal_surface_smooth_sigma_cells", 2.0)),
        base_tol_m=float(cfg.get("als_postfinal_surface_base_tol_m", 0.30)),
        slope_start_deg=float(cfg.get("als_postfinal_surface_slope_start_deg", 10.0)),
        slope_gain_m_per_deg=float(cfg.get("als_postfinal_surface_slope_gain_m_per_deg", 0.020)),
        max_tol_m=float(cfg.get("als_postfinal_surface_max_tol_m", 1.00)),
        max_sheet_excess_m=float(cfg.get("als_postfinal_surface_max_sheet_excess_m", 0.25)),
        hard_vertical_span_m=float(cfg.get("als_postfinal_surface_hard_vertical_span_m", 8.0)),
        hard_span_explained_fraction=float(cfg.get("als_postfinal_surface_hard_span_explained_fraction", 0.80)),
    )


def _fill_nearest(grid: np.ndarray) -> np.ndarray:
    valid = np.isfinite(grid)
    if np.all(valid):
        return grid.copy()
    if not np.any(valid):
        return np.zeros_like(grid, dtype=np.float64)
    _, inds = distance_transform_edt(~valid, return_indices=True)
    return grid[tuple(inds)]


def recover_postfinal_surface(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ground_mask: np.ndarray,
    sensor_mode: str,
    cfg: dict[str, Any] | None = None,
    return_report: bool = True,
):
    """
    ALS/ULS aerial, promotion-only final terrain completion.

    IMPORTANT:
    This function is intended to run AFTER all existing FAST-GC final terrain
    QC / two-way cleanup and immediately BEFORE Classification is serialized.

    It builds a 0.5 m raw lower surface from ALL XYZ points. Points rejected by
    the main classifier are recovered only when they occupy the thin lower sheet
    of a slope-consistent cell.

    Existing ground is never demoted.
    TLS is untouched. ALS and ULS use sensor-scaled configuration.
    """
    sm = str(sensor_mode).upper().strip()
    original = np.asarray(ground_mask, dtype=bool)
    out = original.copy()

    report = {
        "enabled": False,
        "sensor_mode": sm,
        "ground_before": int(np.count_nonzero(original)),
        "ground_after": int(np.count_nonzero(original)),
        "recovered_points": 0,
        "recovered_cells": 0,
        "candidate_points": 0,
        "candidate_cells": 0,
    }

    if sm not in {"ALS", "ULS"}:
        return (out, report) if return_report else out

    c = _cfg(cfg)
    report["enabled"] = bool(c.enabled)
    if not c.enabled:
        return (out, report) if return_report else out

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if x.size == 0 or x.size != original.size or not np.any(valid):
        return (out, report) if return_report else out

    wi = np.flatnonzero(valid)
    xv, yv, zv = x[wi], y[wi], z[wi]
    gv = out[wi].copy()

    cell = max(float(c.cell_m), 0.25)
    x0 = float(np.floor(np.min(xv) / cell) * cell)
    y0 = float(np.floor(np.min(yv) / cell) * cell)

    ix = np.floor((xv - x0) / cell).astype(np.int32)
    iy = np.floor((yv - y0) / cell).astype(np.int32)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1

    key = iy.astype(np.int64) * nx + ix.astype(np.int64)
    ncell = nx * ny

    # One sort provides exact per-cell minimum and P90.
    order = np.lexsort((zv, key))
    ks = key[order]
    zs = zv[order]

    starts = np.r_[0, 1 + np.flatnonzero(ks[1:] != ks[:-1])]
    ends = np.r_[starts[1:], ks.size]
    lens = ends - starts
    keys_u = ks[starts]

    zmin = np.full(ncell, np.nan, dtype=np.float64)
    zp90 = np.full(ncell, np.nan, dtype=np.float64)
    zmin[keys_u] = zs[starts]

    qpos = starts + np.floor(
        (lens - 1) * np.clip(float(c.p90_quantile), 0.5, 0.99)
    ).astype(np.int64)
    zp90[keys_u] = zs[qpos]

    zmin2 = zmin.reshape(ny, nx)
    zp902 = zp90.reshape(ny, nx)

    # Interpolate the raw lower surface only for derivative estimation.
    # Point recovery itself still uses the OBSERVED minimum of its own cell.
    lower_surface = gaussian_filter(
        _fill_nearest(zmin2),
        sigma=max(float(c.smooth_sigma_cells), 0.0),
        mode="nearest",
    )

    gy, gx = np.gradient(lower_surface, cell, cell)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))

    # A steep 0.5 m terrain cell naturally spans Z. Remove the span explained
    # by local slope before deciding whether the cell is volumetric.
    observed_span = zp902 - zmin2
    expected_span = (np.abs(gx) + np.abs(gy)) * cell
    sheet_excess = np.maximum(0.0, observed_span - expected_span)

    # Hard forest-column guard. Very tall vertical columns are allowed only
    # when most of their span is actually explained by terrain slope.
    tall = observed_span > float(c.hard_vertical_span_m)
    explained_fraction = expected_span / np.maximum(observed_span, 1e-6)
    sheet_cell = (
        np.isfinite(zmin2)
        & np.isfinite(sheet_excess)
        & (sheet_excess <= float(c.max_sheet_excess_m))
        & (~tall | (explained_fraction >= float(c.hard_span_explained_fraction)))
    )

    rel = zv - zmin[key]
    slope_p = slope_deg.ravel()[key]
    tol = np.minimum(
        float(c.max_tol_m),
        float(c.base_tol_m)
        + float(c.slope_gain_m_per_deg)
        * np.maximum(slope_p - float(c.slope_start_deg), 0.0),
    )

    candidate = (
        (~gv)
        & sheet_cell.ravel()[key]
        & (rel <= tol)
    )

    report["candidate_points"] = int(np.count_nonzero(candidate))
    if np.any(candidate):
        candidate_cell_count = np.bincount(
            key,
            weights=candidate.astype(np.int8),
            minlength=ncell,
        )
        report["candidate_cells"] = int(np.count_nonzero(candidate_cell_count))

    # Promotion only. Nothing is allowed to demote these afterward because this
    # stage is inserted immediately before final Classification serialization.
    gv[candidate] = True
    out[wi] = gv

    recovered = out & ~original
    report["recovered_points"] = int(np.count_nonzero(recovered))
    if np.any(recovered):
        recovered_cell_count = np.bincount(
            key,
            weights=recovered[wi].astype(np.int8),
            minlength=ncell,
        )
        report["recovered_cells"] = int(np.count_nonzero(recovered_cell_count))

    report["ground_after"] = int(np.count_nonzero(out))
    return (out, report) if return_report else out


__all__ = ["PostFinalSurfaceConfig", "recover_postfinal_surface"]
