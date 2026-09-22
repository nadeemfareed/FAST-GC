from __future__ import annotations

"""
FAST-GC ALS/ULS final surface-impurity guard V4.1.

Purpose
-------
Remove the small detached/near-detached class-2 clusters that create local
pyramids, tents, domes and "pimples" in a 0.5 m triangulated terrain surface.

The guard is DEMOTION ONLY.  It does not use reference labels.  It works from:
  1) a 0.5 m lower-ground raster;
  2) compact local slope anomalies relative to an annular/radial neighborhood;
  3) leave-the-anomaly-out robust plane/quadratic terrain fitting;
  4) terrain-normal point residuals; and
  5) same-surface ground connectivity/support.

Absolute slope is never a rejection criterion.  A steep coherent hillside is
therefore protected when it agrees with its surrounding terrain sheet.
"""

from dataclasses import dataclass
import math
import warnings
from typing import Any

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    label,
    median_filter,
)
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SurfaceImpurityConfig:
    cell_m: float = 0.50
    ring_inner_m: float = 1.00
    ring_outer_m: float = 4.00
    min_ring_cells: int = 14
    min_ring_sectors: int = 5
    max_ring_cells: int = 72

    # Candidate generation: local slope disagreement, not absolute slope.
    slope_window_cells: int = 9
    slope_mad_k: float = 5.0
    slope_delta_min_deg: float = 16.0
    slope_delta_strong_deg: float = 28.0
    candidate_dilate_cells: int = 1
    max_component_area_m2: float = 16.0

    # Terrain-normal separation.
    normal_residual_min_m: float = 0.18
    normal_residual_strong_m: float = 0.45
    normal_mad_k: float = 4.0
    normal_tol_cap_m: float = 0.35

    # Same-sheet support.  This is evaluated in residual space so steep slopes
    # do not look disconnected merely because Z changes rapidly with XY.
    point_support_radius_m: float = 1.25
    same_surface_band_m: float = 0.12
    min_same_surface_ground: int = 3

    # A very detached small cluster may be removed without a non-ground vote,
    # but only when the annular terrain model is strong.
    hard_detached_m: float = 0.60
    hard_max_component_area_m2: float = 9.0

    # Optional corroboration for the harder near-surface case.
    nonground_vote_radius_m: float = 1.25
    nonground_same_surface_band_m: float = 0.16
    nonground_fraction_min: float = 0.55
    nonground_min_neighbors: int = 3

    # Curvature safeguard.
    plane_curve_rmse_m: float = 0.08
    quadratic_improvement_ratio: float = 0.88

    # Iteration is deliberately short; each pass rebuilds the lower terrain after
    # confirmed contamination has been removed.
    passes: int = 2
    max_demote_fraction_per_pass: float = 0.025


def _cfg(sensor_mode: str, cfg: dict | None) -> SurfaceImpurityConfig:
    cfg = cfg or {}
    sm = str(sensor_mode).upper().strip()
    d = dict(SurfaceImpurityConfig().__dict__)
    if sm == "ULS":
        d.update(
            ring_outer_m=3.25,
            min_ring_cells=18,
            normal_residual_min_m=0.14,
            normal_residual_strong_m=0.34,
            hard_detached_m=0.48,
            point_support_radius_m=1.0,
            nonground_vote_radius_m=1.0,
            same_surface_band_m=0.10,
        )
    for k, v in list(d.items()):
        key = f"surface_impurity_{k}"
        if key in cfg:
            if isinstance(v, bool):
                d[k] = bool(cfg[key])
            elif isinstance(v, int):
                d[k] = int(cfg[key])
            else:
                d[k] = float(cfg[key])
    return SurfaceImpurityConfig(**d)


def _sector_count(dx: np.ndarray, dy: np.ndarray) -> int:
    if dx.size == 0:
        return 0
    ang = (np.arctan2(dy, dx) + 2.0 * np.pi) % (2.0 * np.pi)
    return int(np.unique(np.floor(ang / (np.pi / 4.0)).astype(np.int8)).size)


def _robust_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if int(keep.sum()) < 6:
        return None
    xc = float(np.median(x[keep]))
    yc = float(np.median(y[keep]))
    xx = x - xc
    yy = y - yc
    coef = None
    for _ in range(5):
        if int(keep.sum()) < 6:
            return None
        A = np.column_stack((xx[keep], yy[keep], np.ones(int(keep.sum()))))
        try:
            coef, *_ = np.linalg.lstsq(A, z[keep], rcond=None)
        except np.linalg.LinAlgError:
            return None
        r = z - (coef[0] * xx + coef[1] * yy + coef[2])
        med = float(np.median(r[keep]))
        mad = max(1.4826 * float(np.median(np.abs(r[keep] - med))), 0.02)
        # Asymmetric terrain fit: high returns are less trusted.
        nk = keep & (r >= med - 4.0 * mad) & (r <= med + 2.25 * mad)
        if int(nk.sum()) < 6 or np.array_equal(nk, keep):
            break
        keep = nk
    if coef is None:
        return None
    pred = coef[0] * xx[keep] + coef[1] * yy[keep] + coef[2]
    rmse = float(np.sqrt(np.mean((z[keep] - pred) ** 2)))
    return float(coef[0]), float(coef[1]), float(coef[2]), xc, yc, rmse


def _robust_quadratic(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if int(keep.sum()) < 12:
        return None
    xc = float(np.median(x[keep]))
    yc = float(np.median(y[keep]))
    xx = x - xc
    yy = y - yc
    coef = None
    for _ in range(5):
        if int(keep.sum()) < 12:
            return None
        A = np.column_stack(
            (
                np.ones(int(keep.sum())),
                xx[keep], yy[keep],
                xx[keep] ** 2,
                xx[keep] * yy[keep],
                yy[keep] ** 2,
            )
        )
        try:
            coef, *_ = np.linalg.lstsq(A, z[keep], rcond=None)
        except np.linalg.LinAlgError:
            return None
        pred = (
            coef[0] + coef[1] * xx + coef[2] * yy
            + coef[3] * xx ** 2 + coef[4] * xx * yy + coef[5] * yy ** 2
        )
        r = z - pred
        med = float(np.median(r[keep]))
        mad = max(1.4826 * float(np.median(np.abs(r[keep] - med))), 0.02)
        nk = keep & (r >= med - 4.0 * mad) & (r <= med + 2.25 * mad)
        if int(nk.sum()) < 12 or np.array_equal(nk, keep):
            break
        keep = nk
    if coef is None:
        return None
    pred = (
        coef[0] + coef[1] * xx[keep] + coef[2] * yy[keep]
        + coef[3] * xx[keep] ** 2
        + coef[4] * xx[keep] * yy[keep]
        + coef[5] * yy[keep] ** 2
    )
    rmse = float(np.sqrt(np.mean((z[keep] - pred) ** 2)))
    return np.asarray(coef, float), xc, yc, rmse


def _predict(model, x: np.ndarray, y: np.ndarray):
    kind, obj = model
    if kind == "plane":
        a, b, k, xc, yc, _ = obj
        pred = a * (x - xc) + b * (y - yc) + k
        scale = np.full(np.shape(pred), math.sqrt(1.0 + a * a + b * b), dtype=float)
        return pred, scale
    coef, xc, yc, _ = obj
    xx = x - xc
    yy = y - yc
    pred = (
        coef[0] + coef[1] * xx + coef[2] * yy
        + coef[3] * xx ** 2 + coef[4] * xx * yy + coef[5] * yy ** 2
    )
    gx = coef[1] + 2.0 * coef[3] * xx + coef[4] * yy
    gy = coef[2] + coef[4] * xx + 2.0 * coef[5] * yy
    return pred, np.sqrt(1.0 + gx * gx + gy * gy)


def _grid(x: np.ndarray, y: np.ndarray, cell: float):
    x0 = float(np.min(x))
    y0 = float(np.min(y))
    ix = np.floor((x - x0) / cell).astype(np.int32)
    iy = np.floor((y - y0) / cell).astype(np.int32)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1
    key = iy.astype(np.int64) * nx + ix.astype(np.int64)
    order = np.argsort(key, kind="mergesort")
    ks = key[order]
    starts = np.r_[0, 1 + np.flatnonzero(ks[1:] != ks[:-1])] if ks.size else np.empty(0, int)
    ends = np.r_[starts[1:], len(ks)] if starts.size else np.empty(0, int)
    occ = ks[starts].astype(np.int64, copy=False) if starts.size else np.empty(0, np.int64)
    start = np.full(nx * ny, -1, np.int64)
    end = np.full(nx * ny, -1, np.int64)
    if occ.size:
        start[occ] = starts
        end[occ] = ends
    return x0, y0, ix, iy, key, nx, ny, order, occ, start, end


def _cell_points(k: int, order, start, end):
    a = int(start[k])
    if a < 0:
        return np.empty(0, np.int64)
    return order[a:int(end[k])]


def _lower_ground_surface(z, g, nx, ny, occ, order, start, end):
    surf = np.full(nx * ny, np.nan, float)
    for k in occ:
        ii = _cell_points(int(k), order, start, end)
        gi = ii[g[ii]]
        if gi.size == 0:
            continue
        zz = z[gi]
        # Lower representative makes the terrain surface resistant to a few
        # elevated class-2 points in a mixed cell.
        surf[int(k)] = float(np.min(zz) if zz.size <= 2 else np.quantile(zz, 0.12))
    return surf.reshape(ny, nx)


def _slope_deg(surface: np.ndarray, cell: float):
    """Slope only where immediate finite support exists; no global interpolation."""
    s = np.asarray(surface, float)
    valid = np.isfinite(s)
    # Fill holes only for derivative calculation with a short iterative neighbor
    # median.  Candidate cells themselves retain their observed lower-ground value.
    f = s.copy()
    for _ in range(3):
        miss = ~np.isfinite(f)
        if not np.any(miss):
            break
        vals = []
        for dy, dx in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
            vals.append(np.roll(np.roll(f, dy, axis=0), dx, axis=1))
        stack = np.stack(vals)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            with np.errstate(all="ignore"):
                med = np.nanmedian(stack, axis=0)
        fill = miss & np.isfinite(med)
        f[fill] = med[fill]
    if min(f.shape) < 3:
        return np.full_like(f, np.nan)
    gy, gx = np.gradient(f, cell, cell)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    slope[~valid] = np.nan
    return slope


def _candidate_slope_cells(slope: np.ndarray, c: SurfaceImpurityConfig):
    finite = np.isfinite(slope)
    if not np.any(finite):
        return np.zeros_like(finite), np.full_like(slope, np.nan), np.full_like(slope, np.nan)
    # Median local slope supplies the regional expectation.  MAD makes the
    # threshold relative to roughness rather than a fixed "steep = bad" rule.
    fill = np.where(finite, slope, 0.0)
    med = median_filter(fill, size=c.slope_window_cells, mode="nearest")
    absdev = np.abs(fill - med)
    mad = 1.4826 * median_filter(absdev, size=c.slope_window_cells, mode="nearest")
    delta = slope - med
    thr = np.maximum(c.slope_delta_min_deg, c.slope_mad_k * np.maximum(mad, 1.0))
    cand = finite & (delta >= thr)
    return cand, delta, med


def _fit_annular_model(
    gy: int, gx: int, component: np.ndarray,
    surf: np.ndarray, x0: float, y0: float, cell: float,
    c: SurfaceImpurityConfig,
):
    yy, xx = np.nonzero(np.isfinite(surf))
    if xx.size < c.min_ring_cells:
        return None
    px = x0 + (gx + 0.5) * cell
    py = y0 + (gy + 0.5) * cell
    sx = x0 + (xx.astype(float) + 0.5) * cell
    sy = y0 + (yy.astype(float) + 0.5) * cell
    rr = np.hypot(sx - px, sy - py)
    use = (rr >= c.ring_inner_m) & (rr <= c.ring_outer_m)

    # Explicitly leave the entire anomaly component and one-cell halo out.
    halo = binary_dilation(component, iterations=1)
    use &= ~halo[yy, xx]
    ids = np.flatnonzero(use)
    if ids.size < c.min_ring_cells:
        return None
    if _sector_count(sx[ids] - px, sy[ids] - py) < c.min_ring_sectors:
        return None
    if ids.size > c.max_ring_cells:
        ids = ids[np.argsort(rr[ids])[:c.max_ring_cells]]

    pl = _robust_plane(sx[ids], sy[ids], surf[yy[ids], xx[ids]])
    if pl is None:
        return None
    model = ("plane", pl)
    ring_pred, ring_scale = _predict(model, sx[ids], sy[ids])
    ring_rn = (surf[yy[ids], xx[ids]] - ring_pred) / ring_scale
    med = float(np.median(ring_rn))
    mad = max(1.4826 * float(np.median(np.abs(ring_rn - med))), 0.015)

    if pl[-1] >= c.plane_curve_rmse_m and ids.size >= 12:
        q = _robust_quadratic(sx[ids], sy[ids], surf[yy[ids], xx[ids]])
        if q is not None and float(q[-1]) < c.quadratic_improvement_ratio * float(pl[-1]):
            model = ("quad", q)
            ring_pred, ring_scale = _predict(model, sx[ids], sy[ids])
            ring_rn = (surf[yy[ids], xx[ids]] - ring_pred) / ring_scale
            med = float(np.median(ring_rn))
            mad = max(1.4826 * float(np.median(np.abs(ring_rn - med))), 0.015)
    return model, mad


def apply_surface_impurity_guard(
    *,
    x,
    y,
    z,
    ground_mask,
    sensor_mode,
    cfg: dict | None = None,
    return_report: bool = True,
):
    """Apply final ALS/ULS surface-driven class-2 contamination cleanup."""
    sm = str(sensor_mode).upper().strip()
    g = np.asarray(ground_mask, dtype=bool).copy()
    report: dict[str, Any] = {
        "enabled": sm in {"ALS", "ULS"},
        "ground_before": int(g.sum()),
        "ground_after": int(g.sum()),
        "passes": [],
        "demoted_points": 0,
        "candidate_components": 0,
        "validated_components": 0,
    }
    if sm not in {"ALS", "ULS"} or g.sum() < 12:
        return (g, report) if return_report else g

    cfg = cfg or {}
    if not bool(cfg.get("surface_impurity_guard_enabled", True)):
        report["enabled"] = False
        return (g, report) if return_report else g

    c = _cfg(sm, cfg)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    if not (len(x) == len(y) == len(z) == len(g)):
        raise ValueError("x, y, z and ground_mask must have equal length.")

    x0, y0, ix, iy, key, nx, ny, order, occ, start, end = _grid(x, y, c.cell_m)
    all_tree = cKDTree(np.column_stack((x, y)))
    initial_ground = int(g.sum())

    for ipass in range(max(1, int(c.passes))):
        surf = _lower_ground_surface(z, g, nx, ny, occ, order, start, end)
        slope = _slope_deg(surf, c.cell_m)
        slope_seed, slope_delta, slope_med = _candidate_slope_cells(slope, c)

        # A one-cell dilation captures the center of a tent: the highest center can
        # itself have moderate derivative while its sides carry the extreme slope.
        anomaly = binary_dilation(slope_seed, iterations=max(0, int(c.candidate_dilate_cells)))
        labs, nlab = label(anomaly, np.ones((3, 3), np.uint8))
        report["candidate_components"] += int(nlab)

        demote = []
        pass_valid = 0
        pass_components = 0

        for lid in range(1, int(nlab) + 1):
            comp = labs == lid
            ncell = int(np.count_nonzero(comp))
            if ncell == 0:
                continue
            area = ncell * c.cell_m * c.cell_m
            if area > c.max_component_area_m2:
                continue
            yy, xx = np.nonzero(comp)
            # Tile edges have incomplete annular support.
            if np.any(xx <= 1) or np.any(xx >= nx - 2) or np.any(yy <= 1) or np.any(yy >= ny - 2):
                continue

            # At least one strong fine-vs-neighborhood slope disagreement is needed.
            dvals = slope_delta[yy, xx]
            if not np.any(np.isfinite(dvals)):
                continue
            max_delta = float(np.nanmax(dvals))
            if max_delta < c.slope_delta_min_deg:
                continue

            # Model at the component center, using an annulus that explicitly excludes
            # the anomaly itself.
            weights = np.where(np.isfinite(dvals), np.maximum(dvals, 0.0) + 1.0, 1.0)
            gy = int(np.round(np.average(yy, weights=weights)))
            gx = int(np.round(np.average(xx, weights=weights)))
            fitted = _fit_annular_model(gy, gx, comp, surf, x0, y0, c.cell_m, c)
            if fitted is None:
                continue
            model, ring_mad = fitted
            pass_components += 1

            comp_keys = yy.astype(np.int64) * nx + xx.astype(np.int64)
            chunks = [_cell_points(int(k), order, start, end) for k in comp_keys]
            chunks = [v for v in chunks if v.size]
            if not chunks:
                continue
            pidx = np.unique(np.concatenate(chunks))
            pidx = pidx[g[pidx]]
            if pidx.size == 0:
                continue

            pred, scale = _predict(model, x[pidx], y[pidx])
            rn = (z[pidx] - pred) / scale

            adaptive = min(
                c.normal_tol_cap_m,
                max(c.normal_residual_min_m, c.normal_mad_k * ring_mad),
            )
            strong_thr = max(c.normal_residual_strong_m, adaptive)
            hard_thr = max(c.hard_detached_m, strong_thr)

            # Same-sheet support and non-ground corroboration are evaluated per point.
            local_demote = []
            for pi, ri in zip(pidx, rn):
                ri = float(ri)
                if ri < adaptive:
                    continue

                ids = np.asarray(
                    all_tree.query_ball_point([x[pi], y[pi]], c.point_support_radius_m),
                    dtype=np.int64,
                )
                ids = ids[ids != pi]
                same_ground = 0
                ng_frac = 0.0
                ng_n = 0
                if ids.size:
                    pp, ss = _predict(model, x[ids], y[ids])
                    rr = (z[ids] - pp) / ss

                    # IMPORTANT V4.1:
                    # A detached false-ground cluster can be internally dense/coherent.
                    # Points belonging to the same raster anomaly must NOT protect one
                    # another.  Protective support must come from current-ground points
                    # OUTSIDE the candidate anomaly and agree with the candidate in
                    # terrain-normal residual space.
                    ids_keys = key[ids]
                    inside_candidate = np.isin(ids_keys, comp_keys, assume_unique=False)
                    external = ~inside_candidate

                    same = np.abs(rr - ri) <= c.same_surface_band_m
                    same_ground = int(
                        np.count_nonzero(same & g[ids] & external)
                    )

                    # Non-ground corroboration may be inside or immediately around the
                    # anomalous footprint because vegetation returns commonly coexist
                    # with the leaked class-2 cluster.
                    ng_same = np.abs(rr - ri) <= c.nonground_same_surface_band_m
                    ng_n = int(np.count_nonzero(ng_same))
                    if ng_n:
                        ng_frac = float(np.count_nonzero(ng_same & (~g[ids])) / ng_n)

                weak_sheet = same_ground < c.min_same_surface_ground
                hard_detached = (
                    ri >= hard_thr
                    and area <= c.hard_max_component_area_m2
                    and weak_sheet
                )
                surface_pimple = (
                    ri >= strong_thr
                    and max_delta >= c.slope_delta_strong_deg
                    and weak_sheet
                    and ng_n >= c.nonground_min_neighbors
                    and ng_frac >= c.nonground_fraction_min
                )
                if hard_detached or surface_pimple:
                    local_demote.append(int(pi))

            if local_demote:
                demote.extend(local_demote)
                pass_valid += 1

        if demote:
            dem = np.unique(np.asarray(demote, np.int64))
            dem = dem[g[dem]]
            # Catastrophic-loss guard.  Keep only the largest positive normal
            # residuals if an unexpectedly large fraction is proposed.
            max_n = max(1, int(math.ceil(c.max_demote_fraction_per_pass * max(1, int(g.sum())))))
            if dem.size > max_n:
                # Re-score against a coarse local median of Z only as an ordering
                # device; acceptance already required terrain-normal/radial tests.
                order_score = np.argsort(z[dem])[::-1]
                dem = dem[order_score[:max_n]]
            g[dem] = False
            ndem = int(dem.size)
        else:
            ndem = 0

        report["validated_components"] += int(pass_valid)
        report["passes"].append(
            {
                "pass": ipass + 1,
                "slope_seed_cells": int(np.count_nonzero(slope_seed)),
                "candidate_components": int(nlab),
                "modeled_components": int(pass_components),
                "validated_components": int(pass_valid),
                "demoted_points": int(ndem),
            }
        )
        if ndem == 0:
            break

    report["ground_after"] = int(g.sum())
    report["demoted_points"] = int(initial_ground - int(g.sum()))
    return (g, report) if return_report else g


__all__ = ["SurfaceImpurityConfig", "apply_surface_impurity_guard"]
