from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree


def _mad(a):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return np.nan
    m = float(np.median(a))
    return float(1.4826 * np.median(np.abs(a - m)))


def _robust_plane(x, y, z, qx, qy):
    if len(z) < 8:
        return None

    A = np.column_stack(
        (x - qx, y - qy, np.ones(len(z), dtype=np.float64))
    )
    keep = np.ones(len(z), dtype=bool)
    coef = None

    for _ in range(5):
        if int(keep.sum()) < 8:
            return None
        coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        r = z - A @ coef
        rk = r[keep]
        med = float(np.median(rk))
        spread = max(_mad(rk), 0.01)
        keep2 = np.abs(r - med) <= max(0.15, 3.5 * spread)
        if np.array_equal(keep2, keep):
            break
        keep = keep2

    if coef is None or int(keep.sum()) < 8:
        return None

    rr = z[keep] - A[keep] @ coef
    return {
        "coef": coef,
        "spread": max(_mad(rr), 0.01),
        "n": int(keep.sum()),
        "kind": "plane",
    }


def _robust_quadratic(x, y, z, qx, qy):
    if len(z) < 18:
        return None

    dx = x - qx
    dy = y - qy
    A = np.column_stack(
        (
            dx,
            dy,
            dx * dx,
            dx * dy,
            dy * dy,
            np.ones(len(z), dtype=np.float64),
        )
    )
    keep = np.ones(len(z), dtype=bool)
    coef = None

    for _ in range(5):
        if int(keep.sum()) < 18:
            return None
        coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        r = z - A @ coef
        rk = r[keep]
        med = float(np.median(rk))
        spread = max(_mad(rk), 0.01)
        keep2 = np.abs(r - med) <= max(0.18, 3.5 * spread)
        if np.array_equal(keep2, keep):
            break
        keep = keep2

    if coef is None or int(keep.sum()) < 18:
        return None

    rr = z[keep] - A[keep] @ coef
    return {
        "coef": coef,
        "spread": max(_mad(rr), 0.01),
        "n": int(keep.sum()),
        "kind": "quadratic",
    }


def _predict(model, x, y, qx, qy):
    dx = np.asarray(x, dtype=np.float64) - qx
    dy = np.asarray(y, dtype=np.float64) - qy
    c = model["coef"]

    if model["kind"] == "plane":
        return c[0] * dx + c[1] * dy + c[2]

    return (
        c[0] * dx
        + c[1] * dy
        + c[2] * dx * dx
        + c[3] * dx * dy
        + c[4] * dy * dy
        + c[5]
    )


def _choose_surface(x, y, z, qx, qy):
    plane = _robust_plane(x, y, z, qx, qy)
    if plane is None:
        return None

    quad = _robust_quadratic(x, y, z, qx, qy)
    if quad is None:
        return plane

    # Use curvature only when it materially improves the robust residual.
    if (
        plane["spread"] >= 0.10
        and quad["spread"] <= 0.82 * plane["spread"]
        and quad["spread"] <= 0.65
    ):
        return quad

    return plane


def _support_cell_positions(cells, cid):
    cells = np.asarray(cells, dtype=np.int64)
    cid = np.asarray(cid, dtype=np.int64)
    out = np.full(cid.shape, -1, dtype=np.int64)

    if cells.size == 0 or cid.size == 0:
        return out

    pos = np.searchsorted(cells, cid)
    ok_idx = np.flatnonzero(pos < cells.size)
    if ok_idx.size:
        p = pos[ok_idx]
        same = cells[p] == cid[ok_idx]
        out[ok_idx[same]] = p[same]
    return out


def _connected_lower_top_bins(cells, hcell, hbin, *, max_missing_bins=1):
    cells = np.asarray(cells, dtype=np.int64)
    hcell = np.asarray(hcell, dtype=np.int64)
    hbin = np.asarray(hbin, dtype=np.int64)

    top = np.zeros(cells.size, dtype=np.int32)
    if cells.size == 0 or hcell.size == 0:
        return top

    order = np.lexsort((hbin, hcell))
    hc = hcell[order]
    hb = hbin[order]

    starts = np.flatnonzero(np.r_[True, hc[1:] != hc[:-1]])
    ends = np.r_[starts[1:], hc.size]

    support_pos = np.searchsorted(cells, hc[starts])

    for p, s, e in zip(support_pos, starts, ends):
        if p >= cells.size or cells[p] != hc[s]:
            continue

        bins = np.unique(hb[s:e])
        if bins.size == 0:
            continue

        current = int(bins[0])
        for b in bins[1:]:
            b = int(b)
            if b - current - 1 > max_missing_bins:
                break
            current = b

        top[p] = current

    return top


def _candidate_raster_components(x, y, candidates, *, res=0.5):
    """
    Rasterize suspicious final-ground points to physical XY cells.
    Returns connected components with footprint in square metres.
    """
    if len(candidates) == 0:
        return []

    cx = x[candidates]
    cy = y[candidates]

    x0 = np.floor(float(np.min(cx)) / res) * res
    y0 = np.floor(float(np.min(cy)) / res) * res

    ix = np.floor((cx - x0) / res).astype(np.int64)
    iy = np.floor((cy - y0) / res).astype(np.int64)

    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1

    grid = np.zeros((ny, nx), dtype=bool)
    grid[iy, ix] = True

    lab, ncomp = label(grid, structure=np.ones((3, 3), dtype=np.int8))
    point_labels = lab[iy, ix]

    out = []

    for lid in range(1, ncomp + 1):
        point_sel = point_labels == lid
        if not np.any(point_sel):
            continue

        pts = candidates[point_sel]
        occupied = int(np.sum(lab == lid))

        out.append(
            {
                "label": lid,
                "points": pts,
                "area_m2": occupied * res * res,
                "x_min": float(np.min(x[pts])),
                "x_max": float(np.max(x[pts])),
                "y_min": float(np.min(y[pts])),
                "y_max": float(np.max(y[pts])),
                "cx": float(np.median(x[pts])),
                "cy": float(np.median(y[pts])),
            }
        )

    return out


def _annulus_support(
    tree,
    ground_indices,
    x,
    y,
    z,
    comp,
    *,
    inner_pad,
    radii,
    min_points,
    min_sectors,
):
    """
    Build terrain support only OUTSIDE the suspicious component footprint.

    The inner rectangle is expanded by inner_pad. Candidate/blob points and
    their immediate neighborhood therefore cannot confirm themselves.
    """
    qx = comp["cx"]
    qy = comp["cy"]

    for outer in radii:
        ids = tree.query_ball_point([qx, qy], outer)
        if not ids:
            continue

        q = ground_indices[np.asarray(ids, dtype=np.int64)]
        if q.size < min_points:
            continue

        inside_component_zone = (
            (x[q] >= comp["x_min"] - inner_pad)
            & (x[q] <= comp["x_max"] + inner_pad)
            & (y[q] >= comp["y_min"] - inner_pad)
            & (y[q] <= comp["y_max"] + inner_pad)
        )
        q = q[~inside_component_zone]
        if q.size < min_points:
            continue

        dx = x[q] - qx
        dy = y[q] - qy

        # Require surrounding support in several directions.
        ang = np.arctan2(dy, dx)
        sec = (
            np.floor((ang + np.pi) / (2.0 * np.pi) * 8.0)
            .astype(np.int32)
            % 8
        )
        if np.unique(sec).size < min_sectors:
            continue

        if q.size > 500:
            d2 = dx * dx + dy * dy
            q = q[np.argsort(d2)[:500]]

        model = _choose_surface(x[q], y[q], z[q], qx, qy)
        if model is None:
            continue

        # A wildly uncertain external model is not safe enough for demotion.
        if model["spread"] > 0.75:
            continue

        return q, model, outer

    return None, None, None


def _component_decision(
    comp,
    *,
    x,
    y,
    z,
    cp,
    weak,
    canopy,
    lower_top_bin,
    z0,
    dz,
    tree,
    ground_indices,
    max_area,
):
    pts = comp["points"]
    area = float(comp["area_m2"])

    if area > max_area:
        return None

    valid_cp = cp[pts] >= 0
    if not np.any(valid_cp):
        return None

    p = pts[valid_cp]
    k = cp[p]

    rel = z[p] - z0[k]
    pbin = np.floor(np.maximum(rel, 0.0) / max(dz, 1e-6)).astype(np.int64)
    disconnected = pbin >= (lower_top_bin[k].astype(np.int64) + 2)

    frac_disc = float(np.mean(disconnected)) if disconnected.size else 0.0
    frac_weak = float(np.mean(weak[k])) if k.size else 0.0
    frac_canopy = float(np.mean(canopy[k])) if k.size else 0.0

    # Physical size controls how much corroboration is required.
    if area <= 2.0:
        inner_pad = 1.5
        min_points = 12
        min_sectors = 4
        radii = (3.0, 5.0, 7.5, 10.0, 15.0)
        min_disc = 0.25
        base_gap = 0.30
        required_above_fraction = 0.55
    elif area <= 8.0:
        inner_pad = 2.0
        min_points = 14
        min_sectors = 4
        radii = (5.0, 7.5, 10.0, 15.0, 20.0)
        min_disc = 0.35
        base_gap = 0.40
        required_above_fraction = 0.60
    elif area <= 25.0:
        inner_pad = 3.0
        min_points = 18
        min_sectors = 5
        radii = (7.5, 10.0, 15.0, 20.0, 25.0)
        min_disc = 0.45
        base_gap = 0.55
        required_above_fraction = 0.65
    else:
        inner_pad = 5.0
        min_points = 22
        min_sectors = 5
        radii = (10.0, 15.0, 20.0, 25.0, 30.0)
        min_disc = 0.55
        base_gap = 0.70
        required_above_fraction = 0.70

    q, model, used_radius = _annulus_support(
        tree,
        ground_indices,
        x,
        y,
        z,
        comp,
        inner_pad=inner_pad,
        radii=radii,
        min_points=min_points,
        min_sectors=min_sectors,
    )

    if model is None:
        return None

    pred = _predict(model, x[p], y[p], comp["cx"], comp["cy"])
    residual = z[p] - pred

    threshold = max(
        base_gap,
        3.0 * float(model["spread"]),
    )

    above = residual > threshold
    above_fraction = float(np.mean(above)) if above.size else 0.0
    median_residual = float(np.median(residual)) if residual.size else 0.0
    p75_residual = float(np.percentile(residual, 75)) if residual.size else 0.0

    vertical_evidence = (
        frac_disc >= min_disc
        or frac_weak >= 0.50
        or frac_canopy >= 0.40
    )

    # Component is object-like only when most of it sits above a coherent
    # externally predicted terrain sheet AND raw vertical support agrees.
    object_like = bool(
        vertical_evidence
        and above_fraction >= required_above_fraction
        and median_residual > 0.5 * threshold
        and p75_residual > threshold
    )

    return {
        "object_like": object_like,
        "points": p,
        "area_m2": area,
        "used_radius_m": float(used_radius),
        "surface_kind": model["kind"],
        "surface_spread_m": float(model["spread"]),
        "threshold_m": float(threshold),
        "above_fraction": above_fraction,
        "median_residual_m": median_residual,
        "p75_residual_m": p75_residual,
        "frac_disconnected": frac_disc,
        "frac_weak": frac_weak,
        "frac_canopy": frac_canopy,
    }


def apply_vertical_support_final_guard(
    *,
    x,
    y,
    z,
    ground_mask,
    sensor_mode,
    cfg=None,
    return_report=True,
):
    """
    FAST-GC Vertical Support Final Guard V2.3.

    Keeps the working V2.2 point-scale logic, then adds a final physical
    component-vs-background terrain test.

    The V2.3 component stage:
      - groups suspicious final-ground points on a 0.5 m XY topology raster;
      - measures component footprint in m², never point count;
      - excludes the entire component plus a safety margin;
      - expands an external support ring until terrain is adequately observed;
      - fits a robust plane, optionally a quadratic surface when curvature
        materially improves the fit;
      - removes only components whose majority lies above that independently
        supported terrain AND whose raw vertical support is weak/disconnected.

    Absolute elevation and absolute slope are never demotion criteria.
    """
    sm = str(sensor_mode).upper().strip()
    cfg = cfg or {}
    g = np.asarray(ground_mask, dtype=bool).copy()

    rep = {
        "enabled": False,
        "ground_before": int(g.sum()),
        "support_cells": 0,
        "weak_cells": 0,
        "canopy_cells": 0,
        "candidate_points": 0,
        "small_candidates": 0,
        "large_candidates": 0,
        "components": 0,
        "components_modeled": 0,
        "components_removed": 0,
        "component_points_removed": 0,
        "micro_components_removed": 0,
        "small_components_removed": 0,
        "medium_components_removed": 0,
        "large_components_removed": 0,
        "ground_after": int(g.sum()),
    }

    if sm not in {"ALS", "ULS"}:
        return (g, rep) if return_report else g

    fp = cfg.get("vertical_support_file")
    if not fp:
        return (g, rep) if return_report else g

    p = Path(str(fp))
    if not p.exists():
        return (g, rep) if return_report else g

    if not bool(cfg.get("vertical_support_final_guard_enabled", True)):
        return (g, rep) if return_report else g

    rep["enabled"] = True

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    with np.load(p, allow_pickle=False) as s:
        cell = float(s["xy_cell_m"])
        dz = float(s["z_bin_m"])
        x0 = float(s["origin_x"])
        y0 = float(s["origin_y"])
        nx = int(s["nx"])
        ny = int(s["ny"])

        cells = np.asarray(s["cell_id"], dtype=np.int64)
        z0 = np.asarray(s["z0"], dtype=np.float64)
        denslow = np.asarray(s["density_low2"], dtype=np.float64)
        frac2 = np.asarray(s["frac_low2"], dtype=np.float64)
        frac3 = np.asarray(s["frac_low3"], dtype=np.float64)
        span = np.asarray(s["robust_z_span"], dtype=np.float64)
        nt = np.asarray(s["n_total"], dtype=np.float64)
        nu = np.asarray(s["n_upper3"], dtype=np.float64)
        occupied = np.asarray(s["occupied_z_bins"], dtype=np.float64)

        hcell = np.asarray(s["hist_cell_id"], dtype=np.int64)
        hbin = np.asarray(s["hist_zbin"], dtype=np.int64)

    rep["support_cells"] = int(cells.size)
    if cells.size == 0:
        return (g, rep) if return_report else g

    weak = (
        (denslow < 0.65)
        | ((frac2 < 0.10) & (span >= 2.0))
        | ((occupied >= 4) & (frac2 < 0.14) & (denslow < 1.0))
    )

    canopy = (
        (span >= 2.5)
        & (frac3 < 0.28)
        & (nu >= 0.50 * np.maximum(nt, 1.0))
    )

    severe_canopy = (
        (span >= 4.0)
        & (frac3 < 0.20)
        & (nu >= 0.65 * np.maximum(nt, 1.0))
    )

    rep["weak_cells"] = int(weak.sum())
    rep["canopy_cells"] = int(canopy.sum())

    lower_top_bin = _connected_lower_top_bins(
        cells,
        hcell,
        hbin,
        max_missing_bins=1,
    )

    ix = np.floor((x - x0) / cell).astype(np.int64)
    iy = np.floor((y - y0) / cell).astype(np.int64)

    in_grid = (
        (ix >= 0)
        & (ix < nx)
        & (iy >= 0)
        & (iy < ny)
    )

    cid = iy * nx + ix
    cp = _support_cell_positions(cells, cid)
    valid = in_grid & (cp >= 0)

    gi = np.flatnonzero(g & valid)
    if gi.size == 0:
        return (g, rep) if return_report else g

    k = cp[gi]
    rel = z[gi] - z0[k]
    pbin = np.floor(np.maximum(rel, 0.0) / max(dz, 1e-6)).astype(np.int64)

    disconnected = pbin >= (
        lower_top_bin[k].astype(np.int64) + 2
    )

    large_mask = (
        (weak[k] | canopy[k])
        & (rel >= 3.0)
    )

    small_mask = (
        ~large_mask
        & disconnected
        & (rel >= 1.0)
        & (
            (span[k] >= 1.5)
            | weak[k]
            | severe_canopy[k]
        )
    )

    candidates = gi[large_mask | small_mask]

    rep["candidate_points"] = int(candidates.size)
    rep["large_candidates"] = int(np.sum(large_mask))
    rep["small_candidates"] = int(np.sum(small_mask))

    if candidates.size == 0:
        return (g, rep) if return_report else g

    # ------------------------------------------------------------------
    # V2.3 component-level annulus terrain confirmation.
    # ------------------------------------------------------------------
    topology_res = float(
        cfg.get("vertical_support_component_cell_m", 0.5)
    )
    max_component_area = float(
        cfg.get("vertical_support_component_max_area_m2", 80.0)
    )

    comps = _candidate_raster_components(
        x,
        y,
        candidates,
        res=topology_res,
    )
    rep["components"] = len(comps)

    current_ground = np.flatnonzero(g)
    tree = cKDTree(
        np.column_stack((x[current_ground], y[current_ground]))
    )

    to_remove = []

    for comp in comps:
        result = _component_decision(
            comp,
            x=x,
            y=y,
            z=z,
            cp=cp,
            weak=weak,
            canopy=canopy,
            lower_top_bin=lower_top_bin,
            z0=z0,
            dz=dz,
            tree=tree,
            ground_indices=current_ground,
            max_area=max_component_area,
        )

        if result is None:
            continue

        rep["components_modeled"] += 1

        if not result["object_like"]:
            continue

        pts = np.asarray(result["points"], dtype=np.int64)
        if pts.size == 0:
            continue

        to_remove.append(pts)
        rep["components_removed"] += 1

        area = float(result["area_m2"])
        if area <= 2.0:
            rep["micro_components_removed"] += 1
        elif area <= 8.0:
            rep["small_components_removed"] += 1
        elif area <= 25.0:
            rep["medium_components_removed"] += 1
        else:
            rep["large_components_removed"] += 1

    if to_remove:
        remove = np.unique(np.concatenate(to_remove))

        # Conservative global safety cap.  Component decisions are already
        # restrictive; this prevents catastrophic over-cleaning on an unseen
        # acquisition.  Increase only after validation if ever necessary.
        max_fraction = float(
            cfg.get(
                "vertical_support_component_max_demote_fraction",
                0.04,
            )
        )
        cap = max(
            1,
            int(np.ceil(max_fraction * max(1, int(g.sum())))),
        )

        if remove.size > cap:
            # Do not rank by absolute Z. Keep only the first cap points from
            # already-confirmed object-like components; component thresholds
            # are the scientific decision layer, the cap is only a fail-safe.
            remove = remove[:cap]

        g[remove] = False
        rep["component_points_removed"] = int(remove.size)

    rep["ground_after"] = int(g.sum())

    return (g, rep) if return_report else g


__all__ = ["apply_vertical_support_final_guard"]
