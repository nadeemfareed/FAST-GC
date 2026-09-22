from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def _mad(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    m = float(np.median(v))
    return float(np.median(np.abs(v - m)))


def _build_cells(x, y, z, ground, cell_m=4.0, min_pts=6):
    gids = np.flatnonzero(ground)
    if gids.size == 0:
        return None
    gx, gy = x[gids], y[gids]
    xmin, ymin = float(np.min(gx)), float(np.min(gy))
    ix = np.floor((gx - xmin) / cell_m).astype(np.int64)
    iy = np.floor((gy - ymin) / cell_m).astype(np.int64)
    uniq, inv = np.unique(np.column_stack((ix, iy)), axis=0, return_inverse=True)
    ncell = uniq.shape[0]
    counts = np.bincount(inv, minlength=ncell).astype(np.int64)
    zmin = np.full(ncell, np.nan); zmax = np.full(ncell, np.nan); zrange = np.full(ncell, np.nan)
    order = np.argsort(inv, kind="stable")
    invs = inv[order]; ids_sorted = gids[order]
    starts = np.r_[0, np.flatnonzero(np.diff(invs)) + 1]
    stops = np.r_[starts[1:], invs.size]
    for s, e in zip(starts, stops):
        cid = int(invs[s]); ids = ids_sorted[s:e]
        if ids.size < min_pts:
            continue
        zz = z[ids]
        zmin[cid] = float(np.min(zz)); zmax[cid] = float(np.max(zz)); zrange[cid] = zmax[cid] - zmin[cid]
    valid = (counts >= min_pts) & np.isfinite(zrange)
    lookup = {(int(a), int(b)): i for i, (a,b) in enumerate(uniq)}
    aix = np.floor((x - xmin) / cell_m).astype(np.int64)
    aiy = np.floor((y - ymin) / cell_m).astype(np.int64)
    pc = np.full(x.size, -1, dtype=np.int64)
    for i in range(x.size):
        pc[i] = lookup.get((int(aix[i]), int(aiy[i])), -1)
    return dict(valid=valid, zmin=zmin, zmax=zmax, zrange=zrange, point_cell=pc)


def _trigger(v, q=0.90, floor_m=1.0):
    v = np.asarray(v, dtype=np.float64); v = v[np.isfinite(v)]
    if v.size == 0:
        return float("inf")
    med = float(np.median(v)); scale = 1.4826 * max(_mad(v), 1e-6)
    return max(float(floor_m), float(np.quantile(v, q)), med + 2.0 * scale)


def clean_als_facet_support_consistency(*, x, y, z, ground_mask, sensor_mode, cfg=None, return_report=True):
    """V25: aggressive residual cleanup. Points are preserved; only GROUND->NON-GROUND."""
    sm = str(sensor_mode).upper().strip()
    original = np.asarray(ground_mask, dtype=bool)
    out = original.copy()
    rep = dict(enabled=False, sensor_mode=sm, ground_before=int(original.sum()), ground_after=int(original.sum()), reclassified_to_nonground=0, iterations=[])
    if sm not in {"ALS", "ULS"}:
        return (out, rep) if return_report else out

    cfg = cfg or {}
    enabled = bool(cfg.get("als_residual_sheet_outlier_enabled", True))
    rep["enabled"] = enabled
    if not enabled:
        return (out, rep) if return_report else out

    cell_m = float(cfg.get("als_residual_sheet_outlier_cell_m", 4.0))
    min_cell_pts = int(cfg.get("als_residual_sheet_outlier_min_ground_points_per_cell", 6))
    trigger_q = float(cfg.get("als_residual_sheet_outlier_trigger_quantile", 0.90))
    trigger_floor = float(cfg.get("als_residual_sheet_outlier_trigger_floor_m", 1.0))
    sheet_r = float(cfg.get("als_residual_sheet_outlier_sheet_radius_m", 6.0))
    sheet_min = int(cfg.get("als_residual_sheet_outlier_sheet_min_points", 16))
    sheet_k = float(cfg.get("als_residual_sheet_outlier_sheet_mad_k", 3.0))
    sheet_floor = float(cfg.get("als_residual_sheet_outlier_sheet_floor_m", 0.18))
    same_r = float(cfg.get("als_residual_sheet_outlier_same_ground_radius_m", 0.75))
    max_same = int(cfg.get("als_residual_sheet_outlier_max_ground_neighbors", 5))
    below_r = float(cfg.get("als_residual_sheet_outlier_below_xy_radius_m", 1.25))
    below_min = int(cfg.get("als_residual_sheet_outlier_below_min_points", 4))
    below_clear = float(cfg.get("als_residual_sheet_outlier_below_min_clearance_m", 0.20))
    force_resid = float(cfg.get("als_residual_sheet_outlier_force_residual_m", 0.35))
    max_iter = int(cfg.get("als_residual_sheet_outlier_max_iterations", 4))

    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64); z = np.asarray(z, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    vid = np.flatnonzero(finite)
    if vid.size < 20:
        return (out, rep) if return_report else out

    xv, yv, zv = x[vid], y[vid], z[vid]
    xy = np.column_stack((xv, yv)); xyz = np.column_stack((xv, yv, zv))
    g = out[vid].copy()

    for it in range(max_iter):
        cells = _build_cells(xv, yv, zv, g, cell_m=cell_m, min_pts=min_cell_pts)
        if cells is None:
            break
        vr = cells["zrange"][cells["valid"]]
        if vr.size == 0:
            break
        thr = _trigger(vr, trigger_q, trigger_floor)
        flagged = cells["valid"] & (cells["zrange"] >= thr)
        pc = cells["point_cell"]
        cand = np.flatnonzero(g & (pc >= 0) & flagged[np.maximum(pc,0)])
        if cand.size == 0:
            rep["iterations"].append(dict(iteration=it+1, zrange_threshold_m=float(thr), flagged_cells=int(flagged.sum()), candidate_ground_points=0, reclassified_to_nonground=0))
            break

        gids = np.flatnonzero(g)
        tree2 = cKDTree(xy[gids]); tree3 = cKDTree(xyz[gids])
        demote = np.zeros(g.size, dtype=bool)
        tested = isolated = below_ok = forced = 0

        for gid in cand:
            gid = int(gid)
            n2 = tree2.query_ball_point(xy[gid], r=sheet_r)
            if len(n2) < sheet_min + 1:
                continue
            loc = gids[np.asarray(n2, dtype=np.int64)]
            loc = loc[loc != gid]
            if loc.size < sheet_min:
                continue

            X = np.column_stack((xv[loc], yv[loc], np.ones(loc.size)))
            try:
                coef, *_ = np.linalg.lstsq(X, zv[loc], rcond=None)
                resid = zv[loc] - X @ coef
                medr = float(np.median(resid)); scale = 1.4826 * max(_mad(resid), 0.02)
                keep = np.abs(resid - medr) <= 3.5 * scale
                loc2 = loc[keep]
                if loc2.size >= sheet_min:
                    X2 = np.column_stack((xv[loc2], yv[loc2], np.ones(loc2.size)))
                    coef, *_ = np.linalg.lstsq(X2, zv[loc2], rcond=None)
                    resid = zv[loc2] - X2 @ coef
                pred = float(coef[0]*xv[gid] + coef[1]*yv[gid] + coef[2])
            except Exception:
                continue

            pos_resid = float(zv[gid] - pred)
            rscale = 1.4826 * max(_mad(resid), 0.02)
            rthr = max(sheet_floor, sheet_k * rscale)
            if pos_resid <= rthr:
                continue
            tested += 1

            n3 = tree3.query_ball_point(xyz[gid], r=same_r)
            sparse = max(0, len(n3)-1) <= max_same
            if sparse:
                isolated += 1

            nb = tree2.query_ball_point(xy[gid], r=below_r)
            nbids = gids[np.asarray(nb, dtype=np.int64)] if nb else np.empty(0, dtype=np.int64)
            nbids = nbids[nbids != gid]
            lower = nbids[zv[nbids] <= (zv[gid] - below_clear)]
            has_below = lower.size >= below_min
            if has_below:
                below_ok += 1

            force = sparse and pos_resid >= force_resid
            if (sparse and has_below) or force:
                demote[gid] = True
                if force:
                    forced += 1

        demote &= g
        nd = int(demote.sum())
        rep["iterations"].append(dict(iteration=it+1, zrange_threshold_m=float(thr), flagged_cells=int(flagged.sum()), candidate_ground_points=int(cand.size), sheet_residual_candidates=int(tested), spatially_isolated=int(isolated), lower_sheet_supported=int(below_ok), forced_sparse_residuals=int(forced), reclassified_to_nonground=nd))
        if nd > 0:
            g[demote] = False
        if nd == 0:
            break

    out[vid] = g
    rep["ground_after"] = int(out.sum())
    rep["reclassified_to_nonground"] = int(np.count_nonzero(original & ~out))
    return (out, rep) if return_report else out


__all__ = ["clean_als_facet_support_consistency"]
