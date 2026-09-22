from __future__ import annotations

"""
FAST-GC ALS ground-only vertical-profile contamination guard.

Purpose
-------
Remove class-2 points that have leaked upward into vegetation/canopy where
true terrain support is weak or absent.

Design constraints
------------------
* Aerial ALS/ULS only; TLS is excluded.
* Ground -> non-ground only.
* Uses ONLY the current ground-labelled points to build the vertical signal.
* Does NOT use non-ground density/voting.
* Does NOT modify the upstream terrain classifier.
* Multi-resolution XY footprints make the rule usable across varying point density.
* Trusted lower ground modes are protected.
* Cells without a trustworthy lower mode borrow terrain elevation only from
  nearby trusted ground cells through a small local plane/IDW prediction.

The key signal is the vertical point-frequency distribution of CURRENT GROUND
points within an XY footprint. A strong low mode represents terrain. Sparse
ground-labelled modes far above that low mode are canopy-leak candidates.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt, gaussian_filter, label


@dataclass(frozen=True)
class AlsGroundVerticalProfileConfig:
    footprints_m: tuple[float, ...] = (5.0, 7.5, 10.0)
    z_bin_m: float = 0.25
    mode_gap_m: float = 0.75

    min_ground_points_cell: int = 5
    min_ground_points_mode: int = 3
    min_mode_fraction: float = 0.05

    min_above_mode_m: float = 1.25
    hard_above_mode_m: float = 2.50

    neighbor_k: int = 8
    max_neighbor_distance_factor: float = 2.75
    min_plane_neighbors: int = 3
    max_plane_slope: float = 5.0

    min_votes: int = 2
    hard_vote_residual_m: float = 3.0
    protect_margin_m: float = 0.35


def _cfg_from_dict(cfg: dict | None) -> AlsGroundVerticalProfileConfig:
    c = cfg or {}

    raw_scales = c.get("als_ground_profile_footprints_m", (5.0, 7.5, 10.0))
    if isinstance(raw_scales, str):
        vals = []
        for s in raw_scales.replace(",", " ").split():
            try:
                vals.append(float(s))
            except Exception:
                pass
        scales = tuple(vals) if vals else (5.0, 7.5, 10.0)
    elif isinstance(raw_scales, Iterable):
        try:
            scales = tuple(float(v) for v in raw_scales)
        except Exception:
            scales = (5.0, 7.5, 10.0)
    else:
        scales = (5.0, 7.5, 10.0)

    scales = tuple(sorted({max(2.0, float(v)) for v in scales}))
    if not scales:
        scales = (5.0, 7.5, 10.0)

    return AlsGroundVerticalProfileConfig(
        footprints_m=scales,
        z_bin_m=max(0.10, float(c.get("als_ground_profile_z_bin_m", 0.25))),
        mode_gap_m=max(0.35, float(c.get("als_ground_profile_mode_gap_m", 0.75))),
        min_ground_points_cell=max(3, int(c.get("als_ground_profile_min_cell_points", 5))),
        min_ground_points_mode=max(2, int(c.get("als_ground_profile_min_mode_points", 3))),
        min_mode_fraction=float(np.clip(c.get("als_ground_profile_min_mode_fraction", 0.05), 0.05, 0.80)),
        min_above_mode_m=max(0.50, float(c.get("als_ground_profile_min_above_mode_m", 1.25))),
        hard_above_mode_m=max(1.0, float(c.get("als_ground_profile_hard_above_mode_m", 2.50))),
        neighbor_k=max(3, int(c.get("als_ground_profile_neighbor_k", 8))),
        max_neighbor_distance_factor=max(
            1.25, float(c.get("als_ground_profile_max_neighbor_distance_factor", 2.75))
        ),
        min_plane_neighbors=max(3, int(c.get("als_ground_profile_min_plane_neighbors", 3))),
        max_plane_slope=max(0.5, float(c.get("als_ground_profile_max_plane_slope", 5.0))),
        min_votes=max(1, int(c.get("als_ground_profile_min_votes", 2))),
        hard_vote_residual_m=max(1.5, float(c.get("als_ground_profile_hard_vote_residual_m", 3.0))),
        protect_margin_m=max(0.10, float(c.get("als_ground_profile_protect_margin_m", 0.35))),
    )


def _empty_report(n_ground: int) -> dict:
    return {
        "ground_before": int(n_ground),
        "ground_after": int(n_ground),
        "demoted_points": 0,
        "scales_m": [],
        "trusted_cells_per_scale": [],
        "unsupported_cells_per_scale": [],
        "spike_candidates_per_scale": [],
        "consensus_candidates": 0,
        "hard_candidates": 0,
    }


def _split_low_mode_from_bins(
    zvals: np.ndarray,
    *,
    z_bin_m: float,
    mode_gap_m: float,
    min_mode_points: int,
    min_mode_fraction: float,
):
    n = int(zvals.size)
    if n == 0:
        return False, np.nan, np.nan, np.nan

    zmin = float(np.min(zvals))
    bins = np.floor((zvals - zmin) / float(z_bin_m)).astype(np.int32)
    ub, counts = np.unique(bins, return_counts=True)

    gap_bins = max(1, int(np.ceil(float(mode_gap_m) / float(z_bin_m))))
    cuts = np.flatnonzero(np.diff(ub) > gap_bins)
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts + 1, ub.size]

    need = max(
        int(min_mode_points),
        int(np.ceil(float(min_mode_fraction) * n)),
    )

    chosen = None
    for a, b in zip(starts, ends):
        c = int(np.sum(counts[a:b]))
        if c >= need:
            chosen = (a, b)
            break

    if chosen is None:
        return False, np.nan, np.nan, np.nan

    a, b = chosen
    b0 = int(ub[a])
    b1 = int(ub[b - 1])

    lo = zmin + b0 * float(z_bin_m)
    hi = zmin + (b1 + 1) * float(z_bin_m)

    in_mode = (zvals >= lo) & (zvals < hi + 1e-9)
    if int(np.count_nonzero(in_mode)) < int(min_mode_points):
        return False, np.nan, np.nan, np.nan

    ref = float(np.median(zvals[in_mode]))
    return True, float(lo), float(hi), ref


def _local_prediction(
    *,
    qx: np.ndarray,
    qy: np.ndarray,
    trusted_xy: np.ndarray,
    trusted_z: np.ndarray,
    footprint_m: float,
    cfg: AlsGroundVerticalProfileConfig,
) -> np.ndarray:
    out = np.full(qx.shape, np.nan, dtype=np.float64)
    if trusted_xy.shape[0] == 0 or qx.size == 0:
        return out

    tree = cKDTree(trusted_xy)
    k = min(int(cfg.neighbor_k), trusted_xy.shape[0])
    d, idx = tree.query(np.column_stack((qx, qy)), k=k)

    if k == 1:
        d = d[:, None]
        idx = idx[:, None]

    max_dist = float(cfg.max_neighbor_distance_factor) * float(footprint_m)

    for i in range(qx.size):
        di = np.asarray(d[i], dtype=np.float64)
        ii = np.asarray(idx[i], dtype=np.int64)
        good = np.isfinite(di) & (di <= max_dist)
        if np.count_nonzero(good) == 0:
            continue

        di = di[good]
        ii = ii[good]
        xyz = trusted_xy[ii]
        zz = trusted_z[ii]

        if ii.size >= int(cfg.min_plane_neighbors):
            A = np.column_stack(
                (
                    xyz[:, 0] - qx[i],
                    xyz[:, 1] - qy[i],
                    np.ones(ii.size, dtype=np.float64),
                )
            )
            try:
                coef, *_ = np.linalg.lstsq(A, zz, rcond=None)
                slope = float(np.hypot(coef[0], coef[1]))
                if np.all(np.isfinite(coef)) and slope <= float(cfg.max_plane_slope):
                    out[i] = float(coef[2])
                    continue
            except Exception:
                pass

        w = 1.0 / np.maximum(di, 0.25) ** 2
        out[i] = float(np.sum(w * zz) / np.sum(w))

    return out


def _analyze_scale(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ground_idx: np.ndarray,
    footprint_m: float,
    cfg: AlsGroundVerticalProfileConfig,
):
    xg = x[ground_idx]
    yg = y[ground_idx]
    zg = z[ground_idx]

    x0 = float(np.min(x))
    y0 = float(np.min(y))

    ix = np.floor((xg - x0) / footprint_m).astype(np.int32)
    iy = np.floor((yg - y0) / footprint_m).astype(np.int32)

    nx = int(ix.max()) + 1
    key = iy.astype(np.int64) * int(nx) + ix.astype(np.int64)

    order = np.argsort(key, kind="mergesort")
    key_s = key[order]
    starts = np.r_[0, 1 + np.flatnonzero(key_s[1:] != key_s[:-1])]
    ends = np.r_[starts[1:], key_s.size]
    ukeys = key_s[starts]

    ncell = int(ukeys.size)
    trusted = np.zeros(ncell, dtype=bool)
    mode_hi = np.full(ncell, np.nan, dtype=np.float64)
    mode_ref = np.full(ncell, np.nan, dtype=np.float64)

    for j, (a, b) in enumerate(zip(starts, ends)):
        if int(b - a) < int(cfg.min_ground_points_cell):
            continue
        loc = order[a:b]
        ok, _lo, hi, ref = _split_low_mode_from_bins(
            zg[loc],
            z_bin_m=cfg.z_bin_m,
            mode_gap_m=cfg.mode_gap_m,
            min_mode_points=cfg.min_ground_points_mode,
            min_mode_fraction=cfg.min_mode_fraction,
        )
        if ok:
            trusted[j] = True
            mode_hi[j] = hi
            mode_ref[j] = ref

    pos = np.searchsorted(ukeys, key)
    safe_pos = np.clip(pos, 0, max(0, ncell - 1))
    valid_pos = (pos >= 0) & (pos < ncell) & (ukeys[safe_pos] == key)

    protected = np.zeros(ground_idx.size, dtype=bool)
    suspicious = np.zeros(ground_idx.size, dtype=bool)
    hard = np.zeros(ground_idx.size, dtype=bool)

    direct = valid_pos & trusted[safe_pos]
    if np.any(direct):
        p = pos[direct]
        upper = mode_hi[p] + float(cfg.protect_margin_m)
        resid = zg[direct] - mode_ref[p]

        protected[direct] = zg[direct] <= upper

        direct_susp = (zg[direct] > (mode_hi[p] + float(cfg.min_above_mode_m))) & (
            resid >= float(cfg.min_above_mode_m)
        )
        direct_hard = resid >= float(cfg.hard_above_mode_m)

        ids = np.flatnonzero(direct)
        suspicious[ids[direct_susp]] = True
        hard[ids[direct_hard]] = True

    unsupported_cells = np.flatnonzero(~trusted)
    if unsupported_cells.size and np.any(trusted):
        trusted_keys = ukeys[trusted]
        tx = (trusted_keys % nx).astype(np.float64)
        ty = (trusted_keys // nx).astype(np.float64)

        trusted_xy = np.column_stack(
            (
                x0 + (tx + 0.5) * footprint_m,
                y0 + (ty + 0.5) * footprint_m,
            )
        )
        trusted_z = mode_ref[trusted]

        uk = ukeys[unsupported_cells]
        ux = (uk % nx).astype(np.float64)
        uy = (uk // nx).astype(np.float64)
        qx = x0 + (ux + 0.5) * footprint_m
        qy = y0 + (uy + 0.5) * footprint_m

        pred = _local_prediction(
            qx=qx,
            qy=qy,
            trusted_xy=trusted_xy,
            trusted_z=trusted_z,
            footprint_m=footprint_m,
            cfg=cfg,
        )

        pred_by_cell = np.full(ncell, np.nan, dtype=np.float64)
        pred_by_cell[unsupported_cells] = pred

        undirect = valid_pos & (~trusted[safe_pos])
        if np.any(undirect):
            pp = pos[undirect]
            pref = pred_by_cell[pp]
            good = np.isfinite(pref)
            ids = np.flatnonzero(undirect)

            resid = np.full(ids.size, np.nan, dtype=np.float64)
            resid[good] = zg[undirect][good] - pref[good]

            s = good & (resid >= float(cfg.min_above_mode_m))
            h = good & (resid >= float(cfg.hard_vote_residual_m))

            suspicious[ids[s]] = True
            hard[ids[h]] = True

    return suspicious, hard, protected, {
        "scale_m": float(footprint_m),
        "trusted_cells": int(np.count_nonzero(trusted)),
        "unsupported_cells": int(np.count_nonzero(~trusted)),
        "spike_candidates": int(np.count_nonzero(suspicious)),
    }



def _raster_reduce_surface(x, y, z, mask, res, mode):
    idx = np.flatnonzero(mask)
    xs, ys, zs = x[idx], y[idx], z[idx]
    x0 = float(np.floor(np.min(xs) / res) * res)
    y0 = float(np.floor(np.min(ys) / res) * res)
    ix = np.floor((xs - x0) / res).astype(np.int32)
    iy = np.floor((ys - y0) / res).astype(np.int32)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1
    key = iy.astype(np.int64) * nx + ix.astype(np.int64)
    order = np.lexsort((zs, key))
    ks = key[order]
    zz = zs[order]
    starts = np.r_[0, 1 + np.flatnonzero(ks[1:] != ks[:-1])]
    ends = np.r_[starts[1:], ks.size]
    ukeys = ks[starts]
    surf = np.full(nx * ny, np.nan, dtype=np.float64)
    vals = zz[starts] if mode == "min" else zz[ends - 1]
    surf[ukeys] = vals
    return surf.reshape(ny, nx), {"x0": x0, "y0": y0, "res": float(res), "nx": nx, "ny": ny}


def _fill_surface_nn(surface, sigma):
    valid = np.isfinite(surface)
    if not np.any(valid):
        return np.zeros_like(surface)
    if np.all(valid):
        out = surface.copy()
    else:
        _, inds = distance_transform_edt(~valid, return_indices=True)
        out = surface[tuple(inds)]
    if sigma > 0:
        out = gaussian_filter(out, sigma=float(sigma), mode="nearest")
    return out


def _sample_surface_to_grid(surface, src, dst):
    nyf, nxf = int(dst["ny"]), int(dst["nx"])
    xf = dst["x0"] + (np.arange(nxf, dtype=np.float64) + 0.5) * dst["res"]
    yf = dst["y0"] + (np.arange(nyf, dtype=np.float64) + 0.5) * dst["res"]
    fx = (xf - src["x0"]) / src["res"] - 0.5
    fy = (yf - src["y0"]) / src["res"] - 0.5
    x0 = np.floor(fx).astype(np.int64); y0 = np.floor(fy).astype(np.int64)
    wx = fx - x0; wy = fy - y0
    x0 = np.clip(x0, 0, surface.shape[1] - 1); y0 = np.clip(y0, 0, surface.shape[0] - 1)
    x1 = np.clip(x0 + 1, 0, surface.shape[1] - 1); y1 = np.clip(y0 + 1, 0, surface.shape[0] - 1)
    a = surface[y0[:, None], x0[None, :]]
    b = surface[y0[:, None], x1[None, :]]
    c = surface[y1[:, None], x0[None, :]]
    d = surface[y1[:, None], x1[None, :]]
    wx2 = wx[None, :]; wy2 = wy[:, None]
    return (1-wx2)*(1-wy2)*a + wx2*(1-wy2)*b + (1-wx2)*wy2*c + wx2*wy2*d


def _raster_bbox_second_pass(x, y, z, g, cfg):
    # Conservative second pass tuned against Site1 reference.
    fine_res, mid_res, coarse_res = 0.5, 5.0, 10.0
    fine, mf = _raster_reduce_surface(x, y, z, g, fine_res, "max")
    mid, mm = _raster_reduce_surface(x, y, z, g, mid_res, "min")
    coarse, mc = _raster_reduce_surface(x, y, z, g, coarse_res, "min")
    midf = _sample_surface_to_grid(_fill_surface_nn(mid, 0.65), mm, mf)
    coarsef = _sample_surface_to_grid(_fill_surface_nn(coarse, 0.50), mc, mf)
    diff = np.abs(midf - coarsef)
    thr = 0.90 + np.minimum(4.0, 1.25 * diff)
    e5 = fine - midf
    e10 = fine - coarsef
    valid = np.isfinite(fine)
    flagged = valid & (((diff <= 0.65) & ((e5 > thr) | (e10 > thr))) | ((diff > 0.65) & (e5 > thr) & (e10 > thr)) | ((e5 > 3.0) & (e10 > 3.0)))
    comps, ncomp = label(flagged.astype(np.uint8), structure=np.ones((3,3), dtype=np.uint8))
    # map points to fine cells
    ix = np.floor((x - mf["x0"]) / mf["res"]).astype(np.int64)
    iy = np.floor((y - mf["y0"]) / mf["res"]).astype(np.int64)
    inside = (ix>=0)&(iy>=0)&(ix<mf["nx"])&(iy<mf["ny"])
    cid = np.zeros(x.size, dtype=np.int32)
    vi = np.flatnonzero(inside)
    cid[vi] = comps[iy[vi], ix[vi]]
    ref5 = np.full(x.size, np.nan); ref10 = np.full(x.size, np.nan); pthr = np.full(x.size, np.nan)
    ref5[vi] = midf[iy[vi], ix[vi]]
    ref10[vi] = coarsef[iy[vi], ix[vi]]
    pthr[vi] = thr[iy[vi], ix[vi]]
    tref = np.minimum(ref5, ref10)
    residual = z - tref
    raw_candidate = g & (cid > 0) & np.isfinite(residual) & (residual > 0.5) & (residual > 0.70 * pthr)
    if not np.any(raw_candidate):
        return g, {"raster_components": int(ncomp), "raster_candidates": 0, "bbox_validated": 0, "bbox_demoted": 0}
    # trees for efficient XY box queries
    gi = np.flatnonzero(g); ni = np.flatnonzero(~g)
    gtree = cKDTree(np.column_stack((x[gi], y[gi])))
    ntree = cKDTree(np.column_stack((x[ni], y[ni])))
    idxc = np.flatnonzero(cid > 0)
    order = np.argsort(cid[idxc], kind="mergesort")
    ids = idxc[order]; cids = cid[ids]
    starts = np.r_[0, 1 + np.flatnonzero(cids[1:] != cids[:-1])]
    ends = np.r_[starts[1:], cids.size]
    comp_ids = cids[starts]
    demote = np.zeros(x.size, dtype=bool)
    valid_components = 0
    for a,b,cc in zip(starts, ends, comp_ids):
        inds = ids[a:b]
        an = inds[raw_candidate[inds]]
        if an.size == 0:
            continue
        margin = 4.0
        xmin, xmax = float(np.min(x[an]) - margin), float(np.max(x[an]) + margin)
        ymin, ymax = float(np.min(y[an]) - margin), float(np.max(y[an]) + margin)
        cx, cy = 0.5*(xmin+xmax), 0.5*(ymin+ymax)
        rad = float(np.hypot(0.5*(xmax-xmin), 0.5*(ymax-ymin)))
        gl = gi[gtree.query_ball_point([cx,cy], rad)]
        gl = gl[(x[gl]>=xmin)&(x[gl]<=xmax)&(y[gl]>=ymin)&(y[gl]<=ymax)]
        lower = gl[np.isfinite(tref[gl]) & (np.abs(z[gl]-tref[gl]) <= 1.25)]
        if lower.size < 3:
            continue
        lower_z = float(np.median(z[lower])); backend_z = float(np.median(tref[lower]))
        if abs(lower_z - backend_z) > 1.25:
            continue
        if float(np.median(z[an])) - lower_z < 1.25:
            continue
        zlo = float(np.percentile(z[an],5) - 0.75); zhi = float(np.percentile(z[an],95) + 0.75)
        nl = ni[ntree.query_ball_point([cx,cy], rad)]
        nl = nl[(x[nl]>=xmin)&(x[nl]<=xmax)&(y[nl]>=ymin)&(y[nl]<=ymax)&(z[nl]>=zlo)&(z[nl]<=zhi)]
        gv = gl[(z[gl]>=zlo)&(z[gl]<=zhi)]
        if nl.size + gv.size < 4:
            continue
        ng_point_fraction = float(nl.size / max(nl.size + gv.size, 1))
        xyz = np.r_[gv, nl]
        isg = np.r_[np.ones(gv.size, dtype=bool), np.zeros(nl.size, dtype=bool)]
        vv = 0.75
        vx = np.floor((x[xyz]-xmin)/vv).astype(np.int32)
        vy = np.floor((y[xyz]-ymin)/vv).astype(np.int32)
        vz = np.floor((z[xyz]-zlo)/vv).astype(np.int32)
        vox = np.column_stack((vx,vy,vz))
        gvox = np.unique(vox[isg], axis=0).shape[0] if gv.size else 0
        nvox = np.unique(vox[~isg], axis=0).shape[0] if nl.size else 0
        ng_voxel_fraction = float(nvox / max(gvox+nvox,1))
        if ((ng_point_fraction >= 0.50 and ng_voxel_fraction >= 0.50) or ng_point_fraction >= 0.70 or ng_voxel_fraction >= 0.70):
            demote[an] = True
            valid_components += 1
    out = g.copy()
    out[demote] = False
    return out, {"raster_components": int(ncomp), "raster_candidates": int(np.count_nonzero(raw_candidate)), "bbox_validated": int(valid_components), "bbox_demoted": int(np.count_nonzero(demote))}


def _local_robust_anomaly_third_pass(x, y, z, g):
    """
    Iterative canopy-top anomaly operator.

    Candidate generation:
      * 0.5 m MAX final-ground raster
      * 5 m / 10 m MIN lower-envelope backend
      * local robust residual anomaly:
            S = (R - median_local(R)) /
                (1.4826 * MAD_local(R) + 0.25)

    Candidate requirements:
      * S > 4.5
      * height above backend > 1.5 m

    Counterpart safeguards:
      A) same-level neighborhood is majority non-ground, OR
      B) the candidate is at/near a canopy top where the top itself can be
         ground-majority, but the vertical column immediately BELOW it has a
         persistent non-ground population.

    The below-top check is deliberately asymmetric:
      XY radius : 2.5 m
      Z slab    : 0.5 to 5.0 m below candidate
      evidence  : >= 6 points, >= 55% non-ground, and non-ground support in
                  at least two 0.5 m vertical bins.

    This catches leaked ground sitting on the canopy crown while preserving a
    true terrain sheet, because candidate generation already requires the point
    to be a strong positive local anomaly above the 5/10 m terrain backend.

    ALS final-stage only. Ground -> Non-ground only.
    """
    from scipy.ndimage import median_filter

    empty_qc = {
        "local_anomaly_candidates": 0,
        "local_anomaly_validated_cells": 0,
        "local_anomaly_samelevel_cells": 0,
        "local_anomaly_belowtop_cells": 0,
        "local_anomaly_demoted": 0,
    }

    if np.count_nonzero(g) < 20:
        return g, empty_qc.copy()

    fine, mf = _raster_reduce_surface(x, y, z, g, 0.5, "max")
    s5, m5 = _raster_reduce_surface(x, y, z, g, 5.0, "min")
    s10, m10 = _raster_reduce_surface(x, y, z, g, 10.0, "min")

    f5 = _fill_surface_nn(s5, 0.65)
    f10 = _fill_surface_nn(s10, 0.50)

    b5 = _sample_surface_to_grid(f5, m5, mf)
    b10 = _sample_surface_to_grid(f10, m10, mf)
    backend = np.minimum(b5, b10)

    residual = fine - backend
    valid = np.isfinite(fine) & np.isfinite(residual)
    if not np.any(valid):
        return g, empty_qc.copy()

    if np.all(valid):
        rf = residual.copy()
    else:
        _, inds = distance_transform_edt(~valid, return_indices=True)
        rf = residual.copy()
        rf[~valid] = residual[tuple(inds)][~valid]

    # 9 x 9 at 0.5 m = 4.5 m local context.
    local_med = median_filter(rf, size=9, mode="nearest")
    local_mad = median_filter(np.abs(rf - local_med), size=9, mode="nearest")
    robust_score = (residual - local_med) / (1.4826 * local_mad + 0.25)

    cell_candidate = (
        valid
        & (robust_score > 4.5)
        & (residual > 1.5)
    )

    if not np.any(cell_candidate):
        return g, empty_qc.copy()

    ix = np.floor((x - mf["x0"]) / mf["res"]).astype(np.int64)
    iy = np.floor((y - mf["y0"]) / mf["res"]).astype(np.int64)
    inside = (
        (ix >= 0) & (iy >= 0)
        & (ix < mf["nx"]) & (iy < mf["ny"])
    )

    p_backend = np.full(x.size, np.nan, dtype=np.float64)
    p_score = np.full(x.size, np.nan, dtype=np.float64)
    p_cell_candidate = np.zeros(x.size, dtype=bool)

    vi = np.flatnonzero(inside)
    p_backend[vi] = backend[iy[vi], ix[vi]]
    p_score[vi] = robust_score[iy[vi], ix[vi]]
    p_cell_candidate[vi] = cell_candidate[iy[vi], ix[vi]]

    raw = (
        g
        & p_cell_candidate
        & np.isfinite(p_backend)
        & ((z - p_backend) > 1.5)
        & (p_score > 4.5)
    )

    raw_idx = np.flatnonzero(raw)
    if raw_idx.size == 0:
        return g, empty_qc.copy()

    # Validate once per unique 0.5 m candidate cell.
    cell_key = iy[raw_idx].astype(np.int64) * int(mf["nx"]) + ix[raw_idx].astype(np.int64)
    order = np.argsort(cell_key, kind="mergesort")
    rid = raw_idx[order]
    keys = cell_key[order]
    starts = np.r_[0, 1 + np.flatnonzero(keys[1:] != keys[:-1])]
    ends = np.r_[starts[1:], keys.size]

    all_tree = cKDTree(np.column_stack((x, y)))

    demote = np.zeros(x.size, dtype=bool)
    validated_cells = 0
    samelevel_cells = 0
    belowtop_cells = 0

    for a, b in zip(starts, ends):
        pts = rid[a:b]
        if pts.size == 0:
            continue

        cx = float(np.median(x[pts]))
        cy = float(np.median(y[pts]))
        cz = float(np.median(z[pts]))

        nei_xy = np.asarray(
            all_tree.query_ball_point([cx, cy], 2.5),
            dtype=np.int64,
        )
        if nei_xy.size == 0:
            continue

        # A) Same-height counterpart vote.
        same = nei_xy[np.abs(z[nei_xy] - cz) <= 1.0]
        same_ok = False
        if same.size >= 4:
            same_ng = float(np.mean(~g[same]))
            same_ok = same_ng >= 0.50

        # B) Canopy-top safeguard: even if the very top is currently
        # ground-majority, inspect the column immediately BELOW it.
        below = nei_xy[
            (z[nei_xy] <= cz - 0.5)
            & (z[nei_xy] >= cz - 5.0)
        ]
        below_ok = False
        if below.size >= 6:
            below_ng_fraction = float(np.mean(~g[below]))

            # Vertical persistence: non-ground should occur in at least two
            # different 0.5 m bins beneath the candidate.
            ng_below = below[~g[below]]
            if ng_below.size >= 4:
                zb = np.floor((cz - z[ng_below]) / 0.5).astype(np.int32)
                n_bins = int(np.unique(zb).size)
            else:
                n_bins = 0

            below_ok = (
                below_ng_fraction >= 0.55
                and n_bins >= 2
            )

        if not (same_ok or below_ok):
            continue

        demote[pts] = True
        validated_cells += 1
        samelevel_cells += int(same_ok)
        belowtop_cells += int((not same_ok) and below_ok)

    out = g.copy()
    out[demote] = False

    return out, {
        "local_anomaly_candidates": int(raw_idx.size),
        "local_anomaly_validated_cells": int(validated_cells),
        "local_anomaly_samelevel_cells": int(samelevel_cells),
        "local_anomaly_belowtop_cells": int(belowtop_cells),
        "local_anomaly_demoted": int(np.count_nonzero(demote)),
    }


def refine_als_ground_vertical_profile(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ground_mask: np.ndarray,
    sensor_mode: str,
    cfg: dict | None = None,
    return_report: bool = False,
):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    g = np.asarray(ground_mask, dtype=bool).copy()

    report = _empty_report(np.count_nonzero(g))

    sm = str(sensor_mode).upper().strip()
    if sm not in {"ALS", "ULS"}:
        return (g, report) if return_report else g

    gi = np.flatnonzero(g)
    if gi.size < 20 or x.size == 0:
        return (g, report) if return_report else g

    c = _cfg_from_dict(cfg)

    votes = np.zeros(gi.size, dtype=np.uint8)
    hard_votes = np.zeros(gi.size, dtype=np.uint8)
    protected_votes = np.zeros(gi.size, dtype=np.uint8)

    for scale in c.footprints_m:
        susp, hard, protected, diag = _analyze_scale(
            x=x,
            y=y,
            z=z,
            ground_idx=gi,
            footprint_m=float(scale),
            cfg=c,
        )

        votes += susp.astype(np.uint8)
        hard_votes += hard.astype(np.uint8)
        protected_votes += protected.astype(np.uint8)

        report["scales_m"].append(float(scale))
        report["trusted_cells_per_scale"].append(int(diag["trusted_cells"]))
        report["unsupported_cells_per_scale"].append(int(diag["unsupported_cells"]))
        report["spike_candidates_per_scale"].append(int(diag["spike_candidates"]))

    normal = (votes >= int(c.min_votes)) & (protected_votes == 0)
    hard = (hard_votes >= 1) & (protected_votes == 0)
    demote_local = normal | hard

    report["consensus_candidates"] = int(np.count_nonzero(normal))
    report["hard_candidates"] = int(np.count_nonzero(hard))

    bad_idx = gi[demote_local]
    if bad_idx.size:
        g[bad_idx] = False

    report["demoted_points"] = int(bad_idx.size)
    report["ground_after"] = int(np.count_nonzero(g))

    # Conservative raster + 3-D counterpart validation second pass.
    g2, raster_qc = _raster_bbox_second_pass(x, y, z, g, cfg)
    extra = int(np.count_nonzero(g & ~g2))
    g = g2
    report["raster_components"] = int(raster_qc.get("raster_components", 0))
    report["raster_candidates"] = int(raster_qc.get("raster_candidates", 0))
    report["bbox_validated"] = int(raster_qc.get("bbox_validated", 0))
    report["bbox_demoted"] = int(raster_qc.get("bbox_demoted", 0))
    report["demoted_points"] = int(report["demoted_points"] + extra)
    report["ground_after"] = int(np.count_nonzero(g))

    # Iterative local robust anomaly refinement.
    #
    # Recompute the 0.5/5/10 m surfaces after each demotion round.  This is
    # important because removing the first layer of canopy-ground leakage can
    # expose a second lower layer that was previously hidden by the MAX raster.
    local_total = 0
    local_candidates_total = 0
    local_validated_total = 0
    samelevel_total = 0
    belowtop_total = 0
    iterations_run = 0

    for _iter in range(4):
        g_next, local_qc = _local_robust_anomaly_third_pass(x, y, z, g)
        removed = int(np.count_nonzero(g & ~g_next))
        iterations_run += 1

        local_candidates_total += int(
            local_qc.get("local_anomaly_candidates", 0)
        )
        local_validated_total += int(
            local_qc.get("local_anomaly_validated_cells", 0)
        )
        samelevel_total += int(
            local_qc.get("local_anomaly_samelevel_cells", 0)
        )
        belowtop_total += int(
            local_qc.get("local_anomaly_belowtop_cells", 0)
        )

        if removed <= 0:
            break

        g = g_next
        local_total += removed

    report["local_anomaly_iterations"] = int(iterations_run)
    report["local_anomaly_candidates"] = int(local_candidates_total)
    report["local_anomaly_validated_cells"] = int(local_validated_total)
    report["local_anomaly_samelevel_cells"] = int(samelevel_total)
    report["local_anomaly_belowtop_cells"] = int(belowtop_total)
    report["local_anomaly_demoted"] = int(local_total)

    report["demoted_points"] = int(report["demoted_points"] + local_total)
    report["ground_after"] = int(np.count_nonzero(g))

    return (g, report) if return_report else g


__all__ = [
    "AlsGroundVerticalProfileConfig",
    "refine_als_ground_vertical_profile",
]
