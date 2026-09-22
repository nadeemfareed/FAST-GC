from __future__ import annotations

import math
import numpy as np
from scipy import ndimage


def _density(cfg, x, y):
    for key in (
        "adaptive_support_dataset_density_pts_m2",
        "dataset_density_pts_m2",
        "density_pts_m2",
    ):
        value = cfg.get(key)
        if value is not None:
            try:
                value = float(value)
            except Exception:
                value = None
            if value is not None and np.isfinite(value) and value > 0:
                return value
    if x.size < 2:
        return 0.0
    sx = max(float(np.max(x) - np.min(x)), 1e-6)
    sy = max(float(np.max(y) - np.min(y)), 1e-6)
    return float(x.size / (sx * sy))


def _mad(a):
    if a.size == 0:
        return float("nan")
    m = float(np.median(a))
    return float(1.4826 * np.median(np.abs(a - m)))


def _ground_grid(x, y, z, ground, cell=1.0):
    gi = np.flatnonzero(ground)
    if gi.size == 0:
        return None
    gx, gy, gz = x[gi], y[gi], z[gi]
    x0 = math.floor(float(np.min(gx)) / cell) * cell
    y0 = math.floor(float(np.min(gy)) / cell) * cell
    ix = np.floor((gx - x0) / cell).astype(np.int64)
    iy = np.floor((gy - y0) / cell).astype(np.int64)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1
    cid = iy * nx + ix
    order = np.argsort(cid, kind="mergesort")
    cs = cid[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    ends = np.r_[starts[1:], cs.size]
    flat = np.full(nx * ny, np.nan, dtype=np.float64)
    for s, e in zip(starts, ends):
        c = int(cs[s])
        flat[c] = float(np.median(gz[order[s:e]]))
    zz = flat.reshape(ny, nx)
    valid = np.isfinite(zz)
    return dict(z=zz, valid=valid, x0=x0, y0=y0, cell=cell,
                ix=ix, iy=iy, gi=gi)


def _nearest_fill(z, valid):
    if np.all(valid):
        return z.copy()
    inds = ndimage.distance_transform_edt(
        ~valid, return_distances=False, return_indices=True
    )
    out = z.copy()
    out[~valid] = z[tuple(inds[:, ~valid])]
    return out


def _fields(zfill, cell, smooth_m):
    sigma = max(float(smooth_m) / (6.0 * cell), 0.55)
    zs = ndimage.gaussian_filter(zfill, sigma=sigma, mode="nearest")
    dzdy, dzdx = np.gradient(zs, cell, cell)
    slope = np.degrees(np.arctan(np.sqrt(dzdx * dzdx + dzdy * dzdy)))
    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(zs)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    return slope, (nx / norm, ny / norm, nz / norm)


def _normal_stats(normals, mask):
    if np.count_nonzero(mask) < 3:
        return 90.0, np.array([0.0, 0.0, 1.0])
    nx, ny, nz = normals
    V = np.column_stack((nx[mask], ny[mask], nz[mask]))
    mean = np.mean(V, axis=0)
    n = np.linalg.norm(mean)
    if n <= 1e-12:
        return 90.0, np.array([0.0, 0.0, 1.0])
    mean /= n
    ang = np.degrees(np.arccos(np.clip(V @ mean, -1.0, 1.0)))
    return float(np.median(ang)), mean


def _angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _plane(xx, yy, zz):
    if zz.size < 18:
        return None
    A = np.column_stack((xx, yy, np.ones(zz.size)))
    keep = np.ones(zz.size, dtype=bool)
    coef = None
    for _ in range(5):
        if np.count_nonzero(keep) < 12:
            return None
        coef, *_ = np.linalg.lstsq(A[keep], zz[keep], rcond=None)
        r = zz - A @ coef
        rk = r[keep]
        med = float(np.median(rk))
        sig = max(_mad(rk), 0.06)
        nk = np.abs(r - med) <= 3.0 * sig
        if np.array_equal(nk, keep):
            keep = nk
            break
        keep = nk
    if coef is None:
        return None
    r = zz - A @ coef
    return coef, max(_mad(r[keep]), 0.06)


def _elongation(mask):
    yy, xx = np.nonzero(mask)
    if xx.size < 3:
        return 1.0
    X = np.column_stack((xx - np.mean(xx), yy - np.mean(yy)))
    C = (X.T @ X) / max(1, X.shape[0] - 1)
    vals = np.maximum(np.linalg.eigvalsh(C), 1e-9)
    return float(np.sqrt(vals[-1] / vals[0]))


def apply_d2_reverse_context_vote(
    *, x, y, z, ground_mask, sensor_mode, cfg=None, return_report=True
):
    """Reverse D2 audit: Z + slope distribution + normals at coarse scales."""
    cfg = cfg or {}
    sm = str(sensor_mode).upper().strip()
    g = np.asarray(ground_mask, dtype=bool).copy()
    report = dict(
        enabled=False, activated=False, density_pts_m2=0.0,
        trigger_lt_pts_m2=3.0, candidate_components=0,
        audited_components=0, approved_components=0,
        terrain_veto_components=0, z_votes=0, slope_votes=0,
        normal_votes=0, demoted_points=0,
        ground_before=int(np.count_nonzero(g)),
        ground_after=int(np.count_nonzero(g)),
    )
    if sm not in {"ALS", "ULS"}:
        return (g, report) if return_report else g
    if not bool(cfg.get("d2_reverse_context_enabled", True)):
        return (g, report) if return_report else g

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if not (x.size == y.size == z.size == g.size):
        return (g, report) if return_report else g

    density = _density(cfg, x, y)
    trigger = float(cfg.get("d2_reverse_context_trigger_pts_m2", 3.0))
    report["enabled"] = True
    report["density_pts_m2"] = density
    report["trigger_lt_pts_m2"] = trigger
    if density >= trigger:
        return (g, report) if return_report else g
    report["activated"] = True

    cell = float(cfg.get("d2_reverse_grid_cell_m", 1.0))
    G = _ground_grid(x, y, z, g, cell)
    if G is None:
        return (g, report) if return_report else g
    zg, valid = G["z"], G["valid"]
    zfill = _nearest_fill(zg, valid)

    # Candidate discovery from moderate/broad context only.
    k15 = max(3, int(round(15.0 / cell)) | 1)
    k25 = max(3, int(round(25.0 / cell)) | 1)
    med15 = ndimage.median_filter(zfill, size=k15, mode="nearest")
    med25 = ndimage.median_filter(zfill, size=k25, mode="nearest")
    r15 = zfill - med15
    r25 = zfill - med25
    slope3, _ = _fields(zfill, cell, 3.0)
    cand = valid & (((r15 > 0.28) & (slope3 > 38.0)) | (r25 > 0.55))
    cand = ndimage.binary_dilation(cand, iterations=1)
    cand &= valid & ((r15 > 0.12) | (r25 > 0.20))

    labels, nlab = ndimage.label(cand, np.ones((3, 3), dtype=np.uint8))
    report["candidate_components"] = int(nlab)
    if nlab == 0:
        return (g, report) if return_report else g

    contexts = ((18.0, 5.0), (30.0, 8.0), (45.0, 12.0))
    field_cache = {s: _fields(zfill, cell, s) for _, s in contexts}
    remove_cells = np.zeros_like(valid, dtype=bool)

    for lab in range(1, nlab + 1):
        comp = labels == lab
        area = np.count_nonzero(comp) * cell * cell
        if area < 1.0 or area > 450.0:
            continue
        report["audited_components"] += 1

        yy, xx = np.nonzero(comp)
        pad = int(math.ceil(48.0 / cell))
        ya, yb = max(0, yy.min()-pad), min(comp.shape[0], yy.max()+pad+1)
        xa, xb = max(0, xx.min()-pad), min(comp.shape[1], xx.max()+pad+1)
        lc = comp[ya:yb, xa:xb]
        lv = valid[ya:yb, xa:xb]
        lz = zg[ya:yb, xa:xb]
        dist = ndimage.distance_transform_edt(~lc) * cell

        zv = sv = nv = complex_votes = 0
        elong = _elongation(lc)

        for radius, smooth_m in contexts:
            ann = lv & (dist >= 2.0) & (dist <= radius)
            if np.count_nonzero(ann) < 24:
                continue

            ay, ax = np.nonzero(ann)
            fit = _plane(ax.astype(float)*cell, ay.astype(float)*cell, lz[ann])
            if fit is None:
                continue
            coef, sig = fit
            cy, cx = np.nonzero(lc & lv)
            if cx.size == 0:
                continue
            pred = coef[0]*(cx*cell) + coef[1]*(cy*cell) + coef[2]
            prominence = float(np.median(lz[lc & lv] - pred))
            if prominence > max(0.24, 2.5*sig):
                zv += 1

            slope, normals = field_cache[smooth_m]
            ls = slope[ya:yb, xa:xb]
            ln = tuple(a[ya:yb, xa:xb] for a in normals)
            cs, rs = ls[lc], ls[ann]
            if cs.size < 2 or rs.size < 8:
                continue

            cp90, rp90 = np.percentile(cs, [90])[0], np.percentile(rs, [90])[0]
            cp95, rp95 = np.percentile(cs, [95])[0], np.percentile(rs, [95])[0]
            c70, r70 = float(np.mean(cs >= 70.0)), float(np.mean(rs >= 70.0))
            if ((cp90-rp90 >= 18.0 and c70 >= r70+0.15) or
                (cp95 >= 72.0 and rp95 <= 52.0 and c70 >= 0.15)):
                sv += 1

            cdisp, cmean = _normal_stats(ln, lc)
            rdisp, rmean = _normal_stats(ln, ann)
            mismatch = _angle(cmean, rmean)
            if ((rdisp <= 24.0 and cdisp >= rdisp+14.0) or
                (rdisp <= 22.0 and mismatch >= 24.0)):
                nv += 1

            # Coarse terrain-complexity protection.
            if rp90 >= 58.0 or rdisp >= 32.0:
                complex_votes += 1

        report["z_votes"] += zv
        report["slope_votes"] += sv
        report["normal_votes"] += nv

        terrain_veto = (complex_votes >= 2 or (elong >= 3.5 and complex_votes >= 1))
        if terrain_veto:
            report["terrain_veto_components"] += 1
            continue

        # Z prominence is mandatory; slope and normals must independently agree.
        evidence = zv + sv + nv
        if zv >= 2 and sv >= 1 and nv >= 1 and evidence >= 6:
            remove_cells |= comp
            report["approved_components"] += 1

    if np.any(remove_cells):
        pix = remove_cells[G["iy"], G["ix"]]
        remove_global = G["gi"][pix]
        max_remove = max(1, int(math.ceil(0.025 * max(1, np.count_nonzero(g)))))
        if remove_global.size > max_remove:
            scores = r25[G["iy"][pix], G["ix"][pix]]
            keep = np.argsort(scores)[-max_remove:]
            remove_global = remove_global[keep]
        g[remove_global] = False
        report["demoted_points"] = int(remove_global.size)

    report["ground_after"] = int(np.count_nonzero(g))
    return (g, report) if return_report else g


__all__ = ["apply_d2_reverse_context_vote"]
