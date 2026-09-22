from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class TlsInvertDsmVoteConfig:
    """Stable TLS baseline.

    Design goals:
      * preserve the existing strong planar TLS result;
      * use a globally snapped grid so neighboring buffered tiles share phase;
      * never wrap information across tile edges;
      * establish strong/support-rich terrain first and propagate toward weak TLS support;
      * use a robust plane first and a quadratic fallback only where coherent curvature
        makes the plane inadequate;
      * keep the public CLI unchanged (sensor_mode=TLS).

    The field list is intentionally a superset of current/legacy io_las callers so
    this module can replace older tls_vote.py versions without touching io_las.py.
    """

    # Grid / candidate extraction
    cell: float = 0.35
    top_m: int = 12
    candidate_cluster_gap: float = 0.08
    candidate_cluster_span: float = 0.14
    candidate_min_cluster_points: int = 3
    candidate_quantile: float = 0.50
    below_ground_gap_threshold: float = 0.16
    below_ground_max_cluster_points: int = 2
    use_offset_grid: bool = True

    # Neighborhood / robust support
    neighbor_radius_cells: int = 4
    radius_min_cells: int = 2
    radius_max_cells: int = 12
    min_neighbor_cells: int = 8
    min_support_sectors: int = 3
    max_robust_z: float = 2.6
    mad_floor: float = 0.03
    density_reference_quantile: float = 0.60

    # Strong support seeds / wavefront propagation
    seed_confidence: float = 0.68
    seed_min_neighbor_cells: int = 8
    seed_max_vertical_span: float = 0.35
    propagation_iters: int = 24
    max_extrapolation_cells: int = 5
    plane_min_support: int = 6
    plane_max_residual: float = 0.14

    # Conservative hole fill
    fill_iters: int = 24
    fill_min_bank_fraction: float = 0.50
    fill_min_support_sectors: int = 3
    fill_max_distance_cells: int = 5
    reject_edge_connected_voids: bool = True

    # Surface smoothing / classification
    smooth_sigma_cells: float = 0.8
    smooth_iters: int = 1
    bilateral_height_sigma: float = 0.18
    bilateral_slope_sigma: float = 0.60
    ground_threshold: float = 0.10
    slope_adapt_k: float = 0.25
    curvature_adapt_k: float = 0.05
    roughness_adapt_k: float = 0.30
    threshold_min: float = 0.03
    threshold_max: float = 0.28
    lower_threshold_factor: float = 1.75
    min_surface_confidence: float = 0.28
    min_classification_confidence: float = 0.34
    weak_confidence_tighten: float = 0.45

    # Curvature fallback. Internal TLS defaults; not CLI options.
    curvature_enabled: bool = True
    quadratic_min_support: int = 10
    quadratic_trigger_residual: float = 0.055
    quadratic_min_improvement: float = 0.25
    quadratic_max_abs_curvature: float = 1.50

    # Final TLS surface QC. These are internal defaults so io_las/CLI stay unchanged.
    protrusion_guard_enabled: bool = True
    protrusion_guard_radius_cells: int = 6
    protrusion_guard_min_support: int = 10
    protrusion_guard_min_sectors: int = 4
    protrusion_guard_base_m: float = 0.16
    protrusion_guard_roughness_k: float = 2.5
    protrusion_guard_passes: int = 2

    # Vertical-column veto: in cells containing vegetation/woody vertical structure,
    # retain only a thin band around the terrain surface. This prevents the broad
    # slope-adaptive threshold from swallowing stems/logs/shrubs.
    column_guard_enabled: bool = True
    column_guard_span_m: float = 0.55
    column_guard_upper_normal_m: float = 0.075
    column_guard_curvature_k: float = 0.025
    column_guard_max_normal_m: float = 0.12


_SURFACE_DIAGNOSTICS: dict[int, dict[str, np.ndarray]] = {}


def _snap_origin(v: float, cell: float) -> float:
    """Global grid phase independent of tile-local minima."""
    return math.floor(float(v) / float(cell)) * float(cell)


def _grid_index(x: np.ndarray, y: np.ndarray, cell: float, *, x0=None, y0=None):
    if x0 is None:
        x0 = _snap_origin(float(np.min(x)), cell)
    if y0 is None:
        y0 = _snap_origin(float(np.min(y)), cell)
    ix = np.floor((x - x0) / cell).astype(np.int64)
    iy = np.floor((y - y0) / cell).astype(np.int64)
    return float(x0), float(y0), ix, iy


def _grid_shape(ix: np.ndarray, iy: np.ndarray) -> tuple[int, int]:
    return int(ix.max()) + 1, int(iy.max()) + 1


def _pack_key(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    return (ix.astype(np.int64) << 32) | (iy.astype(np.int64) & 0xFFFFFFFF)


def _lowest_coherent_cluster(vals: np.ndarray, cfg: TlsInvertDsmVoteConfig):
    z = np.sort(np.asarray(vals, np.float64))
    if z.size == 0:
        return math.nan, 0, math.nan, 0.0
    z = z[: max(2, int(cfg.top_m))]
    gaps = np.diff(z)
    split = np.flatnonzero(gaps > float(cfg.candidate_cluster_gap)) + 1
    clusters = np.split(z, split)
    chosen = None
    for i, c in enumerate(clusters):
        if c.size == 0:
            continue
        if i + 1 < len(clusters):
            gap_next = float(clusters[i + 1][0] - c[-1])
            if c.size <= int(cfg.below_ground_max_cluster_points) and gap_next >= float(cfg.below_ground_gap_threshold):
                continue
        span = float(np.ptp(c)) if c.size > 1 else 0.0
        if c.size >= int(cfg.candidate_min_cluster_points) and span <= float(cfg.candidate_cluster_span):
            chosen = c
            break
    if chosen is None:
        # Sparse fallback: low-tail median, safer than min(z).
        k = min(z.size, max(2, int(cfg.candidate_min_cluster_points)))
        chosen = z[:k]
    q = float(np.clip(cfg.candidate_quantile, 0.0, 1.0))
    center = float(np.quantile(chosen, q))
    span = float(np.ptp(chosen)) if chosen.size > 1 else 0.0
    support = int(chosen.size)
    support_score = min(1.0, support / max(1.0, 2.0 * float(cfg.candidate_min_cluster_points)))
    compact = math.exp(-((span / max(float(cfg.candidate_cluster_span), 1e-6)) ** 2))
    conf = float(np.clip(0.60 * support_score + 0.40 * compact, 0.0, 1.0))
    return center, support, span, conf


def _build_candidate_grid(x, y, z, cell, cfg, *, x0, y0):
    _, _, ix, iy = _grid_index(x, y, cell, x0=x0, y0=y0)
    valid = (ix >= 0) & (iy >= 0)
    ixv, iyv, zv = ix[valid], iy[valid], z[valid]
    nx, ny = _grid_shape(ixv, iyv)
    cand = np.full((ny, nx), np.nan, np.float64)
    count = np.zeros((ny, nx), np.int32)
    span = np.full((ny, nx), np.nan, np.float64)
    conf = np.zeros((ny, nx), np.float64)
    column_span = np.full((ny, nx), np.nan, np.float64)
    key = _pack_key(ixv, iyv)
    order = np.argsort(key, kind="mergesort")
    keys, zs = key[order], zv[order]
    uniq, starts = np.unique(keys, return_index=True)
    ends = np.r_[starts[1:], zs.size]
    for u, a, b in zip(uniq, starts, ends):
        gx, gy = int(u >> 32), int(u & 0xFFFFFFFF)
        cellz = zs[a:b]
        c, n, s, cf = _lowest_coherent_cluster(cellz, cfg)
        cand[gy, gx], count[gy, gx], span[gy, gx], conf[gy, gx] = c, int(n), s, cf
        if cellz.size > 1:
            # Robust full-column extent. Percentiles are safer than raw min/max for TLS noise.
            q10, q90 = np.quantile(cellz, [0.10, 0.90])
            column_span[gy, gx] = float(q90 - q10)
        else:
            column_span[gy, gx] = 0.0
    return cand, count, span, conf, column_span


def _sample_nearest(grid, x, y, x0, y0, cell, fill=np.nan):
    ix = np.floor((x - x0) / cell).astype(np.int64)
    iy = np.floor((y - y0) / cell).astype(np.int64)
    out = np.full(np.shape(x), fill, dtype=np.float64)
    ok = (ix >= 0) & (iy >= 0) & (ix < grid.shape[1]) & (iy < grid.shape[0])
    out[ok] = grid[iy[ok], ix[ok]]
    return out


def _bilinear_sample(grid, x, y, x0, y0, cell):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    gx = (x - x0) / cell - 0.5
    gy = (y - y0) / cell - 0.5
    ix = np.floor(gx).astype(np.int64); iy = np.floor(gy).astype(np.int64)
    fx = gx - ix; fy = gy - iy
    out = np.full(x.shape, np.nan, np.float64)
    ok = (ix >= 0) & (iy >= 0) & (ix + 1 < grid.shape[1]) & (iy + 1 < grid.shape[0])
    ids = np.flatnonzero(ok)
    if ids.size:
        iix, iiy = ix[ids], iy[ids]
        g00 = grid[iiy, iix]; g10 = grid[iiy, iix + 1]
        g01 = grid[iiy + 1, iix]; g11 = grid[iiy + 1, iix + 1]
        finite = np.isfinite(g00) & np.isfinite(g10) & np.isfinite(g01) & np.isfinite(g11)
        vals = ((1-fx[ids])*(1-fy[ids])*g00 + fx[ids]*(1-fy[ids])*g10 +
                (1-fx[ids])*fy[ids]*g01 + fx[ids]*fy[ids]*g11)
        out[ids[finite]] = vals[finite]
    # nearest supported fallback, without wrapping
    miss = ~np.isfinite(out)
    if np.any(miss):
        out[miss] = _sample_nearest(grid, x[miss], y[miss], x0, y0, cell)
    return out


def _map_offset_to_base(grid, base_shape, x0_base, y0_base, x0_off, y0_off, cell):
    ny, nx = base_shape
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    wx = x0_base + (xx + 0.5) * cell
    wy = y0_base + (yy + 0.5) * cell
    return _bilinear_sample(grid, wx.ravel(), wy.ravel(), x0_off, y0_off, cell).reshape(ny, nx)


def _merge_base_offset(base, base_conf, off, off_conf):
    out = base.copy(); conf = base_conf.copy()
    only = ~np.isfinite(out) & np.isfinite(off)
    out[only], conf[only] = off[only], off_conf[only]
    both = np.isfinite(out) & np.isfinite(off)
    agree = both & (np.abs(out - off) <= 0.10)
    w0 = np.maximum(conf, 1e-6); w1 = np.maximum(off_conf, 1e-6)
    out[agree] = (w0[agree] * out[agree] + w1[agree] * off[agree]) / (w0[agree] + w1[agree])
    conf[agree] = np.clip(0.5 * (conf[agree] + off_conf[agree]) + 0.20, 0, 1)
    prefer = both & ~agree & (off_conf > conf + 0.08)
    out[prefer], conf[prefer] = off[prefer], off_conf[prefer]
    return out, conf


def _support_sectors(xs, ys, cx, cy):
    if len(xs) == 0:
        return 0
    ang = np.arctan2(ys - cy, xs - cx)
    sec = np.floor((ang + np.pi) / (np.pi / 4.0)).astype(np.int32)
    return int(np.unique(np.clip(sec, 0, 7)).size)


def _fit_local(surface, confidence, cx, cy, radius, cfg, *, exclude_center: bool = False):
    ny, nx = surface.shape
    y0, y1 = max(0, cy-radius), min(ny, cy+radius+1)
    x0, x1 = max(0, cx-radius), min(nx, cx+radius+1)
    sub = surface[y0:y1, x0:x1]
    csub = confidence[y0:y1, x0:x1]
    good = np.isfinite(sub) & (csub >= float(cfg.min_surface_confidence))
    if exclude_center and y0 <= cy < y1 and x0 <= cx < x1:
        good[cy - y0, cx - x0] = False
    yy, xx = np.nonzero(good)
    if xx.size < int(cfg.plane_min_support):
        return math.nan, math.inf, 0, 0, 0.0, False
    xxw = xx + x0; yyw = yy + y0; zz = surface[yyw, xxw]; ww = np.maximum(confidence[yyw, xxw], 0.05)
    dx = xxw.astype(np.float64) - float(cx); dy = yyw.astype(np.float64) - float(cy)
    sectors = _support_sectors(xxw, yyw, cx, cy)

    Xp = np.column_stack((dx, dy, np.ones_like(dx)))
    keep = np.ones(zz.size, bool)
    beta = np.array([0.0, 0.0, float(np.median(zz))])
    for _ in range(3):
        if keep.sum() < int(cfg.plane_min_support): break
        sw = np.sqrt(ww[keep])[:, None]
        try:
            beta, *_ = np.linalg.lstsq(Xp[keep] * sw, zz[keep] * sw[:,0], rcond=None)
        except np.linalg.LinAlgError:
            break
        r = zz - Xp @ beta
        med = float(np.median(r[keep])); mad = max(float(np.median(np.abs(r[keep]-med))), float(cfg.mad_floor))
        keep = np.abs(r-med) <= float(cfg.max_robust_z) * 1.4826 * mad
    rp = zz - Xp @ beta
    plane_rmse = float(np.sqrt(np.average(rp[keep]**2, weights=ww[keep]))) if keep.any() else math.inf
    pred = float(beta[2]); curvature = 0.0; used_quad = False

    if cfg.curvature_enabled and zz.size >= int(cfg.quadratic_min_support) and plane_rmse >= float(cfg.quadratic_trigger_residual):
        Xq = np.column_stack((dx, dy, dx*dx, dx*dy, dy*dy, np.ones_like(dx)))
        try:
            sw = np.sqrt(ww)[:, None]
            bq, *_ = np.linalg.lstsq(Xq * sw, zz * sw[:,0], rcond=None)
            rq = zz - Xq @ bq
            qrmse = float(np.sqrt(np.average(rq**2, weights=ww)))
            improvement = (plane_rmse - qrmse) / max(plane_rmse, 1e-9)
            curv = float(math.hypot(2*bq[2], 2*bq[4]))
            if improvement >= float(cfg.quadratic_min_improvement) and curv <= float(cfg.quadratic_max_abs_curvature):
                pred, plane_rmse, curvature, used_quad = float(bq[5]), qrmse, curv, True
        except np.linalg.LinAlgError:
            pass
    return pred, plane_rmse, int(zz.size), sectors, curvature, used_quad


def _adaptive_radius_map(count, cfg):
    finite = count[count > 0]
    if finite.size == 0:
        return np.full(count.shape, int(cfg.neighbor_radius_cells), np.int16)
    ref = max(1.0, float(np.quantile(finite, np.clip(cfg.density_reference_quantile, 0, 1))))
    ratio = np.sqrt(ref / np.maximum(count.astype(np.float64), 1.0))
    r = np.rint(float(cfg.neighbor_radius_cells) * ratio).astype(np.int16)
    return np.clip(r, int(cfg.radius_min_cells), int(cfg.radius_max_cells))


def _propagate_surface(candidate, cand_conf, count, span, radius_map, cfg):
    surface = np.full(candidate.shape, np.nan, np.float64)
    confidence = np.zeros(candidate.shape, np.float64)
    strong = (np.isfinite(candidate) & (cand_conf >= float(cfg.seed_confidence)) &
              (count >= int(cfg.seed_min_neighbor_cells)) &
              np.isfinite(span) & (span <= float(cfg.seed_max_vertical_span)))
    surface[strong] = candidate[strong]; confidence[strong] = cand_conf[strong]
    distance = np.full(candidate.shape, np.inf, np.float64); distance[strong] = 0.0
    if not np.any(strong):
        # graceful fallback: best available observations become seeds
        strong = np.isfinite(candidate) & (cand_conf >= np.nanquantile(cand_conf[np.isfinite(candidate)], 0.70) if np.any(np.isfinite(candidate)) else False)
        surface[strong] = candidate[strong]; confidence[strong] = cand_conf[strong]; distance[strong] = 0.0

    dirs = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))
    frontier = set(map(tuple, np.argwhere(strong)))
    for _ in range(max(1, int(cfg.propagation_iters))):
        candidates = set()
        for y, x in frontier:
            for dy, dx in dirs:
                yy, xx = y+dy, x+dx
                if 0 <= yy < surface.shape[0] and 0 <= xx < surface.shape[1] and not np.isfinite(surface[yy,xx]) and np.isfinite(candidate[yy,xx]):
                    candidates.add((yy,xx))
        if not candidates: break
        new_frontier = set()
        for y, x in candidates:
            pred, rough, ns, sectors, curv, used_quad = _fit_local(surface, confidence, x, y, int(radius_map[y,x]), cfg)
            if not np.isfinite(pred) or ns < int(cfg.min_neighbor_cells) or sectors < int(cfg.min_support_sectors):
                continue
            tol = float(cfg.ground_threshold) + float(cfg.roughness_adapt_k) * max(rough, 0.0) + float(cfg.curvature_adapt_k) * max(curv, 0.0)
            tol = float(np.clip(tol, cfg.threshold_min, cfg.threshold_max))
            resid = float(candidate[y,x] - pred)
            if abs(resid) > min(float(cfg.plane_max_residual) + rough + 0.5*tol, tol + rough):
                continue
            # must remain attached to accepted terrain and within bounded wavefront distance
            nd = min((distance[y+dy, x+dx] for dy,dx in dirs if 0 <= y+dy < surface.shape[0] and 0 <= x+dx < surface.shape[1] and np.isfinite(distance[y+dy,x+dx])), default=np.inf)
            if not np.isfinite(nd) or nd + 1 > float(cfg.max_extrapolation_cells):
                continue
            surface[y,x] = candidate[y,x]
            confidence[y,x] = max(float(cand_conf[y,x]), float(cfg.min_surface_confidence)) * math.exp(-0.12*(nd+1))
            distance[y,x] = nd + 1
            new_frontier.add((y,x))
        if not new_frontier: break
        frontier = new_frontier
    return surface, confidence, distance


def _no_wrap_fill(surface, confidence, candidate, cand_conf, cfg):
    """Bounded frontier fill. Never uses np.roll; no opposite-edge leakage."""
    out = surface.copy(); conf = confidence.copy()
    dirs = ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1))
    for _ in range(max(0, int(cfg.fill_iters))):
        missing = np.argwhere(~np.isfinite(out) & np.isfinite(candidate))
        if missing.size == 0: break
        changed = 0
        for y, x in missing:
            neigh = []
            for dy, dx in dirs:
                yy, xx = int(y+dy), int(x+dx)
                if 0 <= yy < out.shape[0] and 0 <= xx < out.shape[1] and np.isfinite(out[yy,xx]):
                    neigh.append((yy,xx))
            if len(neigh) < max(2, int(cfg.fill_min_support_sectors)):
                continue
            pred, rough, ns, sectors, curv, _ = _fit_local(out, conf, int(x), int(y), max(2,int(cfg.neighbor_radius_cells)), cfg)
            if not np.isfinite(pred) or sectors < int(cfg.fill_min_support_sectors): continue
            tol = float(np.clip(cfg.ground_threshold + cfg.roughness_adapt_k*rough + cfg.curvature_adapt_k*curv, cfg.threshold_min, cfg.threshold_max))
            if abs(float(candidate[y,x]-pred)) <= tol + rough:
                out[y,x] = candidate[y,x]
                conf[y,x] = max(float(cand_conf[y,x]), float(cfg.min_surface_confidence)*0.9)
                changed += 1
        if changed == 0: break
    return out, conf


def _remove_positive_protrusions(surface, confidence, cfg):
    """Remove only well-supported positive islands above the local terrain manifold.

    This is intentionally conservative: the center cell is excluded from the local
    plane/quadratic fit, support must surround the cell in several sectors, and only
    positive residuals are removed. Genuine convex/concave terrain is preserved by
    the quadratic fallback in _fit_local.
    """
    if not bool(cfg.protrusion_guard_enabled):
        return surface, confidence, np.zeros(surface.shape, bool)
    out = surface.copy()
    conf = confidence.copy()
    removed = np.zeros(surface.shape, bool)
    R = max(2, int(cfg.protrusion_guard_radius_cells))
    for _ in range(max(1, int(cfg.protrusion_guard_passes))):
        changed = 0
        ids = np.argwhere(np.isfinite(out))
        for y, x in ids:
            pred, rough, ns, sectors, curv, _ = _fit_local(
                out, conf, int(x), int(y), R, cfg, exclude_center=True
            )
            if (not np.isfinite(pred) or ns < int(cfg.protrusion_guard_min_support)
                    or sectors < int(cfg.protrusion_guard_min_sectors)):
                continue
            resid = float(out[y, x] - pred)
            tol = float(cfg.protrusion_guard_base_m) + float(cfg.protrusion_guard_roughness_k) * max(float(rough), 0.0)
            # Curved terrain receives a little extra tolerance; isolated vertical
            # objects still have much larger positive residuals than this envelope.
            tol += 0.35 * float(cfg.curvature_adapt_k) * max(float(curv), 0.0)
            if resid > tol:
                out[y, x] = np.nan
                conf[y, x] = 0.0
                removed[y, x] = True
                changed += 1
        if changed == 0:
            break
    return out, conf, removed


def _surface_derivatives(surface, cell):
    filled = surface.copy()
    if not np.any(np.isfinite(filled)):
        shape = filled.shape
        return np.full(shape,np.nan), np.full(shape,np.nan), np.full(shape,np.nan)
    # nearest-ish finite fill for derivative diagnostics only, with repeated no-wrap averaging
    for _ in range(6):
        miss = ~np.isfinite(filled)
        if not np.any(miss): break
        num = np.zeros_like(filled); den = np.zeros_like(filled)
        for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
            src = np.full_like(filled,np.nan)
            ys0=max(0,-dy); ys1=min(filled.shape[0],filled.shape[0]-dy) if dy>=0 else filled.shape[0]
            xs0=max(0,-dx); xs1=min(filled.shape[1],filled.shape[1]-dx) if dx>=0 else filled.shape[1]
            yd0=max(0,dy); xd0=max(0,dx); h=max(0,ys1-ys0); w=max(0,xs1-xs0)
            if h and w: src[yd0:yd0+h,xd0:xd0+w]=filled[ys0:ys1,xs0:xs1]
            ok=np.isfinite(src); num[ok]+=src[ok]; den[ok]+=1
        take=miss&(den>0); filled[take]=num[take]/den[take]
    gy,gx=np.gradient(filled,float(cell),float(cell)); slope=np.sqrt(gx*gx+gy*gy)
    gyy,gyx=np.gradient(gy,float(cell),float(cell)); gxy,gxx=np.gradient(gx,float(cell),float(cell))
    curvature=np.sqrt(gxx*gxx+gyy*gyy+0.5*(gxy+gyx)**2)
    # cheap roughness: difference to local tangent prediction not needed; use finite gradient change proxy
    rough=np.sqrt(np.maximum(curvature,0))*float(cell)
    invalid=~np.isfinite(surface); slope[invalid]=np.nan; curvature[invalid]=np.nan; rough[invalid]=np.nan
    return slope,curvature,rough


def build_tls_surface_invert_dsm_vote(x: np.ndarray, y: np.ndarray, z: np.ndarray, cfg: TlsInvertDsmVoteConfig):
    x=np.asarray(x,np.float64); y=np.asarray(y,np.float64); z=np.asarray(z,np.float64)
    finite=np.isfinite(x)&np.isfinite(y)&np.isfinite(z); x,y,z=x[finite],y[finite],z[finite]
    if x.size==0: return np.empty((0,0),np.float64),math.nan,math.nan
    cell=float(cfg.cell); x0=_snap_origin(float(np.min(x)),cell); y0=_snap_origin(float(np.min(y)),cell)
    base,count,span,cconf,column_span=_build_candidate_grid(x,y,z,cell,cfg,x0=x0,y0=y0)
    if bool(cfg.use_offset_grid):
        ox0,oy0=x0-0.5*cell,y0-0.5*cell
        off,_,_,offconf,_=_build_candidate_grid(x,y,z,cell,cfg,x0=ox0,y0=oy0)
        offb=_map_offset_to_base(off,base.shape,x0,y0,ox0,oy0,cell)
        offcb=_map_offset_to_base(offconf,base.shape,x0,y0,ox0,oy0,cell)
        base,cconf=_merge_base_offset(base,cconf,offb,np.nan_to_num(offcb,nan=0.0))
    radius=_adaptive_radius_map(count,cfg)
    surface,confidence,dist=_propagate_surface(base,cconf,count,span,radius,cfg)
    surface,confidence=_no_wrap_fill(surface,confidence,base,cconf,cfg)
    surface,confidence,removed_protrusions=_remove_positive_protrusions(surface,confidence,cfg)
    slope,curvature,roughness=_surface_derivatives(surface,cell)
    seed_mask = np.isfinite(dist) & (dist == 0.0)
    _SURFACE_DIAGNOSTICS[id(surface)]={
        "confidence":confidence,"candidate":base,"candidate_confidence":cconf,
        "support_count":count.astype(np.float64),"vertical_span":span,
        "column_span":column_span,"removed_protrusions":removed_protrusions.astype(np.float64),
        "radius_map":radius.astype(np.float64),"propagation_distance":dist,
        "seed_mask":seed_mask.astype(np.uint8),
        "slope":slope,"curvature":curvature,"roughness":roughness,
    }
    return surface,x0,y0



def get_tls_surface_diagnostics(surf_z: np.ndarray) -> dict[str, np.ndarray]:
    """Return copies of diagnostics associated with a TLS terrain surface.

    This function is diagnostic-only and has no effect on terrain estimation
    or point classification.
    """
    diag = _SURFACE_DIAGNOSTICS.get(id(surf_z), {})
    return {
        name: np.asarray(values).copy()
        for name, values in diag.items()
    }

def classify_tls_by_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray, surf_z: np.ndarray, x0: float, y0: float, cfg: TlsInvertDsmVoteConfig):
    x=np.asarray(x,np.float64); y=np.asarray(y,np.float64); z=np.asarray(z,np.float64)
    if surf_z.size==0: return np.zeros(z.size,bool)
    cell=float(cfg.cell); zhat=_bilinear_sample(surf_z,x,y,x0,y0,cell)
    diag=_SURFACE_DIAGNOSTICS.get(id(surf_z),{})
    if diag:
        slope=_bilinear_sample(diag["slope"],x,y,x0,y0,cell)
        curv=_bilinear_sample(diag["curvature"],x,y,x0,y0,cell)
        rough=_bilinear_sample(diag["roughness"],x,y,x0,y0,cell)
        conf=_bilinear_sample(diag["confidence"],x,y,x0,y0,cell)
    else:
        gy,gx=np.gradient(surf_z,cell,cell); sg=np.sqrt(gx*gx+gy*gy)
        slope=_bilinear_sample(sg,x,y,x0,y0,cell); curv=np.zeros_like(slope); rough=np.zeros_like(slope); conf=np.ones_like(slope)
    slope=np.nan_to_num(slope,nan=0.0,posinf=0.0,neginf=0.0)
    curv=np.nan_to_num(curv,nan=0.0,posinf=0.0,neginf=0.0)
    rough=np.nan_to_num(rough,nan=0.0,posinf=0.0,neginf=0.0)
    conf=np.nan_to_num(conf,nan=0.0)
    thr=(float(cfg.ground_threshold)+float(cfg.slope_adapt_k)*slope+
         float(cfg.curvature_adapt_k)*np.minimum(curv,1.0)+float(cfg.roughness_adapt_k)*np.minimum(rough,0.25))
    # Weak support tightens slightly rather than broadening into woody material.
    thr*=1.0-float(cfg.weak_confidence_tighten)*np.clip(float(cfg.min_classification_confidence)-conf,0,1)
    thr=np.clip(thr,float(cfg.threshold_min),float(cfg.threshold_max))
    resid=z-zhat
    keep=np.isfinite(zhat)&(resid<=thr)&(resid>=-float(cfg.lower_threshold_factor)*thr)

    # Final positive-protrusion/vertical-column guard.  Use surface-normal residual
    # so steep slopes are not penalized merely for having a large vertical dz.
    if bool(cfg.column_guard_enabled) and diag and "column_span" in diag:
        cspan=_sample_nearest(diag["column_span"],x,y,x0,y0,cell)
        vertical=np.isfinite(cspan)&(cspan>=float(cfg.column_guard_span_m))
        if np.any(vertical):
            normal_resid=resid/np.sqrt(1.0+slope*slope)
            upper=(float(cfg.column_guard_upper_normal_m)+
                   float(cfg.column_guard_curvature_k)*np.minimum(curv,1.5))
            upper=np.minimum(upper,float(cfg.column_guard_max_normal_m))
            # Only tighten the positive side. Points on/below the modeled terrain
            # retain the original asymmetric TLS tolerance.
            keep &= (~vertical) | (normal_resid <= upper)
    return keep


def sample_surface(surf_z: np.ndarray, x0: float, y0: float, cell: float, x: float, y: float) -> float:
    return float(_bilinear_sample(surf_z,np.array([x]),np.array([y]),x0,y0,float(cell))[0])


def sample_slope(surf_z: np.ndarray, x0: float, y0: float, cell: float, x: float, y: float) -> float:
    if surf_z.size==0: return float("nan")
    gy,gx=np.gradient(surf_z,float(cell),float(cell)); s=np.sqrt(gx*gx+gy*gy)
    return float(_bilinear_sample(s,np.array([x]),np.array([y]),x0,y0,float(cell))[0])
