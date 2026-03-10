from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class TlsInvertDsmVoteConfig:
    # grid + candidates
    cell: float = 0.35
    top_m: int = 6

    # neighborhood vote on candidate surface
    neighbor_radius_cells: int = 4
    min_neighbor_cells: int = 10
    max_robust_z: float = 2.6
    mad_floor: float = 0.03

    # hole fill / smooth on surface grid
    fill_iters: int = 100
    smooth_sigma_cells: float = 0.8

    # classification (thin-sheet)
    ground_threshold: float = 0.10  # residual threshold at zero slope
    slope_adapt_k: float = 0.25     # increase threshold with local slope (m per m)


def _grid_index(x: np.ndarray, y: np.ndarray, cell: float):
    x0 = float(np.min(x))
    y0 = float(np.min(y))
    ix = np.floor((x - x0) / cell).astype(np.int32)
    iy = np.floor((y - y0) / cell).astype(np.int32)
    return x0, y0, ix, iy


def _accum_topm(zp: np.ndarray, ix: np.ndarray, iy: np.ndarray, top_m: int):
    """
    Per cell keep TOP-M highest zp (zp = -z). Return candidate zp and counts per cell.
    """
    # cell key
    key = (ix.astype(np.int64) << 32) | (iy.astype(np.int64) & 0xFFFFFFFF)
    order = np.argsort(key, kind="mergesort")
    key_s = key[order]
    zp_s = zp[order]

    # group
    uniq, start = np.unique(key_s, return_index=True)
    start = start.astype(np.int64)
    end = np.r_[start[1:], zp_s.size]

    # We'll keep top_m per group (highest zp) -> sort within group by zp desc
    out_key = []
    out_zp = []
    for u, a, b in zip(uniq, start, end):
        g = zp_s[a:b]
        if g.size == 0:
            continue
        # take top_m highest
        if g.size > top_m:
            idx = np.argpartition(g, -top_m)[-top_m:]
            gsel = g[idx]
        else:
            gsel = g
        out_key.append(np.full(gsel.size, u, dtype=np.int64))
        out_zp.append(gsel.astype(np.float64))
    if not out_key:
        return np.empty((0,), np.int64), np.empty((0,), np.float64)
    return np.concatenate(out_key), np.concatenate(out_zp)


def _grid_shape_from_ixiy(ix: np.ndarray, iy: np.ndarray):
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1
    return nx, ny


def _build_initial_surface(zp: np.ndarray, ix: np.ndarray, iy: np.ndarray, top_m: int):
    """
    Make initial surface per cell: max(zp) among top-m candidates, else NaN.
    Returns surface_zp (ny,nx)
    """
    nx, ny = _grid_shape_from_ixiy(ix, iy)
    surf = np.full((ny, nx), np.nan, dtype=np.float64)

    cand_key, cand_zp = _accum_topm(zp, ix, iy, top_m=top_m)
    if cand_key.size == 0:
        return surf
    cx = (cand_key >> 32).astype(np.int32)
    cy = (cand_key & 0xFFFFFFFF).astype(np.int32)

    # per cell max zp
    key2 = (cx.astype(np.int64) << 32) | (cy.astype(np.int64) & 0xFFFFFFFF)
    o = np.argsort(key2, kind="mergesort")
    key2s = key2[o]
    zps = cand_zp[o]
    uniq, st = np.unique(key2s, return_index=True)
    en = np.r_[st[1:], zps.size]
    for u, a, b in zip(uniq, st, en):
        x = int(u >> 32)
        y = int(u & 0xFFFFFFFF)
        surf[y, x] = float(np.nanmax(zps[a:b]))
    return surf


def _neighbor_vote(surface: np.ndarray, radius: int, min_n: int, max_robust_z: float, mad_floor: float):
    """
    For each cell with data, robust filter vs neighborhood (median/MAD).
    """
    ny, nx = surface.shape
    out = surface.copy()
    r = int(max(1, radius))
    for y in range(ny):
        y0 = max(0, y - r); y1 = min(ny, y + r + 1)
        for x in range(nx):
            if not np.isfinite(surface[y, x]):
                continue
            x0 = max(0, x - r); x1 = min(nx, x + r + 1)
            nb = surface[y0:y1, x0:x1].ravel()
            nb = nb[np.isfinite(nb)]
            if nb.size < min_n:
                continue
            med = np.median(nb)
            mad = np.median(np.abs(nb - med))
            mad = max(float(mad), float(mad_floor))
            rz = (surface[y, x] - med) / (1.4826 * mad)
            if np.abs(rz) > max_robust_z:
                out[y, x] = np.nan
    return out


def _fill_holes(surface: np.ndarray, iters: int):
    """
    Simple neighbor-mean fill for NaNs (fast).
    """
    out = surface.copy()
    ny, nx = out.shape
    for _ in range(int(max(0, iters))):
        nan = ~np.isfinite(out)
        if not np.any(nan):
            break
        # 4-neighbor mean
        up = np.roll(out, 1, axis=0)
        dn = np.roll(out, -1, axis=0)
        lf = np.roll(out, 1, axis=1)
        rt = np.roll(out, -1, axis=1)

        stack = np.stack([up, dn, lf, rt], axis=0)
        m = np.nanmean(stack, axis=0)
        # update only nan where mean exists
        fill = nan & np.isfinite(m)
        if not np.any(fill):
            break
        out[fill] = m[fill]
    return out


def _gaussian_smooth_nan(surface: np.ndarray, sigma_cells: float):
    """
    Nan-safe Gaussian smoothing using separable 1D kernels (small sigma).
    """
    sigma = float(sigma_cells)
    if sigma <= 0:
        return surface
    # kernel radius ~ 3 sigma
    rad = int(max(1, round(3.0 * sigma)))
    x = np.arange(-rad, rad + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= np.sum(k)

    def conv1d_nan(a: np.ndarray, axis: int):
        w = np.isfinite(a).astype(np.float64)
        a0 = np.where(np.isfinite(a), a, 0.0)
        # pad reflect
        pad = [(0,0),(0,0)]
        pad[axis] = (rad, rad)
        a1 = np.pad(a0, pad, mode="reflect")
        w1 = np.pad(w, pad, mode="reflect")
        # convolve
        out_num = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), axis, a1)
        out_den = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), axis, w1)
        out = np.where(out_den > 1e-6, out_num / out_den, np.nan)
        return out

    tmp = conv1d_nan(surface, axis=1)
    out = conv1d_nan(tmp, axis=0)
    return out


def build_tls_surface_invert_dsm_vote(x: np.ndarray, y: np.ndarray, z: np.ndarray, cfg: TlsInvertDsmVoteConfig):
    """
    TLS-specific inverted-DSM vote surface.

    Returns:
        surf_z (ny,nx) in ORIGINAL z units,
        x0, y0 (grid origin)
    """
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    z = np.asarray(z, np.float64)

    cell = float(cfg.cell)
    x0, y0, ix, iy = _grid_index(x, y, cell)
    zp = -z  # inverted elevation

    surf_zp = _build_initial_surface(zp, ix, iy, top_m=int(cfg.top_m))
    surf_zp = _neighbor_vote(
        surf_zp,
        radius=int(cfg.neighbor_radius_cells),
        min_n=int(cfg.min_neighbor_cells),
        max_robust_z=float(cfg.max_robust_z),
        mad_floor=float(cfg.mad_floor),
    )
    surf_zp = _fill_holes(surf_zp, iters=int(cfg.fill_iters))
    surf_zp = _gaussian_smooth_nan(surf_zp, sigma_cells=float(cfg.smooth_sigma_cells))

    # back to original z
    surf_z = -surf_zp
    return surf_z, x0, y0


def _bilinear_sample(grid: np.ndarray, x: np.ndarray, y: np.ndarray, x0: float, y0: float, cell: float):
    """
    Bilinear sampling from a regular grid (ny,nx) with origin at (x0,y0).
    """
    ny, nx = grid.shape
    gx = (x - x0) / cell
    gy = (y - y0) / cell
    ix = np.floor(gx).astype(np.int32)
    iy = np.floor(gy).astype(np.int32)
    fx = gx - ix
    fy = gy - iy

    # clamp to valid interior
    ix0 = np.clip(ix, 0, nx - 2)
    iy0 = np.clip(iy, 0, ny - 2)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    g00 = grid[iy0, ix0]
    g10 = grid[iy0, ix1]
    g01 = grid[iy1, ix0]
    g11 = grid[iy1, ix1]

    # if any NaN, fall back to nearest
    nn = grid[np.clip(iy, 0, ny - 1), np.clip(ix, 0, nx - 1)]
    ok = np.isfinite(g00) & np.isfinite(g10) & np.isfinite(g01) & np.isfinite(g11)
    out = np.where(
        ok,
        (1-fx)*(1-fy)*g00 + fx*(1-fy)*g10 + (1-fx)*fy*g01 + fx*fy*g11,
        nn
    )
    return out


def classify_tls_by_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray, surf_z: np.ndarray, x0: float, y0: float, cfg: TlsInvertDsmVoteConfig):
    """
    Thin-sheet classification: ground if z is within (adaptive) threshold of surface.
    The slope-adaptive term is approximated from local surface gradient (grid-space).
    """
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    z = np.asarray(z, np.float64)
    cell = float(cfg.cell)

    zhat = _bilinear_sample(surf_z, x, y, x0, y0, cell)

    # slope magnitude from surface grid gradient (cheap)
    # precompute gradients once
    gy, gx = np.gradient(surf_z, cell, cell)
    slope_mag = np.sqrt(gx*gx + gy*gy)
    slope_here = _bilinear_sample(slope_mag, x, y, x0, y0, cell)

    thr = float(cfg.ground_threshold) + float(cfg.slope_adapt_k) * slope_here
    thr = np.clip(thr, 0.03, 0.30)  # TLS guardrails
    resid = z - zhat

    # ground is "close to surface" AND not above it too much
    # (asymmetric: allow slightly below due to gaps/noise)
    return (resid <= thr) & (resid >= -2.0 * thr)


# ------------------------------------------------------------
# lightweight helpers used by io_las (surface/slope sampling)
# ------------------------------------------------------------

def sample_surface(surf_z: np.ndarray, x0: float, y0: float, cell: float, x: float, y: float) -> float:
    """Bilinear sample of a surface grid at world coordinate (x,y)."""
    out = _bilinear_sample(surf_z, np.array([x], np.float64), np.array([y], np.float64), x0, y0, float(cell))
    v = float(out[0])
    return v


def sample_slope(surf_z: np.ndarray, x0: float, y0: float, cell: float, x: float, y: float) -> float:
    """Approximate local slope magnitude at (x,y) from the surface grid."""
    if surf_z.size == 0:
        return float("nan")
    gy, gx = np.gradient(surf_z, float(cell), float(cell))
    slope_mag = np.sqrt(gx*gx + gy*gy)
    out = _bilinear_sample(slope_mag, np.array([x], np.float64), np.array([y], np.float64), x0, y0, float(cell))
    return float(out[0])
