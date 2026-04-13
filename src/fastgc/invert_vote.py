
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class InvertVoteConfig:
    cell: float = 0.25
    top_m: int = 1
    neighbor_radius_cells: int = 6
    min_neighbor_cells: int = 10
    max_robust_z: float = 2.8
    mad_floor: float = 0.05
    fill_iters: int = 120
    smooth_sigma_cells: float = 1.25
    ground_threshold: float = 0.2
    slope_adapt_k: float = 0.35

    # Global support-layer controls
    use_offset_swipe: bool = True
    # Keep snapping local by default; larger radii can reintroduce banding.
    support_snap_radius_cells: int = 0


def _grid_index(
    x: np.ndarray,
    y: np.ndarray,
    cell: float,
    x0: float | None = None,
    y0: float | None = None,
):
    if x0 is None:
        x0 = float(np.min(x))
    if y0 is None:
        y0 = float(np.min(y))
    ix = np.floor((x - x0) / cell).astype(np.int32)
    iy = np.floor((y - y0) / cell).astype(np.int32)
    return float(x0), float(y0), ix, iy


def _conv_same_len(v: np.ndarray, k1: np.ndarray) -> np.ndarray:
    """
    np.convolve(..., mode="same") returns length max(len(v), len(k1)).
    That breaks when len(k1) > len(v). We always trim back to len(v).
    """
    out = np.convolve(v, k1, mode="same")
    n = int(v.size)
    if out.size == n:
        return out.astype(np.float32, copy=False)
    start = max(0, (out.size - n) // 2)
    end = start + n
    return out[start:end].astype(np.float32, copy=False)


def _conv1d_nan(a: np.ndarray, k1: np.ndarray, axis: int):
    out = np.full_like(a, np.nan, dtype=np.float32)
    if axis == 0:
        for i in range(a.shape[1]):
            col = a[:, i].astype(np.float32, copy=False)
            num = _conv_same_len(np.nan_to_num(col, nan=0.0), k1)
            den = _conv_same_len(np.isfinite(col).astype(np.float32), k1)
            tmp = np.full(col.shape, np.nan, dtype=np.float32)
            np.divide(num, den, out=tmp, where=den > 1e-6)
            out[:, i] = tmp
    else:
        for j in range(a.shape[0]):
            row = a[j, :].astype(np.float32, copy=False)
            num = _conv_same_len(np.nan_to_num(row, nan=0.0), k1)
            den = _conv_same_len(np.isfinite(row).astype(np.float32), k1)
            tmp = np.full(row.shape, np.nan, dtype=np.float32)
            np.divide(num, den, out=tmp, where=den > 1e-6)
            out[j, :] = tmp
    return out


def _build_point_bins(ix: np.ndarray, iy: np.ndarray, nx: int, ny: int):
    bins = [[] for _ in range(nx * ny)]
    for p in range(ix.size):
        i = int(ix[p])
        j = int(iy[p])
        if 0 <= i < nx and 0 <= j < ny:
            bins[j * nx + i].append(p)
    return bins


def _initial_surface_from_swipe(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cell: float,
    x0_base: float,
    y0_base: float,
    nx: int,
    ny: int,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
):
    """
    Provisional scalar support surface in inverted space:
      - invert z -> zp
      - keep highest zp (lowest z) per swipe cell
      - map the selected point back to the base grid
      - merge to one scalar value per base-grid cell
    """
    zp = -z
    x0 = x0_base + x_offset
    y0 = y0_base + y_offset

    ix = np.floor((x - x0) / cell).astype(np.int32)
    iy = np.floor((y - y0) / cell).astype(np.int32)

    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    if not np.any(valid):
        return np.full((ny, nx), np.nan, dtype=np.float32)

    ixv = ix[valid]
    iyv = iy[valid]
    xv = x[valid]
    yv = y[valid]
    zv = z[valid]
    zpv = zp[valid]

    lin = iyv * nx + ixv
    order = np.argsort(lin, kind="mergesort")
    lin_s = lin[order]
    z_s = zv[order]
    zp_s = zpv[order]
    x_s = xv[order]
    y_s = yv[order]

    surf = np.full((ny, nx), np.nan, dtype=np.float32)

    p = 0
    while p < lin_s.size:
        c = int(lin_s[p])
        q = p + 1
        while q < lin_s.size and int(lin_s[q]) == c:
            q += 1

        loc = p + int(np.argmax(zp_s[p:q]))  # max inverted z == min original z
        xs = x_s[loc]
        ys = y_s[loc]
        zs = z_s[loc]

        ib = int(np.floor((xs - x0_base) / cell))
        jb = int(np.floor((ys - y0_base) / cell))
        if 0 <= ib < nx and 0 <= jb < ny:
            cur = surf[jb, ib]
            if not np.isfinite(cur) or zs < cur:
                surf[jb, ib] = float(zs)
        p = q

    return surf


def _merge_surfaces_lower(s1: np.ndarray, s2: np.ndarray):
    out = s1.copy()
    mask2 = np.isfinite(s2)
    mask1 = np.isfinite(out)
    take2 = mask2 & (~mask1 | (s2 < out))
    out[take2] = s2[take2]
    return out


def _bilinear_sample(
    grid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    y0: float,
    cell: float,
):
    ny, nx = grid.shape
    gx = (x - x0) / cell
    gy = (y - y0) / cell

    ix = np.floor(gx).astype(np.int32)
    iy = np.floor(gy).astype(np.int32)
    fx = gx - ix
    fy = gy - iy

    ix0 = np.clip(ix, 0, max(0, nx - 2))
    iy0 = np.clip(iy, 0, max(0, ny - 2))
    ix1 = np.clip(ix0 + 1, 0, nx - 1)
    iy1 = np.clip(iy0 + 1, 0, ny - 1)

    g00 = grid[iy0, ix0]
    g10 = grid[iy0, ix1]
    g01 = grid[iy1, ix0]
    g11 = grid[iy1, ix1]

    nn = grid[np.clip(iy, 0, ny - 1), np.clip(ix, 0, nx - 1)]

    ok = np.isfinite(g00) & np.isfinite(g10) & np.isfinite(g01) & np.isfinite(g11)
    out = np.where(
        ok,
        (1.0 - fx) * (1.0 - fy) * g00
        + fx * (1.0 - fy) * g10
        + (1.0 - fx) * fy * g01
        + fx * fy * g11,
        nn,
    )
    return out


def _select_surface_snap_points(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    provisional: np.ndarray,
    x0: float,
    y0: float,
    cfg: InvertVoteConfig,
):
    """
    For each base-grid support location, choose the closest competitor to the
    provisional surface from the nearest point above and the nearest point below.
    No tolerance band.
    """
    cell = float(cfg.cell)
    x0b, y0b, ix, iy = _grid_index(x, y, cell, x0=x0, y0=y0)
    assert x0b == x0 and y0b == y0

    ny, nx = provisional.shape
    bins = _build_point_bins(ix, iy, nx, ny)
    r = int(max(0, cfg.support_snap_radius_cells))

    chosen = np.full((ny, nx), np.nan, dtype=np.float32)

    for j in range(ny):
        for i in range(nx):
            zs = provisional[j, i]
            if not np.isfinite(zs):
                continue

            j0 = max(0, j - r)
            j1 = min(ny, j + r + 1)
            i0 = max(0, i - r)
            i1 = min(nx, i + r + 1)

            best_above = np.nan
            best_above_abs = np.inf
            best_below = np.nan
            best_below_abs = np.inf

            for jj in range(j0, j1):
                base = jj * nx
                for ii in range(i0, i1):
                    pts = bins[base + ii]
                    if not pts:
                        continue
                    vals = z[np.asarray(pts, dtype=np.int32)]

                    above = vals[vals >= zs]
                    if above.size:
                        idx = int(np.argmin(above - zs))
                        cand = float(above[idx])
                        d = abs(cand - zs)
                        if d < best_above_abs:
                            best_above_abs = d
                            best_above = cand

                    below = vals[vals < zs]
                    if below.size:
                        idx = int(np.argmin(zs - below))
                        cand = float(below[idx])
                        d = abs(cand - zs)
                        if d < best_below_abs:
                            best_below_abs = d
                            best_below = cand

            if np.isfinite(best_above) and np.isfinite(best_below):
                chosen[j, i] = np.float32(best_above if best_above_abs <= best_below_abs else best_below)
            elif np.isfinite(best_below):
                chosen[j, i] = np.float32(best_below)
            elif np.isfinite(best_above):
                chosen[j, i] = np.float32(best_above)

    return chosen


def build_surface_invert_vote(x: np.ndarray, y: np.ndarray, z: np.ndarray, cfg: InvertVoteConfig):
    """
    Global support-layer version:
      1) explicit inversion + scalar provisional support surface
      2) optional half-cell offset swipe
      3) single above/below competition snap per support location
      4) robust vote on the scalar snapped surface
      5) hole-fill + nan-safe smoothing

    Returns: (surf_z_grid, x0, y0)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    if x.size == 0:
        return np.empty((0, 0), dtype=np.float32), 0.0, 0.0

    cell = float(cfg.cell)
    x0 = float(np.min(x))
    y0 = float(np.min(y))
    ix = np.floor((x - x0) / cell).astype(np.int32)
    iy = np.floor((y - y0) / cell).astype(np.int32)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1

    surf0 = _initial_surface_from_swipe(x, y, z, cell, x0, y0, nx, ny, 0.0, 0.0)
    provisional = surf0

    if bool(getattr(cfg, "use_offset_swipe", True)):
        surf1 = _initial_surface_from_swipe(
            x, y, z, cell, x0, y0, nx, ny, 0.5 * cell, 0.5 * cell
        )
        provisional = _merge_surfaces_lower(surf0, surf1)

    snapped = _select_surface_snap_points(x, y, z, provisional, x0, y0, cfg)

    # Robust vote on the snapped scalar surface
    grid = np.full((ny, nx), np.nan, dtype=np.float32)
    r = int(max(1, cfg.neighbor_radius_cells))
    min_nei = int(max(1, cfg.min_neighbor_cells))
    max_rz = float(cfg.max_robust_z)
    mad_floor = float(cfg.mad_floor)

    for j in range(ny):
        j0 = max(0, j - r)
        j1 = min(ny, j + r + 1)
        for i in range(nx):
            i0 = max(0, i - r)
            i1 = min(nx, i + r + 1)

            vals = snapped[j0:j1, i0:i1]
            vals = vals[np.isfinite(vals)]
            if vals.size < min_nei:
                continue

            med = np.nanmedian(vals)
            mad = np.nanmedian(np.abs(vals - med))
            mad = float(max(mad, mad_floor))
            rz = np.abs(vals - med) / (1.4826 * mad)

            good = vals[rz <= max_rz]
            if good.size < min_nei:
                continue

            grid[j, i] = float(np.nanmedian(good))

    # hole fill
    for _ in range(int(max(0, cfg.fill_iters))):
        nan_mask = ~np.isfinite(grid)
        if not nan_mask.any():
            break
        g2 = grid.copy()
        for j in range(ny):
            for i in range(nx):
                if np.isfinite(grid[j, i]):
                    continue
                j0 = max(0, j - 1)
                j1 = min(ny, j + 2)
                i0 = max(0, i - 1)
                i1 = min(nx, i + 2)
                neigh = grid[j0:j1, i0:i1]
                neigh = neigh[np.isfinite(neigh)]
                if neigh.size:
                    g2[j, i] = float(np.nanmedian(neigh))
        grid = g2

    # nan-safe smoothing (separable)
    sigma = float(max(0.01, cfg.smooth_sigma_cells))
    rad = int(np.ceil(3.0 * sigma))
    xs = np.arange(-rad, rad + 1, dtype=np.float32)
    k = np.exp(-(xs * xs) / (2.0 * sigma * sigma)).astype(np.float32)
    k /= float(k.sum())

    grid = _conv1d_nan(grid, k, axis=0)
    grid = _conv1d_nan(grid, k, axis=1)

    return grid, x0, y0


def sample_surface(surf_z: np.ndarray, x0: float, y0: float, cell: float, x: float, y: float) -> float:
    out = _bilinear_sample(
        surf_z,
        np.array([x], dtype=np.float64),
        np.array([y], dtype=np.float64),
        x0,
        y0,
        float(cell),
    )
    return float(out[0])


def sample_slope(surf_z: np.ndarray, x0: float, y0: float, cell: float, x: float, y: float) -> float:
    if surf_z.size == 0:
        return float("nan")
    gy, gx = np.gradient(surf_z, float(cell), float(cell))
    slope_mag = np.sqrt(gx * gx + gy * gy)
    out = _bilinear_sample(
        slope_mag,
        np.array([x], dtype=np.float64),
        np.array([y], dtype=np.float64),
        x0,
        y0,
        float(cell),
    )
    return float(out[0])


def classify_by_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    surf_z: np.ndarray,
    x0: float,
    y0: float,
    cfg: InvertVoteConfig,
) -> np.ndarray:
    """
    ALS/ULS classification using a slope-adaptive asymmetric thin-sheet gate.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    cell = float(cfg.cell)
    thr0 = float(cfg.ground_threshold)
    slope_k = float(cfg.slope_adapt_k)

    zhat = _bilinear_sample(surf_z, x, y, x0, y0, cell)

    gy, gx = np.gradient(surf_z, cell, cell)
    slope_mag = np.sqrt(gx * gx + gy * gy)
    slope_here = _bilinear_sample(slope_mag, x, y, x0, y0, cell)

    thr = thr0 + slope_k * slope_here
    thr = np.clip(thr, 0.05, 0.60)

    resid = z - zhat

    # stricter above surface, more tolerant below surface
    return np.isfinite(zhat) & (resid <= thr) & (resid >= -2.0 * thr)
