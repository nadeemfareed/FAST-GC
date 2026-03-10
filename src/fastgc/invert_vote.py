from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class InvertVoteConfig:
    cell: float = 0.5
    top_m: int = 5
    neighbor_radius_cells: int = 4
    min_neighbor_cells: int = 10
    max_robust_z: float = 2.8
    mad_floor: float = 0.05
    fill_iters: int = 120
    smooth_sigma_cells: float = 1.25
    ground_threshold: float = 0.2
    slope_adapt_k: float = 0.35


def _grid_index(x: np.ndarray, y: np.ndarray, cell: float):
    x0 = float(np.min(x))
    y0 = float(np.min(y))
    ix = np.floor((x - x0) / cell).astype(np.int32)
    iy = np.floor((y - y0) / cell).astype(np.int32)
    return x0, y0, ix, iy


def build_surface_invert_vote(x: np.ndarray, y: np.ndarray, z: np.ndarray, cfg: InvertVoteConfig):
    """
    Strict surface:
      - per cell keep TOP-M lowest Z (equiv highest inverted Z)
      - neighbor vote via median + MAD + robust-z reject
      - hole-fill + nan-safe smoothing
    Returns: (surf_z_grid, x0, y0)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    cell = float(cfg.cell)
    top_m = int(max(1, cfg.top_m))

    x0, y0, ix, iy = _grid_index(x, y, cell)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1

    lin = iy * nx + ix
    order = np.argsort(lin, kind="mergesort")
    lin_s = lin[order]
    z_s = z[order]

    # candidate buffers: TOP-M lowest z per cell
    cell_candidates = [[] for _ in range(nx * ny)]
    p = 0
    while p < lin_s.size:
        c = int(lin_s[p])
        q = p + 1
        while q < lin_s.size and int(lin_s[q]) == c:
            q += 1
        zz = z_s[p:q]
        if zz.size:
            m = min(top_m, zz.size)
            part = np.partition(zz, m - 1)[:m]
            cell_candidates[c] = part.astype(np.float32, copy=False)
        p = q

    # vote grid in (ny, nx)
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

            neigh_vals = []
            for jj in range(j0, j1):
                base = jj * nx
                for ii in range(i0, i1):
                    vals = cell_candidates[base + ii]
                    if len(vals) > 0:
                        neigh_vals.append(vals)
            if not neigh_vals:
                continue

            vals = np.concatenate(neigh_vals).astype(np.float32, copy=False)
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

    def _conv1d_nan(a: np.ndarray, k1: np.ndarray, axis: int):
        out = np.full_like(a, np.nan, dtype=np.float32)
        if axis == 0:
            for i in range(a.shape[1]):
                col = a[:, i]
                num = np.convolve(np.nan_to_num(col, nan=0.0), k1, mode="same")
                den = np.convolve(np.isfinite(col).astype(np.float32), k1, mode="same")
                tmp = np.full_like(num, np.nan, dtype=np.float32)
                np.divide(num, den, out=tmp, where=den > 1e-6)
                out[:, i] = tmp
        else:
            for j in range(a.shape[0]):
                row = a[j, :]
                num = np.convolve(np.nan_to_num(row, nan=0.0), k1, mode="same")
                den = np.convolve(np.isfinite(row).astype(np.float32), k1, mode="same")
                tmp = np.full_like(num, np.nan, dtype=np.float32)
                np.divide(num, den, out=tmp, where=den > 1e-6)
                out[j, :] = tmp
        return out

    grid = _conv1d_nan(grid, k, axis=0)
    grid = _conv1d_nan(grid, k, axis=1)

    return grid, x0, y0


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

    ix0 = np.clip(ix, 0, nx - 2)
    iy0 = np.clip(iy, 0, ny - 2)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

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
    ALS/ULS classification using TLS-style logic:
      - bilinear surface sample
      - bilinear slope sample
      - asymmetric thin-sheet gate
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

    # Asymmetric rule:
    #   stricter above surface (reject canopy leakage)
    #   more tolerant below surface (allow slope underfit / local roughness)
    return np.isfinite(zhat) & (resid <= thr) & (resid >= -2.0 * thr)