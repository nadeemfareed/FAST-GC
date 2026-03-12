from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import binary_dilation, label


def _bilinear_sample(grid: np.ndarray, x: np.ndarray, y: np.ndarray, x0: float, y0: float, cell: float) -> np.ndarray:
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


def _fit_plane(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray):
    """
    Fit z = ax + by + c using least squares.
    Returns (ok, (a,b,c))
    """
    if xs.size < 3:
        return False, (0.0, 0.0, 0.0)

    A = np.column_stack([xs, ys, np.ones_like(xs)])
    try:
        coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
        a, b, c = coef
        return True, (float(a), float(b), float(c))
    except Exception:
        return False, (0.0, 0.0, 0.0)


def recover_ground_in_voids(
    xw: np.ndarray,
    yw: np.ndarray,
    zw: np.ndarray,
    ground_mask: np.ndarray,
    surf_z: np.ndarray,
    sx0: float,
    sy0: float,
    surf_cell: float,
    sensor_mode: str,
    cfg: dict,
) -> np.ndarray:
    """
    Targeted void recovery on candidate-layer points only.

    Workflow:
      1) Build XY occupancy of candidate layer and final ground.
      2) Find voids where candidate exists but final ground is missing.
      3) Restrict to slope / slope-break neighborhoods.
      4) Reject candidate cells with high vertical diversity.
      5) Compare lowest candidate bin with neighboring final-ground bins.
      6) Promote only the lowest candidate bin where terrain continuity is likely.

    This is designed to patch abrupt transition zones without jeopardizing
    the existing stable final ground.
    """
    sm = str(sensor_mode).upper().strip()
    out_mask = np.asarray(ground_mask, dtype=bool).copy()

    if sm not in {"ALS", "ULS"}:
        return out_mask

    if xw.size == 0 or out_mask.size == 0 or np.count_nonzero(out_mask) < 10:
        return out_mask

    enabled = bool(cfg.get("void_recover_enabled", False))
    if not enabled:
        return out_mask

    cell = float(cfg.get("void_recover_cell_m", surf_cell))
    ground_buffer_m = float(cfg.get("void_recover_ground_buffer_m", 5.0))
    slope_thr_deg = float(cfg.get("void_recover_slope_thr_deg", 6.0))
    slope_break_thr = float(cfg.get("void_recover_slope_break_thr", 0.08))
    min_component_cells = int(cfg.get("void_recover_min_component_cells", 2))
    max_component_cells = int(cfg.get("void_recover_max_component_cells", 500))
    bin_dz = float(cfg.get("void_recover_bin_dz_m", 0.20))
    max_bins = int(cfg.get("void_recover_max_bins", 2))
    z_std_thr = float(cfg.get("void_recover_z_std_thr_m", 0.12))
    z_span_thr = float(cfg.get("void_recover_z_span_thr_m", 0.35))
    z_tol = float(cfg.get("void_recover_z_tol_m", 0.18))
    plane_tol = float(cfg.get("void_recover_plane_tol_m", 0.18))
    neighbor_radius_cells = int(cfg.get("void_recover_neighbor_radius_cells", 2))
    min_neighbor_ground_cells = int(cfg.get("void_recover_min_neighbor_ground_cells", 4))

    x0 = float(np.min(xw))
    y0 = float(np.min(yw))

    ix = np.floor((xw - x0) / cell).astype(np.int32)
    iy = np.floor((yw - y0) / cell).astype(np.int32)

    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1

    cand_occ = np.zeros((ny, nx), dtype=bool)
    ground_occ = np.zeros((ny, nx), dtype=bool)
    ground_zmin = np.full((ny, nx), np.nan, dtype=np.float64)

    # cell -> candidate point indices
    cell_to_points: dict[int, list[int]] = {}

    for p in range(xw.size):
        cx = int(ix[p])
        cy = int(iy[p])

        cand_occ[cy, cx] = True
        lin = cy * nx + cx
        if lin not in cell_to_points:
            cell_to_points[lin] = [p]
        else:
            cell_to_points[lin].append(p)

        if out_mask[p]:
            ground_occ[cy, cx] = True
            cur = ground_zmin[cy, cx]
            if not np.isfinite(cur) or zw[p] < cur:
                ground_zmin[cy, cx] = zw[p]

    # Candidate cells where final ground is missing
    void = cand_occ & (~ground_occ)

    if not np.any(void):
        return out_mask

    # Must be near existing final ground
    buffer_cells = max(1, int(math.ceil(ground_buffer_m / cell)))
    ground_near = binary_dilation(ground_occ, iterations=buffer_cells)
    void &= ground_near

    if not np.any(void):
        return out_mask

    # Slope / slope-break mask from the existing voted surface
    gy, gx = np.gradient(surf_z.astype(np.float64), float(surf_cell), float(surf_cell))
    slope_mag = np.sqrt(gx * gx + gy * gy)
    lap = np.zeros_like(surf_z, dtype=np.float64)
    lap[1:-1, 1:-1] = (
        surf_z[1:-1, :-2] + surf_z[1:-1, 2:]
        + surf_z[:-2, 1:-1] + surf_z[2:, 1:-1]
        - 4.0 * surf_z[1:-1, 1:-1]
    ) / max(float(surf_cell), 1e-6)

    cx_coords = x0 + (np.arange(nx, dtype=np.float64) + 0.5) * cell
    cy_coords = y0 + (np.arange(ny, dtype=np.float64) + 0.5) * cell
    XX, YY = np.meshgrid(cx_coords, cy_coords)

    slope_here = _bilinear_sample(slope_mag, XX.ravel(), YY.ravel(), sx0, sy0, float(surf_cell)).reshape(ny, nx)
    lap_here = _bilinear_sample(lap, XX.ravel(), YY.ravel(), sx0, sy0, float(surf_cell)).reshape(ny, nx)

    slope_thr = math.tan(math.radians(slope_thr_deg))
    slope_zone = (slope_here >= slope_thr) | (np.abs(lap_here) >= slope_break_thr)

    void &= slope_zone

    if not np.any(void):
        return out_mask

    # Keep only connected systematic voids
    structure = np.ones((3, 3), dtype=np.int32)
    labels, nlab = label(void, structure=structure)

    keep_void = np.zeros_like(void, dtype=bool)
    for lab_id in range(1, nlab + 1):
        comp = labels == lab_id
        n = int(np.count_nonzero(comp))
        if min_component_cells <= n <= max_component_cells:
            keep_void |= comp

    void = keep_void
    if not np.any(void):
        return out_mask

    # Promotion pass
    for cy in range(ny):
        for cx in range(nx):
            if not void[cy, cx]:
                continue

            lin = cy * nx + cx
            pts = cell_to_points.get(lin, None)
            if not pts:
                continue

            pts_idx = np.asarray(pts, dtype=np.int32)
            zc = zw[pts_idx]
            if zc.size == 0:
                continue

            # Vertical diversity rejection
            z0 = float(np.min(zc))
            bins = np.floor((zc - z0) / max(bin_dz, 1e-6)).astype(np.int32)
            nbins = int(np.unique(bins).size)
            zstd = float(np.std(zc))
            zspan = float(np.max(zc) - np.min(zc))

            if nbins > max_bins:
                continue
            if zstd > z_std_thr:
                continue
            if zspan > z_span_thr:
                continue

            lowest_bin = int(np.min(bins))
            low_sel = bins == lowest_bin
            low_pts = pts_idx[low_sel]
            if low_pts.size == 0:
                continue

            z_low = float(np.median(zw[low_pts]))

            # Surrounding ground cells
            n_z = []
            n_x = []
            n_y = []
            for yy in range(max(0, cy - neighbor_radius_cells), min(ny, cy + neighbor_radius_cells + 1)):
                for xx in range(max(0, cx - neighbor_radius_cells), min(nx, cx + neighbor_radius_cells + 1)):
                    if yy == cy and xx == cx:
                        continue
                    zv = ground_zmin[yy, xx]
                    if np.isfinite(zv):
                        n_z.append(float(zv))
                        n_x.append(x0 + (xx + 0.5) * cell)
                        n_y.append(y0 + (yy + 0.5) * cell)

            if len(n_z) < min_neighbor_ground_cells:
                continue

            n_z = np.asarray(n_z, dtype=np.float64)
            n_x = np.asarray(n_x, dtype=np.float64)
            n_y = np.asarray(n_y, dtype=np.float64)

            # Compare to surrounding ground bins
            z_med = float(np.median(n_z))
            if abs(z_low - z_med) > z_tol:
                continue

            ok_plane, plane = _fit_plane(n_x, n_y, n_z)
            if ok_plane:
                a, b, c = plane
                cx0 = x0 + (cx + 0.5) * cell
                cy0 = y0 + (cy + 0.5) * cell
                z_pred = a * cx0 + b * cy0 + c
                if abs(z_low - z_pred) > plane_tol:
                    continue

            # Passed all tests: promote only the lowest-bin candidate points
            out_mask[low_pts] = True

    return out_mask