from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import binary_dilation, label


def _bilinear_sample(
    grid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    y0: float,
    cell: float,
) -> np.ndarray:
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


def _fit_plane(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray):
    """
    Fit z = ax + by + c using least squares.
    Returns (ok, (a, b, c)).
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


def _build_support_grids(
    xw: np.ndarray,
    yw: np.ndarray,
    zw: np.ndarray,
    ground_mask: np.ndarray,
    *,
    cell: float,
):
    """
    Build candidate occupancy and final-ground occupancy at parent-grid resolution.
    Also stores per-cell point indices and minimum ground z.
    """
    x0 = float(np.min(xw))
    y0 = float(np.min(yw))

    ix = np.floor((xw - x0) / cell).astype(np.int32)
    iy = np.floor((yw - y0) / cell).astype(np.int32)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1

    cand_occ = np.zeros((ny, nx), dtype=bool)
    ground_occ = np.zeros((ny, nx), dtype=bool)
    ground_zmin = np.full((ny, nx), np.nan, dtype=np.float64)
    cell_to_points: dict[int, list[int]] = {}

    for p in range(xw.size):
        cx = int(ix[p])
        cy = int(iy[p])
        cand_occ[cy, cx] = True
        lin = cy * nx + cx
        cell_to_points.setdefault(lin, []).append(p)

        if ground_mask[p]:
            ground_occ[cy, cx] = True
            cur = ground_zmin[cy, cx]
            if not np.isfinite(cur) or zw[p] < cur:
                ground_zmin[cy, cx] = zw[p]

    return x0, y0, ix, iy, nx, ny, cand_occ, ground_occ, ground_zmin, cell_to_points


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
    Void-aware ground recovery.

    Main idea:
      1) Build a support/void raster from the current final ground using a parent grid.
      2) Find coherent voids where candidate points exist but final ground support is absent.
      3) For each void, inspect a small buffer ring (bank) around the void.
      4) Promote only non-ground points inside the void whose z matches the bank-defined
         local surface continuation.

    This is designed for ALS / ULS and avoids promoting canopy points inside big canopy voids.
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

    # Parent grid / void controls
    cell = float(cfg.get("void_recover_cell_m", 2.0))
    void_buffer_m = float(cfg.get("void_recover_void_buffer_m", 0.5))
    min_component_cells = int(cfg.get("void_recover_min_component_cells", 2))
    max_component_cells = int(cfg.get("void_recover_max_component_cells", 500))
    min_void_points = int(cfg.get("void_recover_min_void_points", 1))

    # Compactness rejection for candidate points in a cell
    bin_dz = float(cfg.get("void_recover_bin_dz_m", 0.20))
    max_bins = int(cfg.get("void_recover_max_bins", 2))
    z_std_thr = float(cfg.get("void_recover_z_std_thr_m", 0.12))
    z_span_thr = float(cfg.get("void_recover_z_span_thr_m", 0.35))

    # Bank comparison / interpolation checks
    min_bank_ground_points = int(cfg.get("void_recover_min_bank_ground_points", 6))
    bank_match_tol = float(cfg.get("void_recover_bank_match_tol_m", 0.18))
    plane_tol = float(cfg.get("void_recover_plane_tol_m", 0.18))
    surface_tol = float(cfg.get("void_recover_surface_tol_m", 0.18))
    use_surface_anchor = bool(cfg.get("void_recover_use_surface_anchor", True))

    # Promotion selection
    above_tol = float(cfg.get("void_recover_above_tol_m", 0.04))
    max_abs_resid = float(cfg.get("void_recover_max_abs_resid_m", 0.18))
    promote_cluster_dz = float(cfg.get("void_recover_promote_cluster_dz_m", 0.05))
    promote_one_per_void_cell = bool(cfg.get("void_recover_promote_one_per_void_cell", False))

    (
        x0,
        y0,
        ix,
        iy,
        nx,
        ny,
        cand_occ,
        ground_occ,
        ground_zmin,
        cell_to_points,
    ) = _build_support_grids(
        xw,
        yw,
        zw,
        out_mask,
        cell=cell,
    )

    # Void = candidate points exist but final ground support absent
    void = cand_occ & (~ground_occ)
    if not np.any(void):
        return out_mask

    # Connected voids
    structure = np.ones((3, 3), dtype=np.int32)
    labels, nlab = label(void, structure=structure)

    if nlab == 0:
        return out_mask

    buffer_cells = max(1, int(math.ceil(void_buffer_m / max(cell, 1e-6))))

    for lab_id in range(1, nlab + 1):
        comp = labels == lab_id
        n_cells = int(np.count_nonzero(comp))
        if n_cells < min_component_cells or n_cells > max_component_cells:
            continue

        # ring around the void = bank
        comp_buffer = binary_dilation(comp, iterations=buffer_cells)
        bank = comp_buffer & (~comp)

        if not np.any(bank):
            continue

        bank_ground_cells = bank & ground_occ
        if not np.any(bank_ground_cells):
            continue

        bank_ground_z = ground_zmin[bank_ground_cells]
        bank_ground_z = bank_ground_z[np.isfinite(bank_ground_z)]
        if bank_ground_z.size < min_bank_ground_points:
            continue

        # robust local bank-defined surface
        bank_med = float(np.median(bank_ground_z))
        bank_q10 = float(np.percentile(bank_ground_z, 10.0))

        # collect bank ground cell centers for optional plane fit
        by, bx = np.where(bank_ground_cells)
        bank_x = x0 + (bx.astype(np.float64) + 0.5) * cell
        bank_y = y0 + (by.astype(np.float64) + 0.5) * cell
        ok_plane, plane = _fit_plane(bank_x, bank_y, bank_ground_z)

        comp_cells_y, comp_cells_x = np.where(comp)
        for cy, cx in zip(comp_cells_y.tolist(), comp_cells_x.tolist()):
            lin = cy * nx + cx
            pts = cell_to_points.get(lin, None)
            if not pts:
                continue

            pts_idx = np.asarray(pts, dtype=np.int32)
            if pts_idx.size < min_void_points:
                continue

            # only consider currently non-ground points for recovery
            pts_idx = pts_idx[~out_mask[pts_idx]]
            if pts_idx.size == 0:
                continue

            zc = zw[pts_idx]

            # reject vertically diverse cell contents - likely vegetation
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

            cx0 = x0 + (cx + 0.5) * cell
            cy0 = y0 + (cy + 0.5) * cell

            target_parts = [bank_med, bank_q10]

            if ok_plane:
                a, b, c = plane
                z_plane = a * cx0 + b * cy0 + c
                target_parts.append(float(z_plane))
            else:
                z_plane = np.nan

            if use_surface_anchor:
                z_surf = float(
                    _bilinear_sample(
                        surf_z,
                        np.array([cx0], dtype=np.float64),
                        np.array([cy0], dtype=np.float64),
                        sx0,
                        sy0,
                        float(surf_cell),
                    )[0]
                )
                if np.isfinite(z_surf):
                    target_parts.append(z_surf)
            else:
                z_surf = np.nan

            z_target = float(np.median(np.asarray(target_parts, dtype=np.float64)))

            # consistency checks against bank / plane / surface
            if abs(z_target - bank_med) > bank_match_tol:
                continue
            if np.isfinite(z_plane) and abs(z_target - z_plane) > plane_tol:
                continue
            if np.isfinite(z_surf) and abs(z_target - z_surf) > surface_tol:
                continue

            resid = zc - z_target
            valid = np.abs(resid) <= max_abs_resid
            valid &= resid <= above_tol

            if not np.any(valid):
                continue

            cand_idx = pts_idx[valid]
            cand_resid = resid[valid]
            cand_z = zw[cand_idx]

            best_local = int(np.argmin(np.abs(cand_resid)))
            best_idx = int(cand_idx[best_local])
            best_z = float(zw[best_idx])

            if promote_one_per_void_cell:
                out_mask[best_idx] = True
            else:
                promote_sel = np.abs(cand_z - best_z) <= promote_cluster_dz
                if not np.any(promote_sel):
                    out_mask[best_idx] = True
                else:
                    out_mask[cand_idx[promote_sel]] = True

    return out_mask