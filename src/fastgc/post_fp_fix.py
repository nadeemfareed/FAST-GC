from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import laspy
import numpy as np
import shutil
from scipy.ndimage import binary_dilation, label

from .io_las import (
    PRODUCT_DEM,
    PRODUCT_GC,
    PRODUCT_NORMALIZED,
    _build_dem_bundle,
    _load_product_context,
    _make_product_dirs,
    _sample_bilinear_from_grid,
    _write_tif,
    list_classified_files,
)
from .monster import DEFAULT_BACKEND, log_info, run_stage


def _write_temp_normalized_from_residual(ctx: dict[str, Any], residual: np.ndarray, out_fp: Path) -> None:
    out = laspy.LasData(ctx["las"].header)
    out.points = ctx["las"].points.copy()
    z_norm = np.maximum(np.asarray(residual, dtype=np.float64), 0.0)
    z_scale = float(out.header.scales[2])
    z_offset = float(out.header.offsets[2])
    raw_Z = np.round((z_norm - z_offset) / z_scale).astype(out.points.array["Z"].dtype, copy=False)
    out.points.array["Z"] = raw_Z
    try:
        out.classification = ctx["cls"].astype(np.uint8, copy=False)
    except Exception:
        out.classification = np.asarray(ctx["cls"], dtype=np.uint8)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out.write(out_fp)


def _ctx_with_cls(ctx: dict[str, Any], cls: np.ndarray) -> dict[str, Any]:
    out = dict(ctx)
    out["cls"] = np.asarray(cls, dtype=np.uint8)
    return out


def _sample_residual(ctx: dict[str, Any], dem_pack: dict[str, Any], dem_res: float) -> np.ndarray:
    z_dem = _sample_bilinear_from_grid(
        dem_pack["dem"], ctx["x"], ctx["y"], dem_pack["xmin"], dem_pack["ymax"], float(dem_res)
    )
    return ctx["z"] - z_dem


def _mad(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    med = float(np.median(a))
    return float(np.median(np.abs(a - med)))


def _grid_index(x: np.ndarray, y: np.ndarray, cell: float, x0: float | None = None, y0: float | None = None):
    if x0 is None:
        x0 = float(np.min(x))
    if y0 is None:
        y0 = float(np.min(y))
    ix = np.floor((x - x0) / cell).astype(np.int32)
    iy = np.floor((y - y0) / cell).astype(np.int32)
    return float(x0), float(y0), ix, iy


def _conv_same_len(v: np.ndarray, k1: np.ndarray) -> np.ndarray:
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


def _bilinear_sample(grid: np.ndarray, x: np.ndarray, y: np.ndarray, x0: float, y0: float, cell: float):
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
    return np.where(ok, (1.0 - fx) * (1.0 - fy) * g00 + fx * (1.0 - fy) * g10 + (1.0 - fx) * fy * g01 + fx * fy * g11, nn)


def _coarse_ground_cell_prefilter_demote(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    *,
    cell_m: float = 4.0,
    min_points_cell: int = 8,
    range_abs_thr_m: float = 1.20,
    range_rel_k: float = 3.0,
    mad_abs_thr_m: float = 0.18,
    mad_rel_k: float = 3.0,
    dz_neighbor_abs_m: float = 0.60,
    point_m_score_thr: float = 3.5,
    point_high_offset_m: float = 0.18,
    max_demote_fraction: float = 0.45,
) -> np.ndarray:
    cls = np.asarray(cls, dtype=np.uint8)
    ground_ids = np.flatnonzero(cls == 2)
    out = np.zeros(cls.shape, dtype=bool)
    if ground_ids.size < max(20, min_points_cell):
        return out

    gx, gy, gz = np.asarray(x[ground_ids], np.float64), np.asarray(y[ground_ids], np.float64), np.asarray(z[ground_ids], np.float64)
    x0, y0, ix, iy = _grid_index(gx, gy, float(cell_m))
    nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
    count = np.zeros((ny, nx), dtype=np.int32)
    zmin = np.full((ny, nx), np.nan, dtype=np.float64)
    zmax = np.full((ny, nx), np.nan, dtype=np.float64)
    med = np.full((ny, nx), np.nan, dtype=np.float64)
    madd = np.full((ny, nx), np.nan, dtype=np.float64)
    cell_to_ground: dict[int, list[int]] = {}

    for local_idx, pid in enumerate(ground_ids.tolist()):
        cx, cy = int(ix[local_idx]), int(iy[local_idx])
        lin = cy * nx + cx
        cell_to_ground.setdefault(lin, []).append(int(pid))
        count[cy, cx] += 1
        v = gz[local_idx]
        if not np.isfinite(zmin[cy, cx]) or v < zmin[cy, cx]:
            zmin[cy, cx] = v
        if not np.isfinite(zmax[cy, cx]) or v > zmax[cy, cx]:
            zmax[cy, cx] = v

    for lin, ids in cell_to_ground.items():
        if len(ids) < int(min_points_cell):
            continue
        cy = lin // nx
        cx = lin % nx
        vals = np.asarray(z[ids], dtype=np.float64)
        med[cy, cx] = float(np.median(vals))
        madd[cy, cx] = _mad(vals)

    zrange = zmax - zmin
    valid = np.isfinite(med) & np.isfinite(madd) & np.isfinite(zrange) & (count >= int(min_points_cell))
    if not np.any(valid):
        return out

    for cy, cx in np.argwhere(valid).tolist():
        y0n, y1n = max(0, cy - 1), min(ny, cy + 2)
        x0n, x1n = max(0, cx - 1), min(nx, cx + 2)
        nbr_valid = valid[y0n:y1n, x0n:x1n].copy()
        nbr_valid[cy - y0n, cx - x0n] = False
        if not np.any(nbr_valid):
            continue
        nbr_med = med[y0n:y1n, x0n:x1n][nbr_valid]
        nbr_range = zrange[y0n:y1n, x0n:x1n][nbr_valid]
        nbr_mad = madd[y0n:y1n, x0n:x1n][nbr_valid]
        nbr_med_med = float(np.median(nbr_med))
        nbr_range_med = float(np.median(nbr_range))
        nbr_mad_med = float(np.median(nbr_mad))
        nbr_range_mad = _mad(nbr_range)
        nbr_mad_mad = _mad(nbr_mad)
        dz = float(med[cy, cx] - nbr_med_med)
        is_high_range = zrange[cy, cx] > max(float(range_abs_thr_m), nbr_range_med + float(range_rel_k) * 1.4826 * max(nbr_range_mad, 1e-6))
        is_high_mad = madd[cy, cx] > max(float(mad_abs_thr_m), nbr_mad_med + float(mad_rel_k) * 1.4826 * max(nbr_mad_mad, 1e-6))
        is_high_dz = dz > float(dz_neighbor_abs_m)
        if not (is_high_range or is_high_mad or is_high_dz):
            continue
        ids = np.asarray(cell_to_ground.get(cy * nx + cx, []), dtype=np.int32)
        if ids.size < int(min_points_cell):
            continue
        vals = np.asarray(z[ids], dtype=np.float64)
        cell_med = float(med[cy, cx])
        cell_mad = float(madd[cy, cx])
        robust_scale = 1.4826 * max(cell_mad, 1e-6)
        mscore = 0.6745 * (vals - cell_med) / max(cell_mad, 1e-6)
        high_thr = max(cell_med + float(point_high_offset_m), nbr_med_med + float(point_high_offset_m), cell_med + (float(point_m_score_thr) / 0.6745) * robust_scale)
        demote_local = (vals > high_thr) & (mscore > float(point_m_score_thr))
        if demote_local.size > 0 and (np.count_nonzero(demote_local) / float(demote_local.size)) <= float(max_demote_fraction):
            out[ids[demote_local]] = True
    return out


def _weak_support_swamp_demote(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls: np.ndarray,
    *,
    cell_m: float = 10.0,
    low_count_abs: int = 10,
    low_count_rel_frac: float = 0.35,
    high_zrange_abs_m: float = 1.0,
    high_zrange_rel_k: float = 2.5,
    support_ring_min_count: int = 20,
    point_above_ring_m: float = 0.25,
    component_expand_cells: int = 1,
) -> np.ndarray:
    """
    Targeted weak-support pass for swamp / marsh / water-edge terrain.

    Runs ONLY on current ground points using a low-resolution 10 m grid:
      - ground point count per cell
      - z-range per cell
      - local neighborhood comparison

    Flags only weak-support, high-zrange cells, groups them into connected patches,
    then uses surrounding stronger-support neighborhood cells as the terrain reference.
    Ground points in the flagged patches that sit above the surrounding terrain membrane
    are demoted to non-ground.
    """
    cls = np.asarray(cls, dtype=np.uint8)
    ground_ids = np.flatnonzero(cls == 2)
    out = np.zeros(cls.shape, dtype=bool)
    if ground_ids.size < max(20, low_count_abs):
        return out

    gx = np.asarray(x[ground_ids], dtype=np.float64)
    gy = np.asarray(y[ground_ids], dtype=np.float64)
    gz = np.asarray(z[ground_ids], dtype=np.float64)

    x0, y0, ix, iy = _grid_index(gx, gy, float(cell_m))
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1

    count = np.zeros((ny, nx), dtype=np.int32)
    zmin = np.full((ny, nx), np.nan, dtype=np.float64)
    zmax = np.full((ny, nx), np.nan, dtype=np.float64)
    med = np.full((ny, nx), np.nan, dtype=np.float64)
    cell_to_ground: dict[int, list[int]] = {}

    for local_idx, pid in enumerate(ground_ids.tolist()):
        cx = int(ix[local_idx]); cy = int(iy[local_idx])
        lin = cy * nx + cx
        cell_to_ground.setdefault(lin, []).append(int(pid))
        count[cy, cx] += 1
        v = gz[local_idx]
        if not np.isfinite(zmin[cy, cx]) or v < zmin[cy, cx]:
            zmin[cy, cx] = v
        if not np.isfinite(zmax[cy, cx]) or v > zmax[cy, cx]:
            zmax[cy, cx] = v

    for lin, ids in cell_to_ground.items():
        cy = lin // nx
        cx = lin % nx
        vals = np.asarray(z[ids], dtype=np.float64)
        med[cy, cx] = float(np.median(vals))

    zrange = zmax - zmin
    valid = np.isfinite(med) & np.isfinite(zrange) & (count > 0)
    if not np.any(valid):
        return out

    candidate = np.zeros((ny, nx), dtype=bool)
    for cy, cx in np.argwhere(valid).tolist():
        y0n, y1n = max(0, cy - 1), min(ny, cy + 2)
        x0n, x1n = max(0, cx - 1), min(nx, cx + 2)
        nbr_valid = valid[y0n:y1n, x0n:x1n].copy()
        nbr_valid[cy - y0n, cx - x0n] = False
        if not np.any(nbr_valid):
            continue

        nbr_counts = count[y0n:y1n, x0n:x1n][nbr_valid].astype(np.float64)
        nbr_zrange = zrange[y0n:y1n, x0n:x1n][nbr_valid]
        nbr_count_med = float(np.median(nbr_counts))
        nbr_zrange_med = float(np.median(nbr_zrange))
        nbr_zrange_mad = _mad(nbr_zrange)

        is_low_count = bool(
            (count[cy, cx] <= int(low_count_abs))
            or (count[cy, cx] <= max(1.0, float(low_count_rel_frac) * nbr_count_med))
        )
        is_high_zrange = bool(
            zrange[cy, cx] > max(float(high_zrange_abs_m), nbr_zrange_med + float(high_zrange_rel_k) * 1.4826 * max(nbr_zrange_mad, 1e-6))
        )
        candidate[cy, cx] = is_low_count and is_high_zrange

    if not np.any(candidate):
        return out

    structure = np.ones((3, 3), dtype=np.uint8)
    labels, nlab = label(candidate.astype(np.uint8), structure=structure)
    if nlab <= 0:
        return out

    expanded = binary_dilation(candidate, structure=structure, iterations=max(0, int(component_expand_cells)))

    for lab in range(1, int(nlab) + 1):
        comp = labels == lab
        if not np.any(comp):
            continue

        patch = expanded & binary_dilation(comp, structure=structure, iterations=max(0, int(component_expand_cells)))
        ring = binary_dilation(patch, structure=structure, iterations=1) & (~patch) & valid & (count >= int(support_ring_min_count))
        ring_vals = med[ring]
        ring_vals = ring_vals[np.isfinite(ring_vals)]
        if ring_vals.size < 3:
            continue

        ring_med = float(np.median(ring_vals))
        patch_cells = np.argwhere(patch & valid)
        for cy, cx in patch_cells.tolist():
            ids = np.asarray(cell_to_ground.get(cy * nx + cx, []), dtype=np.int32)
            if ids.size == 0:
                continue
            vals = np.asarray(z[ids], dtype=np.float64)
            demote_local = vals > (ring_med + float(point_above_ring_m))
            out[ids[demote_local]] = True

    return out



class _MembraneVoteConfig:
    def __init__(self, *, cell: float = 0.5, neighbor_radius_cells: int = 4, min_neighbor_cells: int = 6, max_robust_z: float = 2.8, mad_floor: float = 0.05, fill_iters: int = 60, smooth_sigma_cells: float = 1.0, ground_threshold: float = 0.12, slope_adapt_k: float = 0.25) -> None:
        self.cell = float(cell)
        self.neighbor_radius_cells = int(neighbor_radius_cells)
        self.min_neighbor_cells = int(min_neighbor_cells)
        self.max_robust_z = float(max_robust_z)
        self.mad_floor = float(mad_floor)
        self.fill_iters = int(fill_iters)
        self.smooth_sigma_cells = float(smooth_sigma_cells)
        self.ground_threshold = float(ground_threshold)
        self.slope_adapt_k = float(slope_adapt_k)


def _build_refined_membrane_from_ground(xg: np.ndarray, yg: np.ndarray, zg: np.ndarray, cfg: _MembraneVoteConfig):
    xg, yg, zg = np.asarray(xg, np.float64), np.asarray(yg, np.float64), np.asarray(zg, np.float64)
    if xg.size == 0:
        return np.empty((0, 0), dtype=np.float32), 0.0, 0.0
    cell = float(cfg.cell)
    x0, y0, ix, iy = _grid_index(xg, yg, cell)
    nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
    seed = np.full((ny, nx), np.nan, dtype=np.float32)
    bins = [[] for _ in range(nx * ny)]
    for p in range(ix.size):
        i, j = int(ix[p]), int(iy[p])
        if 0 <= i < nx and 0 <= j < ny:
            bins[j * nx + i].append(p)
    for j in range(ny):
        base = j * nx
        for i in range(nx):
            pts = bins[base + i]
            if not pts:
                continue
            vals = zg[np.asarray(pts, dtype=np.int32)]
            seed[j, i] = np.float32(np.percentile(vals, 25.0))
    grid = np.full((ny, nx), np.nan, dtype=np.float32)
    r = int(max(1, cfg.neighbor_radius_cells))
    min_nei = int(max(1, cfg.min_neighbor_cells))
    max_rz = float(cfg.max_robust_z)
    mad_floor = float(cfg.mad_floor)
    for j in range(ny):
        j0, j1 = max(0, j - r), min(ny, j + r + 1)
        for i in range(nx):
            i0, i1 = max(0, i - r), min(nx, i + r + 1)
            vals = seed[j0:j1, i0:i1]
            vals = vals[np.isfinite(vals)]
            if vals.size < min_nei:
                continue
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            mad = float(max(mad, mad_floor))
            rz = np.abs(vals - med) / (1.4826 * mad)
            good = vals[rz <= max_rz]
            if good.size < min_nei:
                continue
            grid[j, i] = np.float32(np.median(good))
    for _ in range(int(max(0, cfg.fill_iters))):
        nan_mask = ~np.isfinite(grid)
        if not nan_mask.any():
            break
        g2 = grid.copy()
        for j in range(ny):
            for i in range(nx):
                if np.isfinite(grid[j, i]):
                    continue
                j0, j1 = max(0, j - 1), min(ny, j + 2)
                i0, i1 = max(0, i - 1), min(nx, i + 2)
                neigh = grid[j0:j1, i0:i1]
                neigh = neigh[np.isfinite(neigh)]
                if neigh.size:
                    g2[j, i] = np.float32(np.median(neigh))
        grid = g2
    sigma = float(max(0.01, cfg.smooth_sigma_cells))
    rad = int(np.ceil(3.0 * sigma))
    xs = np.arange(-rad, rad + 1, dtype=np.float32)
    k = np.exp(-(xs * xs) / (2.0 * sigma * sigma)).astype(np.float32)
    k /= float(k.sum())
    grid = _conv1d_nan(grid, k, axis=0)
    grid = _conv1d_nan(grid, k, axis=1)
    return grid, x0, y0


def _classify_by_refined_membrane(x: np.ndarray, y: np.ndarray, z: np.ndarray, surf_z: np.ndarray, x0: float, y0: float, cfg: _MembraneVoteConfig) -> np.ndarray:
    x, y, z = np.asarray(x, np.float64), np.asarray(y, np.float64), np.asarray(z, np.float64)
    if surf_z.size == 0:
        return np.zeros(z.shape, dtype=bool)
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
    return np.isfinite(zhat) & (resid <= thr) & (resid >= -2.0 * thr)


def _fix_one_classified_tile(classified_fp: str, temp_product_dirs: dict[str, str], dem_res: float, nonground_to_ground_max_z: float, ground_to_nonground_min_z: float, *, keep_temp: bool = False) -> dict[str, Any]:
    ctx = _load_product_context(classified_fp)
    cls = ctx["cls"].copy().astype(np.uint8, copy=False)

    coarse_prefilter_demote = _coarse_ground_cell_prefilter_demote(ctx["x"], ctx["y"], ctx["z"], cls)
    if np.any(coarse_prefilter_demote):
        cls[coarse_prefilter_demote] = 1

    weak_support_demote = _weak_support_swamp_demote(ctx["x"], ctx["y"], ctx["z"], cls)
    if np.any(weak_support_demote):
        cls[weak_support_demote] = 1

    try:
        dem_pack = _build_dem_bundle(_ctx_with_cls(ctx, cls), dem_res)
    except RuntimeError as e:
        msg = str(e)
        if "Too few ground points" in msg:
            return {"tile": Path(classified_fp).name, "changed_points": int(np.count_nonzero(coarse_prefilter_demote)), "coarse_prefilter_demoted_to_nonground": int(np.count_nonzero(coarse_prefilter_demote)), "weak_support_demoted_to_nonground": int(np.count_nonzero(weak_support_demote)), "core_promoted_to_ground": 0, "core_demoted_to_nonground": 0, "membrane_promoted_to_ground": 0, "membrane_demoted_to_nonground": 0, "status": "skipped_too_few_ground", "message": msg}
        raise

    residual = _sample_residual(ctx, dem_pack, dem_res)

    if keep_temp:
        temp_dem_fp = Path(temp_product_dirs[PRODUCT_DEM]) / f"{ctx['base']}.tif"
        _write_tif(dem_pack["dem"], str(temp_dem_fp), dem_pack["xmin"], dem_pack["ymax"], dem_res, crs=ctx["crs"])
        temp_norm_fp = Path(temp_product_dirs[PRODUCT_NORMALIZED]) / f"{ctx['base']}.las"
        _write_temp_normalized_from_residual(_ctx_with_cls(ctx, cls), residual, temp_norm_fp)

    nonground = cls != 2
    ground = cls == 2
    core_promote = nonground & np.isfinite(residual) & (residual <= float(nonground_to_ground_max_z))
    core_demote = ground & np.isfinite(residual) & (residual > float(ground_to_nonground_min_z))
    if np.any(core_promote):
        cls[core_promote] = 2
    if np.any(core_demote):
        cls[core_demote] = 1

    gmask = cls == 2
    membrane_promote = np.zeros(cls.shape, dtype=bool)
    membrane_demote = np.zeros(cls.shape, dtype=bool)
    if np.count_nonzero(gmask) >= 20:
        mem_cfg = _MembraneVoteConfig(cell=max(0.5, float(dem_res)), ground_threshold=max(0.10, float(ground_to_nonground_min_z)))
        surf_z, sx0, sy0 = _build_refined_membrane_from_ground(ctx["x"][gmask], ctx["y"][gmask], ctx["z"][gmask], mem_cfg)
        membrane_mask = _classify_by_refined_membrane(ctx["x"], ctx["y"], ctx["z"], surf_z, sx0, sy0, mem_cfg)
        membrane_promote = (cls != 2) & membrane_mask
        zhat = _bilinear_sample(surf_z, ctx["x"], ctx["y"], sx0, sy0, mem_cfg.cell)
        mem_resid = ctx["z"] - zhat
        membrane_demote = (cls == 2) & (~membrane_mask) & np.isfinite(mem_resid) & (mem_resid > max(0.08, float(ground_to_nonground_min_z)))
        if np.any(membrane_promote):
            cls[membrane_promote] = 2
        if np.any(membrane_demote):
            cls[membrane_demote] = 1

    changed = int(np.count_nonzero(cls != ctx["cls"]))
    if changed:
        out = laspy.LasData(ctx["las"].header)
        out.points = ctx["las"].points.copy()
        try:
            out.classification = cls.astype(np.uint8, copy=False)
        except Exception:
            out.classification = np.asarray(cls, dtype=np.uint8)
        out.write(classified_fp)

    return {
        "tile": Path(classified_fp).name,
        "changed_points": changed,
        "coarse_prefilter_demoted_to_nonground": int(np.count_nonzero(coarse_prefilter_demote)),
        "weak_support_demoted_to_nonground": int(np.count_nonzero(weak_support_demote)),
        "core_promoted_to_ground": int(np.count_nonzero(core_promote)),
        "core_demoted_to_nonground": int(np.count_nonzero(core_demote)),
        "membrane_promoted_to_ground": int(np.count_nonzero(membrane_promote)),
        "membrane_demoted_to_nonground": int(np.count_nonzero(membrane_demote)),
        "status": "ok",
        "message": None,
    }


def apply_fp_fix_to_output_root(out_root: str | Path, sensor_mode: str, *, dem_res: float = 0.25, nonground_to_ground_max_z: float = 0.0, ground_to_nonground_min_z: float = 0.06, keep_temp: bool = False, n_jobs: int = 1, joblib_backend: str = DEFAULT_BACKEND, joblib_batch_size: int | str = "auto", joblib_pre_dispatch: str | int = "2*n_jobs") -> dict[str, Any]:
    out_root = Path(out_root)
    classified_root = out_root / PRODUCT_GC
    if not classified_root.exists():
        raise FileNotFoundError(f"FAST_GC folder not found: {classified_root}")
    classified_files = list_classified_files(classified_root)
    if not classified_files:
        raise FileNotFoundError(f"No classified LAS files found in: {classified_root}")

    temp_root = out_root / "_temp_fp_fix"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_product_dirs = _make_product_dirs(str(temp_root))
    log_info("FP-FIX: 4 m coarse pre-clean -> 10 m weak-support swamp/water pass -> build DEM -> traditional residual core -> normal-space membrane voting refine")

    total_t0 = perf_counter()

    def _worker(gc_fp: str) -> dict[str, Any]:
        return _fix_one_classified_tile(gc_fp, temp_product_dirs, dem_res, nonground_to_ground_max_z, ground_to_nonground_min_z, keep_temp=bool(keep_temp))

    stage_summary = run_stage(
        f"FP-FIX {sensor_mode.upper()}",
        classified_files,
        _worker,
        n_jobs=n_jobs,
        backend=joblib_backend,
        batch_size=joblib_batch_size,
        pre_dispatch=joblib_pre_dispatch,
        source=str(classified_root),
        unit="tile",
        show_banner=True,
        show_progress=True,
    )

    summary_rows, failed_rows = [], []
    for rec in stage_summary.records:
        if isinstance(rec.result, dict):
            row = dict(rec.result)
        else:
            row = {"tile": rec.name, "changed_points": 0, "coarse_prefilter_demoted_to_nonground": 0, "weak_support_demoted_to_nonground": 0, "core_promoted_to_ground": 0, "core_demoted_to_nonground": 0, "membrane_promoted_to_ground": 0, "membrane_demoted_to_nonground": 0, "status": rec.status, "message": rec.error}
        if rec.status == "failed":
            row.setdefault("tile", rec.name)
            row["status"] = "failed"
            row["message"] = rec.error
            failed_rows.append(row)
        elif row.get("status") == "skipped_too_few_ground":
            row["status"] = "skipped"
        summary_rows.append(row)

    total_elapsed = perf_counter() - total_t0
    summary = {
        "sensor_mode": sensor_mode.upper(),
        "dem_res": float(dem_res),
        "nonground_to_ground_max_z": float(nonground_to_ground_max_z),
        "ground_to_nonground_min_z": float(ground_to_nonground_min_z),
        "tile_count": len(classified_files),
        "ok_tile_count": int(stage_summary.ok),
        "skipped_tile_count": sum(1 for r in summary_rows if r.get("status") == "skipped"),
        "failed_tile_count": len(failed_rows),
        "total_changed_points": int(sum(int(r.get("changed_points", 0)) for r in summary_rows)),
        "total_coarse_prefilter_demoted_to_nonground": int(sum(int(r.get("coarse_prefilter_demoted_to_nonground", 0)) for r in summary_rows)),
        "total_weak_support_demoted_to_nonground": int(sum(int(r.get("weak_support_demoted_to_nonground", 0)) for r in summary_rows)),
        "total_core_promoted_to_ground": int(sum(int(r.get("core_promoted_to_ground", 0)) for r in summary_rows)),
        "total_core_demoted_to_nonground": int(sum(int(r.get("core_demoted_to_nonground", 0)) for r in summary_rows)),
        "total_membrane_promoted_to_ground": int(sum(int(r.get("membrane_promoted_to_ground", 0)) for r in summary_rows)),
        "total_membrane_demoted_to_nonground": int(sum(int(r.get("membrane_demoted_to_nonground", 0)) for r in summary_rows)),
        "time_s": float(total_elapsed),
        "tiles": summary_rows,
    }
    summary_fp = temp_root / "fp_fix_summary.json"
    with summary_fp.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if not keep_temp:
        shutil.rmtree(temp_root, ignore_errors=True)
    return summary
