from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

@dataclass(frozen=True)
class StatisticalSheetCleanerConfig:
    enabled: bool = True
    cell_m: float = 1.0
    smooth_sigma_cells: float = 1.0
    iterations: int = 2
    low_zrange_m: float = 1.5
    high_zrange_m: float = 4.0
    low_iqr_m: float = 0.60
    high_iqr_m: float = 1.50
    tol_thin_m: float = 2.50
    tol_moderate_m: float = 1.00
    tol_complex_m: float = 0.60
    min_upper_fraction: float = 0.10
    upper_tail_start_m: float = 1.50
    min_points_per_cell: int = 4

def _cfg(cfg: dict[str, Any] | None) -> StatisticalSheetCleanerConfig:
    cfg = cfg or {}
    return StatisticalSheetCleanerConfig(
        enabled=bool(cfg.get('als_statsheet_enabled', True)),
        cell_m=float(cfg.get('als_statsheet_cell_m', 1.0)),
        smooth_sigma_cells=float(cfg.get('als_statsheet_sigma_cells', 1.0)),
        iterations=int(cfg.get('als_statsheet_iterations', 2)),
        low_zrange_m=float(cfg.get('als_statsheet_low_zrange_m', 1.5)),
        high_zrange_m=float(cfg.get('als_statsheet_high_zrange_m', 4.0)),
        low_iqr_m=float(cfg.get('als_statsheet_low_iqr_m', 0.60)),
        high_iqr_m=float(cfg.get('als_statsheet_high_iqr_m', 1.50)),
        tol_thin_m=float(cfg.get('als_statsheet_tol_thin_m', 2.50)),
        tol_moderate_m=float(cfg.get('als_statsheet_tol_moderate_m', 1.00)),
        tol_complex_m=float(cfg.get('als_statsheet_tol_complex_m', 0.60)),
        min_upper_fraction=float(cfg.get('als_statsheet_min_upper_fraction', 0.10)),
        upper_tail_start_m=float(cfg.get('als_statsheet_upper_tail_start_m', 1.50)),
        min_points_per_cell=int(cfg.get('als_statsheet_min_points_per_cell', 4)),
    )

def _fill_nearest(grid):
    valid = np.isfinite(grid)
    if np.all(valid): return grid.copy()
    if not np.any(valid): return np.zeros_like(grid, dtype=np.float64)
    _, inds = distance_transform_edt(~valid, return_indices=True)
    return grid[tuple(inds)]

def _cell_stats(z, key, ncell):
    order = np.lexsort((z, key)); ks = key[order]; zs = z[order]
    starts = np.r_[0, 1 + np.flatnonzero(ks[1:] != ks[:-1])]
    ends = np.r_[starts[1:], ks.size]; lens = ends - starts; keys_u = ks[starts]
    def q(qv):
        arr = np.full(ncell, np.nan, dtype=np.float64)
        pos = starts + np.floor((lens - 1) * qv).astype(np.int64)
        arr[keys_u] = zs[pos]
        return arr
    count = np.zeros(ncell, dtype=np.int32); count[keys_u] = lens.astype(np.int32)
    zmin = np.full(ncell, np.nan, dtype=np.float64); zmin[keys_u] = zs[starts]
    return {'zmin': zmin, 'z05': q(.05), 'z25': q(.25), 'z50': q(.50), 'z75': q(.75), 'z95': q(.95), 'count': count}

def _ground_min_surface(z, key, ground_mask, ncell, ny, nx):
    gi = np.flatnonzero(ground_mask)
    if gi.size == 0: return np.full((ny, nx), np.nan)
    kg = key[gi]; zg = z[gi]; order = np.lexsort((zg, kg)); ks = kg[order]; zs = zg[order]
    starts = np.r_[0, 1 + np.flatnonzero(ks[1:] != ks[:-1])]; keys_u = ks[starts]
    out = np.full(ncell, np.nan); out[keys_u] = zs[starts]
    return out.reshape(ny, nx)

def clean_statistical_ground_sheet(*, x, y, z, ground_mask, sensor_mode, cfg=None, return_report=True):
    sm = str(sensor_mode).upper().strip(); original = np.asarray(ground_mask, dtype=bool); out = original.copy()
    report = {'enabled': False, 'sensor_mode': sm, 'ground_before': int(original.sum()), 'ground_after': int(original.sum()), 'demoted_points': 0, 'demoted_per_iteration': [], 'thin_cells': 0, 'moderate_cells': 0, 'complex_cells': 0}
    if sm not in {'ALS', 'ULS'}: return (out, report) if return_report else out
    c = _cfg(cfg); report['enabled'] = c.enabled
    if not c.enabled: return (out, report) if return_report else out
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    wi = np.flatnonzero(valid)
    if wi.size < 10: return (out, report) if return_report else out
    xv, yv, zv = x[wi], y[wi], z[wi]; gv = out[wi].copy(); cell = max(c.cell_m, .25)
    x0 = float(np.floor(xv.min()/cell)*cell); y0 = float(np.floor(yv.min()/cell)*cell)
    ix = np.floor((xv-x0)/cell).astype(np.int32); iy = np.floor((yv-y0)/cell).astype(np.int32)
    nx = int(ix.max())+1; ny = int(iy.max())+1; key = iy.astype(np.int64)*nx + ix.astype(np.int64); ncell = nx*ny
    st = _cell_stats(zv, key, ncell); zrange = st['z95']-st['z05']; iqr = st['z75']-st['z25']; count = st['count']
    lower = st['z05'][key]; upper = (zv >= lower + c.upper_tail_start_m).astype(np.int8)
    upper_count = np.bincount(key, weights=upper, minlength=ncell); upper_frac = upper_count/np.maximum(count, 1)
    enough = count >= c.min_points_per_cell

    # Slope compensation for vertical statistics. A steep bare 1 m terrain
    # cell can have a large raw Z-range simply because the terrain crosses
    # the cell. Estimate that expected geometric span from the raw lower
    # surface before deciding whether the column is truly multilayer.
    raw_lower_surface = gaussian_filter(
        _fill_nearest(st['z05'].reshape(ny, nx)),
        sigma=max(c.smooth_sigma_cells, 0.0),
        mode='nearest',
    )
    rgy, rgx = np.gradient(raw_lower_surface, cell, cell)
    expected_span = (np.abs(rgx) + np.abs(rgy)) * cell
    expected_span = expected_span.ravel()
    zrange_excess = np.maximum(0.0, zrange - expected_span)
    iqr_excess = np.maximum(0.0, iqr - 0.5 * expected_span)

    thin = enough & (zrange_excess <= c.low_zrange_m) & (iqr_excess <= c.low_iqr_m)
    complex_ = enough & ((zrange_excess >= c.high_zrange_m) | (iqr_excess >= c.high_iqr_m))
    moderate = enough & ~(thin | complex_)
    report['thin_cells'] = int(thin.sum()); report['moderate_cells'] = int(moderate.sum()); report['complex_cells'] = int(complex_.sum())
    per = []
    for _ in range(max(1, c.iterations)):
        gmin = _ground_min_surface(zv, key, gv, ncell, ny, nx)
        if not np.any(np.isfinite(gmin)): break
        support = gaussian_filter(_fill_nearest(gmin), sigma=max(c.smooth_sigma_cells, 0.0), mode='nearest')
        gy, gx = np.gradient(support, cell, cell); ns = np.sqrt(1 + gx[iy, ix]**2 + gy[iy, ix]**2)
        h = zv - support[iy, ix]; dn = h/np.maximum(ns, 1e-6)
        ct = thin[key]; cc = complex_[key]; cm = moderate[key]; tail = upper_frac[key] >= c.min_upper_fraction
        tol = np.where(cc, c.tol_complex_m, np.where(cm, c.tol_moderate_m, c.tol_thin_m))
        demote = gv & (dn > tol) & tail & (~ct | (h > c.tol_thin_m))
        n = int(demote.sum()); per.append(n)
        if n == 0: break
        gv[demote] = False
    out[wi] = gv; report['ground_after'] = int(out.sum()); report['demoted_points'] = int((original & ~out).sum()); report['demoted_per_iteration'] = per
    return (out, report) if return_report else out

__all__ = ['StatisticalSheetCleanerConfig', 'clean_statistical_ground_sheet']
