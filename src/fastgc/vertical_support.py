from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


VERTICAL_SUPPORT_VERSION = 1
DEFAULT_XY_CELL_M = 5.0
DEFAULT_Z_BIN_M = 1.0


def _finite_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return x[m], y[m], z[m]


def build_vertical_support(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    out_path: str | Path,
    *,
    xy_cell_m: float = DEFAULT_XY_CELL_M,
    z_bin_m: float = DEFAULT_Z_BIN_M,
) -> dict[str, Any]:
    """Build a compact, classification-blind XYZ support descriptor.

    The descriptor is intentionally computed from raw XYZ only.  It does not
    inspect LAS Classification or any reference extra byte.  Each 5 m XY cell
    gets a robust local lower origin (2nd percentile), then a sparse 1 m vertical
    histogram plus cheap lower/upper support summaries.  The saved NPZ is meant
    to be reused later by invert voting and final QC; V1 only records it.
    """
    x, y, z = _finite_xyz(x, y, z)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cell = float(xy_cell_m)
    dz = float(z_bin_m)
    if cell <= 0.0 or dz <= 0.0:
        raise ValueError("xy_cell_m and z_bin_m must be > 0")

    if x.size == 0:
        np.savez_compressed(
            out_path,
            version=np.int16(VERTICAL_SUPPORT_VERSION),
            xy_cell_m=np.float32(cell), z_bin_m=np.float32(dz),
            origin_x=np.float64(0.0), origin_y=np.float64(0.0),
            nx=np.int32(0), ny=np.int32(0),
            cell_id=np.empty(0, np.int32), z0=np.empty(0, np.float32),
            n_total=np.empty(0, np.int32), n_low1=np.empty(0, np.int32),
            n_low2=np.empty(0, np.int32), n_low3=np.empty(0, np.int32),
            n_upper3=np.empty(0, np.int32), frac_low1=np.empty(0, np.float32),
            frac_low2=np.empty(0, np.float32), frac_low3=np.empty(0, np.float32),
            density_total=np.empty(0, np.float32), density_low2=np.empty(0, np.float32),
            occupied_z_bins=np.empty(0, np.int16), robust_z_span=np.empty(0, np.float32),
            hist_cell_id=np.empty(0, np.int32), hist_zbin=np.empty(0, np.int16),
            hist_count=np.empty(0, np.int32),
        )
        return {"version": VERTICAL_SUPPORT_VERSION, "xy_cell_m": cell, "z_bin_m": dz,
                "occupied_xy_cells": 0, "support_file": str(out_path)}

    x0 = np.floor(np.min(x) / cell) * cell
    y0 = np.floor(np.min(y) / cell) * cell
    ix = np.floor((x - x0) / cell).astype(np.int32)
    iy = np.floor((y - y0) / cell).astype(np.int32)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1
    cid = iy.astype(np.int64) * nx + ix.astype(np.int64)

    # One sort supplies group boundaries, robust local lower origin and upper
    # quantile.  This is preprocessing-only and avoids repeated KD-tree work.
    order = np.lexsort((z, cid))
    cs = cid[order]
    zs = z[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    ends = np.r_[starts[1:], cs.size]
    counts = (ends - starts).astype(np.int32)
    cells = cs[starts].astype(np.int32)

    # Robust lower origin: empirical 2nd percentile within each XY cell.
    q02_idx = starts + np.floor(0.02 * np.maximum(counts - 1, 0)).astype(np.int64)
    q98_idx = starts + np.floor(0.98 * np.maximum(counts - 1, 0)).astype(np.int64)
    z0_cell = zs[q02_idx]
    z98_cell = zs[q98_idx]

    # Map local z0 back to every sorted point, then form sparse relative bins.
    group_idx = np.repeat(np.arange(cells.size, dtype=np.int32), counts)
    rel = np.maximum(0.0, zs - z0_cell[group_idx])
    zbin = np.floor(rel / dz).astype(np.int32)

    n_low1 = np.add.reduceat((rel < 1.0).astype(np.int32), starts)
    n_low2 = np.add.reduceat((rel < 2.0).astype(np.int32), starts)
    n_low3 = np.add.reduceat((rel < 3.0).astype(np.int32), starts)
    n_upper3 = counts - n_low3

    denom = np.maximum(counts.astype(np.float64), 1.0)
    frac1 = (n_low1 / denom).astype(np.float32)
    frac2 = (n_low2 / denom).astype(np.float32)
    frac3 = (n_low3 / denom).astype(np.float32)
    area = cell * cell
    dens_total = (counts / area).astype(np.float32)
    dens_low2 = (n_low2 / area).astype(np.float32)
    zspan = np.maximum(0.0, z98_cell - z0_cell).astype(np.float32)

    # Sparse histogram triplets (cell id, relative z bin, count).
    pair = cs.astype(np.int64) * (int(zbin.max()) + 1) + zbin.astype(np.int64)
    hp, hcnt = np.unique(pair, return_counts=True)
    hbase = int(zbin.max()) + 1
    hcell = (hp // hbase).astype(np.int32)
    hbin32 = (hp % hbase).astype(np.int32)
    if hbin32.max(initial=0) <= np.iinfo(np.int16).max:
        hbin = hbin32.astype(np.int16)
    else:
        hbin = hbin32
    # Number of occupied vertical bins per occupied XY cell.
    hgroup = np.searchsorted(cells, hcell)
    occupied = np.bincount(hgroup, minlength=cells.size).astype(np.int16)

    np.savez_compressed(
        out_path,
        version=np.int16(VERTICAL_SUPPORT_VERSION),
        xy_cell_m=np.float32(cell), z_bin_m=np.float32(dz),
        origin_x=np.float64(x0), origin_y=np.float64(y0),
        nx=np.int32(nx), ny=np.int32(ny),
        cell_id=cells, z0=z0_cell.astype(np.float32),
        n_total=counts, n_low1=n_low1.astype(np.int32), n_low2=n_low2.astype(np.int32),
        n_low3=n_low3.astype(np.int32), n_upper3=n_upper3.astype(np.int32),
        frac_low1=frac1, frac_low2=frac2, frac_low3=frac3,
        density_total=dens_total, density_low2=dens_low2,
        occupied_z_bins=occupied, robust_z_span=zspan,
        hist_cell_id=hcell, hist_zbin=hbin, hist_count=hcnt.astype(np.int32),
    )

    # V1 diagnostic labels only.  They do NOT alter Classification.
    weak = (dens_low2 < 0.50) | ((frac2 < 0.05) & (zspan >= 3.0))
    canopy_dom = (zspan >= 3.0) & (frac3 < 0.15) & (n_upper3 > n_low3)
    very_weak = (dens_low2 < 0.20) & (zspan >= 3.0)

    def med(a):
        return float(np.median(a)) if a.size else float("nan")
    def pct(a, q):
        return float(np.quantile(a, q)) if a.size else float("nan")

    return {
        "version": int(VERTICAL_SUPPORT_VERSION),
        "xy_cell_m": cell,
        "z_bin_m": dz,
        "occupied_xy_cells": int(cells.size),
        "xy_grid_cells": int(nx * ny),
        "xy_occupancy_ratio": float(cells.size / max(1, nx * ny)),
        "median_total_density_pts_m2": med(dens_total),
        "median_lower2_density_pts_m2": med(dens_low2),
        "lower2_density_p10": pct(dens_low2, 0.10),
        "lower2_density_p50": pct(dens_low2, 0.50),
        "lower2_density_p90": pct(dens_low2, 0.90),
        "median_lower2_fraction": med(frac2),
        "median_vertical_span_m": med(zspan),
        "weak_lower_support_fraction": float(np.mean(weak)) if weak.size else 0.0,
        "very_weak_lower_support_fraction": float(np.mean(very_weak)) if very_weak.size else 0.0,
        "canopy_dominated_fraction": float(np.mean(canopy_dom)) if canopy_dom.size else 0.0,
        "support_file": str(out_path),
        "classification_effect": "none_v1_diagnostics_only",
    }


def load_vertical_support(path: str | Path):
    """Load the immutable preprocessing support product for later pipeline use."""
    return np.load(Path(path), allow_pickle=False)


__all__ = ["build_vertical_support", "load_vertical_support", "VERTICAL_SUPPORT_VERSION"]

