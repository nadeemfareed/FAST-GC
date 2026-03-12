from __future__ import annotations

import numpy as np


def remove_outliers_xyz(x, y, z, *, max_zscore: float = 6.0) -> np.ndarray:
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if np.count_nonzero(keep) < 10:
        return keep
    zz = z[keep]
    med = float(np.median(zz))
    mad = float(np.median(np.abs(zz - med)))
    mad = max(mad, 1e-9)
    rz = (zz - med) / (1.4826 * mad)
    good = np.abs(rz) <= float(max_zscore)
    out = keep.copy()
    idx = np.flatnonzero(keep)
    out[idx[~good]] = False
    return out


def layer1_mask(
    x,
    y,
    z,
    cell: float | None = None,
    dz: float | None = None,
    *,
    sensor_mode: str = "ALS",
    **_ignored,
) -> np.ndarray:
    """
    Thin-sheet candidate selection.
    Accepts sensor_mode kwarg because older io_las variants call it that way.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    if cell is None or dz is None:
        try:
            from .sensors import sensor_defaults

            cfg = sensor_defaults(str(sensor_mode))
        except Exception:
            cfg = {}
        if cell is None:
            cell = float(cfg.get("cand_cell_m", cfg.get("base_cell_m", 2.0)))
        if dz is None:
            dz = float(cfg.get("cand_dz_m", 0.5))

    cell = float(cell)
    dz = float(dz)

    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if np.count_nonzero(keep) == 0:
        return keep

    x0 = float(np.min(x[keep]))
    y0 = float(np.min(y[keep]))

    ix = np.floor((x - x0) / cell).astype(np.int32)
    iy = np.floor((y - y0) / cell).astype(np.int32)

    w = int(np.max(ix[keep])) + 1
    h = int(np.max(iy[keep])) + 1

    zmin = np.full((h, w), np.nan, dtype=np.float64)
    idx = np.flatnonzero(keep)
    for i in idx:
        cx = int(ix[i])
        cy = int(iy[i])
        zi = float(z[i])
        v = zmin[cy, cx]
        if np.isnan(v) or zi < v:
            zmin[cy, cx] = zi

    ix2 = np.clip(ix, 0, w - 1)
    iy2 = np.clip(iy, 0, h - 1)
    base = zmin[iy2, ix2]

    return keep & np.isfinite(base) & (z <= (base + dz))
