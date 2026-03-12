from __future__ import annotations

import numpy as np

def as_f64(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)

def mad(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    med = np.nanmedian(a)
    return float(np.nanmedian(np.abs(a - med))) + 1e-12

def robust_z(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    med = np.nanmedian(a)
    s = 1.4826 * mad(a)
    return (a - med) / (s + 1e-12)

def robust_quantile(a: np.ndarray, q: float) -> float:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.quantile(a, q))

def otsu_threshold(x: np.ndarray, nbins: int = 256) -> float:
    """Classic Otsu threshold on a 1D array. Returns threshold value in x units.
    If Otsu is ill-posed (flat histogram), returns nan.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 1000:
        return float("nan")

    xmin, xmax = float(np.min(x)), float(np.max(x))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return float("nan")

    hist, edges = np.histogram(x, bins=nbins, range=(xmin, xmax))
    hist = hist.astype(np.float64)
    if hist.sum() <= 0:
        return float("nan")

    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + edges[1:]) * 0.5)
    mu_t = mu[-1]

    denom = omega * (1.0 - omega)
    denom[denom <= 1e-12] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom

    k = int(np.nanargmax(sigma_b2))
    thr = (edges[k] + edges[k + 1]) * 0.5
    return float(thr)
