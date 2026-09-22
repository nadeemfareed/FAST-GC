import numpy as np

from fastgc.tls_vote import (
    TlsInvertDsmVoteConfig,
    build_tls_surface_invert_dsm_vote,
    classify_tls_by_surface,
)


def _grid_cloud(nx=28, ny=28, spacing=0.22):
    xx, yy = np.meshgrid(np.arange(nx) * spacing, np.arange(ny) * spacing)
    return xx.ravel(), yy.ravel()


def test_planar_ground_retained_with_vertical_vegetation():
    x, y = _grid_cloud()
    z_ground = 0.05 * x + 0.025 * y
    # Dense ground plus a vertical vegetation column in the middle.
    cx, cy = float(np.median(x)), float(np.median(y))
    xv = np.full(30, cx)
    yv = np.full(30, cy)
    zv = (0.05 * cx + 0.025 * cy) + np.linspace(0.10, 2.2, 30)
    xa = np.r_[x, xv]
    ya = np.r_[y, yv]
    za = np.r_[z_ground, zv]
    cfg = TlsInvertDsmVoteConfig(cell=0.35)
    surf, x0, y0 = build_tls_surface_invert_dsm_vote(xa, ya, za, cfg)
    g = classify_tls_by_surface(xa, ya, za, surf, x0, y0, cfg)
    # Preserve nearly all ground, reject nearly all elevated vertical structure.
    assert g[: x.size].mean() > 0.93
    assert g[x.size :].mean() < 0.20


def test_curved_surface_survives_qc():
    x, y = _grid_cloud(nx=34, ny=34)
    xc, yc = x.mean(), y.mean()
    z = 0.035 * x - 0.02 * y + 0.010 * (x-xc)**2 - 0.006 * (y-yc)**2
    cfg = TlsInvertDsmVoteConfig(cell=0.35)
    surf, x0, y0 = build_tls_surface_invert_dsm_vote(x, y, z, cfg)
    g = classify_tls_by_surface(x, y, z, surf, x0, y0, cfg)
    assert g.mean() > 0.90


def test_isolated_positive_surface_blob_is_not_ground():
    x, y = _grid_cloud(nx=34, ny=34)
    z = 0.02 * x + 0.01 * y
    # Add a compact false-ground patch 0.75 m above terrain with enough points to
    # look locally dense if no protrusion guard is present.
    mask = ((x - x.mean())**2 + (y - y.mean())**2) < 0.32**2
    xb = np.repeat(x[mask], 4)
    yb = np.repeat(y[mask], 4)
    zb = np.repeat(z[mask] + 0.75, 4)
    xa = np.r_[x, xb]
    ya = np.r_[y, yb]
    za = np.r_[z, zb]
    cfg = TlsInvertDsmVoteConfig(cell=0.35)
    surf, x0, y0 = build_tls_surface_invert_dsm_vote(xa, ya, za, cfg)
    g = classify_tls_by_surface(xa, ya, za, surf, x0, y0, cfg)
    assert g[: x.size].mean() > 0.90
    assert g[x.size :].mean() < 0.10
