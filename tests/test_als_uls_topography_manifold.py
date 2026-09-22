import numpy as np

from fastgc.invert_vote import InvertVoteConfig, build_surface_invert_vote, classify_by_surface


def _surface(x, y):
    return 0.08 * x + 0.55 * np.sin(x / 3.5) + 0.25 * np.cos(y / 2.5)


def _synthetic(seed=4):
    rng = np.random.default_rng(seed)
    gx = np.arange(0.0, 28.0, 0.45)
    gy = np.arange(0.0, 12.0, 0.45)
    X, Y = np.meshgrid(gx, gy)
    xg = X.ravel() + rng.normal(0.0, 0.04, X.size)
    yg = Y.ravel() + rng.normal(0.0, 0.04, Y.size)
    zg = _surface(xg, yg) + rng.normal(0.0, 0.015, X.size)

    n = xg.size
    xv = rng.uniform(0.0, 28.0, n)
    yv = rng.uniform(0.0, 12.0, n)
    zv = _surface(xv, yv) + rng.uniform(0.5, 5.0, n)
    return np.r_[xg, xv], np.r_[yg, yv], np.r_[zg, zv], n


def test_normal_vote_keeps_curved_ground_and_rejects_high_objects():
    x, y, z, ng = _synthetic()
    cfg = InvertVoteConfig(
        cell=0.75,
        top_m=4,
        neighbor_radius_cells=4,
        min_neighbor_cells=8,
        max_robust_z=2.8,
        mad_floor=0.05,
        fill_iters=12,
        smooth_sigma_cells=1.0,
        ground_threshold=0.18,
        slope_adapt_k=0.30,
    )
    surf, x0, y0 = build_surface_invert_vote(x, y, z, cfg)
    pred = classify_by_surface(x, y, z, surf, x0, y0, cfg)
    assert pred[:ng].mean() > 0.94
    assert pred[ng:].mean() < 0.02


def test_surface_remains_data_attached_without_global_smoothing():
    x, y, z, _ = _synthetic(seed=7)
    cfg = InvertVoteConfig(cell=0.75, neighbor_radius_cells=4, min_neighbor_cells=8)
    surf, _, _ = build_surface_invert_vote(x, y, z, cfg)
    assert np.isfinite(surf).mean() > 0.90
    # A curved terrain must retain measurable local variation; a fitted slab would
    # artificially collapse this distribution.
    gy, gx = np.gradient(np.nan_to_num(surf, nan=np.nanmedian(surf)), cfg.cell, cfg.cell)
    assert np.nanstd(gx) > 0.01
