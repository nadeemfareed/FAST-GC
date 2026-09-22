import numpy as np

from fastgc.als_surface_impurity_guard import apply_surface_impurity_guard


def _cloud(surface_fn, nx=35, ny=35, spacing=0.5):
    xx, yy = np.meshgrid(np.arange(nx)*spacing, np.arange(ny)*spacing)
    x = xx.ravel()
    y = yy.ravel()
    z = surface_fn(x, y)
    g = np.ones(x.size, dtype=bool)
    return x, y, z, g


def test_continuous_steep_slope_is_preserved():
    x, y, z, g = _cloud(lambda x, y: 0.9*x + 0.2*y)
    out, rep = apply_surface_impurity_guard(
        x=x, y=y, z=z, ground_mask=g, sensor_mode="ALS", cfg={},
        return_report=True,
    )
    assert np.array_equal(out, g)
    assert rep["demoted_points"] == 0


def test_small_detached_cluster_is_demoted():
    x, y, z, g = _cloud(lambda x, y: 0.12*x + 0.04*y)
    # Add a compact elevated class-2 cluster and non-ground corroboration.
    cx, cy = 8.0, 8.0
    px = np.array([cx, cx+.12, cx-.12, cx+.08, cx-.08])
    py = np.array([cy, cy+.08, cy-.08, cy-.12, cy+.12])
    base = 0.12*px + 0.04*py
    pz = base + 1.0
    x = np.r_[x, px]
    y = np.r_[y, py]
    z = np.r_[z, pz]
    g = np.r_[g, np.ones(px.size, bool)]

    # Nearby non-ground returns at the same elevated layer.
    ngx = px + 0.18
    ngy = py - 0.12
    ngz = 0.12*ngx + 0.04*ngy + 1.0
    x = np.r_[x, ngx]
    y = np.r_[y, ngy]
    z = np.r_[z, ngz]
    g = np.r_[g, np.zeros(ngx.size, bool)]

    out, rep = apply_surface_impurity_guard(
        x=x, y=y, z=z, ground_mask=g, sensor_mode="ALS",
        cfg={"surface_impurity_slope_delta_min_deg": 8.0,
             "surface_impurity_slope_delta_strong_deg": 12.0},
        return_report=True,
    )
    assert np.count_nonzero(g & (~out)) >= 1
    assert rep["demoted_points"] >= 1


def test_guard_never_promotes():
    x, y, z, g = _cloud(lambda x, y: 0.3*x)
    g[::11] = False
    out, _ = apply_surface_impurity_guard(
        x=x, y=y, z=z, ground_mask=g, sensor_mode="ALS", cfg={},
        return_report=True,
    )
    assert not np.any(out & (~g))
