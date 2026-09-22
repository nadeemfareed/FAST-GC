
import numpy as np
from fastgc.depression_sweeper import sweep_ground_depressions


def _grid_cloud(zfun, *, nx=81, ny=31, step=0.20):
    xs = np.arange(nx) * step
    ys = np.arange(ny) * step
    xx, yy = np.meshgrid(xs, ys)
    zz = np.asarray(zfun(xx, yy), dtype=float)
    if zz.ndim == 0:
        zz = np.full_like(xx, float(zz), dtype=float)
    return xx.ravel(), yy.ravel(), zz.ravel()


def test_simple_v_ditch_recovers_bottom():
    def zf(x, y):
        xc = 8.0
        d = np.abs(x - xc)
        return np.where(d < 0.7, -0.55 * (1.0 - d / 0.7), 0.0)

    x, y, z = _grid_cloud(zf)
    ground = np.ones(x.size, dtype=bool)
    ditch = np.abs(x - 8.0) < 0.55
    ground[ditch] = False

    xo = np.linspace(7.7, 8.2, 60)
    yo = np.full(60, 3.0)
    zo = np.linspace(0.3, 1.4, 60)
    x = np.r_[x, xo]; y = np.r_[y, yo]; z = np.r_[z, zo]
    ground = np.r_[ground, np.zeros(60, dtype=bool)]

    out = sweep_ground_depressions(
        x=x, y=y, z=z, ground_mask=ground, sensor_mode="ULS", cfg={}
    )
    original_n = ground.size - 60
    recovered = out[:original_n] & ~ground[:original_n]
    assert recovered.sum() > 100
    assert not np.any(out[-60:])


def test_sloped_simple_ditch_recovers():
    def zf(x, y):
        base = 0.06 * x
        d = np.abs(y - 3.0)
        return base + np.where(d < 0.8, -0.45 * (1.0 - d / 0.8), 0.0)

    x, y, z = _grid_cloud(zf, nx=91, ny=31)
    ground = np.ones(x.size, dtype=bool)
    ditch = np.abs(y - 3.0) < 0.6
    ground[ditch] = False
    out = sweep_ground_depressions(
        x=x, y=y, z=z, ground_mask=ground, sensor_mode="ULS", cfg={}
    )
    assert np.count_nonzero(out & ~ground) > 100


def test_isolated_low_noise_rejected():
    x, y, z = _grid_cloud(lambda x, y: 0.03*x)
    ground = np.ones(x.size, dtype=bool)
    x = np.r_[x, [8.0, 8.05]]
    y = np.r_[y, [3.0, 3.05]]
    z = np.r_[z, [-0.7, -0.75]]
    ground = np.r_[ground, [False, False]]
    out = sweep_ground_depressions(
        x=x, y=y, z=z, ground_mask=ground, sensor_mode="ULS", cfg={}
    )
    assert not out[-1] and not out[-2]


def test_elevated_object_rejected():
    x, y, z = _grid_cloud(lambda x, y: 0.0)
    ground = np.ones(x.size, dtype=bool)
    xo = np.linspace(7.5, 8.5, 100)
    yo = np.full(100, 3.0)
    zo = np.full(100, 0.8)
    x = np.r_[x, xo]; y = np.r_[y, yo]; z = np.r_[z, zo]
    ground = np.r_[ground, np.zeros(100, dtype=bool)]
    out = sweep_ground_depressions(
        x=x, y=y, z=z, ground_mask=ground, sensor_mode="ULS", cfg={}
    )
    assert not np.any(out[-100:])


def test_tls_unchanged():
    x, y, z = _grid_cloud(lambda x, y: 0.0)
    ground = np.ones(x.size, dtype=bool)
    ground[100:150] = False
    out = sweep_ground_depressions(
        x=x, y=y, z=z, ground_mask=ground, sensor_mode="TLS", cfg={}
    )
    assert np.array_equal(out, ground)
