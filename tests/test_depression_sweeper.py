import numpy as np

from fastgc.depression_sweeper import sweep_ground_depressions


def _surface():
    xs = np.arange(0.0, 10.0, 0.2)
    ys = np.arange(0.0, 10.0, 0.2)
    x, y = np.meshgrid(xs, ys)
    x = x.ravel()
    y = y.ravel()
    z = 0.05 * x
    return x, y, z


def test_recovers_coherent_negative_depression_only():
    x, y, z = _surface()
    ground = np.ones(x.size, dtype=bool)
    ditch = (x >= 4.2) & (x <= 5.8) & (y >= 3.5) & (y <= 6.5)
    z = z.copy()
    z[ditch] -= 0.35
    ground[ditch] = False

    out, report = sweep_ground_depressions(
        x=x, y=y, z=z, ground_mask=ground, sensor_mode="ULS", cfg={}, return_report=True
    )

    assert report["validated_components"] == 1
    assert report["recovered_points"] > 0
    assert np.count_nonzero(out & ditch) > 0


def test_does_not_recover_elevated_object_hole():
    x, y, z = _surface()
    ground = np.ones(x.size, dtype=bool)
    obj = (x >= 4.2) & (x <= 5.8) & (y >= 3.5) & (y <= 6.5)
    z = z.copy()
    z[obj] += 1.0
    ground[obj] = False

    out, report = sweep_ground_depressions(
        x=x, y=y, z=z, ground_mask=ground, sensor_mode="ULS", cfg={}, return_report=True
    )

    assert report["validated_components"] == 0
    assert report["recovered_points"] == 0
    assert not np.any(out[obj])


def test_does_not_recover_single_low_noise_return():
    x, y, z = _surface()
    ground = np.ones(x.size, dtype=bool)
    i = int(np.argmin((x - 5.0) ** 2 + (y - 5.0) ** 2))
    z = z.copy()
    z[i] -= 0.5
    ground[i] = False

    out, report = sweep_ground_depressions(
        x=x, y=y, z=z, ground_mask=ground, sensor_mode="ULS", cfg={}, return_report=True
    )

    assert report["recovered_points"] == 0
    assert not bool(out[i])


def test_tls_is_unchanged():
    x, y, z = _surface()
    ground = np.zeros(x.size, dtype=bool)
    ground[:20] = True
    out = sweep_ground_depressions(
        x=x, y=y, z=z, ground_mask=ground, sensor_mode="TLS", cfg={}
    )
    assert np.array_equal(out, ground)


def test_recovers_mixed_label_ditch_bottom_below_ground_banks():
    # Dense ditch case: each support cell can contain a trusted bank/edge ground
    # return AND a lower non-ground ditch-bottom return.  This is the failure
    # mode seen in the ULS roadside-ditch regression.
    xs = np.arange(0.0, 10.0, 0.4)
    ys = np.arange(0.0, 10.0, 0.4)
    gx, gy = np.meshgrid(xs, ys)
    gx = gx.ravel()
    gy = gy.ravel()
    gz = 0.04 * gx

    # Original trusted surface everywhere.
    x = [gx]
    y = [gy]
    z = [gz]
    ground = [np.ones(gx.size, dtype=bool)]

    # Add lower ditch-bottom returns into the SAME XY support cells.
    ditch_xy = (gx >= 4.0) & (gx <= 6.0) & (gy >= 3.2) & (gy <= 6.8)
    for dz in (0.30, 0.32, 0.34):
        x.append(gx[ditch_xy])
        y.append(gy[ditch_xy])
        z.append(gz[ditch_xy] - dz)
        ground.append(np.zeros(np.count_nonzero(ditch_xy), dtype=bool))

    x = np.concatenate(x)
    y = np.concatenate(y)
    z = np.concatenate(z)
    ground = np.concatenate(ground)

    out, report = sweep_ground_depressions(
        x=x, y=y, z=z, ground_mask=ground, sensor_mode="ULS", cfg={}, return_report=True
    )

    added = np.arange(gx.size, x.size)
    assert report["validated_components"] >= 1
    assert report["recovered_points"] > 0
    assert np.count_nonzero(out[added]) > 0
