import numpy as np

from fastgc.als_terrain_blob_guard import (
    apply_terrain_blob_guard,
)


def _run(x, y, z, g, **extra):

    cfg = {
        "terrain_blob_prominence_1_m": 0.12,
        "terrain_blob_prominence_2_m": 0.18,
        "terrain_blob_prominence_3_m": 0.24,
        "terrain_blob_strong_prominence_m": 0.50,
        "terrain_blob_extreme_prominence_m": 0.80,
        "terrain_blob_normal_residual_min_m": 0.22,
        "terrain_blob_normal_residual_strong_m": 0.40,
        "terrain_blob_normal_residual_hard_m": 0.60,
        "terrain_blob_normal_residual_extreme_m": 0.90,
        "terrain_blob_min_mask_score": 2,
    }

    cfg.update(extra)

    return apply_terrain_blob_guard(
        x=x,
        y=y,
        z=z,
        ground_mask=g,
        sensor_mode="ALS",
        cfg=cfg,
        return_report=True,
    )


def test_continuous_steep_plane_is_preserved():

    xx, yy = np.meshgrid(
        np.arange(0, 20, 0.5),
        np.arange(0, 20, 0.5),
    )

    x = xx.ravel()
    y = yy.ravel()

    # Very steep but coherent plane.
    z = (
        1.35 * x
        + 0.55 * y
    )

    g = np.ones(
        x.size,
        dtype=bool,
    )

    out, report = _run(
        x,
        y,
        z,
        g,
    )

    assert np.array_equal(
        out,
        g,
    )

    assert report[
        "demoted_points"
    ] == 0


def test_curved_terrain_is_preserved():

    xx, yy = np.meshgrid(
        np.arange(-10, 10, 0.5),
        np.arange(-10, 10, 0.5),
    )

    x = xx.ravel()
    y = yy.ravel()

    z = (
        100.0
        + 0.11 * x
        - 0.08 * y
        + 0.012 * x * x
        + 0.007 * y * y
        + 0.004 * x * y
    )

    g = np.ones(
        x.size,
        dtype=bool,
    )

    out, report = _run(
        x,
        y,
        z,
        g,
    )

    assert np.array_equal(
        out,
        g,
    )


def test_void_canopy_blob_is_demoted():

    xx, yy = np.meshgrid(
        np.arange(0, 20, 0.5),
        np.arange(0, 20, 0.5),
    )

    tx = xx.ravel()
    ty = yy.ravel()

    tz = (
        0.06 * tx
        + 0.03 * ty
    )

    # Remove true terrain returns from a small canopy-occluded area.
    keep = ~(
        (tx >= 9.5)
        & (tx <= 10.5)
        & (ty >= 9.5)
        & (ty <= 10.5)
    )

    tx = tx[keep]
    ty = ty[keep]
    tz = tz[keep]

    # False ground canopy points occupying that void.
    cx = np.array(
        [
            9.8,
            10.0,
            10.2,
            10.1,
        ]
    )

    cy = np.array(
        [
            9.8,
            10.0,
            10.1,
            10.25,
        ]
    )

    cz = (
        0.06 * cx
        + 0.03 * cy
        + np.array(
            [
                1.6,
                1.9,
                2.0,
                1.75,
            ]
        )
    )

    # Correct non-ground canopy layer.
    ngx = np.array(
        [
            9.4,
            9.7,
            10.4,
            10.6,
            10.1,
            9.9,
        ]
    )

    ngy = np.array(
        [
            10.2,
            9.5,
            9.7,
            10.1,
            10.5,
            10.3,
        ]
    )

    ngz = (
        0.06 * ngx
        + 0.03 * ngy
        + np.array(
            [
                1.7,
                1.8,
                1.9,
                1.65,
                1.85,
                2.0,
            ]
        )
    )

    x = np.r_[
        tx,
        cx,
        ngx,
    ]

    y = np.r_[
        ty,
        cy,
        ngy,
    ]

    z = np.r_[
        tz,
        cz,
        ngz,
    ]

    g = np.r_[
        np.ones(
            tx.size,
            dtype=bool,
        ),
        np.ones(
            cx.size,
            dtype=bool,
        ),
        np.zeros(
            ngx.size,
            dtype=bool,
        ),
    ]

    false_start = tx.size

    out, report = _run(
        x,
        y,
        z,
        g,
    )

    removed = int(
        np.count_nonzero(
            ~out[
                false_start:
                false_start + cx.size
            ]
        )
    )

    assert removed >= 3

    assert np.all(
        out[
            :tx.size
        ]
    )


def test_mixed_cell_removes_high_point_not_real_ground():

    xx, yy = np.meshgrid(
        np.arange(0, 20, 0.5),
        np.arange(0, 20, 0.5),
    )

    tx = xx.ravel()
    ty = yy.ravel()

    tz = (
        0.04 * tx
        - 0.02 * ty
    )

    # Several true ground points and one false canopy point
    # occupy nearly the same 0.5 m raster cell.
    gx = np.array(
        [
            10.02,
            10.08,
            10.14,
        ]
    )

    gy = np.array(
        [
            10.03,
            10.10,
            10.16,
        ]
    )

    gz = (
        0.04 * gx
        - 0.02 * gy
    )

    fx = np.array(
        [
            10.18,
        ]
    )

    fy = np.array(
        [
            10.20,
        ]
    )

    fz = (
        0.04 * fx
        - 0.02 * fy
        + 1.8
    )

    ngx = np.array(
        [
            9.8,
            10.3,
            10.0,
        ]
    )

    ngy = np.array(
        [
            10.2,
            9.9,
            10.4,
        ]
    )

    ngz = (
        0.04 * ngx
        - 0.02 * ngy
        + np.array(
            [
                1.7,
                1.9,
                1.8,
            ]
        )
    )

    x = np.r_[
        tx,
        gx,
        fx,
        ngx,
    ]

    y = np.r_[
        ty,
        gy,
        fy,
        ngy,
    ]

    z = np.r_[
        tz,
        gz,
        fz,
        ngz,
    ]

    g = np.r_[
        np.ones(
            tx.size,
            dtype=bool,
        ),
        np.ones(
            gx.size,
            dtype=bool,
        ),
        np.ones(
            fx.size,
            dtype=bool,
        ),
        np.zeros(
            ngx.size,
            dtype=bool,
        ),
    ]

    ground_extra_start = tx.size
    false_index = (
        tx.size
        + gx.size
    )

    out, report = _run(
        x,
        y,
        z,
        g,
    )

    # True same-cell terrain remains.
    assert np.all(
        out[
            ground_extra_start:
            ground_extra_start + gx.size
        ]
    )

    # High MAX-Z-generating point is demoted.
    assert not bool(
        out[
            false_index
        ]
    )


def test_never_promotes():

    xx, yy = np.meshgrid(
        np.arange(0, 10, 0.5),
        np.arange(0, 10, 0.5),
    )

    x = xx.ravel()
    y = yy.ravel()

    z = (
        0.02 * x
        + 0.01 * y
    )

    g = np.ones(
        x.size,
        dtype=bool,
    )

    g[::25] = False

    original_non_ground = ~g.copy()

    out, report = _run(
        x,
        y,
        z,
        g,
    )

    assert np.all(
        ~out[
            original_non_ground
        ]
    )

def test_sparse_airborne_ground_is_removed_without_blob():

    import numpy as np
    from scipy.spatial import cKDTree

    from fastgc.als_terrain_blob_guard import (
        _cfg,
        _apply_hard_airborne_ground_guard,
    )

    # --------------------------------------------------------
    # Coherent sloping ground sheet
    # --------------------------------------------------------

    xx, yy = np.meshgrid(
        np.arange(0.0, 30.0, 1.0),
        np.arange(0.0, 20.0, 1.0),
    )

    x = xx.ravel()
    y = yy.ravel()

    z = (
        100.0
        + 0.18 * x
        - 0.09 * y
    )

    # --------------------------------------------------------
    # Sparse suspended false-ground points.
    #
    # They do NOT form a dense circular raster blob.
    # --------------------------------------------------------

    fx = np.array(
        [
            13.2,
            14.1,
            15.0,
            22.4,
        ]
    )

    fy = np.array(
        [
            9.8,
            10.4,
            9.6,
            13.1,
        ]
    )

    fz = (
        100.0
        + 0.18 * fx
        - 0.09 * fy
        + np.array(
            [
                2.2,
                2.8,
                3.1,
                4.0,
            ]
        )
    )

    x = np.r_[
        x,
        fx,
    ]

    y = np.r_[
        y,
        fy,
    ]

    z = np.r_[
        z,
        fz,
    ]

    ground = np.ones(
        x.size,
        dtype=bool,
    )

    start = (
        x.size
        - fx.size
    )

    tree = cKDTree(
        np.column_stack(
            (
                x,
                y,
            )
        )
    )

    cfg = _cfg(
        "ALS",
        {},
    )

    out, report = (
        _apply_hard_airborne_ground_guard(
            x=x,
            y=y,
            z=z,
            ground=ground,
            tree=tree,
            cfg=cfg,
        )
    )

    removed = int(
        np.count_nonzero(
            ~out[start:]
        )
    )

    assert removed >= 3
    assert np.all(
        out[:start]
    )


def test_hard_airborne_preserves_very_steep_plane():

    import numpy as np
    from scipy.spatial import cKDTree

    from fastgc.als_terrain_blob_guard import (
        _cfg,
        _apply_hard_airborne_ground_guard,
    )

    xx, yy = np.meshgrid(
        np.arange(0.0, 25.0, 1.0),
        np.arange(0.0, 20.0, 1.0),
    )

    x = xx.ravel()
    y = yy.ravel()

    # Very steep but perfectly coherent terrain.
    z = (
        100.0
        + 1.50 * x
        + 0.65 * y
    )

    ground = np.ones(
        x.size,
        dtype=bool,
    )

    tree = cKDTree(
        np.column_stack(
            (
                x,
                y,
            )
        )
    )

    cfg = _cfg(
        "ALS",
        {},
    )

    out, report = (
        _apply_hard_airborne_ground_guard(
            x=x,
            y=y,
            z=z,
            ground=ground,
            tree=tree,
            cfg=cfg,
        )
    )

    assert np.array_equal(
        out,
        ground,
    )

