import numpy as np

from fastgc.als_neighbor_outlier_guard import (
    apply_neighbor_outlier_guard,
)


def test_suspended_ground_points_removed():

    xx, yy = np.meshgrid(
        np.arange(
            0.0,
            30.0,
            1.0,
        ),
        np.arange(
            0.0,
            25.0,
            1.0,
        ),
    )

    tx = xx.ravel()
    ty = yy.ravel()

    tz = (
        100.0
        + 0.12 * tx
        - 0.06 * ty
    )


    fx = np.array(
        [
            13.1,
            13.8,
            14.2,
            20.5,
        ]
    )

    fy = np.array(
        [
            10.2,
            10.7,
            9.9,
            15.1,
        ]
    )


    fz = (
        100.0
        + 0.12 * fx
        - 0.06 * fy
        + np.array(
            [
                2.0,
                2.5,
                3.0,
                2.2,
            ]
        )
    )


    x = np.r_[
        tx,
        fx,
    ]

    y = np.r_[
        ty,
        fy,
    ]

    z = np.r_[
        tz,
        fz,
    ]

    ground = np.ones(
        x.size,
        dtype=bool,
    )


    start = tx.size


    out, report = (
        apply_neighbor_outlier_guard(
            x=x,
            y=y,
            z=z,
            ground_mask=ground,
            sensor_mode="ALS",
            cfg={},
            return_report=True,
        )
    )


    removed_false = int(
        np.count_nonzero(
            ~out[
                start:
            ]
        )
    )


    removed_real = int(
        np.count_nonzero(
            ~out[
                :start
            ]
        )
    )


    print(
        report
    )


    assert removed_false >= 3
    assert removed_real == 0


def test_very_steep_plane_preserved():

    xx, yy = np.meshgrid(
        np.arange(
            0.0,
            30.0,
            1.0,
        ),
        np.arange(
            0.0,
            25.0,
            1.0,
        ),
    )

    x = xx.ravel()
    y = yy.ravel()


    z = (
        500.0
        + 1.35 * x
        + 0.55 * y
    )


    ground = np.ones(
        x.size,
        dtype=bool,
    )


    out, report = (
        apply_neighbor_outlier_guard(
            x=x,
            y=y,
            z=z,
            ground_mask=ground,
            sensor_mode="ALS",
            cfg={},
            return_report=True,
        )
    )


    print(
        report
    )


    assert np.array_equal(
        out,
        ground,
    )
