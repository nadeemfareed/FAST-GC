from __future__ import annotations

"""
FAST-GC ALS Neighbor Outlier Guard V4
=====================================

Independent final-stage detector inspired by the principle of
ArcGIS Locate Outliers.

The detector does NOT require:
    * a connected raster blob;
    * circularity;
    * a void;
    * a previous airborne detector;
    * or a global low-slope terrain assumption.

Candidate generation
--------------------
For each current predicted-ground point:

1. find nearby current-ground neighbors;
2. compare positive dz against a distance-dependent tolerance;
3. calculate the proportion of neighbor comparisons exceeded;
4. require exceedance across several radial sectors.

This makes isolated elevated points suspicious while protecting
a coherent steep hillside, where a point is primarily higher than
neighbors in downhill directions rather than in most directions.

Confirmation
------------
Candidates are confirmed against external 5 m and 10 m robust
lower-terrain models.

Final operation
---------------
DEMOTION ONLY.
"""

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class NeighborOutlierConfig:

    radius_m: float = 5.0
    k_neighbors: int = 16

    min_neighbors: int = 6

    # Candidate-neighbor test:
    #
    # dz > z_tolerance + slope_ratio * horizontal_distance
    z_tolerance_m: float = 0.18
    slope_ratio: float = 0.30

    exceed_ratio: float = 0.55
    min_exceed_sectors: int = 4

    local_gap_m: float = 0.30
    strong_local_gap_m: float = 0.75

    # External terrain confirmation.
    terrain_radii_m: tuple = (5.0, 10.0)

    terrain_exclusion_m: float = 1.25
    scaffold_cell_m: float = 1.0

    min_support_points: int = 10
    max_support_points: int = 320
    min_support_sectors: int = 4

    normal_gap_m: float = 0.40
    strong_normal_gap_m: float = 0.75
    extreme_normal_gap_m: float = 1.25

    max_prediction_spread_m: float = 1.0

    # Iterative peeling.
    iterations: int = 4

    # Per-pass demotion cap.
    max_demote_fraction: float = 0.005


def _cfg(sensor_mode: str, cfg: dict | None):

    cfg = cfg or {}

    values = dict(
        NeighborOutlierConfig().__dict__
    )

    sm = str(
        sensor_mode
    ).upper().strip()

    if sm == "ULS":

        values.update(
            z_tolerance_m=0.15,
            local_gap_m=0.25,
            strong_local_gap_m=0.65,
            normal_gap_m=0.34,
            strong_normal_gap_m=0.65,
            extreme_normal_gap_m=1.05,
        )

    for name, default in list(
        values.items()
    ):

        key = f"neighbor_outlier_{name}"

        if key not in cfg:
            continue

        value = cfg[key]

        if isinstance(default, bool):

            values[name] = bool(
                value
            )

        elif isinstance(default, int):

            values[name] = int(
                value
            )

        elif isinstance(default, tuple):

            values[name] = tuple(
                value
            )

        else:

            values[name] = float(
                value
            )

    return NeighborOutlierConfig(
        **values
    )


def _sector_ids(
    dx,
    dy,
):

    angle = (
        np.arctan2(
            dy,
            dx,
        )
        + 2.0 * np.pi
    ) % (
        2.0 * np.pi
    )

    return np.floor(
        angle
        / (
            np.pi / 4.0
        )
    ).astype(
        np.int16
    )


def _sector_count(
    dx,
    dy,
):

    if len(dx) == 0:

        return 0

    return int(
        np.unique(
            _sector_ids(
                np.asarray(dx),
                np.asarray(dy),
            )
        ).size
    )


def _lower_scaffold_python_reference(
    x,
    y,
    z,
    ids,
    cell_m,
):

    ids = np.asarray(
        ids,
        dtype=np.int64,
    )

    if ids.size == 0:

        return ids

    x0 = float(
        np.min(
            x[ids]
        )
    )

    y0 = float(
        np.min(
            y[ids]
        )
    )

    ix = np.floor(
        (
            x[ids] - x0
        )
        / float(cell_m)
    ).astype(
        np.int64
    )

    iy = np.floor(
        (
            y[ids] - y0
        )
        / float(cell_m)
    ).astype(
        np.int64
    )

    nx = int(
        ix.max()
    ) + 1

    key = (
        iy * nx
        + ix
    )

    order = np.argsort(
        key,
        kind="mergesort",
    )

    ids2 = ids[
        order
    ]

    key2 = key[
        order
    ]

    starts = np.r_[
        0,
        1 + np.flatnonzero(
            key2[1:]
            != key2[:-1]
        ),
    ]

    ends = np.r_[
        starts[1:],
        len(key2),
    ]

    selected = []

    for a, b in zip(
        starts,
        ends,
    ):

        group = ids2[
            a:b
        ]

        selected.append(
            int(
                group[
                    np.argmin(
                        z[group]
                    )
                ]
            )
        )

    return np.asarray(
        selected,
        dtype=np.int64,
    )


def _lower_scaffold(
    x,
    y,
    z,
    ids,
    cell_m,
):
    #
    # Execution routing only.
    # Scientific reference implementation is retained below
    # as _lower_scaffold_python_reference.
    try:
        from .backend.native import (
            native_available,
            lower_scaffold_native,
        )

        if native_available():
            return lower_scaffold_native(
                x,
                y,
                z,
                ids,
                cell_m,
            )

    except Exception:
        # Native acceleration must never prevent execution of
        # the scientific Python reference implementation.
        pass

    return _lower_scaffold_python_reference(
        x,
        y,
        z,
        ids,
        cell_m,
    )



def _robust_plane(
    x,
    y,
    z,
):

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    z = np.asarray(
        z,
        dtype=float,
    )

    if len(x) < 6:

        return None

    xc = float(
        np.median(x)
    )

    yc = float(
        np.median(y)
    )

    xx = x - xc
    yy = y - yc

    keep = np.ones(
        len(x),
        dtype=bool,
    )

    coef = None

    for _ in range(6):

        if int(
            keep.sum()
        ) < 6:

            return None

        A = np.column_stack(
            (
                xx[keep],
                yy[keep],
                np.ones(
                    int(
                        keep.sum()
                    )
                ),
            )
        )

        try:

            coef, *_ = np.linalg.lstsq(
                A,
                z[keep],
                rcond=None,
            )

        except np.linalg.LinAlgError:

            return None

        pred = (
            coef[0] * xx
            + coef[1] * yy
            + coef[2]
        )

        r = (
            z - pred
        )

        med = float(
            np.median(
                r[keep]
            )
        )

        mad = max(
            1.4826
            * float(
                np.median(
                    np.abs(
                        r[keep] - med
                    )
                )
            ),
            0.025,
        )

        new_keep = (
            (
                r
                >= med - 4.0 * mad
            )
            &
            (
                r
                <= med + 1.75 * mad
            )
        )

        if (
            int(
                new_keep.sum()
            ) < 6
            or np.array_equal(
                new_keep,
                keep,
            )
        ):

            break

        keep = new_keep

    if coef is None:

        return None

    return (
        float(
            coef[0]
        ),
        float(
            coef[1]
        ),
        float(
            coef[2]
        ),
        xc,
        yc,
    )


def _predict_plane(
    model,
    x,
    y,
):

    a, b, c, xc, yc = model

    pred = (
        a * (
            x - xc
        )
        + b * (
            y - yc
        )
        + c
    )

    normal_scale = math.sqrt(
        1.0
        + a * a
        + b * b
    )

    return (
        pred,
        normal_scale,
    )


def _screen_candidates(
    *,
    x,
    y,
    z,
    ground,
    cfg,
):

    gids = np.flatnonzero(
        ground
    )

    report = {
        "tested": 0,
        "candidates": 0,
        "max_ratio": 0.0,
        "max_gap_m": 0.0,
    }

    if (
        gids.size
        < cfg.min_neighbors + 1
    ):

        return (
            np.empty(
                0,
                dtype=np.int64,
            ),
            np.empty(
                0,
                dtype=float,
            ),
            report,
        )


    gxy = np.column_stack(
        (
            x[gids],
            y[gids],
        )
    )

    tree = cKDTree(
        gxy
    )

    k = min(
        int(
            cfg.k_neighbors + 1
        ),
        int(
            gids.size
        ),
    )

    dists, neighbors = tree.query(
        gxy,
        k=k,
        distance_upper_bound=float(
            cfg.radius_m
        ),
        workers=-1,
    )

    if k == 1:

        dists = dists[
            :,
            None
        ]

        neighbors = neighbors[
            :,
            None
        ]


    candidates = []
    scores = []


    for row, point_id in enumerate(
        gids
    ):

        valid = (
            np.isfinite(
                dists[row]
            )
            &
            (
                neighbors[row]
                < gids.size
            )
            &
            (
                dists[row]
                > 0.05
            )
        )

        if (
            int(
                valid.sum()
            )
            < cfg.min_neighbors
        ):

            continue


        nids = gids[
            neighbors[
                row
            ][
                valid
            ]
        ]

        distance = dists[
            row
        ][
            valid
        ]


        local_gap = float(
            z[point_id]
            - np.median(
                z[nids]
            )
        )


        if (
            local_gap
            < cfg.local_gap_m
        ):

            continue


        dz = (
            z[point_id]
            - z[nids]
        )


        tolerance = (
            float(
                cfg.z_tolerance_m
            )
            + float(
                cfg.slope_ratio
            )
            * distance
        )


        exceed = (
            dz
            > tolerance
        )


        ratio = float(
            np.count_nonzero(
                exceed
            )
            / max(
                1,
                exceed.size,
            )
        )


        if np.any(
            exceed
        ):

            eids = nids[
                exceed
            ]

            sector_count = _sector_count(
                x[eids]
                - x[point_id],
                y[eids]
                - y[point_id],
            )

        else:

            sector_count = 0


        normal_case = (
            ratio
            >= cfg.exceed_ratio
            and sector_count
            >= cfg.min_exceed_sectors
        )


        strong_case = (
            local_gap
            >= cfg.strong_local_gap_m
            and ratio
            >= max(
                0.40,
                cfg.exceed_ratio
                - 0.10,
            )
            and sector_count
            >= max(
                3,
                cfg.min_exceed_sectors
                - 1,
            )
        )


        if not (
            normal_case
            or strong_case
        ):

            continue


        score = float(
            local_gap
            * (
                0.50
                + ratio
            )
        )


        candidates.append(
            int(
                point_id
            )
        )

        scores.append(
            score
        )


        report[
            "max_ratio"
        ] = max(
            report[
                "max_ratio"
            ],
            ratio,
        )


        report[
            "max_gap_m"
        ] = max(
            report[
                "max_gap_m"
            ],
            local_gap,
        )


    report[
        "tested"
    ] = int(
        gids.size
    )

    report[
        "candidates"
    ] = int(
        len(
            candidates
        )
    )


    return (
        np.asarray(
            candidates,
            dtype=np.int64,
        ),
        np.asarray(
            scores,
            dtype=float,
        ),
        report,
    )


def _confirm_candidates(
    *,
    x,
    y,
    z,
    ground,
    candidate_ids,
    cfg,
):

    gids = np.flatnonzero(
        ground
    )

    if (
        gids.size == 0
        or candidate_ids.size == 0
    ):

        return (
            [],
            {
                "modeled": 0,
                "confirmed": 0,
                "max_rn": 0.0,
            },
        )


    gtree = cKDTree(
        np.column_stack(
            (
                x[gids],
                y[gids],
            )
        )
    )


    confirmed = []
    modeled = 0
    max_rn = 0.0


    for point_id in candidate_ids:

        predictions = []
        residuals = []


        for radius in cfg.terrain_radii_m:

            local_index = np.asarray(
                gtree.query_ball_point(
                    [
                        x[point_id],
                        y[point_id],
                    ],
                    float(
                        radius
                    ),
                ),
                dtype=np.int64,
            )


            if local_index.size == 0:

                continue


            support = gids[
                local_index
            ]


            support = support[
                support
                != point_id
            ]


            if (
                support.size
                < cfg.min_support_points
            ):

                continue


            dist = np.hypot(
                x[support]
                - x[point_id],
                y[support]
                - y[point_id],
            )


            support = support[
                dist
                >= cfg.terrain_exclusion_m
            ]


            if (
                support.size
                < cfg.min_support_points
            ):

                continue


            scaffold = _lower_scaffold(
                x,
                y,
                z,
                support,
                cfg.scaffold_cell_m,
            )


            if (
                scaffold.size
                < cfg.min_support_points
            ):

                continue


            if (
                _sector_count(
                    x[scaffold]
                    - x[point_id],
                    y[scaffold]
                    - y[point_id],
                )
                < cfg.min_support_sectors
            ):

                continue


            if (
                scaffold.size
                > cfg.max_support_points
            ):

                ds = np.hypot(
                    x[scaffold]
                    - x[point_id],
                    y[scaffold]
                    - y[point_id],
                )


                scaffold = scaffold[
                    np.argsort(
                        ds
                    )[
                        :
                        cfg.max_support_points
                    ]
                ]


            model = _robust_plane(
                x[scaffold],
                y[scaffold],
                z[scaffold],
            )


            if model is None:

                continue


            pred, scale = _predict_plane(
                model,
                np.asarray(
                    [
                        x[point_id]
                    ],
                    dtype=float,
                ),
                np.asarray(
                    [
                        y[point_id]
                    ],
                    dtype=float,
                ),
            )


            terrain_z = float(
                pred[0]
            )


            rn = float(
                (
                    z[point_id]
                    - terrain_z
                )
                / scale
            )


            predictions.append(
                terrain_z
            )

            residuals.append(
                rn
            )


        if len(
            residuals
        ) < 2:

            continue


        modeled += 1


        residuals = np.asarray(
            residuals,
            dtype=float,
        )


        predictions = np.asarray(
            predictions,
            dtype=float,
        )


        median_rn = float(
            np.median(
                residuals
            )
        )


        positive_votes = int(
            np.count_nonzero(
                residuals
                >= cfg.normal_gap_m
            )
        )


        if positive_votes < 2:

            continue


        spread = float(
            np.ptp(
                predictions
            )
        )


        if (
            spread
            > cfg.max_prediction_spread_m
            and median_rn
            < cfg.extreme_normal_gap_m
        ):

            continue


        if (
            median_rn
            >= cfg.normal_gap_m
        ):

            confirmed.append(
                (
                    int(
                        point_id
                    ),
                    median_rn,
                )
            )

            max_rn = max(
                max_rn,
                median_rn,
            )


    return (
        confirmed,
        {
            "modeled":
                int(
                    modeled
                ),

            "confirmed":
                int(
                    len(
                        confirmed
                    )
                ),

            "max_rn":
                float(
                    max_rn
                ),
        },
    )


def apply_neighbor_outlier_guard(
    *,
    x,
    y,
    z,
    ground_mask,
    sensor_mode,
    cfg: dict | None = None,
    return_report: bool = True,
):

    ground = np.asarray(
        ground_mask,
        dtype=bool,
    ).copy()


    c = _cfg(
        sensor_mode,
        cfg,
    )


    initial = int(
        ground.sum()
    )


    report: dict[str, Any] = {
        "ground_before":
            initial,

        "ground_after":
            initial,

        "iterations_run":
            0,

        "tested":
            0,

        "candidates":
            0,

        "modeled":
            0,

        "confirmed":
            0,

        "demoted_points":
            0,

        "max_ratio":
            0.0,

        "max_gap_m":
            0.0,

        "max_rn":
            0.0,

        "passes":
            [],
    }


    for iteration in range(
        max(
            1,
            int(
                c.iterations
            ),
        )
    ):


        before = int(
            ground.sum()
        )


        (
            candidate_ids,
            candidate_scores,
            screen,
        ) = _screen_candidates(
            x=x,
            y=y,
            z=z,
            ground=ground,
            cfg=c,
        )


        (
            confirmed,
            confirmation,
        ) = _confirm_candidates(
            x=x,
            y=y,
            z=z,
            ground=ground,
            candidate_ids=candidate_ids,
            cfg=c,
        )


        if confirmed:

            confirmed.sort(
                key=lambda item:
                    item[1],
                reverse=True,
            )


            maximum = max(
                1,
                int(
                    math.ceil(
                        c.max_demote_fraction
                        * max(
                            1,
                            before,
                        )
                    )
                ),
            )


            confirmed = confirmed[
                :
                maximum
            ]


            demote_ids = np.asarray(
                [
                    item[0]
                    for item in confirmed
                ],
                dtype=np.int64,
            )


            ground[
                demote_ids
            ] = False


            demoted = int(
                demote_ids.size
            )

        else:

            demoted = 0


        pass_report = {
            "iteration":
                int(
                    iteration + 1
                ),

            "before":
                before,

            "tested":
                int(
                    screen[
                        "tested"
                    ]
                ),

            "candidates":
                int(
                    screen[
                        "candidates"
                    ]
                ),

            "modeled":
                int(
                    confirmation[
                        "modeled"
                    ]
                ),

            "confirmed":
                int(
                    confirmation[
                        "confirmed"
                    ]
                ),

            "demoted":
                int(
                    demoted
                ),

            "max_ratio":
                float(
                    screen[
                        "max_ratio"
                    ]
                ),

            "max_gap_m":
                float(
                    screen[
                        "max_gap_m"
                    ]
                ),

            "max_rn":
                float(
                    confirmation[
                        "max_rn"
                    ]
                ),

            "after":
                int(
                    ground.sum()
                ),
        }


        report[
            "passes"
        ].append(
            pass_report
        )


        report[
            "tested"
        ] += pass_report[
            "tested"
        ]


        report[
            "candidates"
        ] += pass_report[
            "candidates"
        ]


        report[
            "modeled"
        ] += pass_report[
            "modeled"
        ]


        report[
            "confirmed"
        ] += pass_report[
            "confirmed"
        ]


        report[
            "max_ratio"
        ] = max(
            report[
                "max_ratio"
            ],
            pass_report[
                "max_ratio"
            ],
        )


        report[
            "max_gap_m"
        ] = max(
            report[
                "max_gap_m"
            ],
            pass_report[
                "max_gap_m"
            ],
        )


        report[
            "max_rn"
        ] = max(
            report[
                "max_rn"
            ],
            pass_report[
                "max_rn"
            ],
        )


        report[
            "iterations_run"
        ] = int(
            iteration + 1
        )


        if demoted == 0:

            break


    report[
        "ground_after"
    ] = int(
        ground.sum()
    )


    report[
        "demoted_points"
    ] = int(
        initial
        - ground.sum()
    )


    return (
        (
            ground,
            report,
        )
        if return_report
        else ground
    )


__all__ = [
    "NeighborOutlierConfig",
    "apply_neighbor_outlier_guard",
]
