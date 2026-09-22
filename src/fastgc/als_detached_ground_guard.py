from __future__ import annotations

"""
FAST-GC ALS/ULS point-first detached-ground guard.

Purpose
-------
Remove small groups of final class-2 points that are physically suspended
above the surrounding terrain sheet.

This stage deliberately does NOT depend on a slope-anomaly raster.

Workflow
--------
1. Build a 1 m lower-ground support surface.
2. Use neighboring lower-ground cells only as a cheap candidate screen.
3. Fit a robust terrain plane in an annulus OUTSIDE each candidate cell.
4. Measure every ground point in the candidate cell by terrain-normal residual.
5. Require weak external same-sheet support.
6. Demote only strongly positive detached points.

The guard is DEMOTION ONLY and never uses reference labels.
"""

from dataclasses import dataclass
from typing import Any
import math

import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class DetachedGroundConfig:

    # Candidate grid. Deliberately coarser than the 0.5 m DTM.
    cell_m: float = 1.0

    # Cheap candidate screening.
    screen_window_cells: int = 5
    screen_vertical_m: float = 0.45

    # External terrain annulus.
    ring_inner_m: float = 1.5
    ring_outer_m: float = 5.0
    min_ring_points: int = 12
    min_ring_sectors: int = 4
    max_ring_points: int = 96

    # Hard terrain-normal separation.
    detached_normal_m: float = 0.55

    # Protect genuine terrain connected to the same sheet.
    support_radius_m: float = 1.25
    same_surface_band_m: float = 0.16
    min_external_support: int = 3

    # Only small suspicious footprints are handled here.
    max_candidate_cells: int = 2500

    # Catastrophic-loss protection.
    max_demote_fraction: float = 0.015


def _cfg(sensor_mode: str, cfg: dict | None) -> DetachedGroundConfig:
    cfg = cfg or {}
    d = dict(DetachedGroundConfig().__dict__)

    if str(sensor_mode).upper().strip() == "ULS":
        d.update(
            cell_m=0.75,
            ring_inner_m=1.25,
            ring_outer_m=4.0,
            detached_normal_m=0.45,
            support_radius_m=1.0,
            same_surface_band_m=0.13,
        )

    for k, v in list(d.items()):
        key = f"detached_ground_{k}"
        if key in cfg:
            if isinstance(v, int):
                d[k] = int(cfg[key])
            else:
                d[k] = float(cfg[key])

    return DetachedGroundConfig(**d)


def _sector_count(dx, dy):
    if len(dx) == 0:
        return 0

    ang = (
        np.arctan2(dy, dx) + 2.0 * np.pi
    ) % (2.0 * np.pi)

    return int(
        np.unique(
            np.floor(ang / (np.pi / 4.0)).astype(np.int8)
        ).size
    )


def _robust_plane(x, y, z):

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)

    finite = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
    )

    if int(finite.sum()) < 6:
        return None

    xc = float(np.median(x[finite]))
    yc = float(np.median(y[finite]))

    xx = x - xc
    yy = y - yc

    keep = finite.copy()
    coef = None

    for _ in range(5):

        if int(keep.sum()) < 6:
            return None

        A = np.column_stack(
            (
                xx[keep],
                yy[keep],
                np.ones(int(keep.sum())),
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

        r = z - pred

        med = float(np.median(r[keep]))

        mad = max(
            1.4826
            * float(
                np.median(
                    np.abs(r[keep] - med)
                )
            ),
            0.02,
        )

        # High points are intentionally less trusted.
        new_keep = (
            finite
            & (r >= med - 4.0 * mad)
            & (r <= med + 2.0 * mad)
        )

        if (
            int(new_keep.sum()) < 6
            or np.array_equal(new_keep, keep)
        ):
            break

        keep = new_keep

    if coef is None:
        return None

    return (
        float(coef[0]),
        float(coef[1]),
        float(coef[2]),
        xc,
        yc,
    )


def _predict_plane(model, x, y):

    a, b, c, xc, yc = model

    pred = (
        a * (x - xc)
        + b * (y - yc)
        + c
    )

    normal_scale = math.sqrt(
        1.0 + a * a + b * b
    )

    return pred, normal_scale


def apply_detached_ground_guard(
    *,
    x,
    y,
    z,
    ground_mask,
    sensor_mode,
    cfg: dict | None = None,
    return_report: bool = True,
):

    sm = str(sensor_mode).upper().strip()

    g = np.asarray(
        ground_mask,
        dtype=bool,
    ).copy()

    report: dict[str, Any] = {
        "enabled": sm in {"ALS", "ULS"},
        "ground_before": int(g.sum()),
        "screen_cells": 0,
        "modeled_cells": 0,
        "candidate_points": 0,
        "demoted_points": 0,
        "ground_after": int(g.sum()),
    }

    if sm not in {"ALS", "ULS"}:
        return (g, report) if return_report else g

    cfg = cfg or {}

    if not bool(
        cfg.get(
            "detached_ground_guard_enabled",
            True,
        )
    ):
        report["enabled"] = False
        return (g, report) if return_report else g

    if int(g.sum()) < 12:
        return (g, report) if return_report else g

    c = _cfg(sm, cfg)

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)

    if not (
        len(x)
        == len(y)
        == len(z)
        == len(g)
    ):
        raise ValueError(
            "x, y, z and ground_mask must have equal length."
        )

    # --------------------------------------------------------
    # GRID
    # --------------------------------------------------------

    cell = float(c.cell_m)

    x0 = float(np.min(x))
    y0 = float(np.min(y))

    ix = np.floor(
        (x - x0) / cell
    ).astype(np.int32)

    iy = np.floor(
        (y - y0) / cell
    ).astype(np.int32)

    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1

    key = (
        iy.astype(np.int64) * nx
        + ix.astype(np.int64)
    )

    # --------------------------------------------------------
    # LOWER-GROUND SURFACE
    # --------------------------------------------------------

    ground_ids = np.flatnonzero(g)

    gkey = key[ground_ids]

    order = np.argsort(
        gkey,
        kind="mergesort",
    )

    ids_sorted = ground_ids[order]
    keys_sorted = gkey[order]

    starts = np.r_[
        0,
        1 + np.flatnonzero(
            keys_sorted[1:]
            != keys_sorted[:-1]
        ),
    ]

    ends = np.r_[
        starts[1:],
        len(keys_sorted),
    ]

    occupied = keys_sorted[starts]

    low = np.full(
        nx * ny,
        np.nan,
        dtype=float,
    )

    high = np.full(
        nx * ny,
        np.nan,
        dtype=float,
    )

    cell_points = {}

    for k, a, b in zip(
        occupied,
        starts,
        ends,
    ):

        pts = ids_sorted[a:b]

        zz = z[pts]

        if zz.size == 0:
            continue

        kk = int(k)

        # Lower representative is resistant to an elevated
        # minority in a mixed terrain/canopy cell.
        low[kk] = float(
            np.min(zz)
            if zz.size <= 2
            else np.quantile(zz, 0.12)
        )

        high[kk] = float(
            np.max(zz)
        )

        cell_points[kk] = pts

    low = low.reshape(ny, nx)
    high = high.reshape(ny, nx)

    finite = np.isfinite(low)

    if not np.any(finite):
        return (g, report) if return_report else g

    # --------------------------------------------------------
    # CHEAP POINT-FIRST SCREEN
    #
    # This does NOT classify anything.
    # It merely identifies cells containing ground points
    # substantially above nearby lower-ground support.
    # --------------------------------------------------------

    fill_value = float(
        np.nanmedian(low[finite])
    )

    dense = np.where(
        finite,
        low,
        fill_value,
    )

    regional = median_filter(
        dense,
        size=max(
            3,
            int(c.screen_window_cells),
        ),
        mode="nearest",
    )

    # Highest class-2 point in each cell versus surrounding
    # lower-ground tendency.
    excess = high - regional

    screen = (
        np.isfinite(high)
        & np.isfinite(regional)
        & (
            excess
            >= float(c.screen_vertical_m)
        )
    )

    sy, sx = np.nonzero(screen)

    # Extreme fallback:
    # If many cells trigger on a very steep/complex tile,
    # retain the strongest excesses for expensive modeling.
    if sy.size > int(c.max_candidate_cells):

        vals = excess[sy, sx]

        keep_n = int(
            c.max_candidate_cells
        )

        take = np.argpartition(
            vals,
            -keep_n,
        )[-keep_n:]

        sy = sy[take]
        sx = sx[take]

    report["screen_cells"] = int(
        len(sy)
    )

    if len(sy) == 0:
        return (g, report) if return_report else g

    # KD tree of CURRENT GROUND only.
    ground_tree = cKDTree(
        np.column_stack(
            (
                x[ground_ids],
                y[ground_ids],
            )
        )
    )

    demote = []
    modeled = 0
    candidate_points = 0

    # --------------------------------------------------------
    # EXTERNAL ANNULAR TERRAIN CONFIRMATION
    # --------------------------------------------------------

    for gy, gx in zip(sy, sx):

        k = int(
            gy * nx + gx
        )

        pts = cell_points.get(k)

        if pts is None or len(pts) == 0:
            continue

        cx = x0 + (
            float(gx) + 0.5
        ) * cell

        cy = y0 + (
            float(gy) + 0.5
        ) * cell

        ring_local = ground_tree.query_ball_point(
            [cx, cy],
            float(c.ring_outer_m),
        )

        if not ring_local:
            continue

        ring_local = np.asarray(
            ring_local,
            dtype=np.int64,
        )

        ring_ids = ground_ids[
            ring_local
        ]

        rr = np.hypot(
            x[ring_ids] - cx,
            y[ring_ids] - cy,
        )

        use = (
            rr >= float(c.ring_inner_m)
        ) & (
            rr <= float(c.ring_outer_m)
        )

        ring_ids = ring_ids[use]

        if (
            ring_ids.size
            < int(c.min_ring_points)
        ):
            continue

        # Require spatially distributed terrain support.
        if _sector_count(
            x[ring_ids] - cx,
            y[ring_ids] - cy,
        ) < int(c.min_ring_sectors):
            continue

        if (
            ring_ids.size
            > int(c.max_ring_points)
        ):

            d = np.hypot(
                x[ring_ids] - cx,
                y[ring_ids] - cy,
            )

            ring_ids = ring_ids[
                np.argsort(d)[
                    : int(c.max_ring_points)
                ]
            ]

        model = _robust_plane(
            x[ring_ids],
            y[ring_ids],
            z[ring_ids],
        )

        if model is None:
            continue

        modeled += 1

        # ----------------------------------------------------
        # TEST EVERY CURRENT-GROUND POINT IN THIS CELL.
        #
        # This is the critical difference from V4.1.
        # The low point in a mixed cell cannot hide a high
        # false-ground point.
        # ----------------------------------------------------

        pts = pts[g[pts]]

        if pts.size == 0:
            continue

        pred, normal_scale = _predict_plane(
            model,
            x[pts],
            y[pts],
        )

        rn = (
            z[pts] - pred
        ) / normal_scale

        suspicious = (
            rn
            >= float(c.detached_normal_m)
        )

        cand = pts[suspicious]

        if cand.size == 0:
            continue

        candidate_points += int(
            cand.size
        )

        # ----------------------------------------------------
        # EXTERNAL SAME-SHEET SUPPORT
        # ----------------------------------------------------

        for pi in cand:

            pred_i, ns_i = _predict_plane(
                model,
                np.asarray([x[pi]]),
                np.asarray([y[pi]]),
            )

            ri = float(
                (
                    z[pi]
                    - pred_i[0]
                )
                / ns_i
            )

            nearby_local = (
                ground_tree.query_ball_point(
                    [x[pi], y[pi]],
                    float(c.support_radius_m),
                )
            )

            if not nearby_local:
                demote.append(
                    (int(pi), ri)
                )
                continue

            nearby = ground_ids[
                np.asarray(
                    nearby_local,
                    dtype=np.int64,
                )
            ]

            nearby = nearby[
                nearby != pi
            ]

            # Candidate's own 1 m cell is NOT allowed
            # to validate the detached object.
            external = (
                key[nearby] != k
            )

            nearby = nearby[external]

            if nearby.size == 0:
                demote.append(
                    (int(pi), ri)
                )
                continue

            pp, ss = _predict_plane(
                model,
                x[nearby],
                y[nearby],
            )

            nr = (
                z[nearby] - pp
            ) / ss

            same_sheet = int(
                np.count_nonzero(
                    np.abs(
                        nr - ri
                    )
                    <= float(
                        c.same_surface_band_m
                    )
                )
            )

            if (
                same_sheet
                < int(
                    c.min_external_support
                )
            ):
                demote.append(
                    (int(pi), ri)
                )

    report["modeled_cells"] = int(
        modeled
    )

    report["candidate_points"] = int(
        candidate_points
    )

    # --------------------------------------------------------
    # DEMOTE
    # --------------------------------------------------------

    if demote:

        # A point may be proposed by more than one cell.
        best = {}

        for pi, score in demote:
            best[pi] = max(
                score,
                best.get(pi, -np.inf),
            )

        ids = np.asarray(
            list(best.keys()),
            dtype=np.int64,
        )

        scores = np.asarray(
            [best[int(i)] for i in ids],
            dtype=float,
        )

        ids = ids[g[ids]]

        max_n = max(
            1,
            int(
                math.ceil(
                    float(
                        c.max_demote_fraction
                    )
                    * max(
                        1,
                        int(g.sum()),
                    )
                )
            ),
        )

        if ids.size > max_n:

            # Rank by terrain-normal residual,
            # NOT raw elevation.
            score_map = {
                int(i): float(s)
                for i, s in zip(
                    list(best.keys()),
                    scores,
                )
            }

            ss = np.asarray(
                [
                    score_map[int(i)]
                    for i in ids
                ],
                dtype=float,
            )

            ids = ids[
                np.argsort(ss)[::-1][
                    :max_n
                ]
            ]

        g[ids] = False

        report["demoted_points"] = int(
            ids.size
        )

    report["ground_after"] = int(
        g.sum()
    )

    return (
        (g, report)
        if return_report
        else g
    )


__all__ = [
    "DetachedGroundConfig",
    "apply_detached_ground_guard",
]
