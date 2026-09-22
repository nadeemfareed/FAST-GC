from __future__ import annotations

"""
FAST-GC Terrain Blob / Canopy Impostor Guard V1
===============================================

Purpose
-------
Final ALS/ULS cleanup of small false-ground canopy islands that produce
localized tomb / pyramid / dome / pimple artifacts in a high-resolution DEM.

The detector is deliberately RASTER-FIRST and confirmation is POINT-FIRST.

Candidate generation
--------------------
1. Rasterize current predicted ground at 0.5 m using MAXIMUM Z.
2. Compute positive local prominence at several scales.
3. Compute terrain-relative slope-excess.
4. Compute robust local Z outlier strength.
5. Compute surrounding ground-support void fraction.
6. Fuse independent masks.
7. Group neighboring candidate cells into compact components.

Confirmation
------------
For every component:

1. map the raster component back to exact predicted-ground LAS points;
2. exclude the component and a halo from terrain support;
3. use an approximately 5 m surrounding region;
4. fit a robust plane;
5. use a robust quadratic model when curvature materially improves the fit;
6. compute terrain-normal residual for every candidate point;
7. verify that surrounding directions return to the terrain sheet;
8. compute ArcGIS-Locate-Outliers-like neighbor disagreement relative to
   the fitted terrain;
9. inspect existing non-ground points for an elevated layer consistent
   with the candidate;
10. demote only individually confirmed false-ground points.

Important
---------
* DEMOTION ONLY.
* No reference/truth labels are accessed.
* Raw slope alone is NEVER a rejection rule.
* Candidate points cannot support their own terrain model.
* Several false-ground points inside one blob cannot protect one another.
* Genuine steep and curved terrain are protected by the external terrain model.
"""

from dataclasses import dataclass
from typing import Any
import math

import numpy as np

from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    label,
    median_filter,
    uniform_filter,
)

from scipy.spatial import cKDTree

from .als_neighbor_outlier_guard import apply_neighbor_outlier_guard


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class TerrainBlobGuardConfig:

    # --------------------------------------------------------
    # Main raster
    # --------------------------------------------------------
    cell_m: float = 0.50

    # Radius of multiscale background windows.
    prominence_r1_m: float = 1.50
    prominence_r2_m: float = 3.00
    prominence_r3_m: float = 5.00

    # --------------------------------------------------------
    # Positive elevation prominence
    # --------------------------------------------------------
    prominence_1_m: float = 0.18
    prominence_2_m: float = 0.25
    prominence_3_m: float = 0.32

    strong_prominence_m: float = 0.70
    extreme_prominence_m: float = 1.00

    min_prominence_votes: int = 2

    # --------------------------------------------------------
    # Robust local outlier
    # --------------------------------------------------------
    robust_z_min: float = 3.25
    mad_floor_m: float = 0.04

    # --------------------------------------------------------
    # Slope EXCESS, not absolute slope
    # --------------------------------------------------------
    slope_background_radius_m: float = 3.00
    slope_excess_deg: float = 12.0

    # --------------------------------------------------------
    # Ground support / void
    # --------------------------------------------------------
    void_radius_m: float = 5.00
    void_support_fraction: float = 0.28
    strong_void_fraction: float = 0.48

    # --------------------------------------------------------
    # Candidate fusion
    # --------------------------------------------------------
    min_mask_score: int = 3
    group_dilate_cells: int = 1

    # --------------------------------------------------------
    # Component morphology
    # --------------------------------------------------------
    max_component_cells: int = 120
    max_component_area_m2: float = 30.0

    # Circularity is supportive, not mandatory.
    compact_circularity: float = 0.18
    max_aspect_for_compact: float = 5.0

    # --------------------------------------------------------
    # External terrain model
    # --------------------------------------------------------
    terrain_radius_m: float = 5.0
    terrain_exclusion_m: float = 1.0

    min_terrain_points: int = 14
    max_terrain_points: int = 240
    min_terrain_sectors: int = 5

    quadratic_trigger_rmse_m: float = 0.10
    quadratic_improvement_ratio: float = 0.86

    # --------------------------------------------------------
    # Radial terrain-return check
    # --------------------------------------------------------
    radial_sectors: int = 8
    radial_return_residual_m: float = 0.22
    min_radial_return_sectors: int = 4

    # --------------------------------------------------------
    # Exact source-point confirmation
    # --------------------------------------------------------
    normal_residual_min_m: float = 0.28
    normal_residual_strong_m: float = 0.55
    normal_residual_hard_m: float = 0.80
    normal_residual_extreme_m: float = 1.20

    # --------------------------------------------------------
    # Same-sheet protection
    # --------------------------------------------------------
    same_sheet_radius_m: float = 1.50
    same_sheet_band_m: float = 0.16
    min_same_sheet_ground: int = 3

    # --------------------------------------------------------
    # ArcGIS-style neighbor disagreement
    # --------------------------------------------------------
    neighbor_radius_m: float = 5.0

    # Difference between observed candidate-neighbor slope and
    # slope predicted by the terrain model.
    slope_deviation_ratio: float = 0.35

    min_neighbor_comparisons: int = 5
    neighbor_exceed_ratio: float = 0.50

    # --------------------------------------------------------
    # Nearby non-ground canopy consistency
    # --------------------------------------------------------
    nonground_radius_m: float = 5.0
    nonground_residual_band_m: float = 0.45
    nonground_min_consistent: int = 2

    # --------------------------------------------------------
    # HARD AIRBORNE-GROUND POINT-FIRST V2
    # --------------------------------------------------------
    # Sparse class-2 points suspended above weakly sampled
    # terrain. Candidate generation is cheap/raster based;
    # final confirmation is multi-radius 3-D terrain fitting.

    airborne_screen_cell_m: float = 1.0

    airborne_r1_m: float = 5.0
    airborne_r2_m: float = 10.0
    airborne_r3_m: float = 15.0

    airborne_screen_gap_m: float = 0.50
    airborne_screen_min_votes: int = 2
    airborne_max_screen_candidates: int = 6000

    airborne_candidate_gap_m: float = 0.55
    airborne_strong_gap_m: float = 0.85
    airborne_extreme_gap_m: float = 1.25
    airborne_absolute_gap_m: float = 1.80

    airborne_min_radius_votes: int = 2

    airborne_scaffold_cell_m: float = 1.0
    airborne_exclusion_radius_m: float = 1.5

    airborne_min_support_points: int = 10
    airborne_max_support_points: int = 320
    airborne_min_support_sectors: int = 4

    airborne_same_sheet_radius_m: float = 1.5
    airborne_same_sheet_band_m: float = 0.20
    airborne_max_same_sheet_neighbors: int = 2

    airborne_max_prediction_spread_m: float = 0.90
    airborne_max_residual_spread_m: float = 0.85

    airborne_max_demote_fraction: float = 0.015

    # --------------------------------------------------------
    # Runtime/safety
    # --------------------------------------------------------
    passes: int = 2
    max_demote_fraction_per_pass: float = 0.015


def _cfg(sensor_mode: str, cfg: dict | None) -> TerrainBlobGuardConfig:

    cfg = cfg or {}

    values = dict(
        TerrainBlobGuardConfig().__dict__
    )

    sm = str(sensor_mode).upper().strip()

    if sm == "ULS":

        values.update(
            prominence_1_m=0.15,
            prominence_2_m=0.21,
            prominence_3_m=0.28,
            strong_prominence_m=0.60,
            normal_residual_min_m=0.24,
            normal_residual_strong_m=0.48,
            normal_residual_hard_m=0.70,
            normal_residual_extreme_m=1.00,
        )

    for name, default in list(values.items()):

        key = f"terrain_blob_{name}"

        if key not in cfg:
            continue

        value = cfg[key]

        if isinstance(default, bool):
            values[name] = bool(value)

        elif isinstance(default, int):
            values[name] = int(value)

        else:
            values[name] = float(value)

    return TerrainBlobGuardConfig(
        **values
    )


# ============================================================
# RASTER UTILITIES
# ============================================================

def _odd_window(
    radius_m: float,
    cell_m: float,
) -> int:

    r = max(
        1,
        int(
            math.ceil(
                float(radius_m)
                / float(cell_m)
            )
        ),
    )

    return int(
        2 * r + 1
    )


def _grid(
    x: np.ndarray,
    y: np.ndarray,
    cell: float,
):

    x0 = float(
        np.min(x)
    )

    y0 = float(
        np.min(y)
    )

    ix = np.floor(
        (x - x0) / cell
    ).astype(np.int32)

    iy = np.floor(
        (y - y0) / cell
    ).astype(np.int32)

    nx = int(
        ix.max()
    ) + 1

    ny = int(
        iy.max()
    ) + 1

    key = (
        iy.astype(np.int64) * nx
        + ix.astype(np.int64)
    )

    return (
        x0,
        y0,
        ix,
        iy,
        key,
        nx,
        ny,
    )


def _max_ground_surface(
    z: np.ndarray,
    ground: np.ndarray,
    key: np.ndarray,
    nx: int,
    ny: int,
):

    out = np.full(
        nx * ny,
        np.nan,
        dtype=float,
    )

    ids = np.flatnonzero(
        ground
    )

    if ids.size == 0:

        return out.reshape(
            ny,
            nx,
        )

    kk = key[ids]

    order = np.argsort(
        kk,
        kind="mergesort",
    )

    ids = ids[order]
    kk = kk[order]

    starts = np.r_[
        0,
        1 + np.flatnonzero(
            kk[1:] != kk[:-1]
        ),
    ]

    ends = np.r_[
        starts[1:],
        len(kk),
    ]

    for a, b in zip(
        starts,
        ends,
    ):

        out[
            int(kk[a])
        ] = float(
            np.max(
                z[
                    ids[a:b]
                ]
            )
        )

    return out.reshape(
        ny,
        nx,
    )


def _nearest_fill(
    surface: np.ndarray,
) -> np.ndarray:

    surface = np.asarray(
        surface,
        dtype=float,
    )

    finite = np.isfinite(
        surface
    )

    if not np.any(finite):

        return surface.copy()

    if np.all(finite):

        return surface.copy()

    _, inds = distance_transform_edt(
        ~finite,
        return_indices=True,
    )

    result = surface.copy()

    missing = ~finite

    result[missing] = surface[
        tuple(
            index[missing]
            for index in inds
        )
    ]

    return result


def _slope_deg(
    surface: np.ndarray,
    cell: float,
) -> np.ndarray:

    finite = np.isfinite(
        surface
    )

    result = np.full_like(
        surface,
        np.nan,
        dtype=float,
    )

    if not np.any(finite):

        return result

    filled = _nearest_fill(
        surface
    )

    if min(
        filled.shape
    ) < 3:

        return result

    gy, gx = np.gradient(
        filled,
        float(cell),
        float(cell),
    )

    result = np.degrees(
        np.arctan(
            np.hypot(
                gx,
                gy,
            )
        )
    )

    result[
        ~finite
    ] = np.nan

    return result


def _void_fraction(
    occupancy: np.ndarray,
    window: int,
) -> np.ndarray:

    occ = occupancy.astype(
        np.float32
    )

    ones = np.ones_like(
        occ,
        dtype=np.float32,
    )

    occ_mean = uniform_filter(
        occ,
        size=window,
        mode="constant",
        cval=0.0,
    )

    valid_mean = uniform_filter(
        ones,
        size=window,
        mode="constant",
        cval=0.0,
    )

    ground_fraction = (
        occ_mean
        / np.maximum(
            valid_mean,
            1.0e-6,
        )
    )

    return np.clip(
        1.0 - ground_fraction,
        0.0,
        1.0,
    )


# ============================================================
# GEOMETRY / MORPHOLOGY
# ============================================================

def _sector_ids(
    dx: np.ndarray,
    dy: np.ndarray,
    nsectors: int = 8,
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
            2.0 * np.pi
            / float(nsectors)
        )
    ).astype(np.int16)


def _sector_count(
    dx: np.ndarray,
    dy: np.ndarray,
    nsectors: int = 8,
) -> int:

    if len(dx) == 0:
        return 0

    return int(
        np.unique(
            _sector_ids(
                np.asarray(dx),
                np.asarray(dy),
                nsectors,
            )
        ).size
    )


def _component_shape(
    component: np.ndarray,
    cell: float,
):

    yy, xx = np.nonzero(
        component
    )

    n = int(
        xx.size
    )

    if n == 0:

        return {
            "cells": 0,
            "area_m2": 0.0,
            "perimeter_m": 0.0,
            "circularity": 0.0,
            "aspect": np.inf,
        }

    area = (
        n
        * cell
        * cell
    )

    eroded = binary_erosion(
        component,
        structure=np.ones(
            (3, 3),
            dtype=bool,
        ),
        border_value=0,
    )

    boundary = (
        component
        & ~eroded
    )

    # Approximate grid perimeter.
    perimeter = max(
        cell,
        float(
            np.count_nonzero(
                boundary
            )
        )
        * cell,
    )

    circularity = float(
        4.0
        * np.pi
        * area
        / max(
            perimeter * perimeter,
            1.0e-12,
        )
    )

    pts = np.column_stack(
        (
            xx.astype(float),
            yy.astype(float),
        )
    )

    if len(pts) < 3:

        aspect = 1.0

    else:

        pts = (
            pts
            - np.mean(
                pts,
                axis=0,
            )
        )

        cov = np.cov(
            pts.T
        )

        vals = np.linalg.eigvalsh(
            cov
        )

        vals = np.maximum(
            vals,
            1.0e-9,
        )

        aspect = float(
            math.sqrt(
                vals[-1]
                / vals[0]
            )
        )

    return {
        "cells": n,
        "area_m2": float(area),
        "perimeter_m": float(perimeter),
        "circularity": float(circularity),
        "aspect": float(aspect),
    }


# ============================================================
# ROBUST TERRAIN MODELS
# ============================================================

def _robust_plane_python_reference(
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

    finite = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
    )

    if int(
        finite.sum()
    ) < 6:

        return None

    xc = float(
        np.median(
            x[finite]
        )
    )

    yc = float(
        np.median(
            y[finite]
        )
    )

    xx = x - xc
    yy = y - yc

    keep = finite.copy()

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

        residual = (
            z - pred
        )

        med = float(
            np.median(
                residual[keep]
            )
        )

        mad = max(
            1.4826
            * float(
                np.median(
                    np.abs(
                        residual[keep]
                        - med
                    )
                )
            ),
            0.025,
        )

        # Asymmetric robust clipping:
        # high returns are deliberately less trusted.
        new_keep = (
            finite
            & (
                residual
                >= med - 4.0 * mad
            )
            & (
                residual
                <= med + 1.8 * mad
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

    pred_keep = (
        coef[0] * xx[keep]
        + coef[1] * yy[keep]
        + coef[2]
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (
                    z[keep]
                    - pred_keep
                ) ** 2
            )
        )
    )

    return {
        "kind": "plane",
        "coef": np.asarray(
            coef,
            dtype=float,
        ),
        "xc": xc,
        "yc": yc,
        "rmse": rmse,
    }

def _robust_plane_native_hybrid(
    x,
    y,
    z,
):
    """
    Hybrid terrain-plane evaluation.

    Scientific algorithm is identical to
    _robust_plane_python_reference.

    np.linalg.lstsq remains NumPy.
    Only residual/MAD/clipping-mask construction is native.
    """
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

    finite = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
    )

    if int(
        finite.sum()
    ) < 6:
        return None

    xc = float(
        np.median(
            x[finite]
        )
    )

    yc = float(
        np.median(
            y[finite]
        )
    )

    xx = x - xc
    yy = y - yc

    keep = finite.copy()

    coef = None

    from .backend.native import (
        uls_terrain_plane_iteration_native,
    )

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

        new_keep = (
            finite
            & uls_terrain_plane_iteration_native(
                xx,
                yy,
                z,
                finite,
                keep,
                coef,
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

    pred_keep = (
        coef[0] * xx[keep]
        + coef[1] * yy[keep]
        + coef[2]
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (
                    z[keep]
                    - pred_keep
                ) ** 2
            )
        )
    )

    return {
        "kind": "plane",
        "coef": np.asarray(
            coef,
            dtype=float,
        ),
        "xc": xc,
        "yc": yc,
        "rmse": rmse,
    }


def _robust_plane(
    x,
    y,
    z,
):
    """
    Native ULS terrain-plane evaluation.

    Dispatch terrain-plane evaluation to the configured backend.

    Default remains the scientific Python reference.
    """
    import os

    backend = os.environ.get(
        "FASTGC_ULS_TERRAIN_PLANE_BACKEND",
        "reference",
    ).strip().lower()

    if backend == "native":
        try:
            return _robust_plane_native_hybrid(
                x,
                y,
                z,
            )
        except Exception:
            # Native acceleration must never prevent
            # execution of the scientific reference.
            pass

    return _robust_plane_python_reference(
        x,
        y,
        z,
    )




def _robust_quadratic(
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

    finite = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
    )

    if int(
        finite.sum()
    ) < 12:

        return None

    xc = float(
        np.median(
            x[finite]
        )
    )

    yc = float(
        np.median(
            y[finite]
        )
    )

    xx = x - xc
    yy = y - yc

    keep = finite.copy()

    coef = None

    for _ in range(6):

        if int(
            keep.sum()
        ) < 12:

            return None

        A = np.column_stack(
            (
                xx[keep] ** 2,
                yy[keep] ** 2,
                xx[keep] * yy[keep],
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
            coef[0] * xx ** 2
            + coef[1] * yy ** 2
            + coef[2] * xx * yy
            + coef[3] * xx
            + coef[4] * yy
            + coef[5]
        )

        residual = (
            z - pred
        )

        med = float(
            np.median(
                residual[keep]
            )
        )

        mad = max(
            1.4826
            * float(
                np.median(
                    np.abs(
                        residual[keep]
                        - med
                    )
                )
            ),
            0.025,
        )

        new_keep = (
            finite
            & (
                residual
                >= med - 4.0 * mad
            )
            & (
                residual
                <= med + 1.8 * mad
            )
        )

        if (
            int(
                new_keep.sum()
            ) < 12
            or np.array_equal(
                new_keep,
                keep,
            )
        ):

            break

        keep = new_keep

    if coef is None:

        return None

    pred_keep = (
        coef[0] * xx[keep] ** 2
        + coef[1] * yy[keep] ** 2
        + coef[2] * xx[keep] * yy[keep]
        + coef[3] * xx[keep]
        + coef[4] * yy[keep]
        + coef[5]
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (
                    z[keep]
                    - pred_keep
                ) ** 2
            )
        )
    )

    return {
        "kind": "quadratic",
        "coef": np.asarray(
            coef,
            dtype=float,
        ),
        "xc": xc,
        "yc": yc,
        "rmse": rmse,
    }


def _predict_model(
    model,
    x,
    y,
):

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    xx = (
        x - model["xc"]
    )

    yy = (
        y - model["yc"]
    )

    c = model["coef"]

    if model["kind"] == "plane":

        pred = (
            c[0] * xx
            + c[1] * yy
            + c[2]
        )

        scale = np.full_like(
            pred,
            math.sqrt(
                1.0
                + c[0] ** 2
                + c[1] ** 2
            ),
            dtype=float,
        )

        return pred, scale

    pred = (
        c[0] * xx ** 2
        + c[1] * yy ** 2
        + c[2] * xx * yy
        + c[3] * xx
        + c[4] * yy
        + c[5]
    )

    dzdx = (
        2.0 * c[0] * xx
        + c[2] * yy
        + c[3]
    )

    dzdy = (
        2.0 * c[1] * yy
        + c[2] * xx
        + c[4]
    )

    scale = np.sqrt(
        1.0
        + dzdx ** 2
        + dzdy ** 2
    )

    return pred, scale


def _choose_terrain_model(
    x,
    y,
    z,
    cfg: TerrainBlobGuardConfig,
):

    plane = _robust_plane(
        x,
        y,
        z,
    )

    if plane is None:

        return None

    if (
        plane["rmse"]
        < cfg.quadratic_trigger_rmse_m
    ):

        return plane

    quad = _robust_quadratic(
        x,
        y,
        z,
    )

    if quad is None:

        return plane

    if (
        quad["rmse"]
        <= (
            cfg.quadratic_improvement_ratio
            * plane["rmse"]
        )
    ):

        return quad

    return plane


# ============================================================
# RADIAL / NEIGHBOR TESTS
# ============================================================

def _radial_return_count(
    model,
    x,
    y,
    z,
    cx,
    cy,
    cfg: TerrainBlobGuardConfig,
):

    pred, scale = _predict_model(
        model,
        x,
        y,
    )

    residual = (
        z - pred
    ) / scale

    sectors = _sector_ids(
        x - cx,
        y - cy,
        cfg.radial_sectors,
    )

    good = 0

    for sector in range(
        cfg.radial_sectors
    ):

        use = (
            sectors == sector
        )

        if int(
            np.count_nonzero(use)
        ) < 2:

            continue

        med = float(
            np.median(
                np.abs(
                    residual[use]
                )
            )
        )

        if (
            med
            <= cfg.radial_return_residual_m
        ):

            good += 1

    return int(good)


def _neighbor_exceed_ratio(
    point_id: int,
    candidate_pred: float,
    model,
    x,
    y,
    z,
    terrain_ids,
    cfg: TerrainBlobGuardConfig,
):

    if len(
        terrain_ids
    ) < cfg.min_neighbor_comparisons:

        return 0.0, 0

    tx = x[
        terrain_ids
    ]

    ty = y[
        terrain_ids
    ]

    tz = z[
        terrain_ids
    ]

    d = np.hypot(
        tx - x[point_id],
        ty - y[point_id],
    )

    use = (
        d > 0.05
    ) & (
        d <= cfg.neighbor_radius_m
    )

    if int(
        np.count_nonzero(use)
    ) < cfg.min_neighbor_comparisons:

        return 0.0, int(
            np.count_nonzero(use)
        )

    tx = tx[use]
    ty = ty[use]
    tz = tz[use]
    d = d[use]

    terrain_pred, _ = _predict_model(
        model,
        tx,
        ty,
    )

    observed_slope = (
        z[point_id] - tz
    ) / d

    expected_slope = (
        candidate_pred
        - terrain_pred
    ) / d

    slope_deviation = np.abs(
        observed_slope
        - expected_slope
    )

    exceed = (
        slope_deviation
        >= cfg.slope_deviation_ratio
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

    return ratio, int(
        exceed.size
    )



# ============================================================
# HARD AIRBORNE-GROUND POINT-FIRST V2
# ============================================================

def _airborne_min_surface(
    z,
    ground,
    key,
    nx,
    ny,
):
    """Minimum current-ground elevation per XY raster cell."""

    out = np.full(
        nx * ny,
        np.nan,
        dtype=float,
    )

    ids = np.flatnonzero(ground)

    if ids.size == 0:
        return out.reshape(ny, nx)

    kk = key[ids]

    order = np.argsort(
        kk,
        kind="mergesort",
    )

    ids = ids[order]
    kk = kk[order]

    starts = np.r_[
        0,
        1 + np.flatnonzero(
            kk[1:] != kk[:-1]
        ),
    ]

    ends = np.r_[
        starts[1:],
        len(kk),
    ]

    for a, b in zip(starts, ends):

        out[int(kk[a])] = float(
            np.min(
                z[ids[a:b]]
            )
        )

    return out.reshape(ny, nx)


def _airborne_lower_scaffold_python_reference(
    x,
    y,
    z,
    ids,
    cell_m,
):
    """
    One lowest current-ground observation per coarse cell.

    Sparse false-ground canopy points therefore cannot dominate
    a dense local terrain fit merely by occurring in clusters.
    """

    ids = np.asarray(
        ids,
        dtype=np.int64,
    )

    if ids.size == 0:
        return ids

    x0 = float(np.min(x[ids]))
    y0 = float(np.min(y[ids]))

    ix = np.floor(
        (x[ids] - x0) / float(cell_m)
    ).astype(np.int64)

    iy = np.floor(
        (y[ids] - y0) / float(cell_m)
    ).astype(np.int64)

    nx = int(ix.max()) + 1

    key = iy * nx + ix

    order = np.argsort(
        key,
        kind="mergesort",
    )

    ids2 = ids[order]
    key2 = key[order]

    starts = np.r_[
        0,
        1 + np.flatnonzero(
            key2[1:] != key2[:-1]
        ),
    ]

    ends = np.r_[
        starts[1:],
        len(key2),
    ]

    selected = []

    for a, b in zip(starts, ends):

        group = ids2[a:b]

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


def _airborne_lower_scaffold(
    x,
    y,
    z,
    ids,
    cell_m,
):
    #
    # Execution routing only.
    # Scientific reference implementation is retained below
    # as _airborne_lower_scaffold_python_reference.
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

    return _airborne_lower_scaffold_python_reference(
        x,
        y,
        z,
        ids,
        cell_m,
    )



def _apply_hard_airborne_ground_guard(
    *,
    x,
    y,
    z,
    ground,
    tree,
    cfg,
):
    """
    Point-first hard detector for sparse suspended class-2 points.

    It intentionally does not require:
      * circular raster morphology,
      * connected blob membership,
      * nearby vegetation,
      * or one successful 5 m-only terrain model.

    It requires terrain support from multiple scales and a
    positive terrain-normal separation.
    """

    result = np.asarray(
        ground,
        dtype=bool,
    ).copy()

    before = int(result.sum())

    report = {
        "ground_before": before,
        "screen_candidates": 0,
        "modeled_candidates": 0,
        "confirmed_candidates": 0,
        "demoted_points": 0,
        "ground_after": before,
        "max_screen_gap_m": 0.0,
        "max_residual_m": 0.0,
        "median_residual_m": 0.0,
    }

    if before < 20:
        return result, report


    # ========================================================
    # 1. CHEAP MULTISCALE SCREEN
    # ========================================================

    (
        _x0,
        _y0,
        ix,
        iy,
        key,
        nx,
        ny,
    ) = _grid(
        x,
        y,
        cfg.airborne_screen_cell_m,
    )

    low = _airborne_min_surface(
        z,
        result,
        key,
        nx,
        ny,
    )

    if not np.any(np.isfinite(low)):
        return result, report

    filled = _nearest_fill(low)

    radii = (
        float(cfg.airborne_r1_m),
        float(cfg.airborne_r2_m),
        float(cfg.airborne_r3_m),
    )

    backgrounds = []

    for radius in radii:

        window = _odd_window(
            radius,
            cfg.airborne_screen_cell_m,
        )

        backgrounds.append(
            median_filter(
                filled,
                size=window,
                mode="nearest",
            )
        )


    gids = np.flatnonzero(result)

    gx = ix[gids]
    gy = iy[gids]

    gaps = np.column_stack(
        [
            z[gids] - background[gy, gx]
            for background in backgrounds
        ]
    )

    votes = np.sum(
        gaps >= cfg.airborne_screen_gap_m,
        axis=1,
    )

    median_gap = np.median(
        gaps,
        axis=1,
    )

    max_gap = np.max(
        gaps,
        axis=1,
    )

    candidate_local = (
        (votes >= cfg.airborne_screen_min_votes)
        &
        (max_gap >= cfg.airborne_screen_gap_m)
    )

    candidate_ids = gids[candidate_local]
    candidate_score = median_gap[candidate_local]

    if candidate_ids.size == 0:
        return result, report

    report["screen_candidates"] = int(
        candidate_ids.size
    )

    report["max_screen_gap_m"] = float(
        np.max(candidate_score)
    )


    # Limit expensive confirmation work.
    if (
        candidate_ids.size
        > cfg.airborne_max_screen_candidates
    ):

        order = np.argsort(
            candidate_score
        )[::-1][
            :cfg.airborne_max_screen_candidates
        ]

        candidate_ids = candidate_ids[order]
        candidate_score = candidate_score[order]


    confirmed = []
    modeled = 0


    # ========================================================
    # 2. MULTI-RADIUS 3-D TERRAIN CONFIRMATION
    # ========================================================

    #
    # Native execution preserves the reference classification logic.
    # Scalar SciPy query_ball_point remains neighborhood authority.
    # Candidate order, neighbor order, configured radii, support
    # semantics and all downstream scientific operations are unchanged.
    import os

    _kernel7c_backend = os.environ.get(
        "FASTGC_ULS_SUPPORT_BACKEND",
        "reference",
    ).strip().lower()

    _kernel7c_support = None

    if _kernel7c_backend == "native_rayon":
        try:
            from .backend.native import (
                airborne_prepare_support_multi_batch_rayon_native,
            )

            _kernel7c_batch_size = int(
                os.environ.get(
                    "FASTGC_ULS_RAYON_BATCH_SIZE",
                    "16",
                )
            )

            if _kernel7c_batch_size < 1:
                raise ValueError(
                    "FASTGC_ULS_RAYON_BATCH_SIZE must be >= 1"
                )

            _kernel7c_support = {}

            _kernel7c_radii = np.ascontiguousarray(
                np.asarray(radii, dtype=np.float64)
            )

            _kernel7c_max_radius = float(max(radii))

            for _batch_start in range(
                0,
                len(candidate_ids),
                _kernel7c_batch_size,
            ):
                _batch_ids = np.ascontiguousarray(
                    np.asarray(
                        candidate_ids[
                            _batch_start:
                            _batch_start + _kernel7c_batch_size
                        ],
                        dtype=np.int64,
                    )
                )

                _neighbor_chunks = []
                _offsets = [0]

                for _batch_point_id in _batch_ids:
                    _nbr = np.ascontiguousarray(
                        np.asarray(
                            tree.query_ball_point(
                                [
                                    x[_batch_point_id],
                                    y[_batch_point_id],
                                ],
                                _kernel7c_max_radius,
                            ),
                            dtype=np.int64,
                        )
                    )

                    _neighbor_chunks.append(_nbr)
                    _offsets.append(
                        _offsets[-1] + _nbr.size
                    )

                if _offsets[-1]:
                    _packed_neighbors = np.ascontiguousarray(
                        np.concatenate(_neighbor_chunks),
                        dtype=np.int64,
                    )
                else:
                    _packed_neighbors = np.empty(
                        0,
                        dtype=np.int64,
                    )

                _offsets_array = np.ascontiguousarray(
                    np.asarray(_offsets, dtype=np.int64)
                )

                _batch_support = (
                    airborne_prepare_support_multi_batch_rayon_native(
                        x,
                        y,
                        z,
                        result,
                        _batch_ids,
                        _packed_neighbors,
                        _offsets_array,
                        _kernel7c_radii,
                        cfg.airborne_exclusion_radius_m,
                        cfg.airborne_scaffold_cell_m,
                        cfg.airborne_min_support_points,
                        cfg.airborne_min_support_sectors,
                    )
                )

                if len(_batch_support) != len(_batch_ids):
                    raise RuntimeError(
                        "Native ULS support candidate count mismatch"
                    )

                for _local_index, _batch_point_id in enumerate(
                    _batch_ids
                ):
                    _support = _batch_support[_local_index]

                    if len(_support) != len(radii):
                        raise RuntimeError(
                            "Native ULS support radius count mismatch"
                        )

                    _kernel7c_support[
                        int(_batch_point_id)
                    ] = _support

                del (
                    _batch_ids,
                    _neighbor_chunks,
                    _packed_neighbors,
                    _offsets_array,
                    _batch_support,
                )

        except Exception:
            # Preserve established fallback behavior.
            _kernel7c_support = None


    for point_id in candidate_ids:

        residuals = []
        predictions = []
        successful_models = []

        #
        # Native multi-radius support preserves the reference classification logic.
        # reference and native (#6A) retain their existing paths.
        # native_multi performs one scalar query at max(radii),
        # then derives the ordered radius-specific support sets
        # inside the exact native support-preparation kernel.
        import os

        _support_backend = os.environ.get(
            "FASTGC_ULS_SUPPORT_BACKEND",
            "reference",
        ).strip().lower()

        _multi_support = None

        if (
            _support_backend == "native_rayon"
            and _kernel7c_support is not None
        ):
            _multi_support = _kernel7c_support.get(
                int(point_id)
            )

        if _support_backend == "native_multi":
            try:
                from .backend.native import (
                    airborne_prepare_support_multi_native,
                )

                _max_radius = max(radii)

                _nbr_max = np.asarray(
                    tree.query_ball_point(
                        [
                            x[point_id],
                            y[point_id],
                        ],
                        _max_radius,
                    ),
                    dtype=np.int64,
                )

                if _nbr_max.size != 0:
                    _multi_support = (
                        airborne_prepare_support_multi_native(
                            x,
                            y,
                            z,
                            result,
                            _nbr_max,
                            point_id,
                            np.asarray(
                                radii,
                                dtype=np.float64,
                            ),
                            cfg.airborne_exclusion_radius_m,
                            cfg.airborne_scaffold_cell_m,
                            cfg.airborne_min_support_points,
                            cfg.airborne_min_support_sectors,
                        )
                    )

                    if len(_multi_support) != len(radii):
                        _multi_support = None

            except Exception:
                _multi_support = None

        for _radius_index, radius in enumerate(radii):

            _native_support_ok = False

            if (
                _support_backend in (
                    "native_multi",
                    "native_rayon",
                )
                and _multi_support is not None
            ):
                scaffold, sectors = _multi_support[
                    _radius_index
                ]

                _native_support_ok = True

            else:
                nbr = np.asarray(
                    tree.query_ball_point(
                        [
                            x[point_id],
                            y[point_id],
                        ],
                        radius,
                    ),
                    dtype=np.int64,
                )

                if nbr.size == 0:
                    continue

                #
                # Existing single-radius support route is retained.
                if _support_backend == "native":
                    try:
                        from .backend.native import (
                            airborne_prepare_support_native,
                        )

                        scaffold, sectors = (
                            airborne_prepare_support_native(
                                x,
                                y,
                                z,
                                result,
                                nbr,
                                point_id,
                                cfg.airborne_exclusion_radius_m,
                                cfg.airborne_scaffold_cell_m,
                                cfg.airborne_min_support_points,
                                cfg.airborne_min_support_sectors,
                            )
                        )

                        _native_support_ok = True

                    except Exception:
                        _native_support_ok = False

            if not _native_support_ok:
                nbr = nbr[
                    result[nbr]
                ]

                nbr = nbr[
                    nbr != point_id
                ]

                if (
                    nbr.size
                    < cfg.airborne_min_support_points
                ):
                    continue

                # -----------------------------------------------
                # Candidate and immediate companions cannot
                # construct their own terrain surface.
                # -----------------------------------------------

                d = np.hypot(
                    x[nbr] - x[point_id],
                    y[nbr] - y[point_id],
                )

                support = nbr[
                    d >= cfg.airborne_exclusion_radius_m
                ]

                if (
                    support.size
                    < cfg.airborne_min_support_points
                ):
                    continue

                # -----------------------------------------------
                # Lower-ground scaffold.
                # -----------------------------------------------

                scaffold = _airborne_lower_scaffold(
                    x,
                    y,
                    z,
                    support,
                    cfg.airborne_scaffold_cell_m,
                )

                if (
                    scaffold.size
                    < cfg.airborne_min_support_points
                ):
                    continue

                sectors = _sector_count(
                    x[scaffold] - x[point_id],
                    y[scaffold] - y[point_id],
                    8,
                )

                if (
                    sectors
                    < cfg.airborne_min_support_sectors
                ):
                    continue

            else:
                # Native kernels return an empty scaffold whenever
                # the exact reference support/sector gates reject.
                if scaffold.size == 0:
                    continue


            if (
                scaffold.size
                > cfg.airborne_max_support_points
            ):

                sd = np.hypot(
                    x[scaffold] - x[point_id],
                    y[scaffold] - y[point_id],
                )

                scaffold = scaffold[
                    np.argsort(sd)[
                        :cfg.airborne_max_support_points
                    ]
                ]


            model = _choose_terrain_model(
                x[scaffold],
                y[scaffold],
                z[scaffold],
                cfg,
            )

            if model is None:
                continue


            pred, scale = _predict_model(
                model,
                np.asarray(
                    [x[point_id]],
                    dtype=float,
                ),
                np.asarray(
                    [y[point_id]],
                    dtype=float,
                ),
            )

            terrain_z = float(pred[0])

            rn = float(
                (
                    z[point_id] - terrain_z
                )
                / scale[0]
            )

            residuals.append(rn)
            predictions.append(terrain_z)

            successful_models.append(
                (
                    radius,
                    model,
                )
            )


        if (
            len(residuals)
            < cfg.airborne_min_radius_votes
        ):
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

        positive_votes = int(
            np.count_nonzero(
                residuals
                >= cfg.airborne_candidate_gap_m
            )
        )

        strong_votes = int(
            np.count_nonzero(
                residuals
                >= cfg.airborne_strong_gap_m
            )
        )

        if (
            positive_votes
            < cfg.airborne_min_radius_votes
        ):
            continue


        median_rn = float(
            np.median(residuals)
        )

        pred_spread = float(
            np.ptp(predictions)
        )

        rn_spread = float(
            np.ptp(residuals)
        )

        absolute_detached = bool(
            median_rn
            >= cfg.airborne_absolute_gap_m
        )


        # -----------------------------------------------
        # Complicated terrain where 5, 10 and 15 m models
        # disagree substantially is protected unless the
        # candidate is extremely detached.
        # -----------------------------------------------

        if (
            not absolute_detached
            and pred_spread
            > cfg.airborne_max_prediction_spread_m
        ):
            continue

        if (
            not absolute_detached
            and rn_spread
            > cfg.airborne_max_residual_spread_m
        ):
            continue


        # ====================================================
        # 3. SAME-SHEET PROTECTION
        # ====================================================

        local = np.asarray(
            tree.query_ball_point(
                [
                    x[point_id],
                    y[point_id],
                ],
                cfg.airborne_same_sheet_radius_m,
            ),
            dtype=np.int64,
        )

        if local.size:

            local = local[
                (
                    local != point_id
                )
                & result[local]
            ]

        same_sheet = 0

        if local.size:

            successful_models.sort(
                key=lambda item: item[0]
            )

            final_model = successful_models[-1][1]

            lp, ls = _predict_model(
                final_model,
                x[local],
                y[local],
            )

            lr = (
                z[local] - lp
            ) / ls

            same_sheet = int(
                np.count_nonzero(
                    np.abs(
                        lr - median_rn
                    )
                    <= cfg.airborne_same_sheet_band_m
                )
            )


        extreme = bool(
            median_rn
            >= cfg.airborne_extreme_gap_m
        )

        strong = bool(
            median_rn
            >= cfg.airborne_strong_gap_m
        )


        # ====================================================
        # 4. FINAL DECISION
        # ====================================================

        accept = bool(

            # Very obvious airborne point.
            (
                absolute_detached
                and positive_votes
                >= cfg.airborne_min_radius_votes
            )

            or

            # Clearly detached and only weakly connected to
            # an elevated class-2 sheet.
            (
                extreme
                and positive_votes
                >= cfg.airborne_min_radius_votes
                and same_sheet
                <= (
                    cfg.airborne_max_same_sheet_neighbors
                    + 2
                )
            )

            or

            # More moderate case requires stronger repeated
            # confirmation and very weak elevated connectivity.
            (
                strong
                and strong_votes
                >= cfg.airborne_min_radius_votes
                and same_sheet
                <= cfg.airborne_max_same_sheet_neighbors
            )
        )

        if accept:

            confirmed.append(
                (
                    int(point_id),
                    median_rn,
                    positive_votes,
                    same_sheet,
                )
            )


    report[
        "modeled_candidates"
    ] = int(modeled)


    if not confirmed:
        return result, report


    # ========================================================
    # 5. SAFETY CAP
    # ========================================================

    confirmed.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    maximum = max(
        1,
        int(
            math.ceil(
                cfg.airborne_max_demote_fraction
                * max(
                    1,
                    int(result.sum()),
                )
            )
        ),
    )

    confirmed = confirmed[:maximum]

    demote_ids = np.asarray(
        [
            item[0]
            for item in confirmed
        ],
        dtype=np.int64,
    )

    # Demotion only.
    result[demote_ids] = False

    rn_values = np.asarray(
        [
            item[1]
            for item in confirmed
        ],
        dtype=float,
    )

    report[
        "confirmed_candidates"
    ] = int(len(confirmed))

    report[
        "demoted_points"
    ] = int(
        demote_ids.size
    )

    report[
        "ground_after"
    ] = int(
        result.sum()
    )

    report[
        "max_residual_m"
    ] = float(
        np.max(rn_values)
    )

    report[
        "median_residual_m"
    ] = float(
        np.median(rn_values)
    )

    return result, report


# ============================================================
# FINAL CANOPY / RIDGE AUTHORIZATION V5
# ============================================================
#
# Contract
# --------
# Geometry may NOMINATE a false-ground candidate.
#
# Geometry alone may NOT authorize final demotion.
#
# Every point removed by the aggressive post-final V4 cleanup
# must satisfy a final object/context test:
#
#   1. determine the XY component / footprint;
#   2. inspect the projected column through ALL points;
#   3. require existing non-ground structure associated with
#      that footprint;
#   4. protect long, coherent, ridge-like components;
#   5. uncertainty restores the original ground decision.
#
# This stage never creates new demotions.  It can only restore
# points that V4 attempted to demote.
# ============================================================


def _v5_candidate_components(
    x,
    y,
    candidate_ids,
    *,
    cell_m=0.75,
):
    """Connected XY components of points V4 attempted to demote."""

    candidate_ids = np.asarray(
        candidate_ids,
        dtype=np.int64,
    )

    if candidate_ids.size == 0:
        return []

    xx = np.asarray(
        x[candidate_ids],
        dtype=float,
    )
    yy = np.asarray(
        y[candidate_ids],
        dtype=float,
    )

    x0 = float(np.min(xx))
    y0 = float(np.min(yy))

    ix = np.floor(
        (xx - x0) / float(cell_m)
    ).astype(np.int64)

    iy = np.floor(
        (yy - y0) / float(cell_m)
    ).astype(np.int64)

    cell_to_points = {}

    for local_i, (cx, cy) in enumerate(
        zip(ix, iy)
    ):
        key = (
            int(cx),
            int(cy),
        )

        cell_to_points.setdefault(
            key,
            [],
        ).append(
            int(candidate_ids[local_i])
        )

    remaining = set(
        cell_to_points.keys()
    )

    components = []

    neighbors = [
        (-1, -1),
        ( 0, -1),
        ( 1, -1),
        (-1,  0),
        ( 1,  0),
        (-1,  1),
        ( 0,  1),
        ( 1,  1),
    ]

    while remaining:

        seed = remaining.pop()

        stack = [
            seed
        ]

        cells = [
            seed
        ]

        while stack:

            cx, cy = stack.pop()

            for dx, dy in neighbors:

                nb = (
                    cx + dx,
                    cy + dy,
                )

                if nb in remaining:

                    remaining.remove(
                        nb
                    )

                    stack.append(
                        nb
                    )

                    cells.append(
                        nb
                    )

        ids = []

        for key in cells:
            ids.extend(
                cell_to_points[
                    key
                ]
            )

        components.append(
            np.asarray(
                ids,
                dtype=np.int64,
            )
        )

    return components


def _v5_component_xy_shape(
    x,
    y,
    ids,
):
    """
    PCA-based XY morphology.

    Long coherent features are potential terrain ridges and
    receive explicit protection.
    """

    ids = np.asarray(
        ids,
        dtype=np.int64,
    )

    if ids.size == 0:
        return {
            "length_m": 0.0,
            "width_m": 0.0,
            "elongation": 1.0,
        }

    pts = np.column_stack(
        (
            np.asarray(
                x[ids],
                dtype=float,
            ),
            np.asarray(
                y[ids],
                dtype=float,
            ),
        )
    )

    if pts.shape[0] < 2:

        return {
            "length_m": 0.0,
            "width_m": 0.0,
            "elongation": 1.0,
        }

    center = np.median(
        pts,
        axis=0,
    )

    q = (
        pts
        - center
    )

    try:

        cov = np.cov(
            q.T
        )

        vals, vecs = np.linalg.eigh(
            cov
        )

        order = np.argsort(
            vals
        )[::-1]

        vecs = vecs[
            :,
            order,
        ]

        major = q @ vecs[:, 0]
        minor = q @ vecs[:, 1]

        # Robust projected extent.
        major_lo, major_hi = np.percentile(
            major,
            [2.0, 98.0],
        )

        minor_lo, minor_hi = np.percentile(
            minor,
            [2.0, 98.0],
        )

        length = float(
            max(
                0.0,
                major_hi - major_lo,
            )
        )

        width = float(
            max(
                0.20,
                minor_hi - minor_lo,
            )
        )

        elongation = float(
            length
            / width
        )

    except Exception:

        length = float(
            np.ptp(
                pts[:, 0]
            )
        )

        width = float(
            max(
                0.20,
                np.ptp(
                    pts[:, 1]
                ),
            )
        )

        elongation = float(
            max(
                length,
                width,
            )
            / max(
                0.20,
                min(
                    length,
                    width,
                ),
            )
        )

    return {
        "length_m":
            float(length),

        "width_m":
            float(width),

        "elongation":
            float(elongation),
    }


def _v5_canopy_ridge_authorization(
    *,
    x,
    y,
    z,
    ground_before,
    ground_after,
    tree,
):
    """
    Final authorization for V4 demotions.

    Any point that was ground before V4 but non-ground after V4
    is only allowed to remain demoted when its component is
    associated with a real non-ground/canopy population.

    Long coherent ridge-like components are restored.

    Returns
    -------
    final_ground, report
    """

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

    before = np.asarray(
        ground_before,
        dtype=bool,
    )

    after = np.asarray(
        ground_after,
        dtype=bool,
    ).copy()

    attempted = np.flatnonzero(
        before
        & ~after
    )

    report = {
        "attempted_demotions":
            int(
                attempted.size
            ),

        "components":
            0,

        "ridge_protected_components":
            0,

        "no_canopy_components":
            0,

        "canopy_confirmed_components":
            0,

        "restored_points":
            0,

        "authorized_demotions":
            0,
    }

    if attempted.size == 0:
        return after, report


    # --------------------------------------------------------
    # Component construction.
    # --------------------------------------------------------

    components = _v5_candidate_components(
        x,
        y,
        attempted,
        cell_m=0.75,
    )

    report[
        "components"
    ] = int(
        len(
            components
        )
    )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Non-ground context must come from the state BEFORE these
    # aggressive V4 demotions. Otherwise the points V4 just
    # demoted would count as their own "non-ground evidence".
    # --------------------------------------------------------

    original_nonground = (
        ~before
    )


    for comp in components:

        if comp.size == 0:
            continue


        # ====================================================
        # A. COMPONENT MORPHOLOGY / RIDGE PROTECTION
        # ====================================================

        shape = _v5_component_xy_shape(
            x,
            y,
            comp,
        )

        length_m = float(
            shape[
                "length_m"
            ]
        )

        elongation = float(
            shape[
                "elongation"
            ]
        )


        # Long and directionally coherent anomaly:
        # strong indication of ridge / bank / terrain edge.
        #
        # Either criterion is intentionally conservative.
        ridge_like = bool(
            (
                length_m
                >= 8.0
                and elongation
                >= 3.0
            )
            or (
                length_m
                >= 15.0
            )
        )


        if ridge_like:

            after[
                comp
            ] = True

            report[
                "ridge_protected_components"
            ] += 1

            report[
                "restored_points"
            ] += int(
                comp.size
            )

            continue


        # ====================================================
        # B. COMPONENT FOOTPRINT + PERIMETER
        # ====================================================

        cx = float(
            np.median(
                x[comp]
            )
        )

        cy = float(
            np.median(
                y[comp]
            )
        )

        xmin = float(
            np.min(
                x[comp]
            )
        )

        xmax = float(
            np.max(
                x[comp]
            )
        )

        ymin = float(
            np.min(
                y[comp]
            )
        )

        ymax = float(
            np.max(
                y[comp]
            )
        )

        # Project a slightly expanded version of the detected
        # footprint upward/downward through the full cloud.
        pad = 1.25

        query_radius = float(
            0.5
            * np.hypot(
                xmax - xmin,
                ymax - ymin,
            )
            + pad
        )

        query_radius = float(
            np.clip(
                query_radius,
                1.75,
                6.0,
            )
        )


        nearby = np.asarray(
            tree.query_ball_point(
                [
                    cx,
                    cy,
                ],
                query_radius,
            ),
            dtype=np.int64,
        )

        if nearby.size == 0:

            after[
                comp
            ] = True

            report[
                "no_canopy_components"
            ] += 1

            report[
                "restored_points"
            ] += int(
                comp.size
            )

            continue


        # Expanded XY component footprint.
        footprint = nearby[
            (
                x[nearby]
                >= xmin - pad
            )
            & (
                x[nearby]
                <= xmax + pad
            )
            & (
                y[nearby]
                >= ymin - pad
            )
            & (
                y[nearby]
                <= ymax + pad
            )
        ]


        if footprint.size == 0:

            after[
                comp
            ] = True

            report[
                "no_canopy_components"
            ] += 1

            report[
                "restored_points"
            ] += int(
                comp.size
            )

            continue


        ng = footprint[
            original_nonground[
                footprint
            ]
        ]


        # Absolutely no pre-existing non-ground population:
        # this cannot be authorized as canopy cleanup.
        if ng.size == 0:

            after[
                comp
            ] = True

            report[
                "no_canopy_components"
            ] += 1

            report[
                "restored_points"
            ] += int(
                comp.size
            )

            continue


        # ====================================================
        # C. UPWARD + DOWNWARD COLUMN PROJECTION
        # ====================================================

        comp_z_low = float(
            np.percentile(
                z[comp],
                10.0,
            )
        )

        comp_z_high = float(
            np.percentile(
                z[comp],
                90.0,
            )
        )

        comp_z_med = float(
            np.median(
                z[comp]
            )
        )


        ng_z = z[
            ng
        ]


        # Do NOT use the old narrow +/-0.45 m residual band.
        #
        # We want evidence that the candidate is embedded in a
        # vegetation column, so inspect non-ground both above
        # and below the component.
        ng_above = ng[
            ng_z
            >= comp_z_low + 0.20
        ]

        ng_below = ng[
            ng_z
            <= comp_z_high - 0.20
        ]

        ng_near = ng[
            np.abs(
                ng_z
                - comp_z_med
            )
            <= 1.50
        ]


        # ====================================================
        # D. RADIAL / PERIMETER CANOPY OCCUPANCY
        # ====================================================

        dx = (
            x[ng]
            - cx
        )

        dy = (
            y[ng]
            - cy
        )

        rr = np.hypot(
            dx,
            dy,
        )

        valid_r = (
            rr
            > 0.15
        )

        if np.any(
            valid_r
        ):

            angle = (
                np.arctan2(
                    dy[valid_r],
                    dx[valid_r],
                )
                + 2.0 * np.pi
            ) % (
                2.0 * np.pi
            )

            sectors = np.floor(
                angle
                / (
                    2.0
                    * np.pi
                    / 8.0
                )
            ).astype(
                np.int64
            )

            ng_sectors = int(
                np.unique(
                    sectors
                ).size
            )

        else:

            ng_sectors = 0


        # ====================================================
        # E. CANOPY MEMBERSHIP
        # ====================================================
        #
        # Require real pre-existing non-ground structure.
        #
        # Strong routes:
        #   - non-ground above AND below;
        #   - non-ground above + near + multi-directional;
        #   - dense vertically extended non-ground population.
        #
        # Absence/ambiguity protects terrain.
        # ====================================================

        vertical_span = float(
            np.ptp(
                ng_z
            )
        ) if ng_z.size >= 2 else 0.0


        column_both_sides = bool(
            ng_above.size
            >= 2
            and ng_below.size
            >= 2
        )


        crown_envelope = bool(
            ng_above.size
            >= 3
            and ng_near.size
            >= 2
            and ng_sectors
            >= 2
        )


        vertically_extended = bool(
            ng.size
            >= 6
            and vertical_span
            >= 1.50
            and ng_sectors
            >= 2
        )


        # ====================================================
        # V5.1 SAME-LAYER CANOPY MEMBERSHIP
        # ====================================================
        #
        # Important mixed-cell case:
        #
        # real terrain points and one leaked canopy-ground point
        # can occupy the same 0.5 m cell.  The leaked point may
        # sit inside a crown layer where nearby PRE-EXISTING
        # non-ground points occur at approximately the same Z,
        # without necessarily extending both above AND below it.
        #
        # That is valid canopy evidence.
        #
        # This route remains component/context constrained:
        #
        #   - component must already have been geometrically
        #     nominated by the existing V4 detectors;
        #   - long ridge-like components were already protected;
        #   - evidence comes only from PRE-EXISTING non-ground;
        #   - at least three nearby non-ground points are needed.
        #
        # Therefore bare ridges cannot satisfy this route merely
        # because they have extreme slope or elevation.
        # ====================================================

        same_layer_canopy = bool(
            ng_near.size
            >= 3
        )


        canopy_confirmed = bool(
            column_both_sides
            or crown_envelope
            or vertically_extended
            or same_layer_canopy
        )


        if not canopy_confirmed:

            # No convincing canopy envelope:
            # restore original ground status.
            after[
                comp
            ] = True

            report[
                "no_canopy_components"
            ] += 1

            report[
                "restored_points"
            ] += int(
                comp.size
            )

            continue


        # ====================================================
        # F. FINAL AUTHORIZATION
        # ====================================================

        report[
            "canopy_confirmed_components"
        ] += 1


    report[
        "authorized_demotions"
    ] = int(
        np.count_nonzero(
            before
            & ~after
        )
    )

    return (
        after,
        report,
    )




# ============================================================
# MAIN GUARD
# ============================================================


def apply_hard_airborne_ground_guard(
    *,
    x,
    y,
    z,
    ground_mask,
    sensor_mode,
    cfg: dict | None = None,
    return_report: bool = True,
):
    """
    Sensor-shared point-first suspended-ground guard.

    Reuses FAST-GC's established ALS/ULS hard-airborne detector without
    enabling the complete airborne terrain-blob/morphology pipeline.

    The operation is strictly demotion-only.
    """
    sm = str(sensor_mode).upper().strip()

    ground = np.asarray(
        ground_mask,
        dtype=bool,
    ).copy()

    report = {
        "enabled": sm in {"ALS", "ULS", "TLS"},
        "ground_before": int(ground.sum()),
        "screen_candidates": 0,
        "modeled_candidates": 0,
        "confirmed_candidates": 0,
        "demoted_points": 0,
        "ground_after": int(ground.sum()),
        "max_screen_gap_m": 0.0,
        "max_residual_m": 0.0,
        "median_residual_m": 0.0,
    }

    if sm not in {"ALS", "ULS", "TLS"}:
        return (ground, report) if return_report else ground

    user_cfg = cfg or {}

    if not bool(
        user_cfg.get(
            "hard_airborne_ground_guard_enabled",
            True,
        )
    ):
        report["enabled"] = False
        return (ground, report) if return_report else ground

    if int(ground.sum()) < 20:
        return (ground, report) if return_report else ground

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    if not (
        len(x)
        == len(y)
        == len(z)
        == len(ground)
    ):
        raise ValueError(
            "x, y, z and ground_mask must have equal length."
        )

    # Reuse the established sensor configuration. TLS deliberately starts
    # from the conservative ALS geometry rather than inventing a new
    # detector. Individual airborne_* values remain configurable.
    config_mode = "ULS" if sm == "ULS" else "ALS"

    c = _cfg(
        config_mode,
        user_cfg,
    )

    tree = cKDTree(
        np.column_stack((x, y))
    )

    result, airborne = _apply_hard_airborne_ground_guard(
        x=x,
        y=y,
        z=z,
        ground=ground,
        tree=tree,
        cfg=c,
    )

    airborne["enabled"] = True
    airborne["ground_before"] = int(ground.sum())
    airborne["ground_after"] = int(result.sum())

    # Hard invariant: this guard must never recruit ground.
    if np.any(result & ~ground):
        raise RuntimeError(
            "Hard-airborne guard violated demotion-only contract."
        )

    return (
        (result, airborne)
        if return_report
        else result
    )


def apply_terrain_blob_guard(
    *,
    x,
    y,
    z,
    ground_mask,
    sensor_mode,
    cfg: dict | None = None,
    return_report: bool = True,
):

    sm = str(
        sensor_mode
    ).upper().strip()

    ground = np.asarray(
        ground_mask,
        dtype=bool,
    ).copy()

    report: dict[str, Any] = {
        "enabled":
            sm in {
                "ALS",
                "ULS",
            },

        "ground_before":
            int(
                ground.sum()
            ),

        "ground_after":
            int(
                ground.sum()
            ),

        "candidate_components":
            0,

        "modeled_components":
            0,

        "validated_components":
            0,

        "candidate_points":
            0,

        "demoted_points":
            0,

        "passes": [],
    }

    if sm not in {
        "ALS",
        "ULS",
    }:

        return (
            (ground, report)
            if return_report
            else ground
        )

    cfg = cfg or {}

    if not bool(
        cfg.get(
            "terrain_blob_guard_enabled",
            True,
        )
    ):

        report[
            "enabled"
        ] = False

        return (
            (ground, report)
            if return_report
            else ground
        )

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

    if not (
        len(x)
        == len(y)
        == len(z)
        == len(ground)
    ):

        raise ValueError(
            "x, y, z and ground_mask must have equal length."
        )

    if int(
        ground.sum()
    ) < 20:

        return (
            (ground, report)
            if return_report
            else ground
        )

    c = _cfg(
        sm,
        cfg,
    )

    initial_ground = int(
        ground.sum()
    )

    (
        x0,
        y0,
        ix,
        iy,
        point_key,
        nx,
        ny,
    ) = _grid(
        x,
        y,
        c.cell_m,
    )

    all_xy = np.column_stack(
        (
            x,
            y,
        )
    )

    all_tree = cKDTree(
        all_xy
    )

    # ========================================================
    # FINAL CANOPY / RIDGE AUTHORIZATION V5
    # ========================================================
    #
    # Preserve the classification state BEFORE Neighbor
    # Outlier / Airborne / MAX-Z cleanup.  Any point these
    # aggressive positive-anomaly operators remove must later
    # pass the common canopy/ridge authorization gate.
    #
    # This snapshot is also the only valid source of
    # pre-existing non-ground context. Newly demoted points
    # must never become evidence against themselves.
    #
    _v5_ground_before_cleanup = ground.copy()

    # ========================================================
    # INDEPENDENT NEIGHBOR OUTLIER LOCATOR V4
    # ========================================================
    #
    # ArcGIS Locate-Outliers-style operation.
    #
    # This runs independently of raster-blob morphology.
    # It iteratively removes points whose local Z/slope
    # relationships disagree with surrounding current-ground
    # neighbors in several directions, after independent
    # external-terrain confirmation.

    ground, neighbor_outlier_qc = apply_neighbor_outlier_guard(
        x=x,
        y=y,
        z=z,
        ground_mask=ground,
        sensor_mode=sm,
        cfg=cfg,
        return_report=True,
    )

    report["neighbor_outlier"] = neighbor_outlier_qc


    # ========================================================
    # HARD AIRBORNE-GROUND POINT-FIRST V2
    # ========================================================
    #
    # Run AFTER the neighbor-outlier peel so external terrain
    # support is cleaner and mutually supporting false-ground
    # points are less likely to survive.

    ground, airborne_qc = _apply_hard_airborne_ground_guard(
        x=x,
        y=y,
        z=z,
        ground=ground,
        tree=all_tree,
        cfg=c,
    )

    report["airborne"] = airborne_qc

    for ipass in range(
        max(
            1,
            c.passes,
        )
    ):

        # ====================================================
        # A. 0.5 m MAX-Z predicted-ground DEM
        # ====================================================

        maxz = _max_ground_surface(
            z,
            ground,
            point_key,
            nx,
            ny,
        )

        occupancy = np.isfinite(
            maxz
        )

        if not np.any(
            occupancy
        ):

            break

        filled = _nearest_fill(
            maxz
        )

        # ====================================================
        # B. MULTISCALE POSITIVE PROMINENCE
        # ====================================================

        w1 = _odd_window(
            c.prominence_r1_m,
            c.cell_m,
        )

        w2 = _odd_window(
            c.prominence_r2_m,
            c.cell_m,
        )

        w3 = _odd_window(
            c.prominence_r3_m,
            c.cell_m,
        )

        bg1 = median_filter(
            filled,
            size=w1,
            mode="nearest",
        )

        bg2 = median_filter(
            filled,
            size=w2,
            mode="nearest",
        )

        bg3 = median_filter(
            filled,
            size=w3,
            mode="nearest",
        )

        p1 = (
            filled - bg1
        )

        p2 = (
            filled - bg2
        )

        p3 = (
            filled - bg3
        )

        m1 = (
            occupancy
            & (
                p1
                >= c.prominence_1_m
            )
        )

        m2 = (
            occupancy
            & (
                p2
                >= c.prominence_2_m
            )
        )

        m3 = (
            occupancy
            & (
                p3
                >= c.prominence_3_m
            )
        )

        prominence_votes = (
            m1.astype(
                np.int8
            )
            + m2.astype(
                np.int8
            )
            + m3.astype(
                np.int8
            )
        )

        strong_prominence = (
            occupancy
            & (
                p3
                >= c.strong_prominence_m
            )
        )

        extreme_prominence = (
            occupancy
            & (
                p3
                >= c.extreme_prominence_m
            )
        )

        # ====================================================
        # C. ROBUST LOCAL Z OUTLIER
        # ====================================================

        local_med = bg2

        local_abs = np.abs(
            filled - local_med
        )

        local_mad = (
            1.4826
            * median_filter(
                local_abs,
                size=w2,
                mode="nearest",
            )
        )

        local_mad = np.maximum(
            local_mad,
            c.mad_floor_m,
        )

        robust_z = (
            filled - local_med
        ) / local_mad

        outlier_mask = (
            occupancy
            & (
                robust_z
                >= c.robust_z_min
            )
            & (
                p2
                >= c.prominence_2_m
            )
        )

        # ====================================================
        # D. SLOPE-EXCESS MASK
        #
        # IMPORTANT:
        # absolute steepness is not contamination.
        # ====================================================

        fine_slope = _slope_deg(
            maxz,
            c.cell_m,
        )

        slope_window = _odd_window(
            c.slope_background_radius_m,
            c.cell_m,
        )

        slope_fill = np.where(
            np.isfinite(
                fine_slope
            ),
            fine_slope,
            0.0,
        )

        regional_slope = median_filter(
            slope_fill,
            size=slope_window,
            mode="nearest",
        )

        slope_excess = (
            fine_slope
            - regional_slope
        )

        slope_mask = (
            occupancy
            & np.isfinite(
                slope_excess
            )
            & (
                slope_excess
                >= c.slope_excess_deg
            )
        )

        # ====================================================
        # E. GROUND SUPPORT / VOID MASK
        # ====================================================

        void_window = _odd_window(
            c.void_radius_m,
            c.cell_m,
        )

        void_fraction = _void_fraction(
            occupancy,
            void_window,
        )

        void_mask = (
            occupancy
            & (
                void_fraction
                >= c.void_support_fraction
            )
        )

        strong_void = (
            occupancy
            & (
                void_fraction
                >= c.strong_void_fraction
            )
        )

        # ====================================================
        # F. MULTIPLE-INDEPENDENT-MASK FUSION
        # ====================================================

        mask_score = (
            (
                prominence_votes
                >= c.min_prominence_votes
            ).astype(
                np.int8
            )
            + slope_mask.astype(
                np.int8
            )
            + outlier_mask.astype(
                np.int8
            )
            + void_mask.astype(
                np.int8
            )
            + strong_void.astype(
                np.int8
            )
        )

        seed = (
            extreme_prominence
            |
            strong_prominence
            |
            (
                m3
                & (
                    mask_score
                    >= c.min_mask_score
                )
            )
            |
            (
                outlier_mask
                & m2
                & (
                    slope_mask
                    | void_mask
                )
            )
        )

        # ----------------------------------------------------
        # Group nearby candidate cells but preserve original
        # seed cells for exact point selection.
        # ----------------------------------------------------

        if c.group_dilate_cells > 0:

            grouped = binary_dilation(
                seed,
                iterations=c.group_dilate_cells,
            )

        else:

            grouped = seed.copy()

        labels, nlabels = label(
            grouped,
            structure=np.ones(
                (3, 3),
                dtype=np.uint8,
            ),
        )

        report[
            "candidate_components"
        ] += int(
            nlabels
        )

        pass_candidates = 0
        pass_modeled = 0
        pass_validated = 0
        pass_demote = []

        # ====================================================
        # G. COMPONENT LOOP
        # ====================================================

        for lid in range(
            1,
            int(nlabels) + 1,
        ):

            grouped_component = (
                labels == lid
            )

            component_seed = (
                grouped_component
                & seed
            )

            if not np.any(
                component_seed
            ):

                continue

            shape = _component_shape(
                component_seed,
                c.cell_m,
            )

            if (
                shape["cells"]
                > c.max_component_cells
            ):

                continue

            if (
                shape["area_m2"]
                > c.max_component_area_m2
            ):

                continue

            cy_cells, cx_cells = np.nonzero(
                component_seed
            )

            seed_keys = (
                cy_cells.astype(
                    np.int64
                )
                * nx
                + cx_cells.astype(
                    np.int64
                )
            )

            candidate_ids = np.flatnonzero(
                ground
                & np.isin(
                    point_key,
                    seed_keys,
                    assume_unique=False,
                )
            )

            if candidate_ids.size == 0:

                continue

            pass_candidates += int(
                candidate_ids.size
            )

            # Raster cell centres of the actual anomaly.
            seed_x = (
                x0
                + (
                    cx_cells.astype(
                        float
                    )
                    + 0.5
                )
                * c.cell_m
            )

            seed_y = (
                y0
                + (
                    cy_cells.astype(
                        float
                    )
                    + 0.5
                )
                * c.cell_m
            )

            seed_xy = np.column_stack(
                (
                    seed_x,
                    seed_y,
                )
            )

            component_tree = cKDTree(
                seed_xy
            )

            cx = float(
                np.median(
                    seed_x
                )
            )

            cy = float(
                np.median(
                    seed_y
                )
            )

            # ------------------------------------------------
            # 5 m bounding search around entire component
            # ------------------------------------------------

            xmin = float(
                np.min(seed_x)
                - c.terrain_radius_m
            )

            xmax = float(
                np.max(seed_x)
                + c.terrain_radius_m
            )

            ymin = float(
                np.min(seed_y)
                - c.terrain_radius_m
            )

            ymax = float(
                np.max(seed_y)
                + c.terrain_radius_m
            )

            bbox_ids = np.flatnonzero(
                (x >= xmin)
                & (x <= xmax)
                & (y >= ymin)
                & (y <= ymax)
            )

            if bbox_ids.size == 0:

                continue

            # ------------------------------------------------
            # External current ground only.
            #
            # Candidate component plus halo cannot support
            # its own terrain model.
            # ------------------------------------------------

            external = bbox_ids[
                ground[
                    bbox_ids
                ]
            ]

            if external.size == 0:

                continue

            dcomp, _ = component_tree.query(
                np.column_stack(
                    (
                        x[external],
                        y[external],
                    )
                ),
                k=1,
            )

            external = external[
                dcomp
                >= c.terrain_exclusion_m
            ]

            if (
                external.size
                < c.min_terrain_points
            ):

                continue

            # Keep only points reasonably close to component.
            dcenter = np.hypot(
                x[external] - cx,
                y[external] - cy,
            )

            use = (
                dcenter
                <= (
                    c.terrain_radius_m
                    + max(
                        1.0,
                        math.sqrt(
                            shape["area_m2"]
                            / np.pi
                        ),
                    )
                )
            )

            external = external[
                use
            ]

            if (
                external.size
                < c.min_terrain_points
            ):

                continue

            sector_support = _sector_count(
                x[external] - cx,
                y[external] - cy,
                c.radial_sectors,
            )

            if (
                sector_support
                < c.min_terrain_sectors
            ):

                continue

            if (
                external.size
                > c.max_terrain_points
            ):

                d = np.hypot(
                    x[external] - cx,
                    y[external] - cy,
                )

                external = external[
                    np.argsort(
                        d
                    )[
                        :
                        c.max_terrain_points
                    ]
                ]

            # ------------------------------------------------
            # Plane → optional quadratic terrain
            # ------------------------------------------------

            model = _choose_terrain_model(
                x[external],
                y[external],
                z[external],
                c,
            )

            if model is None:

                continue

            pass_modeled += 1

            # ------------------------------------------------
            # Does surrounding support return to fitted terrain
            # in multiple radial directions?
            # ------------------------------------------------

            radial_return = _radial_return_count(
                model,
                x[external],
                y[external],
                z[external],
                cx,
                cy,
                c,
            )

            # ------------------------------------------------
            # Exact candidate residuals
            # ------------------------------------------------

            cand_pred, cand_scale = _predict_model(
                model,
                x[candidate_ids],
                y[candidate_ids],
            )

            cand_residual = (
                z[candidate_ids]
                - cand_pred
            ) / cand_scale

            # ------------------------------------------------
            # Existing non-ground observations in search area
            # ------------------------------------------------

            nonground_ids = bbox_ids[
                ~ground[
                    bbox_ids
                ]
            ]

            if nonground_ids.size:

                ng_pred, ng_scale = _predict_model(
                    model,
                    x[nonground_ids],
                    y[nonground_ids],
                )

                ng_residual = (
                    z[nonground_ids]
                    - ng_pred
                ) / ng_scale

            else:

                ng_residual = np.empty(
                    0,
                    dtype=float,
                )

            component_demoted = False

            compact = bool(
                (
                    shape["circularity"]
                    >= c.compact_circularity
                )
                or (
                    shape["aspect"]
                    <= c.max_aspect_for_compact
                )
            )

            # =================================================
            # H. EXACT POINT-BY-POINT DECISION
            # =================================================

            for local_i, point_id in enumerate(
                candidate_ids
            ):

                rn = float(
                    cand_residual[
                        local_i
                    ]
                )

                if (
                    rn
                    < c.normal_residual_min_m
                ):

                    continue

                predicted_here = float(
                    cand_pred[
                        local_i
                    ]
                )

                # ---------------------------------------------
                # Current raster evidence at this point
                # ---------------------------------------------

                gx = int(
                    ix[
                        point_id
                    ]
                )

                gy = int(
                    iy[
                        point_id
                    ]
                )

                local_void = float(
                    void_fraction[
                        gy,
                        gx,
                    ]
                )

                local_prominence = float(
                    p3[
                        gy,
                        gx,
                    ]
                )

                local_outlier = bool(
                    outlier_mask[
                        gy,
                        gx,
                    ]
                )

                local_slope_anomaly = bool(
                    slope_mask[
                        gy,
                        gx,
                    ]
                )

                # ---------------------------------------------
                # Same elevated GROUND sheet protection
                # ---------------------------------------------

                support = np.asarray(
                    all_tree.query_ball_point(
                        [
                            x[point_id],
                            y[point_id],
                        ],
                        c.same_sheet_radius_m,
                    ),
                    dtype=np.int64,
                )

                if support.size:

                    support = support[
                        support
                        != point_id
                    ]

                    support = support[
                        ground[
                            support
                        ]
                    ]

                    # Candidate component cannot self-protect.
                    if support.size:

                        ds, _ = component_tree.query(
                            np.column_stack(
                                (
                                    x[support],
                                    y[support],
                                )
                            ),
                            k=1,
                        )

                        support = support[
                            ds
                            >= c.terrain_exclusion_m
                        ]

                same_sheet = 0

                if support.size:

                    sp, ss = _predict_model(
                        model,
                        x[support],
                        y[support],
                    )

                    sr = (
                        z[support]
                        - sp
                    ) / ss

                    same_sheet = int(
                        np.count_nonzero(
                            np.abs(
                                sr - rn
                            )
                            <= c.same_sheet_band_m
                        )
                    )

                # ---------------------------------------------
                # ArcGIS-style neighbor slope/Z disagreement,
                # but relative to the reconstructed terrain.
                # ---------------------------------------------

                exceed_ratio, ncomparisons = (
                    _neighbor_exceed_ratio(
                        int(point_id),
                        predicted_here,
                        model,
                        x,
                        y,
                        z,
                        external,
                        c,
                    )
                )

                neighbor_outlier = bool(
                    (
                        ncomparisons
                        >= c.min_neighbor_comparisons
                    )
                    and (
                        exceed_ratio
                        >= c.neighbor_exceed_ratio
                    )
                )

                # ---------------------------------------------
                # Nearby non-ground canopy-layer consistency
                # ---------------------------------------------

                ng_consistent = 0

                if nonground_ids.size:

                    dng = np.hypot(
                        x[nonground_ids]
                        - x[point_id],
                        y[nonground_ids]
                        - y[point_id],
                    )

                    use_ng = (
                        dng
                        <= c.nonground_radius_m
                    )

                    if np.any(
                        use_ng
                    ):

                        ng_consistent = int(
                            np.count_nonzero(
                                (
                                    np.abs(
                                        ng_residual[
                                            use_ng
                                        ]
                                        - rn
                                    )
                                    <= c.nonground_residual_band_m
                                )
                                & (
                                    ng_residual[
                                        use_ng
                                    ]
                                    > c.normal_residual_min_m
                                )
                            )
                        )

                ng_supported = bool(
                    ng_consistent
                    >= c.nonground_min_consistent
                )

                # ---------------------------------------------
                # Independent confirmation evidence
                # ---------------------------------------------

                radial_supported = bool(
                    radial_return
                    >= c.min_radial_return_sectors
                )

                void_supported = bool(
                    local_void
                    >= c.void_support_fraction
                )

                strong_void_supported = bool(
                    local_void
                    >= c.strong_void_fraction
                )

                prominent = bool(
                    local_prominence
                    >= c.prominence_3_m
                )

                strong_prominent = bool(
                    local_prominence
                    >= c.strong_prominence_m
                )

                # ---------------------------------------------
                # Preserve a genuine connected terrain sheet.
                # Extreme detached contamination can override
                # this protection.
                # ---------------------------------------------

                if (
                    same_sheet
                    >= c.min_same_sheet_ground
                    and rn
                    < c.normal_residual_extreme_m
                ):

                    continue

                # ---------------------------------------------
                # Final decision
                # ---------------------------------------------

                extreme = bool(
                    rn
                    >= c.normal_residual_extreme_m
                )

                hard = bool(
                    rn
                    >= c.normal_residual_hard_m
                )

                strong = bool(
                    rn
                    >= c.normal_residual_strong_m
                )

                moderate = bool(
                    rn
                    >= c.normal_residual_min_m
                )

                accept = bool(

                    # Very strongly detached point with
                    # trustworthy surrounding terrain.
                    (
                        extreme
                        and radial_supported
                    )

                    or

                    # Strong clear tomb/blob.
                    (
                        hard
                        and radial_supported
                        and (
                            neighbor_outlier
                            or strong_prominent
                            or strong_void_supported
                            or ng_supported
                        )
                    )

                    or

                    # Moderate/strong anomaly requires several
                    # independent evidence sources.
                    (
                        strong
                        and prominent
                        and radial_supported
                        and (
                            neighbor_outlier
                            or ng_supported
                        )
                        and (
                            void_supported
                            or compact
                            or local_slope_anomaly
                            or local_outlier
                        )
                    )

                    or

                    # Lower residual only accepted when both
                    # canopy-layer and local outlier evidence
                    # agree.
                    (
                        moderate
                        and ng_supported
                        and neighbor_outlier
                        and radial_supported
                        and (
                            void_supported
                            or (
                                compact
                                and local_outlier
                            )
                        )
                    )
                )

                if accept:

                    pass_demote.append(
                        (
                            int(point_id),
                            rn,
                        )
                    )

                    component_demoted = True

            if component_demoted:

                pass_validated += 1

        # ====================================================
        # I. SAFE DEMOTION
        # ====================================================

        if pass_demote:

            best_score = {}

            for point_id, score in pass_demote:

                best_score[
                    point_id
                ] = max(
                    float(score),
                    best_score.get(
                        point_id,
                        -np.inf,
                    ),
                )

            demote_ids = np.asarray(
                list(
                    best_score.keys()
                ),
                dtype=np.int64,
            )

            demote_ids = demote_ids[
                ground[
                    demote_ids
                ]
            ]

            maximum = max(
                1,
                int(
                    math.ceil(
                        c.max_demote_fraction_per_pass
                        * max(
                            1,
                            int(
                                ground.sum()
                            ),
                        )
                    )
                ),
            )

            if (
                demote_ids.size
                > maximum
            ):

                score = np.asarray(
                    [
                        best_score[
                            int(i)
                        ]
                        for i in demote_ids
                    ],
                    dtype=float,
                )

                demote_ids = demote_ids[
                    np.argsort(
                        score
                    )[::-1][
                        :maximum
                    ]
                ]

            ground[
                demote_ids
            ] = False

            n_demoted = int(
                demote_ids.size
            )

        else:

            n_demoted = 0

        report[
            "candidate_points"
        ] += int(
            pass_candidates
        )

        report[
            "modeled_components"
        ] += int(
            pass_modeled
        )

        report[
            "validated_components"
        ] += int(
            pass_validated
        )

        report[
            "passes"
        ].append(
            {
                "pass":
                    int(
                        ipass + 1
                    ),

                "occupied_cells":
                    int(
                        np.count_nonzero(
                            occupancy
                        )
                    ),

                "prominence_1":
                    int(
                        np.count_nonzero(
                            m1
                        )
                    ),

                "prominence_2":
                    int(
                        np.count_nonzero(
                            m2
                        )
                    ),

                "prominence_3":
                    int(
                        np.count_nonzero(
                            m3
                        )
                    ),

                "strong_prominence":
                    int(
                        np.count_nonzero(
                            strong_prominence
                        )
                    ),

                "slope_excess":
                    int(
                        np.count_nonzero(
                            slope_mask
                        )
                    ),

                "local_outliers":
                    int(
                        np.count_nonzero(
                            outlier_mask
                        )
                    ),

                "void_support":
                    int(
                        np.count_nonzero(
                            void_mask
                        )
                    ),

                "seed_cells":
                    int(
                        np.count_nonzero(
                            seed
                        )
                    ),

                "components":
                    int(
                        nlabels
                    ),

                "modeled_components":
                    int(
                        pass_modeled
                    ),

                "validated_components":
                    int(
                        pass_validated
                    ),

                "candidate_points":
                    int(
                        pass_candidates
                    ),

                "demoted_points":
                    int(
                        n_demoted
                    ),
            }
        )

        if n_demoted == 0:

            break

    # ========================================================
    # FINAL CANOPY / RIDGE AUTHORIZATION V5
    # ========================================================
    #
    # Geometry nominated the points above.  Now determine
    # whether those demotions are actually canopy cleanup.
    #
    # Long coherent ridge-like components are restored.
    #
    # Compact/local candidates require a pre-existing
    # non-ground canopy envelope in the vertically projected
    # component footprint.
    #
    # No canopy evidence -> KEEP GROUND.
    #
    ground, canopy_ridge_qc = _v5_canopy_ridge_authorization(
        x=x,
        y=y,
        z=z,
        ground_before=_v5_ground_before_cleanup,
        ground_after=ground,
        tree=all_tree,
    )

    report[
        "canopy_ridge_authorization"
    ] = canopy_ridge_qc

    report[
        "ground_after"
    ] = int(
        ground.sum()
    )

    report[
        "demoted_points"
    ] = int(
        initial_ground
        - ground.sum()
    )

    return (
        (ground, report)
        if return_report
        else ground
    )


__all__ = [
    "TerrainBlobGuardConfig",
    "apply_terrain_blob_guard",
]
