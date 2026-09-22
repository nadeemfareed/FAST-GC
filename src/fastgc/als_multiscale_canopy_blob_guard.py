from __future__ import annotations

"""
FAST-GC ALS/ULS final multiscale canopy-contamination guard V2.

Raster evidence
---------------
* 0.5 m lower/upper predicted-ground surface
* 1 m terrain envelope
* 2 m terrain envelope
* 5 m terrain envelope
* local slope anomaly
* fine-cell vertical span
* surrounding ground-support void fraction

Point confirmation
------------------
For every raster anomaly:
* select the exact predicted-ground source points;
* search a 5 m neighborhood;
* exclude the anomalous component from terrain support;
* fit surrounding ground with a robust terrain plane;
* calculate terrain-normal residuals;
* inspect nearby non-ground points for the same elevated layer;
* demote only confirmed elevated predicted-ground points.

DEMOTION ONLY.
No reference/class truth is used.
"""

from dataclasses import dataclass
from typing import Any
import math

import numpy as np

from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    label,
    median_filter,
    uniform_filter,
)

from scipy.spatial import cKDTree


@dataclass(frozen=True)
class MultiScaleCanopyBlobConfig:

    # Raster geometry
    fine_cell_m: float = 0.50
    scale_1m: float = 1.0
    scale_2m: float = 2.0
    scale_5m: float = 5.0

    # Positive elevation anomaly
    dz_1m_m: float = 0.20
    dz_2m_m: float = 0.28
    dz_5m_m: float = 0.38

    strong_dz_5m_m: float = 0.70
    min_scale_votes: int = 2

    # Mixed cell: low terrain + high false class-2 point
    fine_vertical_span_m: float = 0.22

    # Local surface anomaly
    slope_window_cells: int = 9
    slope_excess_min_deg: float = 12.0

    # Ground-support void raster.
    # Window is a RADIUS in metres around candidate cell.
    void_radius_m: float = 5.0
    void_fraction_support: float = 0.35
    void_fraction_strong: float = 0.55

    # Candidate components
    candidate_dilate_cells: int = 1
    max_component_cells: int = 100
    max_component_area_m2: float = 25.0

    # External 5 m terrain model
    search_radius_m: float = 5.0
    component_exclusion_m: float = 1.0

    min_terrain_points: int = 12
    min_terrain_sectors: int = 4
    max_terrain_points: int = 180

    # Exact point residual
    point_normal_min_m: float = 0.30
    point_normal_strong_m: float = 0.55
    point_normal_very_strong_m: float = 0.80

    # Nearby external-ground protection
    same_ground_radius_m: float = 1.50
    same_ground_band_m: float = 0.15
    min_same_ground: int = 3

    # Nearby non-ground layer corroboration
    nonground_radius_m: float = 5.0
    nonground_residual_band_m: float = 0.40
    nonground_min_consistent: int = 2

    # Safety
    max_demote_fraction: float = 0.02
    passes: int = 2


def _cfg(sensor_mode: str, cfg: dict | None):
    cfg = cfg or {}
    d = dict(MultiScaleCanopyBlobConfig().__dict__)

    if str(sensor_mode).upper().strip() == "ULS":
        d.update(
            dz_1m_m=0.17,
            dz_2m_m=0.24,
            dz_5m_m=0.32,
            strong_dz_5m_m=0.60,
            point_normal_min_m=0.25,
            point_normal_strong_m=0.48,
            point_normal_very_strong_m=0.70,
        )

    for name, value in list(d.items()):
        key = f"canopy_blob_{name}"

        if key not in cfg:
            continue

        if isinstance(value, bool):
            d[name] = bool(cfg[key])
        elif isinstance(value, int):
            d[name] = int(cfg[key])
        else:
            d[name] = float(cfg[key])

    return MultiScaleCanopyBlobConfig(**d)


def _sector_count(dx, dy):
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)

    if dx.size == 0:
        return 0

    angle = (
        np.arctan2(dy, dx) +
        2.0 * np.pi
    ) % (2.0 * np.pi)

    sector = np.floor(
        angle / (np.pi / 4.0)
    ).astype(np.int8)

    return int(np.unique(sector).size)


def _robust_plane(x, y, z):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    finite = (
        np.isfinite(x) &
        np.isfinite(y) &
        np.isfinite(z)
    )

    if np.count_nonzero(finite) < 6:
        return None

    xc = float(np.median(x[finite]))
    yc = float(np.median(y[finite]))

    xx = x - xc
    yy = y - yc

    keep = finite.copy()
    coef = None

    for _ in range(6):

        if np.count_nonzero(keep) < 6:
            return None

        A = np.column_stack(
            (
                xx[keep],
                yy[keep],
                np.ones(np.count_nonzero(keep)),
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
            coef[0] * xx +
            coef[1] * yy +
            coef[2]
        )

        residual = z - pred

        med = float(
            np.median(residual[keep])
        )

        mad = max(
            1.4826 * float(
                np.median(
                    np.abs(
                        residual[keep] - med
                    )
                )
            ),
            0.02,
        )

        # High observations are less trusted because this is
        # specifically a lower terrain-support model.
        new_keep = (
            finite &
            (residual >= med - 4.0 * mad) &
            (residual <= med + 1.75 * mad)
        )

        if (
            np.count_nonzero(new_keep) < 6 or
            np.array_equal(new_keep, keep)
        ):
            break

        keep = new_keep

    if coef is None:
        return None

    pred = (
        coef[0] * xx[keep] +
        coef[1] * yy[keep] +
        coef[2]
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (z[keep] - pred) ** 2
            )
        )
    )

    return (
        float(coef[0]),
        float(coef[1]),
        float(coef[2]),
        xc,
        yc,
        rmse,
    )


def _predict_plane(model, x, y):

    a, b, c, xc, yc, _ = model

    pred = (
        a * (x - xc) +
        b * (y - yc) +
        c
    )

    normal_scale = math.sqrt(
        1.0 + a * a + b * b
    )

    return pred, normal_scale


def _nearest_fill(surface):

    surface = np.asarray(
        surface,
        dtype=float,
    )

    finite = np.isfinite(surface)

    if not np.any(finite):
        return surface.copy()

    if np.all(finite):
        return surface.copy()

    _, indices = distance_transform_edt(
        ~finite,
        return_indices=True,
    )

    out = surface.copy()

    missing = ~finite

    out[missing] = surface[
        tuple(
            idx[missing]
            for idx in indices
        )
    ]

    return out


def _grid(x, y, cell):

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
        iy.astype(np.int64) * nx +
        ix.astype(np.int64)
    )

    return x0, y0, ix, iy, key, nx, ny


def _raster_ground(
    z,
    ground,
    key,
    nx,
    ny,
    mode,
):

    ids = np.flatnonzero(ground)

    out = np.full(
        nx * ny,
        np.nan,
        dtype=float,
    )

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

        cell_key = int(kk[a])

        values = z[ids[a:b]]

        if values.size == 0:
            continue

        if mode == "upper":
            value = float(np.max(values))
        else:
            value = float(
                np.min(values)
                if values.size <= 2
                else np.quantile(
                    values,
                    0.12,
                )
            )

        out[cell_key] = value

    return out.reshape(ny, nx)


def _regional_surface(
    x,
    y,
    z,
    ground,
    *,
    scale,
    fine_x0,
    fine_y0,
    fine_cell,
    fine_nx,
    fine_ny,
):

    width = fine_nx * fine_cell
    height = fine_ny * fine_cell

    nx = max(
        1,
        int(np.ceil(width / scale)),
    )

    ny = max(
        1,
        int(np.ceil(height / scale)),
    )

    ix = np.floor(
        (x - fine_x0) / scale
    ).astype(np.int64)

    iy = np.floor(
        (y - fine_y0) / scale
    ).astype(np.int64)

    valid = (
        ground &
        (ix >= 0) &
        (iy >= 0) &
        (ix < nx) &
        (iy < ny)
    )

    ids = np.flatnonzero(valid)

    raster = np.full(
        nx * ny,
        np.nan,
        dtype=float,
    )

    if ids.size == 0:
        return np.full(
            (fine_ny, fine_nx),
            np.nan,
            dtype=float,
        )

    key = (
        iy[ids] * nx +
        ix[ids]
    )

    order = np.argsort(
        key,
        kind="mergesort",
    )

    ids = ids[order]
    key = key[order]

    starts = np.r_[
        0,
        1 + np.flatnonzero(
            key[1:] != key[:-1]
        ),
    ]

    ends = np.r_[
        starts[1:],
        len(key),
    ]

    for a, b in zip(starts, ends):

        values = z[ids[a:b]]

        if values.size == 0:
            continue

        # Lower-envelope representation.
        raster[int(key[a])] = float(
            np.min(values)
            if values.size <= 3
            else np.quantile(
                values,
                0.10,
            )
        )

    raster = raster.reshape(
        ny,
        nx,
    )

    if not np.any(np.isfinite(raster)):
        return np.full(
            (fine_ny, fine_nx),
            np.nan,
            dtype=float,
        )

    filled = _nearest_fill(raster)

    # Robust regional terrain tendency.
    filled = median_filter(
        filled,
        size=3,
        mode="nearest",
    )

    fy, fx = np.indices(
        (fine_ny, fine_nx)
    )

    world_x = (
        fine_x0 +
        (fx.astype(float) + 0.5) *
        fine_cell
    )

    world_y = (
        fine_y0 +
        (fy.astype(float) + 0.5) *
        fine_cell
    )

    rx = np.floor(
        (world_x - fine_x0) / scale
    ).astype(np.int64)

    ry = np.floor(
        (world_y - fine_y0) / scale
    ).astype(np.int64)

    rx = np.clip(
        rx,
        0,
        nx - 1,
    )

    ry = np.clip(
        ry,
        0,
        ny - 1,
    )

    return filled[ry, rx]


def _slope(surface, cell):

    s = np.asarray(
        surface,
        dtype=float,
    )

    finite = np.isfinite(s)

    if not np.any(finite):
        return np.full_like(
            s,
            np.nan,
        )

    filled = _nearest_fill(s)

    if min(filled.shape) < 3:
        return np.full_like(
            s,
            np.nan,
        )

    gy, gx = np.gradient(
        filled,
        cell,
        cell,
    )

    result = np.degrees(
        np.arctan(
            np.hypot(gx, gy)
        )
    )

    result[~finite] = np.nan

    return result


def apply_multiscale_canopy_blob_guard(
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

    ground = np.asarray(
        ground_mask,
        dtype=bool,
    ).copy()

    report: dict[str, Any] = {
        "enabled": sm in {"ALS", "ULS"},
        "ground_before": int(ground.sum()),
        "ground_after": int(ground.sum()),
        "candidate_components": 0,
        "validated_components": 0,
        "candidate_points": 0,
        "demoted_points": 0,
        "passes": [],
    }

    if sm not in {"ALS", "ULS"}:
        return (
            (ground, report)
            if return_report
            else ground
        )

    cfg = cfg or {}

    if not bool(
        cfg.get(
            "multiscale_canopy_blob_guard_enabled",
            True,
        )
    ):
        report["enabled"] = False

        return (
            (ground, report)
            if return_report
            else ground
        )

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    if not (
        len(x) ==
        len(y) ==
        len(z) ==
        len(ground)
    ):
        raise ValueError(
            "x, y, z and ground_mask must have equal length."
        )

    if np.count_nonzero(ground) < 12:
        return (
            (ground, report)
            if return_report
            else ground
        )

    c = _cfg(sm, cfg)

    initial_ground = int(
        ground.sum()
    )

    (
        x0,
        y0,
        ix,
        iy,
        key,
        nx,
        ny,
    ) = _grid(
        x,
        y,
        c.fine_cell_m,
    )

    xy = np.column_stack(
        (x, y)
    )

    all_tree = cKDTree(xy)

    for ipass in range(
        max(
            1,
            int(c.passes),
        )
    ):

        # ====================================================
        # 0.5 m ground surfaces
        # ====================================================

        low05 = _raster_ground(
            z,
            ground,
            key,
            nx,
            ny,
            "lower",
        )

        high05 = _raster_ground(
            z,
            ground,
            key,
            nx,
            ny,
            "upper",
        )

        occupancy = np.isfinite(
            low05
        )

        # ====================================================
        # VOID RASTER
        #
        # Candidate cells themselves are occupied.
        # What matters is lack of ground support AROUND them.
        # ====================================================

        radius_cells = max(
            1,
            int(
                round(
                    c.void_radius_m /
                    c.fine_cell_m
                )
            ),
        )

        window = (
            2 * radius_cells + 1
        )

        ground_fraction = uniform_filter(
            occupancy.astype(np.float32),
            size=window,
            mode="constant",
            cval=0.0,
        )

        void_fraction = (
            1.0 - ground_fraction
        )

        # ====================================================
        # MULTISCALE TERRAIN ENVELOPES
        # ====================================================

        ref1 = _regional_surface(
            x,
            y,
            z,
            ground,
            scale=c.scale_1m,
            fine_x0=x0,
            fine_y0=y0,
            fine_cell=c.fine_cell_m,
            fine_nx=nx,
            fine_ny=ny,
        )

        ref2 = _regional_surface(
            x,
            y,
            z,
            ground,
            scale=c.scale_2m,
            fine_x0=x0,
            fine_y0=y0,
            fine_cell=c.fine_cell_m,
            fine_nx=nx,
            fine_ny=ny,
        )

        ref5 = _regional_surface(
            x,
            y,
            z,
            ground,
            scale=c.scale_5m,
            fine_x0=x0,
            fine_y0=y0,
            fine_cell=c.fine_cell_m,
            fine_nx=nx,
            fine_ny=ny,
        )

        valid = (
            np.isfinite(high05) &
            np.isfinite(ref1) &
            np.isfinite(ref2) &
            np.isfinite(ref5)
        )

        dz1 = high05 - ref1
        dz2 = high05 - ref2
        dz5 = high05 - ref5

        vote1 = (
            dz1 >= c.dz_1m_m
        )

        vote2 = (
            dz2 >= c.dz_2m_m
        )

        vote5 = (
            dz5 >= c.dz_5m_m
        )

        z_votes = (
            vote1.astype(np.int8) +
            vote2.astype(np.int8) +
            vote5.astype(np.int8)
        )

        multi_z = (
            valid &
            (z_votes >= c.min_scale_votes) &
            vote5
        )

        strong_z = (
            valid &
            (dz5 >= c.strong_dz_5m_m)
        )

        # ====================================================
        # Fine vertical range
        # ====================================================

        span = (
            high05 - low05
        )

        span_mask = (
            np.isfinite(span) &
            (span >= c.fine_vertical_span_m)
        )

        # ====================================================
        # Slope anomaly relative to both 5 m terrain
        # and local neighborhood.
        # ====================================================

        fine_slope = _slope(
            high05,
            c.fine_cell_m,
        )

        regional_slope = _slope(
            ref5,
            c.fine_cell_m,
        )

        fine_fill = np.where(
            np.isfinite(fine_slope),
            fine_slope,
            0.0,
        )

        local_slope = median_filter(
            fine_fill,
            size=max(
                3,
                int(
                    c.slope_window_cells
                ),
            ),
            mode="nearest",
        )

        slope_excess = np.maximum(
            fine_slope - regional_slope,
            fine_slope - local_slope,
        )

        slope_mask = (
            np.isfinite(slope_excess) &
            (
                slope_excess >=
                c.slope_excess_min_deg
            )
        )

        void_support = (
            void_fraction >=
            c.void_fraction_support
        )

        strong_void = (
            void_fraction >=
            c.void_fraction_strong
        )

        # ====================================================
        # MULTI-MASK FUSION
        #
        # Strong Z anomaly stands alone.
        #
        # Moderate anomaly requires corroboration from
        # slope, vertical span, or surrounding void space.
        #
        # Void-rich candidates receive additional support
        # because canopy leaks commonly survive where the
        # genuine ground layer is sparsely sampled.
        # ====================================================

        anomaly = (
            strong_z
            |
            (
                multi_z &
                (
                    slope_mask |
                    span_mask |
                    void_support
                )
            )
            |
            (
                vote5 &
                strong_void &
                (
                    slope_mask |
                    span_mask
                )
            )
        )

        if c.candidate_dilate_cells > 0:
            anomaly = binary_dilation(
                anomaly,
                iterations=int(
                    c.candidate_dilate_cells
                ),
            )

        labels, n_labels = label(
            anomaly,
            structure=np.ones(
                (3, 3),
                dtype=np.uint8,
            ),
        )

        report[
            "candidate_components"
        ] += int(n_labels)

        pass_demotions = []
        pass_candidates = 0
        pass_modeled = 0
        pass_validated = 0

        # ====================================================
        # COMPONENT CONFIRMATION
        # ====================================================

        for component_id in range(
            1,
            int(n_labels) + 1,
        ):

            component = (
                labels == component_id
            )

            n_cells = int(
                np.count_nonzero(
                    component
                )
            )

            if n_cells == 0:
                continue

            if (
                n_cells >
                c.max_component_cells
            ):
                continue

            component_area = (
                n_cells *
                c.fine_cell_m *
                c.fine_cell_m
            )

            if (
                component_area >
                c.max_component_area_m2
            ):
                continue

            gy, gx = np.nonzero(
                component
            )

            component_keys = (
                gy.astype(np.int64) *
                nx +
                gx.astype(np.int64)
            )

            candidate_ids = np.flatnonzero(
                ground &
                np.isin(
                    key,
                    component_keys,
                    assume_unique=False,
                )
            )

            if candidate_ids.size == 0:
                continue

            pass_candidates += int(
                candidate_ids.size
            )

            cx = float(
                np.median(
                    x[candidate_ids]
                )
            )

            cy = float(
                np.median(
                    y[candidate_ids]
                )
            )

            neighborhood = np.asarray(
                all_tree.query_ball_point(
                    [cx, cy],
                    c.search_radius_m,
                ),
                dtype=np.int64,
            )

            if neighborhood.size == 0:
                continue

            # -----------------------------------------------
            # Exclude candidate component and nearby halo from
            # the terrain model.
            # -----------------------------------------------

            external_ground = neighborhood[
                ground[neighborhood] &
                ~np.isin(
                    key[neighborhood],
                    component_keys,
                    assume_unique=False,
                )
            ]

            if external_ground.size == 0:
                continue

            candidate_tree = cKDTree(
                np.column_stack(
                    (
                        x[candidate_ids],
                        y[candidate_ids],
                    )
                )
            )

            distance_to_component, _ = (
                candidate_tree.query(
                    np.column_stack(
                        (
                            x[external_ground],
                            y[external_ground],
                        )
                    ),
                    k=1,
                )
            )

            external_ground = external_ground[
                distance_to_component >=
                c.component_exclusion_m
            ]

            if (
                external_ground.size <
                c.min_terrain_points
            ):
                continue

            if (
                _sector_count(
                    x[external_ground] - cx,
                    y[external_ground] - cy,
                ) <
                c.min_terrain_sectors
            ):
                continue

            if (
                external_ground.size >
                c.max_terrain_points
            ):

                distance = np.hypot(
                    x[external_ground] - cx,
                    y[external_ground] - cy,
                )

                external_ground = (
                    external_ground[
                        np.argsort(distance)[
                            :c.max_terrain_points
                        ]
                    ]
                )

            model = _robust_plane(
                x[external_ground],
                y[external_ground],
                z[external_ground],
            )

            if model is None:
                continue

            pass_modeled += 1

            pred, scale = _predict_plane(
                model,
                x[candidate_ids],
                y[candidate_ids],
            )

            candidate_residual = (
                z[candidate_ids] - pred
            ) / scale

            # -----------------------------------------------
            # Non-ground points from requested 5 m search area
            # -----------------------------------------------

            nonground_ids = neighborhood[
                ~ground[neighborhood]
            ]

            if nonground_ids.size:

                ng_pred, ng_scale = _predict_plane(
                    model,
                    x[nonground_ids],
                    y[nonground_ids],
                )

                ng_residual = (
                    z[nonground_ids] -
                    ng_pred
                ) / ng_scale

            else:
                ng_residual = np.empty(
                    0,
                    dtype=float,
                )

            component_demoted = False

            for point_id, residual in zip(
                candidate_ids,
                candidate_residual,
            ):

                residual = float(residual)

                if (
                    residual <
                    c.point_normal_min_m
                ):
                    continue

                # -------------------------------------------
                # Current candidate pixel's void evidence
                # -------------------------------------------

                px = int(ix[point_id])
                py = int(iy[point_id])

                if (
                    px < 0 or
                    py < 0 or
                    px >= nx or
                    py >= ny
                ):
                    continue

                point_void = float(
                    void_fraction[
                        py,
                        px,
                    ]
                )

                # -------------------------------------------
                # External SAME-SHEET ground support
                # -------------------------------------------

                support = np.asarray(
                    all_tree.query_ball_point(
                        [
                            x[point_id],
                            y[point_id],
                        ],
                        c.same_ground_radius_m,
                    ),
                    dtype=np.int64,
                )

                if support.size:

                    support = support[
                        support != point_id
                    ]

                    support = support[
                        ground[support] &
                        ~np.isin(
                            key[support],
                            component_keys,
                            assume_unique=False,
                        )
                    ]

                same_ground = 0

                if support.size:

                    sp, ss = _predict_plane(
                        model,
                        x[support],
                        y[support],
                    )

                    sr = (
                        z[support] - sp
                    ) / ss

                    same_ground = int(
                        np.count_nonzero(
                            np.abs(
                                sr - residual
                            ) <=
                            c.same_ground_band_m
                        )
                    )

                # Strong real terrain connectivity protects.
                if (
                    same_ground >=
                    c.min_same_ground
                ):
                    continue

                # -------------------------------------------
                # Non-ground Z/residual consistency
                # -------------------------------------------

                consistent_ng = 0

                if nonground_ids.size:

                    ng_xy_distance = np.hypot(
                        x[nonground_ids] -
                        x[point_id],
                        y[nonground_ids] -
                        y[point_id],
                    )

                    local_ng = (
                        ng_xy_distance <=
                        c.nonground_radius_m
                    )

                    if np.any(local_ng):

                        consistent_ng = int(
                            np.count_nonzero(
                                np.abs(
                                    ng_residual[
                                        local_ng
                                    ] -
                                    residual
                                ) <=
                                c.nonground_residual_band_m
                            )
                        )

                ng_supported = (
                    consistent_ng >=
                    c.nonground_min_consistent
                )

                # -------------------------------------------
                # FINAL DECISION
                # -------------------------------------------

                very_strong = (
                    residual >=
                    c.point_normal_very_strong_m
                )

                strong = (
                    residual >=
                    c.point_normal_strong_m
                )

                void_rich = (
                    point_void >=
                    c.void_fraction_support
                )

                extreme_void = (
                    point_void >=
                    c.void_fraction_strong
                )

                accept = (
                    very_strong
                    |
                    (
                        strong &
                        ng_supported
                    )
                    |
                    (
                        strong &
                        extreme_void
                    )
                    |
                    (
                        (
                            residual >=
                            c.point_normal_min_m
                        ) &
                        ng_supported &
                        void_rich
                    )
                )

                if accept:

                    pass_demotions.append(
                        (
                            int(point_id),
                            residual,
                        )
                    )

                    component_demoted = True

            if component_demoted:
                pass_validated += 1

        # ====================================================
        # SAFE DEMOTION
        # ====================================================

        if pass_demotions:

            best = {}

            for point_id, score in pass_demotions:
                best[point_id] = max(
                    score,
                    best.get(
                        point_id,
                        -np.inf,
                    ),
                )

            demote_ids = np.asarray(
                list(best.keys()),
                dtype=np.int64,
            )

            demote_ids = demote_ids[
                ground[demote_ids]
            ]

            max_demotions = max(
                1,
                int(
                    math.ceil(
                        c.max_demote_fraction *
                        max(
                            1,
                            int(
                                ground.sum()
                            ),
                        )
                    )
                ),
            )

            if (
                demote_ids.size >
                max_demotions
            ):

                score = np.asarray(
                    [
                        best[int(i)]
                        for i in demote_ids
                    ],
                    dtype=float,
                )

                demote_ids = demote_ids[
                    np.argsort(score)[::-1][
                        :max_demotions
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
        ] += pass_candidates

        report[
            "validated_components"
        ] += pass_validated

        report["passes"].append(
            {
                "pass":
                    int(ipass + 1),

                "zmask_cells":
                    int(
                        np.count_nonzero(
                            multi_z
                        )
                    ),

                "strong_z_cells":
                    int(
                        np.count_nonzero(
                            strong_z
                        )
                    ),

                "slope_cells":
                    int(
                        np.count_nonzero(
                            slope_mask
                        )
                    ),

                "span_cells":
                    int(
                        np.count_nonzero(
                            span_mask
                        )
                    ),

                "void_support_cells":
                    int(
                        np.count_nonzero(
                            occupancy &
                            void_support
                        )
                    ),

                "strong_void_cells":
                    int(
                        np.count_nonzero(
                            occupancy &
                            strong_void
                        )
                    ),

                "candidate_components":
                    int(n_labels),

                "modeled_components":
                    int(pass_modeled),

                "validated_components":
                    int(pass_validated),

                "candidate_points":
                    int(pass_candidates),

                "demoted_points":
                    int(n_demoted),
            }
        )

        if n_demoted == 0:
            break

    report[
        "ground_after"
    ] = int(
        ground.sum()
    )

    report[
        "demoted_points"
    ] = int(
        initial_ground -
        ground.sum()
    )

    return (
        (ground, report)
        if return_report
        else ground
    )


__all__ = [
    "MultiScaleCanopyBlobConfig",
    "apply_multiscale_canopy_blob_guard",
]
