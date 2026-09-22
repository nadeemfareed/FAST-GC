from __future__ import annotations

from dataclasses import dataclass
import math
import os
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class TlsFinalGroundVoteResult:
    final_ground: np.ndarray
    removed: np.ndarray
    support_surface_z: np.ndarray
    ground_before: int
    ground_after: int
    demoted_points: int
    support_points: int
    enabled: bool
    reason: str


def _sector_count(dx, dy):
    if len(dx) == 0:
        return 0

    ang = np.arctan2(dy, dx)

    sec = np.floor(
        (ang + np.pi)
        / (2.0 * np.pi)
        * 8.0
    ).astype(np.int32)

    sec = np.clip(sec, 0, 7)

    return int(
        np.unique(sec).size
    )


def _cell_representatives(
    x,
    y,
    z,
    cell,
    ox,
    oy,
):
    """
    Robust representative XYZ for each occupied provisional-ground cell.

    Median XYZ is used. No minimum-Z rasterization is performed.
    """

    ix = np.floor(
        (x - ox) / cell
    ).astype(np.int64)

    iy = np.floor(
        (y - oy) / cell
    ).astype(np.int64)

    key = (
        (ix << 32)
        | (iy & 0xFFFFFFFF)
    )

    order = np.argsort(
        key,
        kind="mergesort",
    )

    ks = key[order]
    xs = x[order]
    ys = y[order]
    zs = z[order]

    uniq, starts = np.unique(
        ks,
        return_index=True,
    )

    ends = np.r_[
        starts[1:],
        len(order),
    ]

    ncell = len(uniq)

    rx = np.empty(
        ncell,
        dtype=np.float64,
    )
    ry = np.empty(
        ncell,
        dtype=np.float64,
    )
    rz = np.empty(
        ncell,
        dtype=np.float64,
    )
    rn = np.empty(
        ncell,
        dtype=np.int32,
    )

    for j, (a, b) in enumerate(
        zip(starts, ends)
    ):
        rx[j] = np.median(xs[a:b])
        ry[j] = np.median(ys[a:b])
        rz[j] = np.median(zs[a:b])
        rn[j] = b - a

    return (
        rx,
        ry,
        rz,
        rn,
    )


def _normal_scale_plane(beta):
    return math.sqrt(
        1.0
        + float(beta[0]) ** 2
        + float(beta[1]) ** 2
    )


def _fit_plane(
    px,
    py,
    pz,
    weights,
    cx,
    cy,
):
    dx = px - cx
    dy = py - cy

    X = np.column_stack(
        (
            dx,
            dy,
            np.ones_like(dx),
        )
    )

    w = np.sqrt(
        np.maximum(
            weights,
            0.05,
        )
    )

    try:
        beta, *_ = np.linalg.lstsq(
            X * w[:, None],
            pz * w,
            rcond=None,
        )
    except np.linalg.LinAlgError:
        return None

    pred = X @ beta

    scale = _normal_scale_plane(
        beta
    )

    nr = (
        pz - pred
    ) / scale

    return (
        beta,
        pred,
        nr,
        scale,
    )


def _robust_plane(
    px,
    py,
    pz,
    weights,
    cx,
    cy,
    robust_z,
    mad_floor,
):
    """
    Iteratively robust terrain-normal plane.
    """

    keep = np.ones(
        len(pz),
        dtype=bool,
    )

    result = None

    for _ in range(5):

        if np.count_nonzero(keep) < 6:
            return None

        result = _fit_plane(
            px[keep],
            py[keep],
            pz[keep],
            weights[keep],
            cx,
            cy,
        )

        if result is None:
            return None

        beta, _, _, _ = result

        _tls_robust_backend = os.environ.get(
            "FASTGC_TLS_ROBUST_BACKEND",
            "reference",
        ).strip().lower()

        if _tls_robust_backend == "native":
            from .backend.native import (
                tls_final_plane_iteration_native,
            )

            new_keep = tls_final_plane_iteration_native(
                px,
                py,
                pz,
                keep,
                beta,
                cx,
                cy,
                robust_z,
                mad_floor,
            )

        else:
            # Reference implementation remains mathematically and
            # operationally unchanged. These statistics are required
            # only by the NumPy reference iteration.
            dx = px - cx
            dy = py - cy

            pred_all = (
                beta[0] * dx
                + beta[1] * dy
                + beta[2]
            )

            scale = _normal_scale_plane(
                beta
            )

            nr = (
                pz - pred_all
            ) / scale

            med = float(
                np.median(
                    nr[keep]
                )
            )

            mad = max(
                float(
                    np.median(
                        np.abs(
                            nr[keep] - med
                        )
                    )
                ),
                float(mad_floor),
            )

            sigma = (
                1.4826 * mad
            )

            new_keep = (
                np.abs(
                    nr - med
                )
                <= float(robust_z) * sigma
            )

        if np.array_equal(
            new_keep,
            keep,
        ):
            break

        keep = new_keep

    if np.count_nonzero(keep) < 6:
        return None

    result = _fit_plane(
        px[keep],
        py[keep],
        pz[keep],
        weights[keep],
        cx,
        cy,
    )

    if result is None:
        return None

    beta, _, nr_keep, scale = result

    med = float(
        np.median(nr_keep)
    )

    mad = max(
        float(
            np.median(
                np.abs(
                    nr_keep - med
                )
            )
        ),
        float(mad_floor),
    )

    return (
        beta,
        scale,
        mad,
        keep,
    )


def _lower_scaffold(
    px,
    py,
    pz,
    weights,
    cx,
    cy,
    *,
    robust_z,
    mad_floor,
    upper_sigma,
    lower_sigma,
    min_keep,
):
    """
    Build a conservative terrain scaffold.

    Stage 1:
        establish local terrain orientation with a robust plane.

    Stage 2:
        BEFORE asymmetric upper-side pruning, test whether the robust
        neighborhood contains coherent quadratic curvature.

    If a quadratic surface materially improves the robust plane and its
    curvature remains plausible, preserve a symmetric coherent
    neighborhood. This protects bowls, hollows, convex/concave terrain
    and smoothly curved TLS surfaces.

    Otherwise use the established asymmetric lower scaffold, which is
    deliberately stricter above the terrain model so elevated
    vegetation cannot easily become terrain support.
    """

    first = _robust_plane(
        px,
        py,
        pz,
        weights,
        cx,
        cy,
        robust_z,
        mad_floor,
    )

    if first is None:
        return None

    beta, scale, mad, robust_keep = first

    if np.count_nonzero(robust_keep) < int(min_keep):
        return None

    dx = px - cx
    dy = py - cy

    pred = (
        beta[0] * dx
        + beta[1] * dy
        + beta[2]
    )

    nr = (
        pz - pred
    ) / scale

    base = nr[robust_keep]

    med = float(
        np.median(base)
    )

    sigma = max(
        1.4826
        * float(
            np.median(
                np.abs(
                    base - med
                )
            )
        ),
        float(mad_floor),
    )

    # --------------------------------------------------------
    # CURVATURE TEST BEFORE ASYMMETRIC SCAFFOLD PRUNING
    # --------------------------------------------------------
    #
    # Only the robust preliminary-plane core participates in this test.
    # Therefore a large detached canopy layer does not automatically
    # become evidence for curvature.
    #
    # This local quadratic test is intentionally conservative.
    # --------------------------------------------------------

    rk = np.flatnonzero(
        robust_keep
    )

    use_curved_scaffold = False
    qpred_all = None
    qscale = None
    q_sigma = None

    if rk.size >= max(
        int(min_keep),
        12,
    ):
        qfit = _fit_quadratic(
            px[rk],
            py[rk],
            pz[rk],
            weights[rk],
            cx,
            cy,
            mad_floor=mad_floor,
        )

        if qfit is not None:
            (
                qbeta,
                qscale_fit,
                qmad,
                qrmse,
                qcurv,
            ) = qfit

            # Plane RMSE over exactly the same robust support.
            plane_nr = nr[rk]

            plane_rmse = float(
                np.sqrt(
                    np.mean(
                        plane_nr * plane_nr
                    )
                )
            )

            improvement = (
                plane_rmse - qrmse
            ) / max(
                plane_rmse,
                1e-9,
            )

            # Require substantial improvement. This is deliberately
            # stronger than the later quadratic prediction threshold:
            # scaffold mode changes which support is allowed to survive.
            if (
                improvement >= 0.30
                and qcurv <= 1.50
            ):
                qpred_all = _predict_quadratic(
                    qbeta,
                    px,
                    py,
                    cx,
                    cy,
                )

                qscale = float(
                    qscale_fit
                )

                qnr_all = (
                    pz - qpred_all
                ) / qscale

                qcore = qnr_all[rk]

                qmed = float(
                    np.median(qcore)
                )

                q_sigma = max(
                    1.4826
                    * float(
                        np.median(
                            np.abs(
                                qcore - qmed
                            )
                        )
                    ),
                    float(mad_floor),
                )

                # Curved terrain must also be coherent, not merely
                # mathematically better than a plane.
                coherent_fraction = float(
                    np.mean(
                        np.abs(
                            qcore - qmed
                        )
                        <= 3.0 * q_sigma
                    )
                )

                if coherent_fraction >= 0.80:
                    use_curved_scaffold = True

    # --------------------------------------------------------
    # CURVATURE-PRESERVING SCAFFOLD
    # --------------------------------------------------------

    if use_curved_scaffold:
        qnr_all = (
            pz - qpred_all
        ) / qscale

        qmed = float(
            np.median(
                qnr_all[rk]
            )
        )

        # Symmetric support around a coherent curved terrain surface.
        #
        # We still cap it robustly; this is NOT permission for arbitrary
        # upper vegetation to enter the scaffold.
        curved_sigma = 3.25

        scaffold = (
            np.abs(
                qnr_all - qmed
            )
            <= curved_sigma * q_sigma
        )

        # Preserve the preliminary robust terrain core.
        scaffold |= robust_keep

        if np.count_nonzero(
            scaffold
        ) >= int(min_keep):
            return scaffold

    # --------------------------------------------------------
    # STANDARD LOWER / ASYMMETRIC SCAFFOLD
    # --------------------------------------------------------

    scaffold = (
        (
            nr
            <= med
            + float(upper_sigma) * sigma
        )
        &
        (
            nr
            >= med
            - float(lower_sigma) * sigma
        )
    )

    # Preserve the robust core.
    scaffold |= robust_keep

    if np.count_nonzero(
        scaffold
    ) < int(min_keep):
        return None

    return scaffold

def _fit_quadratic(
    px,
    py,
    pz,
    weights,
    cx,
    cy,
    *,
    mad_floor,
):
    """
    Fit:
        z = a*x + b*y + c
            + d*x^2 + e*x*y + f*y^2

    in local coordinates centered at target XY.

    Returns coefficients and terrain-normal residual statistics.
    """

    dx = px - cx
    dy = py - cy

    X = np.column_stack(
        (
            dx,
            dy,
            np.ones_like(dx),
            dx * dx,
            dx * dy,
            dy * dy,
        )
    )

    w = np.sqrt(
        np.maximum(
            weights,
            0.05,
        )
    )

    try:
        beta, *_ = np.linalg.lstsq(
            X * w[:, None],
            pz * w,
            rcond=None,
        )
    except np.linalg.LinAlgError:
        return None

    pred = X @ beta

    # At target center dx=dy=0, the local gradient is beta[0], beta[1].
    scale = math.sqrt(
        1.0
        + float(beta[0]) ** 2
        + float(beta[1]) ** 2
    )

    nr = (
        pz - pred
    ) / scale

    med = float(
        np.median(nr)
    )

    mad = max(
        float(
            np.median(
                np.abs(
                    nr - med
                )
            )
        ),
        float(mad_floor),
    )

    rmse = float(
        np.sqrt(
            np.mean(
                nr * nr
            )
        )
    )

    # Conservative curvature magnitude.
    curvature = max(
        abs(float(beta[3])),
        abs(float(beta[4])),
        abs(float(beta[5])),
    )

    return (
        beta,
        scale,
        mad,
        rmse,
        curvature,
    )


def _predict_quadratic(
    beta,
    x,
    y,
    cx,
    cy,
):
    dx = x - cx
    dy = y - cy

    return (
        beta[0] * dx
        + beta[1] * dy
        + beta[2]
        + beta[3] * dx * dx
        + beta[4] * dx * dy
        + beta[5] * dy * dy
    )


def _one_grid_vote(
    x,
    y,
    z,
    incoming,
    *,
    cell,
    offset_fraction,
    radius_m,
    min_cells,
    min_sectors,
    normal_tol,
    roughness_k,
    robust_z,
    mad_floor,
    scaffold_upper_sigma,
    scaffold_lower_sigma,
    quadratic_min_support,
    quadratic_trigger_rmse,
    quadratic_min_improvement,
    quadratic_max_curvature,
    coarse_veto_enabled,
    coarse_radius_m,
    coarse_exclusion_m,
    coarse_gap_m,
    coarse_min_support,
):
    """
    One grid-phase final terrain-sheet vote.

    Important safeguards:
      * target cell excluded from fit;
      * elevated neighbor cells removed by lower scaffold;
      * support must occupy multiple directions;
      * plane-normal distances are used;
      * quadratic fallback only when materially superior.
    """

    gx = x[incoming]
    gy = y[incoming]
    gz = z[incoming]

    vote = np.zeros(
        len(x),
        dtype=bool,
    )

    if len(gx) == 0:
        return vote

    ox = (
        math.floor(
            float(np.min(gx))
            / cell
        ) * cell
        - offset_fraction * cell
    )

    oy = (
        math.floor(
            float(np.min(gy))
            / cell
        ) * cell
        - offset_fraction * cell
    )

    rx, ry, rz, rn = (
        _cell_representatives(
            gx,
            gy,
            gz,
            cell,
            ox,
            oy,
        )
    )

    if len(rx) < min_cells:
        return vote

    tree = cKDTree(
        np.column_stack(
            (rx, ry)
        )
    )

    # --------------------------------------------------------
    # COARSE LOWER-CONTEXT SUPPORT
    # --------------------------------------------------------
    #
    # This is a TLS canopy/suspended-sheet veto only.
    #
    # It never creates ground and absence of coarse support
    # never removes ground.
    #
    # The coarse model is built independently for the target
    # cell from surrounding representative cells outside an
    # exclusion radius. The lower half is used only to obtain
    # a conservative surrounding terrain continuation.
    # --------------------------------------------------------

    coarse_veto_enabled = bool(
        coarse_veto_enabled
    )

    rix = np.floor(
        (rx - ox) / cell
    ).astype(np.int64)

    riy = np.floor(
        (ry - oy) / cell
    ).astype(np.int64)

    rep_key = (
        (rix << 32)
        | (riy & 0xFFFFFFFF)
    )

    lookup = {
        int(k): i
        for i, k in enumerate(
            rep_key
        )
    }

    idx = np.flatnonzero(
        incoming
    )

    pix = np.floor(
        (x[idx] - ox) / cell
    ).astype(np.int64)

    piy = np.floor(
        (y[idx] - oy) / cell
    ).astype(np.int64)

    pkey = (
        (pix << 32)
        | (piy & 0xFFFFFFFF)
    )

    # --------------------------------------------------------
    # GROUP TARGET POINTS BY CELL ONCE
    # --------------------------------------------------------
    #
    # Previous implementation used:
    #
    #     np.flatnonzero(inv == j)
    #
    # inside the per-cell loop. That repeatedly scanned the complete
    # point-to-cell array and scales poorly for large TLS clouds.
    #
    # Stable sorting makes every target cell a contiguous point range.
    # Geometry and classification mathematics are unchanged.
    # --------------------------------------------------------

    target_order = np.argsort(
        pkey,
        kind="mergesort",
    )

    sorted_pkey = pkey[
        target_order
    ]

    grouped_idx = idx[
        target_order
    ]

    uniq_target, starts = np.unique(
        sorted_pkey,
        return_index=True,
    )

    ends = np.r_[
        starts[1:],
        len(grouped_idx),
    ]

    for key, start, end in zip(
        uniq_target,
        starts,
        ends,
    ):

        pts = grouped_idx[
            start:end
        ]

        if pts.size == 0:
            continue

        cx = float(
            np.median(
                x[pts]
            )
        )
        cy = float(
            np.median(
                y[pts]
            )
        )

        nbr = tree.query_ball_point(
            [cx, cy],
            r=float(radius_m),
        )

        if not nbr:
            continue

        nbr = np.asarray(
            nbr,
            dtype=np.int64,
        )

        own_rep = lookup.get(
            int(key),
            None,
        )

        if own_rep is not None:
            nbr = nbr[
                nbr != own_rep
            ]

        if nbr.size < int(min_cells):
            continue

        dx = rx[nbr] - cx
        dy = ry[nbr] - cy

        if _sector_count(
            dx,
            dy,
        ) < int(min_sectors):
            continue

        # ----------------------------------------------------
        # LOWER COHERENT TERRAIN SCAFFOLD
        # ----------------------------------------------------

        scaffold = _lower_scaffold(
            rx[nbr],
            ry[nbr],
            rz[nbr],
            rn[nbr].astype(
                np.float64
            ),
            cx,
            cy,
            robust_z=robust_z,
            mad_floor=mad_floor,
            upper_sigma=scaffold_upper_sigma,
            lower_sigma=scaffold_lower_sigma,
            min_keep=min_cells,
        )

        if scaffold is None:
            continue

        sn = nbr[scaffold]

        if sn.size < int(min_cells):
            continue

        if _sector_count(
            rx[sn] - cx,
            ry[sn] - cy,
        ) < int(min_sectors):
            continue

        # ----------------------------------------------------
        # FINAL ROBUST PLANE ON CLEAN SCAFFOLD
        # ----------------------------------------------------

        plane = _robust_plane(
            rx[sn],
            ry[sn],
            rz[sn],
            rn[sn].astype(
                np.float64
            ),
            cx,
            cy,
            robust_z,
            mad_floor,
        )

        if plane is None:
            continue

        (
            pbeta,
            pscale,
            pmad,
            pkeep,
        ) = plane

        fn = sn[pkeep]

        if fn.size < int(min_cells):
            continue

        if _sector_count(
            rx[fn] - cx,
            ry[fn] - cy,
        ) < int(min_sectors):
            continue

        pdx = rx[fn] - cx
        pdy = ry[fn] - cy

        ppred = (
            pbeta[0] * pdx
            + pbeta[1] * pdy
            + pbeta[2]
        )

        pres = (
            rz[fn] - ppred
        ) / pscale

        plane_rmse = float(
            np.sqrt(
                np.mean(
                    pres * pres
                )
            )
        )

        use_quad = False
        qfit = None

        # ----------------------------------------------------
        # CURVATURE FALLBACK
        # ----------------------------------------------------

        if (
            fn.size
            >= int(quadratic_min_support)
            and plane_rmse
            >= float(quadratic_trigger_rmse)
        ):
            qfit = _fit_quadratic(
                rx[fn],
                ry[fn],
                rz[fn],
                rn[fn].astype(
                    np.float64
                ),
                cx,
                cy,
                mad_floor=mad_floor,
            )

            if qfit is not None:

                (
                    qbeta,
                    qscale,
                    qmad,
                    qrmse,
                    qcurv,
                ) = qfit

                improvement = (
                    plane_rmse - qrmse
                ) / max(
                    plane_rmse,
                    1e-9,
                )

                if (
                    improvement
                    >= float(
                        quadratic_min_improvement
                    )
                    and qcurv
                    <= float(
                        quadratic_max_curvature
                    )
                ):
                    use_quad = True

        # ----------------------------------------------------
        # COARSE LOWER-CONTEXT VETO
        # ----------------------------------------------------
        #
        # A locally coherent elevated sheet can otherwise
        # validate itself through neighboring canopy cells.
        #
        # Important:
        #   * target/local companions are excluded;
        #   * coarse support must be independently sufficient;
        #   * unsupported coarse geometry does NOT veto;
        #   * no absolute slope criterion is used;
        #   * this stage can only reject an otherwise valid vote.
        # ----------------------------------------------------

        coarse_veto = False

        if coarse_veto_enabled:

            coarse_nbr = np.asarray(
                tree.query_ball_point(
                    [cx, cy],
                    r=float(coarse_radius_m),
                ),
                dtype=np.int64,
            )

            if coarse_nbr.size >= int(
                coarse_min_support
            ):

                coarse_d = np.hypot(
                    rx[coarse_nbr] - cx,
                    ry[coarse_nbr] - cy,
                )

                coarse_support = coarse_nbr[
                    coarse_d
                    >= float(coarse_exclusion_m)
                ]

                if coarse_support.size >= int(
                    coarse_min_support
                ):

                    coarse_z = rz[
                        coarse_support
                    ]

                    coarse_cut = np.quantile(
                        coarse_z,
                        0.50,
                    )

                    coarse_lower = coarse_support[
                        coarse_z <= coarse_cut
                    ]

                    if coarse_lower.size >= int(
                        coarse_min_support
                    ):

                        coarse_fit = _robust_plane(
                            rx[coarse_lower],
                            ry[coarse_lower],
                            rz[coarse_lower],
                            rn[coarse_lower].astype(
                                np.float64
                            ),
                            cx,
                            cy,
                            robust_z,
                            mad_floor,
                        )

                        if coarse_fit is not None:

                            (
                                cbeta,
                                cscale,
                                cmad,
                                ckeep,
                            ) = coarse_fit

                            if np.count_nonzero(
                                ckeep
                            ) >= int(
                                coarse_min_support
                            ):

                                coarse_gap = (
                                    float(
                                        np.median(
                                            z[pts]
                                        )
                                    )
                                    - float(cbeta[2])
                                ) / float(cscale)

                                if (
                                    coarse_gap
                                    > float(coarse_gap_m)
                                ):
                                    coarse_veto = True

        # ----------------------------------------------------
        # CLASSIFY POINTS IN TARGET CELL
        # ----------------------------------------------------

        if coarse_veto:
            continue

        if use_quad:

            (
                qbeta,
                qscale,
                qmad,
                qrmse,
                qcurv,
            ) = qfit

            pred = _predict_quadratic(
                qbeta,
                x[pts],
                y[pts],
                cx,
                cy,
            )

            normal_resid = (
                z[pts] - pred
            ) / qscale

            rough = qmad

        else:

            pred = (
                pbeta[0]
                * (x[pts] - cx)
                + pbeta[1]
                * (y[pts] - cy)
                + pbeta[2]
            )

            normal_resid = (
                z[pts] - pred
            ) / pscale

            rough = pmad

        tol = (
            float(normal_tol)
            + float(roughness_k)
            * float(rough)
        )

        tol = float(
            np.clip(
                tol,
                0.035,
                0.18,
            )
        )

        ok = (
            (normal_resid <= tol)
            & (
                normal_resid
                >= -1.85 * tol
            )
        )

        vote[pts[ok]] = True

    return vote


def refine_tls_final_ground_vote(
    *,
    x,
    y,
    z,
    ground_mask,
    cfg: dict | None = None,
) -> TlsFinalGroundVoteResult:

    cfg = cfg or {}

    x = np.asarray(
        x,
        dtype=np.float64,
    )
    y = np.asarray(
        y,
        dtype=np.float64,
    )
    z = np.asarray(
        z,
        dtype=np.float64,
    )

    incoming_original = np.asarray(
        ground_mask,
        dtype=bool,
    ).copy()

    if not (
        len(x)
        == len(y)
        == len(z)
        == len(incoming_original)
    ):
        raise ValueError(
            "x, y, z and ground_mask "
            "must have equal length."
        )

    before = int(
        np.count_nonzero(
            incoming_original
        )
    )

    def unchanged(reason):
        return TlsFinalGroundVoteResult(
            final_ground=(
                incoming_original.copy()
            ),
            removed=np.zeros(
                len(incoming_original),
                dtype=bool,
            ),
            support_surface_z=np.empty(
                (0, 0),
                dtype=np.float32,
            ),
            ground_before=before,
            ground_after=before,
            demoted_points=0,
            support_points=before,
            enabled=False,
            reason=reason,
        )

    if not bool(
        cfg.get(
            "tls_final_ground_vote_enabled",
            True,
        )
    ):
        return unchanged(
            "disabled"
        )

    finite = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
    )

    incoming = (
        incoming_original
        & finite
    )

    min_support_points = int(
        cfg.get(
            "tls_final_ground_vote_min_support_points",
            100,
        )
    )

    if (
        np.count_nonzero(incoming)
        < min_support_points
    ):
        return unchanged(
            "insufficient_support"
        )

    cell = float(
        cfg.get(
            "tls_final_ground_vote_cell_m",
            0.35,
        )
    )

    radius_m = float(
        cfg.get(
            "tls_final_ground_vote_radius_m",
            2.5,
        )
    )

    min_cells = int(
        cfg.get(
            "tls_final_ground_vote_min_neighbor_cells",
            8,
        )
    )

    min_sectors = int(
        cfg.get(
            "tls_final_ground_vote_min_support_sectors",
            3,
        )
    )

    normal_tol = float(
        cfg.get(
            "tls_final_ground_vote_normal_tol_m",
            0.075,
        )
    )

    roughness_k = float(
        cfg.get(
            "tls_final_ground_vote_roughness_k",
            2.0,
        )
    )

    robust_z = float(
        cfg.get(
            "tls_final_ground_vote_max_robust_z",
            2.5,
        )
    )

    mad_floor = float(
        cfg.get(
            "tls_final_ground_vote_mad_floor_m",
            0.015,
        )
    )

    scaffold_upper_sigma = float(
        cfg.get(
            "tls_final_ground_vote_scaffold_upper_sigma",
            2.0,
        )
    )

    scaffold_lower_sigma = float(
        cfg.get(
            "tls_final_ground_vote_scaffold_lower_sigma",
            4.0,
        )
    )

    quadratic_min_support = int(
        cfg.get(
            "tls_final_ground_vote_quadratic_min_support",
            12,
        )
    )

    quadratic_trigger_rmse = float(
        cfg.get(
            "tls_final_ground_vote_quadratic_trigger_rmse_m",
            0.045,
        )
    )

    quadratic_min_improvement = float(
        cfg.get(
            "tls_final_ground_vote_quadratic_min_improvement",
            0.20,
        )
    )

    quadratic_max_curvature = float(
        cfg.get(
            "tls_final_ground_vote_quadratic_max_curvature",
            1.50,
        )
    )

    coarse_veto_enabled = bool(
        cfg.get(
            "tls_final_ground_vote_coarse_veto_enabled",
            True,
        )
    )

    coarse_radius_m = float(
        cfg.get(
            "tls_final_ground_vote_coarse_radius_m",
            5.0,
        )
    )

    coarse_exclusion_m = float(
        cfg.get(
            "tls_final_ground_vote_coarse_exclusion_m",
            2.75,
        )
    )

    coarse_gap_m = float(
        cfg.get(
            "tls_final_ground_vote_coarse_gap_m",
            2.0,
        )
    )

    coarse_min_support = int(
        cfg.get(
            "tls_final_ground_vote_coarse_min_support",
            8,
        )
    )

    phases = (
        0.0,
        0.5,
    )

    vote_count = np.zeros(
        len(x),
        dtype=np.uint8,
    )

    for phase in phases:

        v = _one_grid_vote(
            x,
            y,
            z,
            incoming,
            cell=cell,
            offset_fraction=phase,
            radius_m=radius_m,
            min_cells=min_cells,
            min_sectors=min_sectors,
            normal_tol=normal_tol,
            roughness_k=roughness_k,
            robust_z=robust_z,
            mad_floor=mad_floor,
            scaffold_upper_sigma=(
                scaffold_upper_sigma
            ),
            scaffold_lower_sigma=(
                scaffold_lower_sigma
            ),
            quadratic_min_support=(
                quadratic_min_support
            ),
            quadratic_trigger_rmse=(
                quadratic_trigger_rmse
            ),
            quadratic_min_improvement=(
                quadratic_min_improvement
            ),
            quadratic_max_curvature=(
                quadratic_max_curvature
            ),
            coarse_veto_enabled=(
                coarse_veto_enabled
            ),
            coarse_radius_m=(
                coarse_radius_m
            ),
            coarse_exclusion_m=(
                coarse_exclusion_m
            ),
            coarse_gap_m=(
                coarse_gap_m
            ),
            coarse_min_support=(
                coarse_min_support
            ),
        )

        vote_count += (
            v.astype(np.uint8)
        )

    required_votes = int(
        cfg.get(
            "tls_final_ground_vote_required_votes",
            1,
        )
    )

    required_votes = int(
        np.clip(
            required_votes,
            1,
            len(phases),
        )
    )

    supported = (
        vote_count
        >= required_votes
    )

    final_ground = (
        incoming
        & supported
    )

    # --------------------------------------------------------
    # HARD DEMOTION-ONLY CONTRACT
    # --------------------------------------------------------

    if np.any(
        final_ground
        & ~incoming_original
    ):
        raise RuntimeError(
            "TLS final vote recruited new ground."
        )

    after = int(
        np.count_nonzero(
            final_ground
        )
    )

    if after > before:
        raise RuntimeError(
            "TLS final vote increased ground count."
        )

    removed = (
        incoming_original
        & ~final_ground
    )

    return TlsFinalGroundVoteResult(
        final_ground=final_ground,
        removed=removed,
        support_surface_z=np.empty(
            (0, 0),
            dtype=np.float32,
        ),
        ground_before=before,
        ground_after=after,
        demoted_points=int(
            np.count_nonzero(
                removed
            )
        ),
        support_points=int(
            np.count_nonzero(
                incoming
            )
        ),
        enabled=True,
        reason="ok",
    )
