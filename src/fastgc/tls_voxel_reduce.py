"""Conservative TLS terrain-domain voxel reduction.

This is a computational prefilter, NOT a ground classifier.

The reducer identifies only points that are safely detached above a
locally supported lower terrain manifold.  Ambiguous points are always
retained for the full FAST-GC TLS classifier.

V2 principles
-------------
1. Fine XY cells provide low occupied observations.
2. Surrounding low observations describe a local 3-D terrain manifold.
3. A robust local tangent plane is fitted without imposing an absolute
   terrain-slope limit.
4. The center observation is excluded from its own spatial validation.
5. Only well-supported positive departures can establish a lower
   terrain envelope beneath an apparently elevated cell.
6. Height above that envelope is quantized into vertical voxel bands.
7. Only complete bands safely above the terrain corridor are deferred.

Thus steep terrain is allowed.  What matters is detachment from the
locally coherent lower manifold, not absolute elevation or slope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TlsVoxelReduceResult:
    keep_mask: np.ndarray
    deferred_mask: np.ndarray
    envelope_z: np.ndarray
    supported_mask: np.ndarray
    input_points: int
    kept_points: int
    deferred_points: int


def _grid_low_quantile(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cell: float,
    q: float,
):
    """Build low occupied observation grid."""

    x0 = np.floor(np.min(x) / cell) * cell
    y0 = np.floor(np.min(y) / cell) * cell

    ix = np.floor((x - x0) / cell).astype(np.int64)
    iy = np.floor((y - y0) / cell).astype(np.int64)

    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1

    key = iy * nx + ix
    order = np.argsort(key, kind="mergesort")

    keys = key[order]
    zs = z[order]

    uniq, starts = np.unique(keys, return_index=True)
    ends = np.r_[starts[1:], zs.size]

    low = np.full((ny, nx), np.nan, dtype=np.float64)
    count = np.zeros((ny, nx), dtype=np.int32)

    for k, a, b in zip(uniq, starts, ends):
        gy = int(k // nx)
        gx = int(k % nx)

        vals = zs[a:b]
        count[gy, gx] = vals.size

        if vals.size:
            low[gy, gx] = float(
                np.quantile(vals, q)
            )

    return low, count, x0, y0, ix, iy


def _robust_plane_predict(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    xc: float,
    yc: float,
):
    """Robust local tangent-plane prediction.

    No coefficient/slope limit is imposed.

    Returns
    -------
    pred : float
        Predicted terrain Z at center.
    rmse : float
        Robust residual scale after fitting.
    n_used : int
        Number of observations retained by robust fitting.
    sectors : int
        Number of occupied directional sectors around center.
    """

    if zs.size < 3:
        return np.nan, np.inf, 0, 0

    dx = xs - xc
    dy = ys - yc

    # Directional support prevents a one-sided row of observations
    # from defining terrain beneath an unsupported cell.
    angle = np.arctan2(dy, dx)
    sector = np.floor(
        (angle + np.pi) / (0.25 * np.pi)
    ).astype(np.int64)

    sector = np.clip(sector, 0, 7)
    sectors = int(np.unique(sector).size)

    A = np.column_stack(
        (
            np.ones(zs.size, dtype=np.float64),
            dx,
            dy,
        )
    )

    mask = np.ones(zs.size, dtype=bool)

    coef = None

    # Small fixed robust loop; this is per occupied CELL, not point.
    for _ in range(3):
        if np.count_nonzero(mask) < 3:
            break

        try:
            coef, *_ = np.linalg.lstsq(
                A[mask],
                zs[mask],
                rcond=None,
            )
        except np.linalg.LinAlgError:
            return np.nan, np.inf, 0, sectors

        resid = zs - A @ coef

        rr = resid[mask]
        med = float(np.median(rr))
        mad = float(
            1.4826
            * np.median(np.abs(rr - med))
        )

        scale = max(mad, 0.03)

        new_mask = np.abs(resid - med) <= 3.0 * scale

        if np.array_equal(new_mask, mask):
            mask = new_mask
            break

        mask = new_mask

    if coef is None or np.count_nonzero(mask) < 3:
        return np.nan, np.inf, 0, sectors

    # Refit after robust rejection.
    try:
        coef, *_ = np.linalg.lstsq(
            A[mask],
            zs[mask],
            rcond=None,
        )
    except np.linalg.LinAlgError:
        return np.nan, np.inf, 0, sectors

    resid = zs[mask] - A[mask] @ coef

    rmse = float(
        np.sqrt(np.mean(resid * resid))
    )

    # Because coordinates are centered on the target cell,
    # the intercept is the prediction at that cell.
    pred = float(coef[0])

    return (
        pred,
        rmse,
        int(np.count_nonzero(mask)),
        sectors,
    )


def _build_local_manifold(
    low: np.ndarray,
    count: np.ndarray,
    *,
    cell: float,
    radius: int,
    min_cells: int,
    min_points_per_cell: int,
    min_sectors: int,
    max_roughness_m: float,
):
    """Build conservative lower terrain manifold.

    A cell is assigned a predicted terrain level only when neighboring
    lower occupied cells surround it sufficiently and fit a coherent
    local tangent plane.

    The target cell itself is excluded from the fit.  Consequently a
    dense vegetation cell cannot establish its own terrain reference.

    Absolute plane slope is deliberately unrestricted.
    """

    ny, nx = low.shape

    manifold = np.full(
        low.shape,
        np.nan,
        dtype=np.float64,
    )

    supported = np.zeros(
        low.shape,
        dtype=bool,
    )

    roughness = np.full(
        low.shape,
        np.nan,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(low)
        & (count >= int(min_points_per_cell))
    )

    targets = np.argwhere(valid)

    for gy, gx in targets:
        gy = int(gy)
        gx = int(gx)

        ya = max(0, gy - radius)
        yb = min(ny, gy + radius + 1)
        xa = max(0, gx - radius)
        xb = min(nx, gx + radius + 1)

        block_valid = valid[ya:yb, xa:xb].copy()

        cy = gy - ya
        cx = gx - xa

        # Center cannot validate itself.
        block_valid[cy, cx] = False

        by, bx = np.nonzero(block_valid)

        if by.size < int(min_cells):
            continue

        yy = by + ya
        xx = bx + xa

        zs = low[yy, xx]

        xc = (gx + 0.5) * cell
        yc = (gy + 0.5) * cell

        xs = (xx.astype(np.float64) + 0.5) * cell
        ys = (yy.astype(np.float64) + 0.5) * cell

        pred, rmse, n_used, sectors = _robust_plane_predict(
            xs,
            ys,
            zs,
            xc,
            yc,
        )

        if not np.isfinite(pred):
            continue

        if n_used < int(min_cells):
            continue

        if sectors < int(min_sectors):
            continue

        # If the lower observations do not describe a coherent local
        # manifold, do nothing.  Ambiguity means KEEP.
        if (
            not np.isfinite(rmse)
            or rmse > float(max_roughness_m)
        ):
            continue

        manifold[gy, gx] = pred
        roughness[gy, gx] = rmse
        supported[gy, gx] = True

    return manifold, supported, roughness


def reduce_tls_terrain_domain(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    fine_cell_m: float = 0.50,
    support_window_m: float = 5.0,
    low_quantile: float = 0.05,
    min_points_per_cell: int = 3,
    min_support_cells: int = 5,
    vertical_voxel_m: float = 0.25,
    safe_height_m: float = 2.0,
    min_support_sectors: int = 3,
    max_manifold_roughness_m: float = 0.35,
) -> TlsVoxelReduceResult:
    """Return conservative TLS terrain-processing mask.

    The reducer distinguishes two cases.

    A cell whose own lower observation agrees with the neighboring
    terrain manifold uses its own low observation as the envelope.
    This strongly protects real terrain.

    A cell whose entire observed column begins substantially above a
    coherent surrounding manifold may use the neighboring prediction
    as its terrain envelope.  This allows vegetation-only columns to
    be reduced.

    No absolute terrain-slope threshold is used.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    if not (x.size == y.size == z.size):
        raise ValueError(
            "x, y and z must have equal length"
        )

    n = int(z.size)

    if n == 0:
        eb = np.zeros(0, dtype=bool)
        ef = np.zeros(0, dtype=np.float64)

        return TlsVoxelReduceResult(
            keep_mask=eb,
            deferred_mask=eb.copy(),
            envelope_z=ef,
            supported_mask=eb.copy(),
            input_points=0,
            kept_points=0,
            deferred_points=0,
        )

    # Safety policy: everything starts as KEEP.
    keep = np.ones(n, dtype=bool)
    deferred = np.zeros(n, dtype=bool)

    point_env = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    point_supported = np.zeros(
        n,
        dtype=bool,
    )

    finite = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
    )

    ids = np.flatnonzero(finite)

    if ids.size == 0:
        return TlsVoxelReduceResult(
            keep_mask=keep,
            deferred_mask=deferred,
            envelope_z=point_env,
            supported_mask=point_supported,
            input_points=n,
            kept_points=n,
            deferred_points=0,
        )

    xf = x[ids]
    yf = y[ids]
    zf = z[ids]

    cell = max(
        float(fine_cell_m),
        0.05,
    )

    voxel_h = max(
        float(vertical_voxel_m),
        0.05,
    )

    safe_h = max(
        float(safe_height_m),
        voxel_h,
    )

    low, count, _, _, ix, iy = _grid_low_quantile(
        xf,
        yf,
        zf,
        cell,
        float(
            np.clip(
                low_quantile,
                0.0,
                0.25,
            )
        ),
    )

    radius = max(
        1,
        int(
            round(
                0.5
                * float(support_window_m)
                / cell
            )
        ),
    )

    manifold, support, rough = _build_local_manifold(
        low,
        count,
        cell=cell,
        radius=radius,
        min_cells=max(
            3,
            int(min_support_cells),
        ),
        min_points_per_cell=max(
            1,
            int(min_points_per_cell),
        ),
        min_sectors=max(
            2,
            int(min_support_sectors),
        ),
        max_roughness_m=max(
            0.05,
            float(max_manifold_roughness_m),
        ),
    )

    local_low = low[iy, ix]
    pred = manifold[iy, ix]
    supp = support[iy, ix] & np.isfinite(pred)
    local_rough = rough[iy, ix]

    # --------------------------------------------------------
    # Terrain corridor.
    #
    # If a cell's lowest observation is close to the predicted
    # manifold, trust the actual local low observation.
    #
    # If the whole cell starts well above a coherent neighboring
    # manifold, the neighboring prediction may expose a detached
    # vegetation-only column.
    #
    # Rough terrain receives extra safety margin.
    # --------------------------------------------------------

    agreement_margin = (
        2.0 * voxel_h
        + np.nan_to_num(
            local_rough,
            nan=0.0,
        )
    )

    local_agrees = (
        supp
        & np.isfinite(local_low)
        & (
            np.abs(local_low - pred)
            <= agreement_margin
        )
    )

    elevated_column = (
        supp
        & np.isfinite(local_low)
        & (
            local_low - pred
            > safe_h
        )
    )

    # Default unsupported cells remain NaN => KEEP.
    env = np.full(
        zf.shape,
        np.nan,
        dtype=np.float64,
    )

    # Real/near-real terrain cell:
    # use its observed lower support.
    env[local_agrees] = local_low[local_agrees]

    # Strongly detached entire column:
    # use surrounding terrain prediction.
    env[elevated_column] = pred[elevated_column]

    usable = np.isfinite(env)

    point_env[ids] = env
    point_supported[ids] = usable

    dz = zf - env

    # --------------------------------------------------------
    # Vertical voxel corridor.
    #
    # The first safe_height_m above terrain is retained.
    # Only complete vertical bands above it are deferred.
    # --------------------------------------------------------

    # Compute voxel bands only where a valid terrain envelope exists.
    # Unsupported cells intentionally remain KEEP and never need an
    # integer band value.
    band = np.zeros(
        dz.shape,
        dtype=np.int64,
    )

    band_valid = usable & np.isfinite(dz)

    band[band_valid] = np.floor(
        np.maximum(dz[band_valid], 0.0)
        / voxel_h
    ).astype(np.int64)

    first_defer_band = int(
        np.ceil(
            safe_h / voxel_h
        )
    )

    defer_finite = (
        usable
        & np.isfinite(dz)
        & (dz > 0.0)
        & (band >= first_defer_band)
    )

    deferred[
        ids[defer_finite]
    ] = True

    keep[deferred] = False

    return TlsVoxelReduceResult(
        keep_mask=keep,
        deferred_mask=deferred,
        envelope_z=point_env,
        supported_mask=point_supported,
        input_points=n,
        kept_points=int(
            np.count_nonzero(keep)
        ),
        deferred_points=int(
            np.count_nonzero(deferred)
        ),
    )


@dataclass(frozen=True)
class TlsEvidenceSampleResult:
    """Density-normalized TLS terrain-evidence selection.

    This is deliberately separate from TlsVoxelReduceResult.

    domain_mask
        Points still considered potentially relevant after the
        conservative terrain-domain reducer.

    active_mask
        Deterministic representative points used as terrain evidence.

    redundant_mask
        Domain points not needed for terrain construction because their
        local 3-D voxel already contains sufficient representative
        evidence.

    Redundant points are NOT classified as vegetation and are NOT
    removed from the original point cloud.
    """

    active_mask: np.ndarray
    redundant_mask: np.ndarray
    domain_mask: np.ndarray

    input_points: int
    domain_points: int
    active_points: int
    redundant_points: int

    occupied_voxels: int
    dense_voxels: int


def select_tls_active_evidence(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    domain_mask: np.ndarray,
    *,
    xy_voxel_m: float = 0.20,
    z_voxel_m: float = 0.10,
    max_points_per_voxel: int = 8,
) -> TlsEvidenceSampleResult:
    """Return deterministic density-normalized TLS evidence.

    V3 is intentionally scanner-independent.  It removes sampling
    redundancy, not terrain geometry.

    Sparse occupied voxels retain every point.  Dense occupied voxels
    retain a deterministic set spanning their vertical ordering.

    The implementation sorts the domain points once by 3-D voxel and Z.
    There is no Python loop over points or voxels.

    No slope test is used.
    No range-from-scanner test is used.
    No random sampling is used.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    domain = np.asarray(domain_mask, dtype=bool)

    if not (
        x.size == y.size == z.size == domain.size
    ):
        raise ValueError(
            "x, y, z and domain_mask must have equal length"
        )

    n = int(z.size)

    active = np.zeros(n, dtype=bool)
    redundant = np.zeros(n, dtype=bool)

    finite = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
    )

    eligible = domain & finite
    ids = np.flatnonzero(eligible)

    if ids.size == 0:
        return TlsEvidenceSampleResult(
            active_mask=active,
            redundant_mask=redundant,
            domain_mask=domain.copy(),
            input_points=n,
            domain_points=int(np.count_nonzero(domain)),
            active_points=0,
            redundant_points=0,
            occupied_voxels=0,
            dense_voxels=0,
        )

    xyh = max(float(xy_voxel_m), 0.02)
    zh = max(float(z_voxel_m), 0.02)
    cap = max(int(max_points_per_voxel), 1)

    xe = x[ids]
    ye = y[ids]
    ze = z[ids]

    # Local origins keep integer voxel coordinates compact.
    x0 = float(np.min(xe))
    y0 = float(np.min(ye))
    z0 = float(np.min(ze))

    ix = np.floor((xe - x0) / xyh).astype(np.int64)
    iy = np.floor((ye - y0) / xyh).astype(np.int64)
    iz = np.floor((ze - z0) / zh).astype(np.int64)

    # Lexicographic sort avoids constructing a potentially overflowing
    # packed integer voxel key.  Primary order is voxel XYZ; Z is also
    # used as the deterministic within-voxel ordering.
    order = np.lexsort(
        (
            ids,
            ze,
            iz,
            iy,
            ix,
        )
    )

    six = ix[order]
    siy = iy[order]
    siz = iz[order]
    sz = ze[order]
    sids = ids[order]

    new_group = np.empty(sids.size, dtype=bool)
    new_group[0] = True

    if sids.size > 1:
        new_group[1:] = (
            (six[1:] != six[:-1])
            | (siy[1:] != siy[:-1])
            | (siz[1:] != siz[:-1])
        )

    starts = np.flatnonzero(new_group)
    ends = np.r_[starts[1:], sids.size]
    counts = ends - starts

    occupied_voxels = int(starts.size)
    dense_voxels = int(np.count_nonzero(counts > cap))

    # Position of every sorted point within its voxel.
    group_id = np.cumsum(new_group) - 1
    rank = np.arange(
        sids.size,
        dtype=np.int64,
    ) - starts[group_id]

    group_count = counts[group_id]

    # Sparse voxel:
    #     keep every point.
    #
    # Dense voxel:
    #     retain approximately cap evenly distributed order statistics
    #     through the voxel's vertical ordering.  This includes the
    #     lower and upper ends rather than simply taking the first cap.
    #
    # For a group of n points and cap c, selected ranks are equivalent
    # to round(linspace(0, n-1, c)), implemented vectorially.
    sparse = group_count <= cap

    if cap == 1:
        selected_dense = rank == 0
    else:
        target = np.rint(
            rank.astype(np.float64)
            * float(cap - 1)
            / np.maximum(
                group_count - 1,
                1,
            ).astype(np.float64)
        ).astype(np.int64)

        # Select the first observation assigned to each target slot.
        prev_target = np.empty_like(target)
        prev_target[0] = -1

        if target.size > 1:
            prev_target[1:] = target[:-1]

        selected_dense = (
            (rank == 0)
            | new_group
            | (target != prev_target)
        )

    selected = sparse | selected_dense

    active_ids = sids[selected]
    active[active_ids] = True

    # Non-selected points are merely redundant terrain evidence.
    # They remain in the original cloud and remain eligible for final
    # classification once the terrain surface has been estimated.
    redundant = eligible & ~active

    return TlsEvidenceSampleResult(
        active_mask=active,
        redundant_mask=redundant,
        domain_mask=domain.copy(),
        input_points=n,
        domain_points=int(np.count_nonzero(domain)),
        active_points=int(np.count_nonzero(active)),
        redundant_points=int(np.count_nonzero(redundant)),
        occupied_voxels=occupied_voxels,
        dense_voxels=dense_voxels,
    )

