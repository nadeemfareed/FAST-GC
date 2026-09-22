
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, maximum_filter, label


def _bilinear_sample(
    grid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    y0: float,
    cell: float,
) -> np.ndarray:
    """Bilinear sample with nearest-cell fallback for incomplete neighborhoods."""
    ny, nx = grid.shape
    gx = (x - x0) / float(cell)
    gy = (y - y0) / float(cell)

    ix = np.floor(gx).astype(np.int32)
    iy = np.floor(gy).astype(np.int32)
    fx = gx - ix
    fy = gy - iy

    if nx <= 1 or ny <= 1:
        return grid[
            np.clip(iy, 0, max(0, ny - 1)),
            np.clip(ix, 0, max(0, nx - 1)),
        ]

    ix0 = np.clip(ix, 0, nx - 2)
    iy0 = np.clip(iy, 0, ny - 2)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    g00 = grid[iy0, ix0]
    g10 = grid[iy0, ix1]
    g01 = grid[iy1, ix0]
    g11 = grid[iy1, ix1]

    nn = grid[
        np.clip(iy, 0, ny - 1),
        np.clip(ix, 0, nx - 1),
    ]
    ok = np.isfinite(g00) & np.isfinite(g10) & np.isfinite(g01) & np.isfinite(g11)

    out = np.where(
        ok,
        (1.0 - fx) * (1.0 - fy) * g00
        + fx * (1.0 - fy) * g10
        + (1.0 - fx) * fy * g01
        + fx * fy * g11,
        nn,
    )
    return out


def _shift2(a: np.ndarray, dy: int, dx: int, fill=np.nan) -> np.ndarray:
    """Translate a 2-D array without wraparound."""
    out = np.full(a.shape, fill, dtype=a.dtype)
    ny, nx = a.shape

    sy0 = max(0, -dy)
    sy1 = min(ny, ny - dy) if dy >= 0 else ny
    sx0 = max(0, -dx)
    sx1 = min(nx, nx - dx) if dx >= 0 else nx

    dy0 = max(0, dy)
    dy1 = dy0 + max(0, sy1 - sy0)
    dx0 = max(0, dx)
    dx1 = dx0 + max(0, sx1 - sx0)

    if sy1 > sy0 and sx1 > sx0:
        out[dy0:dy1, dx0:dx1] = a[sy0:sy1, sx0:sx1]
    return out


def _point_cell_stats(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ground_mask: np.ndarray,
    *,
    cell: float,
):
    """Vectorized candidate/ground cell summaries used by recovery."""
    x0 = float(np.min(x))
    y0 = float(np.min(y))

    ix = np.floor((x - x0) / cell).astype(np.int32)
    iy = np.floor((y - y0) / cell).astype(np.int32)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1
    lin = iy.astype(np.int64) * int(nx) + ix.astype(np.int64)
    ncell = int(nx * ny)

    count = np.bincount(lin, minlength=ncell).astype(np.int32)
    zmin = np.full(ncell, np.inf, dtype=np.float64)
    zmax = np.full(ncell, -np.inf, dtype=np.float64)
    np.minimum.at(zmin, lin, z)
    np.maximum.at(zmax, lin, z)
    zmin[count == 0] = np.nan
    zmax[count == 0] = np.nan

    gcount = np.bincount(lin[ground_mask], minlength=ncell).astype(np.int32)
    gzmin = np.full(ncell, np.inf, dtype=np.float64)
    if np.any(ground_mask):
        np.minimum.at(gzmin, lin[ground_mask], z[ground_mask])
    gzmin[gcount == 0] = np.nan

    return (
        x0,
        y0,
        ix,
        iy,
        lin,
        nx,
        ny,
        count.reshape(ny, nx),
        zmin.reshape(ny, nx),
        zmax.reshape(ny, nx),
        gcount.reshape(ny, nx),
        gzmin.reshape(ny, nx),
    )


def _directional_facet_support(
    candidate_z: np.ndarray,
    ground_z: np.ndarray,
    ground_occ: np.ndarray,
    candidate_occ: np.ndarray,
    *,
    base_tol: float,
    slope_k: float,
    max_tol: float,
    min_support_dirs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Orientation-neutral terrain continuation.

    For each of 8 directions, use two already-supported ground cells on the
    same side of a candidate to extrapolate one grid step:
        z_pred = 2*z_near - z_far

    This prevents the opposite side of a ridge/valley from dominating the
    prediction.  The final decision depends only on local geometry, not on
    north/south/east/west orientation.
    """
    dirs = (
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),           ( 0, 1),
        ( 1, -1), ( 1, 0),  ( 1, 1),
    )

    support_count = np.zeros(candidate_z.shape, dtype=np.uint8)
    best_abs_resid = np.full(candidate_z.shape, np.inf, dtype=np.float64)
    best_pred = np.full(candidate_z.shape, np.nan, dtype=np.float64)

    for dy, dx in dirs:
        z1 = _shift2(ground_z, -dy, -dx, fill=np.nan)
        z2 = _shift2(ground_z, -2 * dy, -2 * dx, fill=np.nan)
        o1 = _shift2(ground_occ, -dy, -dx, fill=False)
        o2 = _shift2(ground_occ, -2 * dy, -2 * dx, fill=False)

        valid = candidate_occ & o1 & o2 & np.isfinite(z1) & np.isfinite(z2)
        if not np.any(valid):
            continue

        pred = 2.0 * z1 - z2
        local_step = np.abs(z1 - z2)
        tol = np.minimum(
            float(max_tol),
            float(base_tol) + float(slope_k) * local_step,
        )
        resid = np.abs(candidate_z - pred)
        ok = valid & np.isfinite(candidate_z) & (resid <= tol)

        support_count[ok] += 1
        better = ok & (resid < best_abs_resid)
        best_abs_resid[better] = resid[better]
        best_pred[better] = pred[better]

    supported = support_count >= int(max(1, min_support_dirs))
    return supported, support_count, best_pred



def _planar_neighbor_count(zmin: np.ndarray, occ: np.ndarray, tol: float) -> np.ndarray:
    """Count occupied 8-neighbors at nearly the same elevation as each cell."""
    count = np.zeros(zmin.shape, dtype=np.uint8)
    for dy, dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
        zn = _shift2(zmin, -dy, -dx, fill=np.nan)
        on = _shift2(occ, -dy, -dx, fill=False)
        same = occ & on & np.isfinite(zmin) & np.isfinite(zn) & (np.abs(zmin - zn) <= float(tol))
        count[same] += 1
    return count


def _elevated_planar_island_veto(
    zmin: np.ndarray,
    candidate_occ: np.ndarray,
    original_ground_z: np.ndarray,
    original_ground_occ: np.ndarray,
    *,
    planar_tol: float,
    planar_min_neighbors: int,
    height_thr: float,
    radius_cells: int,
) -> np.ndarray:
    """Reject roof-like flat islands elevated above all nearby trusted ground.

    This is deliberately a veto on *recovery only*.  Existing FAST-GC ground
    is never demoted.  A cell is vetoed only when it is locally planar and no
    original trusted ground of comparable elevation exists nearby.
    """
    r = max(1, int(radius_cells))
    neg_inf = np.float64(-np.inf)
    trusted = np.where(original_ground_occ & np.isfinite(original_ground_z), original_ground_z, neg_inf)
    local_highest_ground = maximum_filter(trusted, size=(2*r+1, 2*r+1), mode='constant', cval=neg_inf)
    planar_n = _planar_neighbor_count(zmin, candidate_occ, float(planar_tol))
    no_comparable_ground = np.isfinite(zmin) & np.isfinite(local_highest_ground) & ((zmin - local_highest_ground) > float(height_thr))
    planar_patch = planar_n >= int(max(1, planar_min_neighbors))
    return candidate_occ & planar_patch & no_comparable_ground


def _canopy_minority_veto(
    zmin: np.ndarray,
    count: np.ndarray,
    original_ground_z: np.ndarray,
    original_ground_occ: np.ndarray,
    *,
    height_gap_m: float,
    radius_cells: int,
    min_local_points: int,
) -> np.ndarray:
    """Reject elevated non-planar islands likely belonging to canopy.

    Recovery is allowed to use only the compact bottom layer of a candidate
    cell.  A sparse branch/crown patch can therefore masquerade as a terrain
    continuation when the cell itself has a small vertical span.  This guard
    asks a stricter question: is the candidate low layer still substantially
    above the highest ORIGINAL trusted ground nearby, and is it embedded in a
    point-rich local neighborhood?  If so, treat it as canopy/object support,
    not terrain.

    Existing FAST-GC ground is never modified, and recovered cells never become
    trusted evidence for this test.
    """
    r = max(1, int(radius_cells))
    neg_inf = np.float64(-np.inf)
    trusted = np.where(
        original_ground_occ & np.isfinite(original_ground_z),
        original_ground_z,
        neg_inf,
    )
    local_highest_ground = maximum_filter(
        trusted, size=(2 * r + 1, 2 * r + 1), mode='constant', cval=neg_inf
    )
    local_points = maximum_filter(
        np.asarray(count, dtype=np.int32),
        size=(2 * r + 1, 2 * r + 1),
        mode='constant',
        cval=0,
    )
    elevated = (
        np.isfinite(zmin)
        & np.isfinite(local_highest_ground)
        & ((zmin - local_highest_ground) > float(height_gap_m))
    )
    embedded = local_points >= int(max(1, min_local_points))
    return elevated & embedded


def _large_hole_edge_mask(
    original_ground_occ: np.ndarray,
    candidate_occ: np.ndarray,
    *,
    min_hole_cells: int,
    edge_width_cells: int,
) -> np.ndarray:
    """Return a narrow band around large original-ground holes.

    The band is not an automatic veto.  It simply marks locations where the
    recovery pass must demand stronger directional support so vegetation/object
    edges do not get pulled into ground while genuine terrain gaps can still be
    recovered when the facet evidence is strong.
    """
    hole = np.asarray(candidate_occ, dtype=bool) & (~np.asarray(original_ground_occ, dtype=bool))
    if not np.any(hole):
        return np.zeros_like(hole, dtype=bool)

    labs, nlab = label(hole, structure=np.ones((3, 3), dtype=np.uint8))
    if nlab <= 0:
        return np.zeros_like(hole, dtype=bool)

    sizes = np.bincount(labs.ravel())
    keep = np.zeros(nlab + 1, dtype=bool)
    if sizes.size > 1:
        keep[1:] = sizes[1:] >= int(max(1, min_hole_cells))
    large = keep[labs]
    if not np.any(large):
        return np.zeros_like(hole, dtype=bool)

    w = max(1, int(edge_width_cells))
    outer = binary_dilation(large, structure=np.ones((3, 3), dtype=bool), iterations=w)
    inner = large & (~binary_dilation(~large, structure=np.ones((3, 3), dtype=bool), iterations=w))
    return outer & (~inner)


def recover_ground_in_voids(
    xw: np.ndarray,
    yw: np.ndarray,
    zw: np.ndarray,
    ground_mask: np.ndarray,
    surf_z: np.ndarray,
    sx0: float,
    sy0: float,
    surf_cell: float,
    sensor_mode: str,
    cfg: dict,
) -> np.ndarray:
    """
    Conservative ground-supported facet recovery for ALS / ULS.

    The primary FAST-GC classifier remains authoritative. This pass only
    promotes currently non-ground candidates when existing ground supplies a
    coherent same-facet continuation. It is designed to repair false
    non-ground ribbons near ridges, valleys, and slope-transition zones.

    Safety properties:
    - ALS/ULS only; TLS is unchanged.
    - Existing ground is never demoted.
    - No global threshold relaxation.
    - Candidate cells must contain a compact low layer.
    - Elevated planar islands without comparable-height trusted ground are vetoed.
    - Elevated canopy/object islands are vetoed using original-ground evidence.
    - Large-hole edges require stronger multi-direction support.
    - Recovery requires directional support from existing ground.
    - Propagation is bounded to a small number of iterations/cells.
    """
    sm = str(sensor_mode).upper().strip()
    out_mask = np.asarray(ground_mask, dtype=bool).copy()

    if sm not in {"ALS", "ULS"}:
        return out_mask
    if xw.size == 0 or out_mask.size != xw.size:
        return out_mask
    if np.count_nonzero(out_mask) < 10:
        return out_mask
    if not bool(cfg.get("void_recover_enabled", True)):
        return out_mask

    # Use the native vote-grid scale by default; this keeps the recovery local.
    cell_default = max(float(surf_cell), 0.50 if sm == "ALS" else 0.35)
    cell = float(cfg.get("void_recover_cell_m", cell_default))

    # Conservative compact-low-layer gate. Tree/shrub cells are vertically
    # diverse and therefore fail this test before directional propagation.
    max_cell_span = float(cfg.get(
        "void_recover_z_span_thr_m",
        0.42 if sm == "ALS" else 0.30,
    ))
    low_cluster_dz = float(cfg.get(
        "void_recover_promote_cluster_dz_m",
        0.10 if sm == "ALS" else 0.07,
    ))
    min_points = int(cfg.get("void_recover_min_void_points", 1))

    # Same-facet directional continuation controls.
    base_tol = float(cfg.get(
        "void_recover_facet_base_tol_m",
        0.16 if sm == "ALS" else 0.12,
    ))
    slope_k = float(cfg.get(
        "void_recover_facet_slope_k",
        0.35 if sm == "ALS" else 0.30,
    ))
    max_tol = float(cfg.get(
        "void_recover_facet_max_tol_m",
        0.42 if sm == "ALS" else 0.30,
    ))
    min_support_dirs = int(cfg.get("void_recover_min_support_dirs", 2))
    max_iters = int(cfg.get(
        "void_recover_propagation_iters",
        3 if sm == "ALS" else 2,
    ))

    # Do not let recovery walk arbitrarily far from trusted ground.
    max_seed_distance_cells = int(cfg.get("void_recover_max_seed_distance_cells", 3))

    # Roof/building protection.  Flat elevated patches can look like excellent
    # terrain facets geometrically, so recovery must also remain attached to
    # trusted ground at a comparable elevation.  This veto is applied only to
    # newly recovered cells; primary FAST-GC ground is untouched.
    roof_planar_tol = float(cfg.get(
        "void_recover_roof_planar_tol_m",
        0.12 if sm == "ALS" else 0.09,
    ))
    roof_planar_min_neighbors = int(cfg.get(
        "void_recover_roof_planar_min_neighbors",
        4,
    ))
    roof_height_thr = float(cfg.get(
        "void_recover_roof_elevation_gap_m",
        0.70 if sm == "ALS" else 0.50,
    ))
    roof_radius_cells = int(cfg.get(
        "void_recover_roof_guard_radius_cells",
        2,
    ))

    # Canopy/object minority guard.  Unlike the planar roof veto, this catches
    # sparse elevated branch/crown islands that can locally look like a compact
    # slope continuation.  It remains anchored to ORIGINAL trusted ground.
    canopy_height_gap = float(cfg.get(
        "void_recover_canopy_elevation_gap_m",
        0.55 if sm == "ALS" else 0.40,
    ))
    canopy_radius_cells = int(cfg.get(
        "void_recover_canopy_guard_radius_cells",
        2,
    ))
    canopy_min_local_points = int(cfg.get(
        "void_recover_canopy_min_local_points",
        4,
    ))

    # Large-hole edge protection.  At the perimeter of broad vegetation/object
    # holes, require more independent directional votes instead of freely
    # propagating a minority of non-ground points into the terrain class.
    hole_min_cells = int(cfg.get(
        "void_recover_large_hole_min_cells",
        12 if sm == "ALS" else 16,
    ))
    hole_edge_width = int(cfg.get(
        "void_recover_large_hole_edge_width_cells",
        1,
    ))
    hole_edge_extra_dirs = int(cfg.get(
        "void_recover_large_hole_extra_support_dirs",
        1,
    ))

    (
        x0,
        y0,
        ix,
        iy,
        lin,
        nx,
        ny,
        count,
        zmin,
        zmax,
        ground_count,
        ground_z,
    ) = _point_cell_stats(
        np.asarray(xw, dtype=np.float64),
        np.asarray(yw, dtype=np.float64),
        np.asarray(zw, dtype=np.float64),
        out_mask,
        cell=cell,
    )

    cand_occ = count >= int(max(1, min_points))
    compact = cand_occ & np.isfinite(zmin) & np.isfinite(zmax)
    compact &= (zmax - zmin) <= max_cell_span

    # Seed-distance mask is based on ORIGINAL trusted ground only.
    seed_occ = ground_count > 0
    original_ground_z = ground_z.copy()
    original_ground_occ = seed_occ.copy()

    # Compute once from ORIGINAL trusted ground.  Recovered cells never become
    # evidence that can legitimize an elevated planar roof in later iterations.
    roof_veto = _elevated_planar_island_veto(
        zmin,
        cand_occ,
        original_ground_z,
        original_ground_occ,
        planar_tol=roof_planar_tol,
        planar_min_neighbors=roof_planar_min_neighbors,
        height_thr=roof_height_thr,
        radius_cells=roof_radius_cells,
    )

    canopy_veto = _canopy_minority_veto(
        zmin,
        count,
        original_ground_z,
        original_ground_occ,
        height_gap_m=canopy_height_gap,
        radius_cells=canopy_radius_cells,
        min_local_points=canopy_min_local_points,
    )

    large_hole_edge = _large_hole_edge_mask(
        original_ground_occ,
        cand_occ,
        min_hole_cells=hole_min_cells,
        edge_width_cells=hole_edge_width,
    )

    near_seed = seed_occ.copy()
    if max_seed_distance_cells > 0:
        near_seed = binary_dilation(
            seed_occ,
            structure=np.ones((3, 3), dtype=bool),
            iterations=max_seed_distance_cells,
        )

    # Optional sanity anchor to the original vote surface. This is deliberately
    # loose; directional ground evidence is the main decision-maker.
    cx = x0 + (np.arange(nx, dtype=np.float64) + 0.5) * cell
    cy = y0 + (np.arange(ny, dtype=np.float64) + 0.5) * cell
    gx, gy = np.meshgrid(cx, cy)
    surf_anchor = _bilinear_sample(
        np.asarray(surf_z, dtype=np.float64),
        gx.ravel(),
        gy.ravel(),
        float(sx0),
        float(sy0),
        float(surf_cell),
    ).reshape(ny, nx)
    surface_guard = float(cfg.get(
        "void_recover_surface_guard_m",
        0.65 if sm == "ALS" else 0.45,
    ))

    recovered_cells = np.zeros((ny, nx), dtype=bool)

    for _ in range(max(0, max_iters)):
        # Refresh ground support from the growing mask, but preserve the
        # original near-seed distance envelope.
        (
            _x0,
            _y0,
            _ix,
            _iy,
            _lin,
            _nx,
            _ny,
            _count,
            _zmin,
            _zmax,
            ground_count,
            ground_z,
        ) = _point_cell_stats(
            np.asarray(xw, dtype=np.float64),
            np.asarray(yw, dtype=np.float64),
            np.asarray(zw, dtype=np.float64),
            out_mask,
            cell=cell,
        )

        ground_occ = ground_count > 0
        target = compact & (~ground_occ) & near_seed
        if not np.any(target):
            break

        supported, support_count, best_pred = _directional_facet_support(
            zmin,
            ground_z,
            ground_occ,
            target,
            base_tol=base_tol,
            slope_k=slope_k,
            max_tol=max_tol,
            min_support_dirs=min_support_dirs,
        )

        # Loose original-surface guard to prevent propagation to unrelated
        # elevated structures while still allowing breakline correction.
        surf_ok = ~np.isfinite(surf_anchor) | (
            np.abs(zmin - surf_anchor) <= surface_guard
        )
        accept = supported & surf_ok & target
        accept &= ~roof_veto
        accept &= ~canopy_veto

        # Preserve the edge of large original-ground holes by requiring one
        # additional independent same-facet direction there.  This is a
        # stronger-evidence rule, not a hard edge veto.
        if hole_edge_extra_dirs > 0:
            edge_need = int(max(1, min_support_dirs + hole_edge_extra_dirs))
            accept &= (~large_hole_edge) | (support_count >= edge_need)

        accept &= ~recovered_cells

        if not np.any(accept):
            break

        accepted_lin = np.flatnonzero(accept.ravel())
        accept_lookup = np.zeros(nx * ny, dtype=bool)
        accept_lookup[accepted_lin] = True
        point_accept_cell = accept_lookup[lin]

        # Promote only the compact bottom cluster in accepted cells.
        cell_low = zmin.ravel()[lin]
        promote = point_accept_cell & (~out_mask)
        promote &= np.asarray(zw, dtype=np.float64) <= (cell_low + low_cluster_dz)

        if not np.any(promote):
            break

        out_mask[promote] = True
        recovered_cells |= accept

    return out_mask


__all__ = ["recover_ground_in_voids"]
