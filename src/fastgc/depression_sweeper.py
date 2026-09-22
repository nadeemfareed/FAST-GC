from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label


@dataclass(frozen=True)
class DepressionSweepReport:
    candidate_cells: int = 0
    candidate_components: int = 0
    validated_components: int = 0
    recovered_points: int = 0
    rejected_components: int = 0
    basin_components: int = 0
    linear_components: int = 0
    nested_pass_recoveries: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "candidate_cells": int(self.candidate_cells),
            "candidate_components": int(self.candidate_components),
            "validated_components": int(self.validated_components),
            "recovered_points": int(self.recovered_points),
            "rejected_components": int(self.rejected_components),
            "basin_components": int(self.basin_components),
            "linear_components": int(self.linear_components),
            "nested_pass_recoveries": int(self.nested_pass_recoveries),
        }


def _grid_stats(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ground_mask: np.ndarray,
    *,
    cell: float,
    low_cluster_dz: float,
) -> dict[str, Any]:
    x0 = float(np.min(x))
    y0 = float(np.min(y))
    ix = np.floor((x - x0) / cell).astype(np.int32)
    iy = np.floor((y - y0) / cell).astype(np.int32)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1
    lin = iy.astype(np.int64) * nx + ix.astype(np.int64)
    ncell = nx * ny

    count = np.bincount(lin, minlength=ncell).astype(np.int32)
    gcount = np.bincount(lin[ground_mask], minlength=ncell).astype(np.int32)
    ngmask = ~ground_mask
    ngcount = np.bincount(lin[ngmask], minlength=ncell).astype(np.int32)

    gz = np.full(ncell, np.inf, dtype=np.float64)
    if np.any(ground_mask):
        np.minimum.at(gz, lin[ground_mask], z[ground_mask])
    gz[gcount == 0] = np.nan

    ngzmin = np.full(ncell, np.inf, dtype=np.float64)
    if np.any(ngmask):
        np.minimum.at(ngzmin, lin[ngmask], z[ngmask])
    ngzmin[ngcount == 0] = np.nan

    low_count = np.zeros(ncell, dtype=np.int32)
    if np.any(ngmask):
        ng_lin = lin[ngmask]
        ng_z = z[ngmask]
        low = ng_z <= (ngzmin[ng_lin] + float(low_cluster_dz))
        if np.any(low):
            low_count = np.bincount(ng_lin[low], minlength=ncell).astype(np.int32)

    return {
        "x0": x0,
        "y0": y0,
        "ix": ix,
        "iy": iy,
        "lin": lin,
        "nx": nx,
        "ny": ny,
        "count": count.reshape(ny, nx),
        "gcount": gcount.reshape(ny, nx),
        "ngcount": ngcount.reshape(ny, nx),
        "ground_z": gz.reshape(ny, nx),
        "ng_zmin": ngzmin.reshape(ny, nx),
        "low_count": low_count.reshape(ny, nx),
    }



def _shift_sample(arr: np.ndarray, dy: int, dx: int, distance: int) -> np.ndarray:
    """Sample ``arr`` at an offset without wraparound.

    Output[y, x] = arr[y + dy*distance, x + dx*distance].
    """
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    sy = int(dy) * int(distance)
    sx = int(dx) * int(distance)

    y_dst0 = max(0, -sy)
    y_dst1 = min(arr.shape[0], arr.shape[0] - sy)
    x_dst0 = max(0, -sx)
    x_dst1 = min(arr.shape[1], arr.shape[1] - sx)
    if y_dst0 >= y_dst1 or x_dst0 >= x_dst1:
        return out

    y_src0 = y_dst0 + sy
    y_src1 = y_dst1 + sy
    x_src0 = x_dst0 + sx
    x_src1 = x_dst1 + sx
    out[y_dst0:y_dst1, x_dst0:x_dst1] = arr[y_src0:y_src1, x_src0:x_src1]
    return out


def _directional_negative_bridge(
    ground_z: np.ndarray,
    ng_zmin: np.ndarray,
    *,
    max_radius_cells: int,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flag low non-ground cells supported by trusted ground on opposite sides."""
    shape = ground_z.shape
    best_depth = np.full(shape, -np.inf, dtype=np.float64)
    support_axes = np.zeros(shape, dtype=np.int8)
    best_pred = np.full(shape, np.nan, dtype=np.float64)

    axes = ((0, 1), (1, 0), (1, 1), (1, -1))
    R = max(1, int(max_radius_cells))

    for dy, dx in axes:
        pz = np.full(shape, np.nan, dtype=np.float64)
        nz = np.full(shape, np.nan, dtype=np.float64)
        pd = np.full(shape, np.inf, dtype=np.float64)
        nd = np.full(shape, np.inf, dtype=np.float64)

        for d in range(1, R + 1):
            sp = _shift_sample(ground_z, dy, dx, d)
            sn = _shift_sample(ground_z, -dy, -dx, d)
            takep = ~np.isfinite(pz) & np.isfinite(sp)
            taken = ~np.isfinite(nz) & np.isfinite(sn)
            if np.any(takep):
                pz[takep] = sp[takep]
                pd[takep] = float(d)
            if np.any(taken):
                nz[taken] = sn[taken]
                nd[taken] = float(d)

        bilateral = np.isfinite(pz) & np.isfinite(nz) & np.isfinite(ng_zmin)
        if not np.any(bilateral):
            continue

        denom = pd + nd
        pred = np.full(shape, np.nan, dtype=np.float64)
        pred[bilateral] = (
            pz[bilateral] * nd[bilateral]
            + nz[bilateral] * pd[bilateral]
        ) / denom[bilateral]

        depth = pred - ng_zmin
        axis_ok = bilateral & (depth >= float(min_depth)) & (depth <= float(max_depth))
        support_axes[axis_ok] += 1

        better = axis_ok & (depth > best_depth)
        best_depth[better] = depth[better]
        best_pred[better] = pred[better]

    candidate = support_axes > 0
    best_depth[~candidate] = np.nan
    return candidate, best_depth, support_axes

def _robust_plane_fit(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, float] | None:
    if x.size < 4:
        return None
    A = np.column_stack((x, y, np.ones(x.size, dtype=np.float64)))
    try:
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    except np.linalg.LinAlgError:
        return None

    pred = A @ coef
    resid = z - pred
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    scale = max(1.4826 * mad, 0.015)
    keep = np.abs(resid - med) <= 2.75 * scale
    if np.count_nonzero(keep) >= 4 and np.count_nonzero(keep) < x.size:
        try:
            coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        except np.linalg.LinAlgError:
            return None
    return coef.astype(np.float64), scale


def _component_pca(cyi: np.ndarray, cxi: np.ndarray, cell: float) -> dict[str, Any]:
    xy = np.column_stack((cxi.astype(np.float64), cyi.astype(np.float64))) * float(cell)
    center = np.mean(xy, axis=0)
    q = xy - center
    if xy.shape[0] < 2:
        major = np.array([1.0, 0.0], dtype=np.float64)
        minor = np.array([0.0, 1.0], dtype=np.float64)
        evals = np.array([0.0, 0.0], dtype=np.float64)
    else:
        cov = np.cov(q, rowvar=False, bias=True)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals = np.maximum(evals[order], 0.0)
        major = evecs[:, order[0]].astype(np.float64)
        minor = evecs[:, order[1]].astype(np.float64)
    u = q @ major
    v = q @ minor
    length = float(np.ptp(u) + cell) if u.size else float(cell)
    width = float(np.ptp(v) + cell) if v.size else float(cell)
    elongation = float(length / max(width, cell))
    return {
        "center": center,
        "major": major,
        "minor": minor,
        "length": length,
        "width": width,
        "elongation": elongation,
        "u": u,
        "v": v,
        "evals": evals,
    }


def _has_opposing_support_in_component_frame(
    ry: np.ndarray,
    rx: np.ndarray,
    *,
    pca: dict[str, Any],
    cell: float,
) -> bool:
    if rx.size < 4:
        return False
    ring_xy = np.column_stack((rx.astype(np.float64), ry.astype(np.float64))) * float(cell)
    q = ring_xy - pca["center"]
    v = q @ pca["minor"]
    eps = max(0.35 * cell, 1e-6)
    return bool(np.any(v > eps) and np.any(v < -eps))


def _validate_basin(
    *,
    comp: np.ndarray,
    cyi: np.ndarray,
    cxi: np.ndarray,
    s: dict[str, Any],
    gocc: np.ndarray,
    cell: float,
    ring_width_cells: int,
    min_ring_cells: int,
    min_depth: float,
    max_depth: float,
    min_negative_fraction: float,
    max_component_depth_spread: float,
) -> tuple[bool, np.ndarray | None, float]:
    ring = binary_dilation(comp, structure=np.ones((3, 3), dtype=bool), iterations=max(1, ring_width_cells))
    ring &= ~comp
    ring &= gocc
    ry, rx = np.nonzero(ring)
    if rx.size < max(4, min_ring_cells):
        return False, None, 0.0

    pca = _component_pca(cyi, cxi, cell)
    if not _has_opposing_support_in_component_frame(ry, rx, pca=pca, cell=cell):
        return False, None, 0.0

    gx = s["x0"] + (rx.astype(np.float64) + 0.5) * cell
    gy = s["y0"] + (ry.astype(np.float64) + 0.5) * cell
    gz = s["ground_z"][ry, rx]
    good = np.isfinite(gz)
    fit = _robust_plane_fit(gx[good], gy[good], gz[good])
    if fit is None:
        return False, None, 0.0
    coef, plane_scale = fit
    if plane_scale > max(0.35, 0.5 * max_depth):
        return False, None, 0.0

    px = s["x0"] + (cxi.astype(np.float64) + 0.5) * cell
    py = s["y0"] + (cyi.astype(np.float64) + 0.5) * cell
    pred = coef[0] * px + coef[1] * py + coef[2]
    lowz = s["ng_zmin"][cyi, cxi]
    finite = np.isfinite(lowz)
    if not np.any(finite):
        return False, None, 0.0
    depth = pred[finite] - lowz[finite]
    negative_fraction = float(np.mean(depth >= min_depth))
    med_depth = float(np.median(depth))
    spread = float(np.percentile(depth, 90) - np.percentile(depth, 10)) if depth.size > 2 else 0.0
    ok = (
        negative_fraction >= min_negative_fraction
        and min_depth <= med_depth <= max_depth
        and spread <= max_component_depth_spread
    )
    return bool(ok), coef if ok else None, med_depth


def _validate_linear_ditch(
    *,
    comp: np.ndarray,
    cyi: np.ndarray,
    cxi: np.ndarray,
    s: dict[str, Any],
    gocc: np.ndarray,
    cell: float,
    pca: dict[str, Any],
    ring_width_cells: int,
    min_ring_cells: int,
    min_depth: float,
    max_depth: float,
    min_valid_sections: int,
    min_valid_section_fraction: float,
    section_step_m: float,
    section_half_length_m: float,
    min_side_offset_m: float,
    max_side_offset_m: float,
    max_longitudinal_resid_m: float,
) -> tuple[bool, dict[str, Any] | None, float]:
    """Validate an elongated ditch using PCA-aligned cross sections.

    The important evidence is bilateral trusted ground across the MINOR axis,
    repeated along the MAJOR axis.  This is orientation-neutral and therefore
    works for ditches at arbitrary XY azimuths.
    """
    ring = binary_dilation(comp, structure=np.ones((3, 3), dtype=bool), iterations=max(2, ring_width_cells))
    ring &= ~comp
    ring &= gocc
    ry, rx = np.nonzero(ring)
    if rx.size < max(6, min_ring_cells):
        return False, None, 0.0

    # Coordinates in metres using cell centres.
    comp_xy = np.column_stack((
        (cxi.astype(np.float64) + 0.5) * cell,
        (cyi.astype(np.float64) + 0.5) * cell,
    ))
    ring_xy = np.column_stack((
        (rx.astype(np.float64) + 0.5) * cell,
        (ry.astype(np.float64) + 0.5) * cell,
    ))
    center = (pca["center"] + 0.5 * cell)
    major = pca["major"]
    minor = pca["minor"]
    cu = (comp_xy - center) @ major
    cv = (comp_xy - center) @ minor
    ru = (ring_xy - center) @ major
    rv = (ring_xy - center) @ minor

    lowz = s["ng_zmin"][cyi, cxi]
    rgz = s["ground_z"][ry, rx]
    valid_low = np.isfinite(lowz)
    valid_rg = np.isfinite(rgz)
    if np.count_nonzero(valid_low) < 2 or np.count_nonzero(valid_rg) < 4:
        return False, None, 0.0

    cu = cu[valid_low]
    cv = cv[valid_low]
    lowz = lowz[valid_low]
    ru = ru[valid_rg]
    rv = rv[valid_rg]
    rgz = rgz[valid_rg]

    umin = float(np.min(cu))
    umax = float(np.max(cu))
    if (umax - umin) < max(section_step_m, 1.5 * cell):
        return False, None, 0.0

    centers = np.arange(umin, umax + 0.5 * section_step_m, max(section_step_m, cell))
    section_records: list[tuple[float, float, float]] = []  # u, candidate z, depth

    for uc in centers:
        csel = np.abs(cu - uc) <= section_half_length_m
        if np.count_nonzero(csel) < 1:
            continue
        # Use the compact bottom layer in this longitudinal slice.
        zc = float(np.median(lowz[csel]))

        rsel = np.abs(ru - uc) <= max(section_half_length_m, 1.5 * cell)
        if np.count_nonzero(rsel) < 2:
            continue

        left = rsel & (rv <= -min_side_offset_m) & (rv >= -max_side_offset_m)
        right = rsel & (rv >= min_side_offset_m) & (rv <= max_side_offset_m)
        if np.count_nonzero(left) < 1 or np.count_nonzero(right) < 1:
            continue

        zl = float(np.median(rgz[left]))
        zr = float(np.median(rgz[right]))
        # Interpolate the expected terrain at the centre of the cross section.
        # Median of the two sides is deliberately conservative and does not
        # flatten their lateral slope into the recovered points themselves.
        zbridge = 0.5 * (zl + zr)
        depth = zbridge - zc
        if min_depth <= depth <= max_depth:
            section_records.append((float(uc), zc, float(depth)))

    if len(section_records) < max(2, min_valid_sections):
        return False, None, 0.0

    valid_fraction = len(section_records) / max(1, len(centers))
    if valid_fraction < min_valid_section_fraction:
        return False, None, 0.0

    # Require longitudinal continuity of the ditch bottom.  A real drainage
    # line may slope, so fit z against major-axis distance and check residuals,
    # rather than requiring a flat bottom.
    arr = np.asarray(section_records, dtype=np.float64)
    A = np.column_stack((arr[:, 0], np.ones(arr.shape[0], dtype=np.float64)))
    try:
        coef_z, *_ = np.linalg.lstsq(A, arr[:, 1], rcond=None)
    except np.linalg.LinAlgError:
        return False, None, 0.0
    zpred = A @ coef_z
    resid = arr[:, 1] - zpred
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    robust_resid = max(1.4826 * mad, float(np.percentile(np.abs(resid - med), 80)))
    if robust_resid > max_longitudinal_resid_m:
        return False, None, 0.0

    model = {
        "major": major,
        "minor": minor,
        "center": center,
        "z_longitudinal_coef": coef_z.astype(np.float64),
        "u_min": umin,
        "u_max": umax,
    }
    return True, model, float(np.median(arr[:, 2]))


def _promote_component_low_layer(
    *,
    comp_id: int,
    labels: np.ndarray,
    s: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    out: np.ndarray,
    low_cluster_dz: float,
    min_depth: float,
    max_depth: float,
    basin_coef: np.ndarray | None,
    linear_model: dict[str, Any] | None,
) -> np.ndarray:
    lin = s["lin"]
    point_labels = labels.ravel()[lin]
    in_comp = (~out) & (point_labels == int(comp_id))
    promote = np.zeros(out.size, dtype=bool)
    if not np.any(in_comp):
        return promote

    idx = np.flatnonzero(in_comp)
    cell_low = s["ng_zmin"].ravel()[lin[idx]]
    local = z[idx] <= (cell_low + low_cluster_dz)

    if basin_coef is not None:
        terrain = basin_coef[0] * x[idx] + basin_coef[1] * y[idx] + basin_coef[2]
        depth = terrain - z[idx]
        local &= depth >= (0.45 * min_depth)
        local &= depth <= max_depth
    elif linear_model is not None:
        xy = np.column_stack((x[idx], y[idx]))
        u = (xy - linear_model["center"]) @ linear_model["major"]
        bottom_pred = (
            linear_model["z_longitudinal_coef"][0] * u
            + linear_model["z_longitudinal_coef"][1]
        )
        # Keep only points close to the validated longitudinal bottom envelope.
        # This prevents higher vegetation/object points sharing the same XY
        # component from being promoted.
        local &= np.abs(z[idx] - bottom_pred) <= max(low_cluster_dz * 2.5, 0.10)
    else:
        return promote

    promote[idx[local]] = True
    return promote


def _sweep_once(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ground_mask: np.ndarray,
    sm: str,
    cfg: dict,
    pass_index: int,
) -> tuple[np.ndarray, dict[str, int]]:
    out = np.asarray(ground_mask, dtype=bool).copy()

    cell = float(cfg.get("ditch_sweeper_cell_m", 0.40 if sm == "ULS" else 0.55))
    low_cluster_dz = float(cfg.get("ditch_sweeper_low_cluster_dz_m", 0.08 if sm == "ULS" else 0.11))
    min_low_points = int(cfg.get("ditch_sweeper_min_low_points", 2))
    min_component_points = int(cfg.get("ditch_sweeper_min_component_points", 4))
    ring_width_cells = int(cfg.get("ditch_sweeper_ring_width_cells", 3))
    max_ground_gap_cells = int(cfg.get("ditch_sweeper_max_ground_gap_cells", 4 if sm == "ULS" else 5))
    min_ring_cells = int(cfg.get("ditch_sweeper_min_ring_ground_cells", 6))
    min_depth = float(cfg.get("ditch_sweeper_min_depth_m", 0.07 if sm == "ULS" else 0.10))
    max_depth = float(cfg.get("ditch_sweeper_max_depth_m", 1.40 if sm == "ULS" else 1.80))
    max_basin_width_m = float(cfg.get("ditch_sweeper_max_width_m", 6.0 if sm == "ULS" else 8.0))
    min_negative_fraction = float(cfg.get("ditch_sweeper_min_negative_fraction", 0.60))
    max_component_depth_spread = float(cfg.get("ditch_sweeper_max_depth_spread_m", 0.70 if sm == "ULS" else 0.90))

    # Linear / trench validator defaults.
    min_elongation = float(cfg.get("ditch_sweeper_linear_min_elongation", 2.2))
    max_linear_width_m = float(cfg.get("ditch_sweeper_linear_max_width_m", 4.0 if sm == "ULS" else 6.0))
    min_linear_length_m = float(cfg.get("ditch_sweeper_linear_min_length_m", 1.6 if sm == "ULS" else 2.2))
    min_valid_sections = int(cfg.get("ditch_sweeper_linear_min_sections", 3))
    min_valid_section_fraction = float(cfg.get("ditch_sweeper_linear_min_section_fraction", 0.45))
    section_step_m = float(cfg.get("ditch_sweeper_linear_section_step_m", max(0.8, 2.0 * cell)))
    section_half_length_m = float(cfg.get("ditch_sweeper_linear_section_half_length_m", max(0.45, 1.25 * cell)))
    min_side_offset_m = float(cfg.get("ditch_sweeper_linear_min_side_offset_m", max(0.35, 0.75 * cell)))
    max_side_offset_m = float(cfg.get("ditch_sweeper_linear_max_side_offset_m", max(2.5, 5.0 * cell)))
    max_longitudinal_resid_m = float(cfg.get("ditch_sweeper_linear_max_longitudinal_resid_m", 0.20 if sm == "ULS" else 0.28))

    s = _grid_stats(x, y, z, out, cell=cell, low_cluster_dz=low_cluster_dz)
    gocc = s["gcount"] > 0
    ngocc = s["ngcount"] > 0
    stable_low = s["low_count"] >= max(1, min_low_points)

    bridge_candidate, bridge_depth, bridge_axes = _directional_negative_bridge(
        s["ground_z"],
        s["ng_zmin"],
        max_radius_cells=max_ground_gap_cells,
        min_depth=min_depth,
        max_depth=max_depth,
    )

    mixed_negative = gocc & ngocc & np.isfinite(s["ground_z"]) & np.isfinite(s["ng_zmin"])
    mixed_negative &= s["ng_zmin"] <= (s["ground_z"] - max(0.5 * min_depth, 0.03))

    candidate = ngocc & stable_low & (bridge_candidate | mixed_negative)
    candidate = binary_dilation(
        candidate,
        structure=np.ones((3, 3), dtype=bool),
        iterations=1,
    ) & ngocc & stable_low

    labels, ncomp = label(candidate, structure=np.ones((3, 3), dtype=np.int8))
    candidate_cells = int(np.count_nonzero(candidate))
    if ncomp == 0:
        return out, {
            "candidate_cells": candidate_cells,
            "candidate_components": 0,
            "validated_components": 0,
            "recovered_points": 0,
            "rejected_components": 0,
            "basin_components": 0,
            "linear_components": 0,
        }

    promote_all = np.zeros(out.size, dtype=bool)
    rejected = 0
    basin_count = 0
    linear_count = 0
    validated = 0

    for comp_id in range(1, int(ncomp) + 1):
        comp = labels == comp_id
        cyi, cxi = np.nonzero(comp)
        if cxi.size == 0:
            continue
        if int(np.sum(s["low_count"][cyi, cxi])) < min_component_points:
            rejected += 1
            continue

        direct_bridge_fraction = float(np.mean(bridge_candidate[cyi, cxi]))
        if direct_bridge_fraction < float(cfg.get("ditch_sweeper_min_bridge_fraction", 0.20)):
            rejected += 1
            continue

        pca = _component_pca(cyi, cxi, cell)
        is_linear = (
            pca["elongation"] >= min_elongation
            and pca["length"] >= min_linear_length_m
            and pca["width"] <= max_linear_width_m
        )

        basin_ok = False
        basin_coef = None
        linear_ok = False
        linear_model = None

        # Long narrow components are checked as ditches first.  If the linear
        # geometry is inconclusive, they may still pass the conservative basin
        # validator when their minor width is small enough.
        if is_linear:
            linear_ok, linear_model, _ = _validate_linear_ditch(
                comp=comp,
                cyi=cyi,
                cxi=cxi,
                s=s,
                gocc=gocc,
                cell=cell,
                pca=pca,
                ring_width_cells=ring_width_cells,
                min_ring_cells=min_ring_cells,
                min_depth=min_depth,
                max_depth=max_depth,
                min_valid_sections=min_valid_sections,
                min_valid_section_fraction=min_valid_section_fraction,
                section_step_m=section_step_m,
                section_half_length_m=section_half_length_m,
                min_side_offset_m=min_side_offset_m,
                max_side_offset_m=max_side_offset_m,
                max_longitudinal_resid_m=max_longitudinal_resid_m,
            )

        if not linear_ok and pca["width"] <= max_basin_width_m:
            basin_ok, basin_coef, _ = _validate_basin(
                comp=comp,
                cyi=cyi,
                cxi=cxi,
                s=s,
                gocc=gocc,
                cell=cell,
                ring_width_cells=ring_width_cells,
                min_ring_cells=min_ring_cells,
                min_depth=min_depth,
                max_depth=max_depth,
                min_negative_fraction=min_negative_fraction,
                max_component_depth_spread=max_component_depth_spread,
            )

        if not linear_ok and not basin_ok:
            rejected += 1
            continue

        promote = _promote_component_low_layer(
            comp_id=comp_id,
            labels=labels,
            s=s,
            x=x,
            y=y,
            z=z,
            out=out,
            low_cluster_dz=low_cluster_dz,
            min_depth=min_depth,
            max_depth=max_depth,
            basin_coef=basin_coef if basin_ok else None,
            linear_model=linear_model if linear_ok else None,
        )
        if not np.any(promote):
            rejected += 1
            continue

        promote_all |= promote
        validated += 1
        if linear_ok:
            linear_count += 1
        else:
            basin_count += 1

    out[promote_all] = True
    return out, {
        "candidate_cells": candidate_cells,
        "candidate_components": int(ncomp),
        "validated_components": int(validated),
        "recovered_points": int(np.count_nonzero(promote_all)),
        "rejected_components": int(rejected),
        "basin_components": int(basin_count),
        "linear_components": int(linear_count),
    }


def sweep_ground_depressions(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ground_mask: np.ndarray,
    sensor_mode: str,
    cfg: dict,
    return_report: bool = False,
):
    """Conservative ditch / nested-depression validator for ALS and ULS.

    The sweeper is intentionally separate from the primary FAST-GC classifier.
    It first flags coherent non-ground low layers embedded in or adjacent to the
    trusted ground support, then validates either:

      * ordinary compact/broad depressions; or
      * elongated linear drainage ditches using PCA-aligned cross sections.

    A second bounded pass is allowed.  Only points recovered with strong
    evidence in pass 1 become local parent-ground support for pass 2, enabling a
    narrow nested trench to be recovered inside an already validated broad
    depression.  The recursion is capped and never changes XYZ coordinates.

    Existing ground is never demoted.  Only the compact validated low envelope
    can change from non-ground to ground.
    """
    sm = str(sensor_mode).upper().strip()
    out = np.asarray(ground_mask, dtype=bool).copy()
    empty = DepressionSweepReport()

    if sm not in {"ALS", "ULS"}:
        return (out, empty.as_dict()) if return_report else out
    if not bool(cfg.get("ditch_sweeper_enabled", True)):
        return (out, empty.as_dict()) if return_report else out

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if x.size < 20 or out.size != x.size or np.count_nonzero(out) < 10:
        return (out, empty.as_dict()) if return_report else out

    max_passes = int(cfg.get("ditch_sweeper_nested_passes", 2))
    max_passes = max(1, min(max_passes, 2))

    totals = {
        "candidate_cells": 0,
        "candidate_components": 0,
        "validated_components": 0,
        "recovered_points": 0,
        "rejected_components": 0,
        "basin_components": 0,
        "linear_components": 0,
        "nested_pass_recoveries": 0,
    }

    for pass_index in range(max_passes):
        before = out.copy()
        out, rep = _sweep_once(
            x=x,
            y=y,
            z=z,
            ground_mask=out,
            sm=sm,
            cfg=cfg,
            pass_index=pass_index,
        )
        newly = int(np.count_nonzero(out & ~before))
        for k in (
            "candidate_cells",
            "candidate_components",
            "validated_components",
            "recovered_points",
            "rejected_components",
            "basin_components",
            "linear_components",
        ):
            totals[k] += int(rep.get(k, 0))
        if pass_index > 0:
            totals["nested_pass_recoveries"] += newly
        if newly == 0:
            break

    report = DepressionSweepReport(**totals)
    return (out, report.as_dict()) if return_report else out


__all__ = ["DepressionSweepReport", "sweep_ground_depressions"]
