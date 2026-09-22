from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class FinalSurfaceVoteConfig:
    density_trigger_pts_m2: float = 5.0
    very_low_density_trigger_pts_m2: float = 3.0
    offset_fraction: float = 0.5

    # Robust terrain-sheet fitting.
    min_patch_points: int = 20
    max_patch_points: int = 1800
    mad_floor_m: float = 0.07
    robust_k: float = 3.25
    fit_iterations: int = 4

    # A contaminant must remain a minority of the local terrain context.
    max_positive_tail_fraction: float = 0.22

    # Physical consolidation.
    cell_m: float = 0.50
    min_cell_candidate_fraction: float = 0.30

    # Global fail-safe per pass.
    max_demote_fraction_low: float = 0.035
    max_demote_fraction_normal: float = 0.010


@dataclass(frozen=True)
class _PassSpec:
    name: str
    patch_sizes_m: tuple[float, float]
    min_vote_fraction: float
    min_positive_residual_m: float
    max_demote_fraction: float


def _mad(a: np.ndarray) -> float:
    if a.size == 0:
        return float("nan")
    med = float(np.median(a))
    return float(1.4826 * np.median(np.abs(a - med)))


def _design(dx: np.ndarray, dy: np.ndarray, quadratic: bool) -> np.ndarray:
    if quadratic:
        return np.column_stack(
            (dx, dy, dx * dx, dx * dy, dy * dy, np.ones(dx.size))
        )
    return np.column_stack((dx, dy, np.ones(dx.size)))


def _robust_fit(x, y, z, qx, qy, quadratic, cfg):
    min_n = 18 if quadratic else 10
    if z.size < min_n:
        return None

    dx = x - qx
    dy = y - qy
    A = _design(dx, dy, quadratic)
    keep = np.ones(z.size, dtype=bool)
    coef = None

    for _ in range(int(cfg.fit_iterations)):
        if np.count_nonzero(keep) < min_n:
            return None
        try:
            coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        except np.linalg.LinAlgError:
            return None

        r = z - A @ coef
        rk = r[keep]
        med = float(np.median(rk))
        sigma = max(_mad(rk), float(cfg.mad_floor_m))
        new_keep = np.abs(r - med) <= float(cfg.robust_k) * sigma

        if np.array_equal(new_keep, keep):
            keep = new_keep
            break
        keep = new_keep

    if coef is None or np.count_nonzero(keep) < min_n:
        return None

    r = z - A @ coef
    rk = r[keep]
    return {
        "coef": coef,
        "quadratic": bool(quadratic),
        "sigma": max(_mad(rk), float(cfg.mad_floor_m)),
    }


def _choose_fit(x, y, z, qx, qy, cfg):
    plane = _robust_fit(x, y, z, qx, qy, False, cfg)
    if plane is None:
        return None

    quad = _robust_fit(x, y, z, qx, qy, True, cfg)
    if quad is None:
        return plane

    # Curvature is used only where it materially improves a broad terrain sheet.
    if plane["sigma"] >= 0.10 and quad["sigma"] <= 0.82 * plane["sigma"] and quad["sigma"] <= 0.65:
        return quad
    return plane


def _predict(model, x, y, qx, qy):
    dx = x - qx
    dy = y - qy
    c = model["coef"]
    if model["quadratic"]:
        return c[0]*dx + c[1]*dy + c[2]*dx*dx + c[3]*dx*dy + c[4]*dy*dy + c[5]
    return c[0]*dx + c[1]*dy + c[2]


def _measured_density(x, y):
    if x.size < 2:
        return 0.0
    sx = max(float(np.max(x) - np.min(x)), 1e-6)
    sy = max(float(np.max(y) - np.min(y)), 1e-6)
    return float(x.size / (sx * sy))


def _density_from_cfg(cfg, fallback):
    for key in (
        "adaptive_support_dataset_density_pts_m2",
        "dataset_density_pts_m2",
        "density_pts_m2",
    ):
        value = cfg.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except Exception:
            continue
        if np.isfinite(value) and value > 0:
            return value
    return float(fallback)


def _iter_patch_groups(x, y, patch_m, off_x, off_y):
    x0 = math.floor((float(np.min(x)) - off_x) / patch_m) * patch_m + off_x
    y0 = math.floor((float(np.min(y)) - off_y) / patch_m) * patch_m + off_y

    ix = np.floor((x - x0) / patch_m).astype(np.int64)
    iy = np.floor((y - y0) / patch_m).astype(np.int64)
    nx = int(ix.max()) + 1
    pid = iy * nx + ix

    order = np.argsort(pid, kind="mergesort")
    ps = pid[order]
    starts = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]])
    ends = np.r_[starts[1:], ps.size]

    for s, e in zip(starts, ends):
        yield order[s:e]


def _single_surface_vote_pass(x, y, z, ground_mask, cfg, spec):
    """
    One terrain-relative offset-voting pass.

    IMPORTANT: x/y/z are unchanged, but ground_mask is the UPDATED mask
    supplied by the previous pass. Therefore every iteration rebuilds its
    terrain sheet after earlier contaminants have been removed.
    """
    g = np.asarray(ground_mask, dtype=bool).copy()
    gi = np.flatnonzero(g)

    report = {
        "name": spec.name,
        "patch_sizes_m": list(spec.patch_sizes_m),
        "ground_before": int(gi.size),
        "patches_tested": 0,
        "patches_modeled": 0,
        "upper_tail_patches": 0,
        "voted_points": 0,
        "required_votes": 0,
        "max_votes": 0,
        "candidate_cells": 0,
        "candidate_points": 0,
        "demoted_points": 0,
        "median_candidate_residual_m": 0.0,
        "ground_after": int(gi.size),
    }

    if gi.size < int(cfg.min_patch_points):
        return g, report

    gx, gy, gz = x[gi], y[gi], z[gi]
    votes = np.zeros(gi.size, dtype=np.uint16)
    residual_sum = np.zeros(gi.size, dtype=np.float64)
    residual_count = np.zeros(gi.size, dtype=np.uint16)

    offsets = (
        (0.0, 0.0),
        (cfg.offset_fraction, 0.0),
        (0.0, cfg.offset_fraction),
        (cfg.offset_fraction, cfg.offset_fraction),
    )
    total_votes = len(spec.patch_sizes_m) * len(offsets)
    required_votes = max(2, int(math.ceil(spec.min_vote_fraction * total_votes)))
    report["required_votes"] = int(required_votes)

    for patch_m in spec.patch_sizes_m:
        patch_m = float(patch_m)
        for fx, fy in offsets:
            for loc in _iter_patch_groups(
                gx, gy, patch_m,
                float(fx) * patch_m,
                float(fy) * patch_m,
            ):
                report["patches_tested"] += 1
                if loc.size < int(cfg.min_patch_points):
                    continue

                fit_loc = loc
                if fit_loc.size > int(cfg.max_patch_points):
                    # Deterministic spatially distributed sample.
                    order_xy = np.lexsort((gy[fit_loc], gx[fit_loc]))
                    ranks = np.linspace(
                        0, order_xy.size - 1, int(cfg.max_patch_points)
                    ).astype(np.int64)
                    fit_loc = fit_loc[order_xy[ranks]]

                qx = float(np.median(gx[fit_loc]))
                qy = float(np.median(gy[fit_loc]))
                model = _choose_fit(
                    gx[fit_loc], gy[fit_loc], gz[fit_loc],
                    qx, qy, cfg,
                )
                if model is None:
                    continue

                report["patches_modeled"] += 1

                pred = _predict(model, gx[loc], gy[loc], qx, qy)
                residual = gz[loc] - pred
                med = float(np.median(residual))
                sigma = max(_mad(residual), float(cfg.mad_floor_m))

                threshold = max(
                    float(spec.min_positive_residual_m),
                    med + float(cfg.robust_k) * sigma,
                )

                positive = residual > threshold
                positive_fraction = float(np.mean(positive))

                # A large positive fraction is more likely real terrain structure.
                if positive_fraction > float(cfg.max_positive_tail_fraction):
                    continue

                if np.any(positive):
                    report["upper_tail_patches"] += 1
                    p = loc[positive]
                    votes[p] += 1
                    residual_sum[p] += residual[positive]
                    residual_count[p] += 1

    report["voted_points"] = int(np.count_nonzero(votes))
    report["max_votes"] = int(votes.max()) if votes.size else 0

    mean_positive_residual = np.zeros(gi.size, dtype=np.float64)
    rc = residual_count > 0
    mean_positive_residual[rc] = residual_sum[rc] / residual_count[rc]

    raw = (
        (votes >= required_votes)
        & (mean_positive_residual >= float(spec.min_positive_residual_m))
    )

    if not np.any(raw):
        return g, report

    # 0.5 m consolidation: require local cell agreement.
    cell = max(float(cfg.cell_m), 0.10)
    x0 = math.floor(float(np.min(gx)) / cell) * cell
    y0 = math.floor(float(np.min(gy)) / cell) * cell
    ix = np.floor((gx - x0) / cell).astype(np.int64)
    iy = np.floor((gy - y0) / cell).astype(np.int64)
    nx = int(ix.max()) + 1
    cid = iy * nx + ix

    order = np.argsort(cid, kind="mergesort")
    cs = cid[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    ends = np.r_[starts[1:], cs.size]

    final = np.zeros(gi.size, dtype=bool)
    cell_count = 0

    for s, e in zip(starts, ends):
        loc = order[s:e]
        frac = float(np.mean(raw[loc]))
        if frac < float(cfg.min_cell_candidate_fraction):
            continue
        # Consensus in the physical DEM support cell.
        if float(np.max(votes[loc])) < required_votes:
            continue
        final[loc] = raw[loc]
        cell_count += 1

    report["candidate_cells"] = int(cell_count)
    report["candidate_points"] = int(np.count_nonzero(final))

    if not np.any(final):
        return g, report

    candidate = np.flatnonzero(final)

    # Per-pass fail-safe.
    max_remove = max(
        1,
        int(math.ceil(float(spec.max_demote_fraction) * max(1, gi.size))),
    )
    if candidate.size > max_remove:
        strength = (
            votes[candidate].astype(np.float64)
            + np.clip(mean_positive_residual[candidate], 0.0, None)
        )
        candidate = candidate[np.argsort(strength)[-max_remove:]]

    remove_global = gi[candidate]
    g[remove_global] = False

    report["demoted_points"] = int(remove_global.size)
    report["ground_after"] = int(np.count_nonzero(g))
    if candidate.size:
        report["median_candidate_residual_m"] = float(
            np.median(mean_positive_residual[candidate])
        )

    return g, report


def apply_final_surface_vote(
    *,
    x,
    y,
    z,
    ground_mask,
    sensor_mode,
    cfg=None,
    return_report=True,
):
    """
    FAST-GC final iterative surface voting V2.2.

    ALS/ULS:
      density < 5 pts/m2:
        pass 1: 20 + 30 m
        pass 2: 12 + 20 m
        pass 3:  8 + 12 m
      density >= 5 pts/m2:
        one conservative 20 + 30 m pass

    All passes are terrain-relative, offset-voted, and DEMOTION ONLY.
    """
    sm = str(sensor_mode).upper().strip()
    user_cfg = cfg or {}
    c = FinalSurfaceVoteConfig()

    g = np.asarray(ground_mask, dtype=bool).copy()

    report = {
        "enabled": False,
        "activated": False,
        "density_pts_m2": 0.0,
        "density_trigger_pts_m2": float(c.density_trigger_pts_m2),
        "low_density": False,
        "very_low_density": False,
        "density_regime": "DISABLED",
        "iterations_requested": 0,
        "iterations_completed": 0,
        "ground_before": int(np.count_nonzero(g)),
        "ground_after": int(np.count_nonzero(g)),
        "demoted_points": 0,
        "passes": [],
    }

    if sm not in {"ALS", "ULS"}:
        return (g, report) if return_report else g

    if not bool(user_cfg.get("final_surface_vote_enabled", True)):
        return (g, report) if return_report else g

    report["enabled"] = True

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    if not (x.size == y.size == z.size == g.size):
        return (g, report) if return_report else g

    density = _density_from_cfg(user_cfg, _measured_density(x, y))
    low_density = bool(density < float(c.density_trigger_pts_m2))
    very_low_density = bool(density < float(c.very_low_density_trigger_pts_m2))

    report["density_pts_m2"] = float(density)
    report["low_density"] = low_density
    report["very_low_density"] = very_low_density
    report["density_regime"] = "VERY_LOW" if very_low_density else ("LOW" if low_density else "NORMAL")
    report["activated"] = True

    if very_low_density:
        # VERY_LOW (<3 pts/m2): broad-to-moderate refinement.
        # Keep wide context longer to prevent sparse contaminant clusters
        # from supporting their own local terrain fit.
        specs = (
            _PassSpec("very_low_broad",    (20.0, 30.0), 0.625, 0.45, 0.035),
            _PassSpec("very_low_refine_1", (18.0, 25.0), 0.625, 0.34, 0.025),
            _PassSpec("very_low_refine_2", (15.0, 22.0), 0.750, 0.28, 0.020),
            _PassSpec("very_low_refine_3", (12.0, 18.0), 0.750, 0.23, 0.015),
            _PassSpec("very_low_refine_4", (10.0, 15.0), 0.875, 0.20, 0.010),
        )

    elif low_density:
        # LOW (3-5 pts/m2): preserve successful D4 behavior exactly.
        specs = (
            _PassSpec("broad",        (20.0, 30.0), 0.625, 0.45, c.max_demote_fraction_low),
            _PassSpec("intermediate", (12.0, 20.0), 0.750, 0.30, 0.025),
            _PassSpec("local",         (8.0, 12.0), 0.875, 0.22, 0.015),
        )

    else:
        specs = (
            _PassSpec(
                "normal_density_broad",
                (20.0, 30.0),
                0.750,   # 6/8
                0.55,
                c.max_demote_fraction_normal,
            ),
        )

    report["iterations_requested"] = len(specs)

    for spec in specs:
        before = int(np.count_nonzero(g))

        # CRITICAL: every pass gets the UPDATED g from the previous pass.
        g, pr = _single_surface_vote_pass(x, y, z, g, c, spec)
        report["passes"].append(pr)
        report["iterations_completed"] += 1

        after = int(np.count_nonzero(g))

        # If no changes occur, later smaller scales can still reveal residuals,
        # so do not terminate early in the low-density 3-pass channel.
        if not low_density and after == before:
            break

    report["ground_after"] = int(np.count_nonzero(g))
    report["demoted_points"] = int(
        report["ground_before"] - report["ground_after"]
    )

    # Backward-compatible aggregate fields expected by current io_las logging.
    if report["passes"]:
        report["patches_tested"] = int(sum(p["patches_tested"] for p in report["passes"]))
        report["patches_modeled"] = int(sum(p["patches_modeled"] for p in report["passes"]))
        report["patches_positive_tail"] = int(sum(p["upper_tail_patches"] for p in report["passes"]))
        report["points_with_votes"] = int(sum(p["voted_points"] for p in report["passes"]))
        report["candidate_cells"] = int(sum(p["candidate_cells"] for p in report["passes"]))
        report["candidate_points"] = int(sum(p["candidate_points"] for p in report["passes"]))
        report["required_votes"] = int(report["passes"][-1]["required_votes"])
        report["max_votes"] = int(max(p["max_votes"] for p in report["passes"]))
        vals = [
            p["median_candidate_residual_m"]
            for p in report["passes"]
            if p["demoted_points"] > 0
        ]
        report["median_candidate_residual_m"] = float(np.median(vals)) if vals else 0.0

    return (g, report) if return_report else g


__all__ = ["FinalSurfaceVoteConfig", "apply_final_surface_vote"]
