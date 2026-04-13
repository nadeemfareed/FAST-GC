from __future__ import annotations

import math
from typing import Any


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _finite_or_none(v: Any) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def _rounded_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(v))))


def _als_static_defaults() -> dict:
    return dict(
        vote_cell_m=1.00,
        vote_top_m=3,
        vote_neighbor_radius_cells=3,
        vote_min_neighbor_cells=6,
        vote_max_robust_z=3.0,
        vote_mad_floor=0.05,
        vote_fill_iters=80,
        vote_smooth_sigma_cells=1.2,
        vote_ground_threshold_m=0.30,
        vote_slope_adapt_k=0.40,
        cand_cell_m=2.0,
        cand_dz_m=0.75,
        base_cell_m=2.0,
        base_radius_m=1.00,
        void_recover_enabled=True,
        chm_method_default="pitfree",
        chm_smooth_method_default="none",
        chm_percentile_default=99.0,
        chm_pitfree_thresholds=[0.0, 2.0, 5.0, 10.0, 15.0],
        chm_use_first_returns_default=True,
        chm_spikefree_freeze_distance=0.40,
        chm_spikefree_insertion_buffer=0.50,
        chm_gaussian_sigma_default=1.0,
        chm_median_size_default=3,
    )


def _uls_static_defaults() -> dict:
    return dict(
        vote_cell_m=0.75,
        vote_top_m=4,
        vote_neighbor_radius_cells=4,
        vote_min_neighbor_cells=8,
        vote_max_robust_z=2.8,
        vote_mad_floor=0.05,
        vote_fill_iters=100,
        vote_smooth_sigma_cells=1.0,
        vote_ground_threshold_m=0.18,
        vote_slope_adapt_k=0.30,
        cand_cell_m=1.5,
        cand_dz_m=0.60,
        base_cell_m=1.5,
        base_radius_m=0.50,
        void_recover_enabled=True,
        chm_method_default="pitfree",
        chm_smooth_method_default="none",
        chm_percentile_default=99.0,
        chm_pitfree_thresholds=[0.0, 1.0, 3.0, 6.0, 10.0, 15.0],
        chm_use_first_returns_default=True,
        chm_spikefree_freeze_distance=0.40,
        chm_spikefree_insertion_buffer=0.50,
        chm_gaussian_sigma_default=1.0,
        chm_median_size_default=3,
    )


def _tls_static_defaults() -> dict:
    return dict(
        vote_cell_m=0.35,
        vote_top_m=6,
        vote_neighbor_radius_cells=4,
        vote_min_neighbor_cells=10,
        vote_max_robust_z=2.6,
        vote_mad_floor=0.05,
        vote_fill_iters=100,
        vote_smooth_sigma_cells=0.8,
        vote_ground_threshold_m=0.10,
        vote_slope_adapt_k=0.25,
        cand_cell_m=2.0,
        cand_dz_m=0.20,
        base_cell_m=2.0,
        base_radius_m=0.25,
        void_recover_enabled=False,
        chm_method_default="p99",
        chm_smooth_method_default="none",
        chm_percentile_default=99.0,
        chm_pitfree_thresholds=[0.0, 2.0, 5.0, 10.0, 15.0],
        chm_use_first_returns_default=False,
        chm_spikefree_freeze_distance=0.40,
        chm_spikefree_insertion_buffer=0.50,
        chm_gaussian_sigma_default=1.0,
        chm_median_size_default=3,
    )


def _lookup_scale(x: float | None, bands: list[tuple[float, float]]) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    for upper, scale in bands:
        if x < upper:
            return float(scale)
    return float(bands[-1][1])


def _sensor_profile(sensor_mode: str) -> dict[str, Any]:
    sm = sensor_mode.upper().strip()
    if sm == "ALS":
        return {
            "pc2_bands": [(10, 1.60), (25, 1.40), (60, 1.20), (160, 1.00), (350, 0.90), (float("inf"), 0.82)],
            "density_bands": [(5, 1.60), (15, 1.35), (40, 1.15), (120, 1.00), (250, 0.90), (float("inf"), 0.82)],
            "scale_bounds": (0.80, 1.80),
            "vote_cell_bounds": (0.75, 2.00),
            "cand_cell_bounds": (1.50, 4.00),
            "base_cell_bounds": (1.50, 4.00),
            "base_radius_bounds": (0.75, 2.50),
            "vote_neighbor_bounds": (4, 8),
            "tile_nudge_bounds": (-0.10, 0.12),
        }
    if sm == "ULS":
        return {
            "pc2_bands": [(20, 1.45), (50, 1.25), (120, 1.10), (300, 1.00), (700, 0.92), (float("inf"), 0.86)],
            "density_bands": [(10, 1.40), (30, 1.22), (90, 1.10), (220, 1.00), (500, 0.92), (float("inf"), 0.86)],
            "scale_bounds": (0.80, 1.60),
            "vote_cell_bounds": (0.50, 1.50),
            "cand_cell_bounds": (1.00, 3.50),
            "base_cell_bounds": (1.00, 3.50),
            "base_radius_bounds": (0.35, 1.50),
            "vote_neighbor_bounds": (6, 10),
            "tile_nudge_bounds": (-0.08, 0.10),
        }
    return {
        "pc2_bands": [(40, 1.50), (100, 1.30), (250, 1.15), (700, 1.00), (1800, 0.92), (float("inf"), 0.86)],
        "density_bands": [(25, 1.45), (80, 1.25), (200, 1.12), (600, 1.00), (1400, 0.92), (float("inf"), 0.86)],
        "scale_bounds": (0.80, 1.55),
        "vote_cell_bounds": (0.25, 0.80),
        "cand_cell_bounds": (1.00, 3.00),
        "base_cell_bounds": (1.00, 3.00),
        "base_radius_bounds": (0.15, 0.90),
        "vote_neighbor_bounds": (8, 14),
        "tile_nudge_bounds": (-0.08, 0.10),
    }


def _derive_scale_from_support_stats(
    sensor_mode: str,
    tile_support_stats: dict[str, Any] | None,
    dataset_support_stats: dict[str, Any] | None,
) -> tuple[float, dict[str, float]]:
    tile_support_stats = tile_support_stats or {}
    dataset_support_stats = dataset_support_stats or {}
    profile = _sensor_profile(sensor_mode)

    tile_pc2 = _finite_or_none(tile_support_stats.get("grid_2m_pointcount_median"))
    tile_occ2 = _finite_or_none(tile_support_stats.get("grid_2m_occupancy_ratio"))
    tile_density = _finite_or_none(tile_support_stats.get("density_pts_m2"))

    ref_pc2 = _finite_or_none(dataset_support_stats.get("grid_2m_pointcount_median_median"))
    ref_density = _finite_or_none(dataset_support_stats.get("density_pts_m2_median"))
    ref_occ2 = _finite_or_none(dataset_support_stats.get("grid_2m_occupancy_ratio_median"))

    dataset_scale = _lookup_scale(ref_pc2, profile["pc2_bands"])
    if dataset_scale is None:
        dataset_scale = _lookup_scale(ref_density, profile["density_bands"])
    if dataset_scale is None:
        dataset_scale = 1.0

    tile_nudge = 0.0
    if tile_pc2 is not None and ref_pc2 is not None and ref_pc2 > 0:
        ratio = tile_pc2 / ref_pc2
        if ratio < 0.50:
            tile_nudge += 0.10
        elif ratio < 0.80:
            tile_nudge += 0.05
        elif ratio > 2.50:
            tile_nudge -= 0.08
        elif ratio > 1.80:
            tile_nudge -= 0.05

    if tile_occ2 is not None and ref_occ2 is not None and math.isfinite(tile_occ2) and math.isfinite(ref_occ2):
        d_occ = tile_occ2 - ref_occ2
        if d_occ < -0.20:
            tile_nudge += 0.05
        elif d_occ < -0.10:
            tile_nudge += 0.025
        elif d_occ > 0.20:
            tile_nudge -= 0.03
        elif d_occ > 0.10:
            tile_nudge -= 0.015

    if tile_density is not None and ref_density is not None and ref_density > 0:
        dens_ratio = tile_density / ref_density
        if dens_ratio < 0.50:
            tile_nudge += 0.04
        elif dens_ratio > 2.00:
            tile_nudge -= 0.03

    tile_nudge = _clamp(tile_nudge, *profile["tile_nudge_bounds"])
    final_scale = _clamp(dataset_scale + tile_nudge, *profile["scale_bounds"])
    return final_scale, {
        "tile_pc2": tile_pc2 if tile_pc2 is not None else math.nan,
        "tile_density": tile_density if tile_density is not None else math.nan,
        "tile_occ2": tile_occ2 if tile_occ2 is not None else math.nan,
        "ref_pc2": ref_pc2 if ref_pc2 is not None else math.nan,
        "ref_density": ref_density if ref_density is not None else math.nan,
        "ref_occ2": ref_occ2 if ref_occ2 is not None else math.nan,
        "dataset_scale": dataset_scale,
        "tile_nudge": tile_nudge,
        "final_scale": final_scale,
    }


def _adapt_defaults(
    sensor_mode: str,
    base_cfg: dict,
    tile_support_stats: dict[str, Any] | None = None,
    dataset_support_stats: dict[str, Any] | None = None,
) -> dict:
    if not dataset_support_stats and not tile_support_stats:
        cfg = dict(base_cfg)
        cfg["adaptive_support_enabled"] = False
        cfg["adaptive_support_scale"] = 1.0
        return cfg

    sm = sensor_mode.upper().strip()
    profile = _sensor_profile(sm)
    scale, diag = _derive_scale_from_support_stats(sm, tile_support_stats, dataset_support_stats)

    cfg = dict(base_cfg)
    cfg["vote_cell_m"] = _clamp(base_cfg["vote_cell_m"] * scale, *profile["vote_cell_bounds"])
    cfg["cand_cell_m"] = _clamp(base_cfg["cand_cell_m"] * scale, *profile["cand_cell_bounds"])
    cfg["base_cell_m"] = _clamp(base_cfg["base_cell_m"] * scale, *profile["base_cell_bounds"])
    cfg["base_radius_m"] = _clamp(base_cfg["base_radius_m"] * scale, *profile["base_radius_bounds"])

    lo_n, hi_n = profile["vote_neighbor_bounds"]
    cfg["vote_min_neighbor_cells"] = _rounded_int(base_cfg["vote_min_neighbor_cells"] / math.sqrt(scale), lo_n, hi_n)

    # Very small bounded nudges only; preserve algorithm identity.
    if scale > 1.20:
        cfg["vote_top_m"] = max(int(base_cfg["vote_top_m"]), int(base_cfg["vote_top_m"]) + 1)
    elif scale < 0.90:
        cfg["vote_top_m"] = max(1, int(base_cfg["vote_top_m"]) - 1)

    cfg["adaptive_support_enabled"] = True
    cfg["adaptive_support_scale"] = float(scale)
    cfg["adaptive_support_tile_pc2_median"] = diag["tile_pc2"]
    cfg["adaptive_support_dataset_pc2_median"] = diag["ref_pc2"]
    cfg["adaptive_support_dataset_density_pts_m2"] = diag["ref_density"]
    cfg["adaptive_support_tile_occ2"] = diag["tile_occ2"]
    cfg["adaptive_support_dataset_scale"] = diag["dataset_scale"]
    cfg["adaptive_support_tile_nudge"] = diag["tile_nudge"]
    return cfg


def sensor_defaults(
    sensor_mode: str,
    tile_support_stats: dict[str, Any] | None = None,
    dataset_support_stats: dict[str, Any] | None = None,
    *,
    adaptive: bool = True,
) -> dict:
    sm = (sensor_mode or "").upper().strip()
    if sm not in {"ALS", "ULS", "TLS"}:
        raise ValueError(f"sensor_mode must be one of ALS|ULS|TLS (got {sensor_mode!r})")

    if sm == "TLS":
        cfg = _tls_static_defaults()
    elif sm == "ULS":
        cfg = _uls_static_defaults()
    else:
        cfg = _als_static_defaults()

    if adaptive:
        cfg = _adapt_defaults(sm, cfg, tile_support_stats=tile_support_stats, dataset_support_stats=dataset_support_stats)
    else:
        cfg = dict(cfg)
        cfg["adaptive_support_enabled"] = False
        cfg["adaptive_support_scale"] = 1.0

    cfg["sensor_mode"] = sm
    return cfg
