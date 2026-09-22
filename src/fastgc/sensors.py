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

        # ------------------------------------------------------------
        # Mature aerial terrain-QC stages.
        #
        # These modules were originally developed under the als_*
        # namespace.  ULS now uses the same terrain reasoning, but at
        # finer spatial scales and with conservative tolerances because
        # total ULS density can be high while true ground support is
        # locally weak under canopy and near swath edges.
        # ------------------------------------------------------------

        # Post-final terrain-sheet completion: promotion only.
        als_postfinal_surface_enabled=True,
        als_postfinal_surface_cell_m=0.40,
        als_postfinal_surface_p90_quantile=0.90,
        als_postfinal_surface_smooth_sigma_cells=2.0,
        als_postfinal_surface_base_tol_m=0.22,
        als_postfinal_surface_slope_start_deg=10.0,
        als_postfinal_surface_slope_gain_m_per_deg=0.016,
        als_postfinal_surface_max_tol_m=0.80,
        als_postfinal_surface_max_sheet_excess_m=0.20,
        als_postfinal_surface_hard_vertical_span_m=7.0,
        als_postfinal_surface_hard_span_explained_fraction=0.82,

        # Statistical terrain-sheet cleaner: demotion only.
        als_statsheet_enabled=True,
        als_statsheet_cell_m=0.75,
        als_statsheet_sigma_cells=1.0,
        als_statsheet_iterations=2,
        als_statsheet_low_zrange_m=1.25,
        als_statsheet_high_zrange_m=3.5,
        als_statsheet_low_iqr_m=0.50,
        als_statsheet_high_iqr_m=1.25,
        als_statsheet_tol_thin_m=2.00,
        als_statsheet_tol_moderate_m=0.85,
        als_statsheet_tol_complex_m=0.55,
        als_statsheet_min_upper_fraction=0.10,
        als_statsheet_upper_tail_start_m=1.25,
        als_statsheet_min_points_per_cell=5,

        # 3-D facet / residual support consistency.
        # Deliberately conservative in Phase 1: ULS may have sparse
        # terrain returns despite very high vegetation-return density.
        als_residual_sheet_outlier_enabled=True,
        als_residual_sheet_outlier_cell_m=3.0,
        als_residual_sheet_outlier_min_ground_points_per_cell=8,
        als_residual_sheet_outlier_trigger_quantile=0.92,
        als_residual_sheet_outlier_trigger_floor_m=0.90,
        als_residual_sheet_outlier_sheet_radius_m=5.0,
        als_residual_sheet_outlier_sheet_min_points=18,
        als_residual_sheet_outlier_sheet_mad_k=3.2,
        als_residual_sheet_outlier_sheet_floor_m=0.16,
        als_residual_sheet_outlier_same_ground_radius_m=0.65,
        als_residual_sheet_outlier_max_ground_neighbors=8,
        als_residual_sheet_outlier_below_xy_radius_m=1.00,
        als_residual_sheet_outlier_below_min_points=4,
        als_residual_sheet_outlier_below_min_clearance_m=0.18,
        als_residual_sheet_outlier_force_residual_m=0.40,
        als_residual_sheet_outlier_max_iterations=3,

        # Ground-only vertical-profile contamination guard.
        als_ground_profile_footprints_m=(4.0, 6.0, 8.0),
        als_ground_profile_z_bin_m=0.20,
        als_ground_profile_mode_gap_m=0.65,
        als_ground_profile_min_cell_points=6,
        als_ground_profile_min_mode_points=3,
        als_ground_profile_min_mode_fraction=0.05,
        als_ground_profile_min_above_mode_m=1.10,
        als_ground_profile_hard_above_mode_m=2.25,
        als_ground_profile_neighbor_k=8,
        als_ground_profile_max_neighbor_distance_factor=2.75,
        als_ground_profile_min_plane_neighbors=3,
        als_ground_profile_max_plane_slope=5.0,
        als_ground_profile_min_votes=2,
        als_ground_profile_hard_vote_residual_m=2.75,
        als_ground_profile_protect_margin_m=0.30,

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
        vote_top_m=12,
        vote_neighbor_radius_cells=4,
        vote_min_neighbor_cells=8,
        vote_max_robust_z=2.6,
        vote_mad_floor=0.03,
        vote_fill_iters=24,
        vote_smooth_sigma_cells=0.65,
        vote_ground_threshold_m=0.10,
        vote_slope_adapt_k=0.22,
        # TLS conservative terrain-domain voxel reduction.
        # This is a preprocessing accelerator, not a ground classifier.
        tls_voxel_reduce_enabled=True,
        tls_voxel_fine_cell_m=0.50,
        tls_voxel_support_window_m=5.0,
        tls_voxel_low_quantile=0.05,
        tls_voxel_min_points_per_cell=3,
        tls_voxel_min_support_cells=5,
        tls_voxel_vertical_m=0.25,
        tls_voxel_safe_height_m=2.0,

        # TLS-v2 internal support-surface controls
        tls_candidate_cluster_gap_m=0.08,
        tls_candidate_cluster_span_m=0.14,
        tls_candidate_min_cluster_points=3,
        tls_below_ground_gap_m=0.16,
        tls_below_ground_max_cluster_points=2,
        tls_use_offset_grid=True,
        tls_radius_min_cells=2,
        tls_radius_max_cells=12,
        tls_min_support_sectors=3,
        tls_seed_confidence=0.68,
        tls_seed_max_vertical_span_m=0.35,
        tls_propagation_iters=24,
        tls_max_extrapolation_cells=5,
        tls_plane_min_support=6,
        tls_plane_max_residual_m=0.14,
        tls_fill_min_bank_fraction=0.50,
        tls_fill_max_distance_cells=5,
        tls_reject_edge_connected_voids=True,
        tls_smooth_iters=2,
        tls_bilateral_height_sigma_m=0.18,
        tls_bilateral_slope_sigma=0.60,
        tls_curvature_adapt_k=0.05,
        tls_roughness_adapt_k=0.30,
        tls_threshold_min_m=0.03,
        tls_threshold_max_m=0.28,
        tls_lower_threshold_factor=1.75,
        tls_min_surface_confidence=0.28,
        tls_min_classification_confidence=0.34,
        tls_weak_confidence_tighten=0.45,
        cand_cell_m=2.0,
        cand_dz_m=0.20,
        base_cell_m=2.0,
        base_radius_m=0.25,
        void_recover_enabled=False,

        # TLS legacy compatibility shims.
        #
        # tls_membrane_refine.py and tls_terrain_recover.py intentionally
        # preserve their historical public APIs but are NO-OPs in the
        # stable production TLS pipeline. Keep the switches explicit and
        # disabled so configuration reflects actual runtime behavior.
        tls_membrane_enabled=False,
        tls_terrain_recovery_enabled=False,

        # ----------------------------------------------------
        # TLS V3 ACTIVE-EVIDENCE SAMPLING
        # ----------------------------------------------------
        # Density-normalized 3-D evidence used only to construct the
        # TLS terrain surface. Classification is subsequently applied
        # to the complete V2 terrain domain.
        tls_evidence_xy_voxel_m=0.20,
        tls_evidence_z_voxel_m=0.10,
        tls_evidence_max_points_per_voxel=8,

        # ----------------------------------------------------
        # FINAL TLS HIGH-CONFIDENCE TERRAIN VOTE
        # ----------------------------------------------------
        # Runs after every other TLS/common QC operation.
        # Strictly demotion-only.
        tls_final_ground_vote_enabled=True,
        tls_final_ground_vote_min_support_points=100,

        tls_final_ground_vote_cell_m=0.35,



        tls_final_ground_vote_min_neighbor_cells=8,
        tls_final_ground_vote_min_support_sectors=3,

        tls_final_ground_vote_max_robust_z=2.5,
        tls_final_ground_vote_mad_floor_m=0.025,







        tls_final_ground_vote_roughness_k=0.25,

        # Lower coherent terrain scaffold.
        tls_final_ground_vote_radius_m=2.5,
        tls_final_ground_vote_normal_tol_m=0.075,
        tls_final_ground_vote_scaffold_upper_sigma=2.0,
        tls_final_ground_vote_scaffold_lower_sigma=4.0,

        # Curvature fallback. Used only when materially better
        # than the local plane.
        tls_final_ground_vote_quadratic_min_support=12,
        tls_final_ground_vote_quadratic_trigger_rmse_m=0.030,
        tls_final_ground_vote_coarse_veto_enabled=True,
        tls_final_ground_vote_coarse_radius_m=5.0,
        tls_final_ground_vote_coarse_exclusion_m=2.75,
        tls_final_ground_vote_coarse_gap_m=2.0,
        tls_final_ground_vote_coarse_min_support=8,
        tls_final_ground_vote_quadratic_min_improvement=0.20,
        tls_final_ground_vote_quadratic_max_curvature=1.50,

        # Either globally snapped grid phase may validate a point.
        tls_final_ground_vote_required_votes=1,




        # Final point-first canopy-leak rejection. Reuses the
        # established ALS hard-airborne terrain confirmation only;
        # the full ALS raster/blob pipeline is not enabled for TLS.
        tls_hard_airborne_guard_enabled=False,
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
