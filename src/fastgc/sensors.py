
from __future__ import annotations


def sensor_defaults(sensor_mode: str) -> dict:
    sm = (sensor_mode or "").upper().strip()
    if sm not in {"ALS", "ULS", "TLS"}:
        raise ValueError(f"sensor_mode must be one of ALS|ULS|TLS (got {sensor_mode!r})")

    if sm == "TLS":
        cfg = dict(
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
    elif sm == "ULS":
        cfg = dict(
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
    else:
        cfg = dict(
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

    cfg["sensor_mode"] = sm
    return cfg
