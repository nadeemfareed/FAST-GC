from __future__ import annotations


def sensor_defaults(sensor_mode: str) -> dict:
    sm = (sensor_mode or "").upper().strip()
    if sm not in {"ALS", "ULS", "TLS"}:
        raise ValueError(f"sensor_mode must be one of ALS|ULS|TLS (got {sensor_mode!r})")

    if sm == "TLS":
        vote_cell_m = 0.35
        vote_top_m = 6
        vote_neighbor_radius_cells = 4
        vote_min_neighbor_cells = 10
        vote_max_robust_z = 2.6
        vote_mad_floor = 0.05
        vote_fill_iters = 100
        vote_smooth_sigma_cells = 0.8
        vote_ground_threshold_m = 0.10
        vote_slope_adapt_k = 0.25

        cand_cell_m = 2.0
        cand_dz_m = 0.20
        base_cell_m = 2.0
        base_radius_m = 0.25

        # Disabled for TLS for now
        void_recover_enabled = False

    elif sm == "ULS":
        vote_cell_m = 0.75
        vote_top_m = 4
        vote_neighbor_radius_cells = 4
        vote_min_neighbor_cells = 8
        vote_max_robust_z = 2.8
        vote_mad_floor = 0.05
        vote_fill_iters = 100
        vote_smooth_sigma_cells = 1.0
        vote_ground_threshold_m = 0.18
        vote_slope_adapt_k = 0.30

        cand_cell_m = 1.5
        cand_dz_m = 0.60
        base_cell_m = 1.5
        base_radius_m = 0.50

        void_recover_enabled = True

    else:  # ALS
        vote_cell_m = 1.00
        vote_top_m = 3
        vote_neighbor_radius_cells = 3
        vote_min_neighbor_cells = 6
        vote_max_robust_z = 3.0
        vote_mad_floor = 0.05
        vote_fill_iters = 80
        vote_smooth_sigma_cells = 1.2
        vote_ground_threshold_m = 0.30
        vote_slope_adapt_k = 0.40

        cand_cell_m = 2.0
        cand_dz_m = 0.75
        base_cell_m = 2.0
        base_radius_m = 1.00

        void_recover_enabled = True

    return dict(
        sensor_mode=sm,

        # support layer
        cand_cell_m=cand_cell_m,
        cand_dz_m=cand_dz_m,

        # compatibility / metadata
        base_cell_m=base_cell_m,
        base_radius_m=base_radius_m,

        # initial vote classifier
        vote_cell_m=vote_cell_m,
        vote_top_m=vote_top_m,
        vote_neighbor_radius_cells=vote_neighbor_radius_cells,
        vote_min_neighbor_cells=vote_min_neighbor_cells,
        vote_max_robust_z=vote_max_robust_z,
        vote_mad_floor=vote_mad_floor,
        vote_fill_iters=vote_fill_iters,
        vote_smooth_sigma_cells=vote_smooth_sigma_cells,
        vote_ground_threshold_m=vote_ground_threshold_m,
        vote_slope_adapt_k=vote_slope_adapt_k,

        # targeted void recovery
        void_recover_enabled=void_recover_enabled,
        void_recover_cell_m=vote_cell_m,
        void_recover_ground_buffer_m=5.0,
        void_recover_slope_thr_deg=6.0,
        void_recover_slope_break_thr=0.08,
        void_recover_min_component_cells=2,
        void_recover_max_component_cells=500,
        void_recover_bin_dz_m=0.20 if sm == "ULS" else 0.25,
        void_recover_max_bins=2,
        void_recover_z_std_thr_m=0.12 if sm == "ULS" else 0.15,
        void_recover_z_span_thr_m=0.35 if sm == "ULS" else 0.45,
        void_recover_z_tol_m=0.18 if sm == "ULS" else 0.22,
        void_recover_plane_tol_m=0.18 if sm == "ULS" else 0.22,
        void_recover_neighbor_radius_cells=2,
        void_recover_min_neighbor_ground_cells=4,
    )