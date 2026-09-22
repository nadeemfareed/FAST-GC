"""Compatibility shim for the stable TLS baseline.

The previous TLS membrane stage was capable of deleting near-surface woody material.
For the stable baseline it is intentionally a NO-OP while retaining the exact public
symbols expected by io_las.py. This lets the TLS route be simplified without touching
io_las.py, cli.py, ALS, or ULS.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class TlsMembraneConfig:
    enabled: bool = True
    cell: float = 0.15
    offset_fraction: float = 0.50
    cluster_gap_m: float = 0.035
    cluster_span_max_m: float = 0.040
    cluster_min_points: int = 2
    cluster_max_points_examined: int = 32
    below_noise_gap_m: float = 0.12
    below_noise_max_points: int = 2
    offset_agreement_m: float = 0.035
    anchor_min_offset_votes: int = 3
    anchor_min_score: float = 0.58
    anchor_min_local_occupancy: float = 0.22
    anchor_density_quantile: float = 0.45
    anchor_max_cluster_thickness_m: float = 0.035
    propagation_iters: int = 60
    propagation_radius_cells: int = 2
    max_anchor_distance_cells: int = 20
    weak_observation_gate_m: float = 0.080
    anchor_lock_weight: float = 12.0
    weak_observation_weight: float = 0.20
    relaxation: float = 0.65
    sheet_thickness_min_m: float = 0.010
    sheet_thickness_base_m: float = 0.018
    sheet_thickness_max_m: float = 0.035
    upper_factor: float = 1.00
    lower_factor: float = 1.35
    thickness_regularize_iters: int = 6
    min_keep_confidence: float = 0.22
    recover_ground: bool = False
    membership_min_offset_votes: int = 3
    membership_gap_min_m: float = 0.008
    membership_gap_sigma_k: float = 3.0
    membership_search_above_m: float = 0.080
    membership_anchor_upper_scale: float = 1.00
    membership_min_points_cell: int = 3
    min_candidate_points_cell: int = 1
    lower_quantile: float = 0.15
    upper_quantile: float = 0.85
    lower_cluster_quantile: float = 0.30
    max_lower_cluster_thickness_m: float = 0.08
    fine_window_cells: int = 3
    medium_window_cells: int = 7
    coarse_window_cells: int = 13
    min_support_fraction: float = 0.18
    convolution_iters: int = 3
    spatial_radius_cells: int = 2
    height_sigma_m: float = 0.06
    slope_sigma: float = 0.45
    max_fill_distance_cells: int = 12
    slope_thickness_k: float = 0.0
    roughness_thickness_k: float = 0.0
    curvature_thickness_k: float = 0.0
    max_vertical_spread_m: float = 0.22
    max_roughness_m: float = 0.08
    max_curvature: float = 1.20
    min_recovery_confidence: float = 0.70
    recovery_upper_m: float = 0.02
    recovery_lower_m: float = 0.03
    normal_sigma_deg: float = 28.0
    anchor_min_confidence: float = 0.50

@dataclass
class TlsMembraneResult:
    final_ground: np.ndarray
    removed_contaminants: np.ndarray
    recovered_ground: np.ndarray
    surface_z: np.ndarray
    confidence_grid: np.ndarray
    vertical_spread_grid: np.ndarray
    lower_cluster_thickness_grid: np.ndarray
    roughness_grid: np.ndarray
    slope_grid: np.ndarray
    curvature_grid: np.ndarray
    anchor_grid: np.ndarray
    offset_vote_grid: np.ndarray
    thickness_grid: np.ndarray
    x0: float
    y0: float
    cell: float

def refine_tls_ground_membrane(*, x, y, z, candidate_ground, config=None):
    g=np.asarray(candidate_ground,bool).copy(); n=g.size
    empty=np.empty((0,0),np.float32); emptyb=np.empty((0,0),bool)
    return TlsMembraneResult(g,np.zeros(n,bool),np.zeros(n,bool),empty,empty,empty,empty,empty,empty,empty,emptyb,np.empty((0,0),np.uint8),empty,
                            float(np.min(x)) if np.size(x) else 0.0,float(np.min(y)) if np.size(y) else 0.0,float(getattr(config,'cell',0.15)))
