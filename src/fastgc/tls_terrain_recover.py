"""Compatibility shim for the stable TLS baseline.

The stable baseline performs support propagation and curvature handling in tls_vote.py.
This former post-membrane recovery stage is intentionally a NO-OP so current io_las.py
can remain unchanged and ALS/ULS code paths are untouched.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class TlsTerrainRecoveryConfig:
    enabled: bool = True
    cell: float = 0.15
    max_passes: int = 3
    low_cluster_gap_m: float = 0.040
    low_cluster_span_max_m: float = 0.060
    min_cluster_points: int = 2
    max_points_examined: int = 48
    detached_low_noise_gap_m: float = 0.12
    detached_low_noise_max_points: int = 2
    min_directional_support: int = 2
    bilateral_radius_cells: int = 4
    base_residual_m: float = 0.050
    slope_residual_k: float = 0.42
    bilateral_base_residual_m: float = 0.070
    max_residual_m: float = 0.135
    max_neighbor_step_m: float = 0.18
    point_lower_m: float = 0.030
    point_upper_m: float = 0.050

@dataclass
class TlsTerrainRecoveryResult:
    final_ground: np.ndarray
    recovered_ground: np.ndarray
    recovered_cells: np.ndarray
    observation_z: np.ndarray
    support_z: np.ndarray
    support_count: np.ndarray
    x0: float
    y0: float
    cell: float
    passes_used: int

def recover_tls_microtopography(*, x, y, z, trusted_ground, candidate_pool, config=None):
    g=np.asarray(trusted_ground,bool).copy(); n=g.size
    return TlsTerrainRecoveryResult(g,np.zeros(n,bool),np.empty((0,0),bool),np.empty((0,0),np.float32),np.empty((0,0),np.float32),np.empty((0,0),np.uint8),
                                    float(np.min(x)) if np.size(x) else 0.0,float(np.min(y)) if np.size(y) else 0.0,float(getattr(config,'cell',0.15)),0)
