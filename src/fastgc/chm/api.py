from __future__ import annotations

"""Stable public CHM API for FAST-GC.

The functions exported here preserve the API formerly provided by
``fastgc.chm``. Existing methods are delegated to the frozen 0.2.1 engine.
Future registered methods are dispatched without changing FAST-GC core/CLI
wiring.
"""

import os
from typing import Any

from . import legacy
from .registry import (
    CHM_METHOD_CHOICES,
    CHM_SURFACE_METHOD_CHOICES,
    CHM_SMOOTH_CHOICES,
    discover_builtin_methods,
    get_root_builder,
)

discover_builtin_methods()

chm_output_label = legacy.chm_output_label
chm_method_output_dir = legacy.chm_method_output_dir
resolve_normalized_root = legacy.resolve_normalized_root
build_chm_from_dem_and_dsm = legacy.build_chm_from_dem_and_dsm


def build_chm_from_normalized_root(
    normalized_root: str | os.PathLike[str],
    out_root: str | os.PathLike[str],
    *,
    sensor_mode: str,
    method: str = "p2r",
    surface_method: str | None = None,
    grid_res: float = 0.5,
    smooth_method: str = "none",
    percentile: float = 99.0,
    percentile_low: float | None = None,
    percentile_high: float | None = None,
    pitfree_thresholds: list[float] | None = None,
    use_first_returns: bool = False,
    spikefree_freeze_distance: float | None = None,
    spikefree_insertion_buffer: float | None = None,
    median_size: int = 0,
    gaussian_sigma: float = 1.0,
    min_height: float = 0.0,
    fill_ground_voids_zero: bool = True,
    void_ground_threshold: float = 0.15,
    overwrite: bool = False,
    n_jobs: int = 1,
    joblib_backend: str = "loky",
    joblib_batch_size: int | str = "auto",
    joblib_pre_dispatch: str | int = "2*n_jobs",
    pitfree_max_edge: float | list[float] | None = None,
    pitfree_subcircle: float | None = None,
    pitfree_highest: bool = True,
) -> str:
    key = str(method).strip().lower()

    # Preserve every established 0.2.1 pathway exactly.
    if key in legacy.CHM_METHOD_CHOICES:
        return legacy.build_chm_from_normalized_root(
            normalized_root,
            out_root,
            sensor_mode=sensor_mode,
            method=method,
            surface_method=surface_method,
            grid_res=grid_res,
            smooth_method=smooth_method,
            percentile=percentile,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
            pitfree_thresholds=pitfree_thresholds,
            use_first_returns=use_first_returns,
            spikefree_freeze_distance=spikefree_freeze_distance,
            spikefree_insertion_buffer=spikefree_insertion_buffer,
            median_size=median_size,
            gaussian_sigma=gaussian_sigma,
            min_height=min_height,
            fill_ground_voids_zero=fill_ground_voids_zero,
            void_ground_threshold=void_ground_threshold,
            overwrite=overwrite,
            n_jobs=n_jobs,
            joblib_backend=joblib_backend,
            joblib_batch_size=joblib_batch_size,
            joblib_pre_dispatch=joblib_pre_dispatch,
            pitfree_max_edge=pitfree_max_edge,
            pitfree_subcircle=pitfree_subcircle,
            pitfree_highest=pitfree_highest,
        )

    builder = get_root_builder(key)
    if builder is None:
        raise ValueError(f"Unsupported CHM method: {method}")

    return builder(
        normalized_root=normalized_root,
        out_root=out_root,
        sensor_mode=sensor_mode,
        method=key,
        surface_method=surface_method,
        grid_res=grid_res,
        smooth_method=smooth_method,
        percentile=percentile,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
        pitfree_thresholds=pitfree_thresholds,
        use_first_returns=use_first_returns,
        spikefree_freeze_distance=spikefree_freeze_distance,
        spikefree_insertion_buffer=spikefree_insertion_buffer,
        median_size=median_size,
        gaussian_sigma=gaussian_sigma,
        min_height=min_height,
        fill_ground_voids_zero=fill_ground_voids_zero,
        void_ground_threshold=void_ground_threshold,
        overwrite=overwrite,
        n_jobs=n_jobs,
        joblib_backend=joblib_backend,
        joblib_batch_size=joblib_batch_size,
        joblib_pre_dispatch=joblib_pre_dispatch,
        pitfree_max_edge=pitfree_max_edge,
        pitfree_subcircle=pitfree_subcircle,
        pitfree_highest=pitfree_highest,
    )


__all__ = [
    "CHM_METHOD_CHOICES",
    "CHM_SURFACE_METHOD_CHOICES",
    "CHM_SMOOTH_CHOICES",
    "build_chm_from_normalized_root",
    "build_chm_from_dem_and_dsm",
    "resolve_normalized_root",
    "chm_output_label",
    "chm_method_output_dir",
]
