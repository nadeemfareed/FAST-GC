"""FAST-GC canopy-height-model subsystem.

Public imports intentionally match the former single-file ``fastgc.chm`` API.
"""

from .api import (
    CHM_METHOD_CHOICES,
    CHM_SURFACE_METHOD_CHOICES,
    CHM_SMOOTH_CHOICES,
    build_chm_from_normalized_root,
    build_chm_from_dem_and_dsm,
    resolve_normalized_root,
    chm_output_label,
    chm_method_output_dir,
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
