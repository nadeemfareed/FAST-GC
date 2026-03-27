from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    dual_gaussian_filter,
    local_maxima_mask,
    peaks_to_geojson,
    prune_peaks_by_distance,
    read_chm_raster,
    write_geojson,
)


def run_itd_on_chm(chm_raster: str | Path, out_root: str | Path, **kwargs: Any) -> dict[str, Any]:
    chm_raster = Path(chm_raster)
    out_root = Path(out_root)
    stem = chm_raster.stem

    min_height = float(kwargs.get("itd_min_height", 2.0))
    crown_window_m = float(kwargs.get("itd_crown_window_m", 3.0))
    min_peak_sep_m = float(kwargs.get("itd_min_peak_separation_m", 1.5))

    arr, _, transform, _ = read_chm_raster(chm_raster)
    filtered = dual_gaussian_filter(arr)
    canopy_mask = np.isfinite(arr) & (arr >= min_height)

    mean_res = max((abs(float(transform.a)) + abs(float(transform.e))) / 2.0, 1e-6)
    window_pixels = max(3, int(round(crown_window_m / mean_res)))
    if window_pixels % 2 == 0:
        window_pixels += 1

    peaks = local_maxima_mask(filtered, window_pixels=window_pixels, min_height=min_height, valid_mask=canopy_mask)
    rows, cols = np.where(peaks)
    values = filtered[rows, cols]
    rows, cols, values = prune_peaks_by_distance(rows, cols, values, transform=transform, min_separation_m=min_peak_sep_m)

    peaks_fp = out_root / "treetops" / f"{stem}_lmf_treetops.geojson"
    out = write_geojson(peaks_fp, peaks_to_geojson(rows, cols, values, transform=transform))

    return {
        "status": "ok",
        "method": "lmf",
        "source_raster": str(chm_raster),
        "outputs": {"treetops_geojson": out},
        "tree_count": int(len(rows)),
        "notes": [
            "LMF currently returns treetop detections only; crown delineation is not part of this module.",
            "The same adaptive smoothing backbone used by the watershed baseline is applied before local-maxima extraction.",
        ],
        "parameters": {
            "itd_min_height": min_height,
            "itd_crown_window_m": crown_window_m,
            "itd_min_peak_separation_m": min_peak_sep_m,
        },
    }
