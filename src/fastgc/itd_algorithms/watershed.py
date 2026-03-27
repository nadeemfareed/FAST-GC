from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    build_marker_raster,
    dual_gaussian_filter,
    label_peak_counts,
    labels_to_geojson,
    local_maxima_mask,
    peaks_to_geojson,
    prune_peaks_by_distance,
    read_chm_raster,
    relabel_sequential,
    remove_small_segments,
    screen_false_peaks,
    watershed_labels,
    write_float_raster,
    write_label_raster,
    write_shapefile,
)


DEFAULTS = {
    "itd_min_height": 2.0,
    "itd_crown_window_m": 3.0,
    "itd_min_peak_separation_m": 1.5,
    "itd_angle_threshold_deg": 110.0,
    "itd_screen_max_pair_distance_m": 6.0,
    "itd_banded_neighborhood_px": 1,
    "itd_min_crown_area_m2": 0.75,
    "itd_write_filtered_chm": True,
}


def run_itd_on_chm(
    chm_raster: str | Path,
    out_root: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    chm_raster = Path(chm_raster)
    out_root = Path(out_root)
    stem = chm_raster.stem

    min_height = float(kwargs.get("itd_min_height", DEFAULTS["itd_min_height"]))
    crown_window_m = float(kwargs.get("itd_crown_window_m", DEFAULTS["itd_crown_window_m"]))
    min_peak_sep_m = float(kwargs.get("itd_min_peak_separation_m", DEFAULTS["itd_min_peak_separation_m"]))
    angle_threshold_deg = float(kwargs.get("itd_angle_threshold_deg", DEFAULTS["itd_angle_threshold_deg"]))
    screen_max_pair_distance_m = float(kwargs.get("itd_screen_max_pair_distance_m", DEFAULTS["itd_screen_max_pair_distance_m"]))
    banded_neighborhood_px = int(kwargs.get("itd_banded_neighborhood_px", DEFAULTS["itd_banded_neighborhood_px"]))
    min_crown_area_m2 = float(kwargs.get("itd_min_crown_area_m2", DEFAULTS["itd_min_crown_area_m2"]))
    write_filtered = bool(kwargs.get("itd_write_filtered_chm", DEFAULTS["itd_write_filtered_chm"]))

    arr, profile, transform, crs = read_chm_raster(chm_raster)
    canopy_mask = np.isfinite(arr) & (arr >= min_height)

    filtered = dual_gaussian_filter(arr)
    filtered[~np.isfinite(arr)] = np.nan

    px = abs(float(transform.a))
    py = abs(float(transform.e))
    mean_res = max((px + py) / 2.0, 1e-6)
    window_pixels = max(3, int(round(crown_window_m / mean_res)))
    if window_pixels % 2 == 0:
        window_pixels += 1

    peak_mask_raw = local_maxima_mask(
        filtered,
        window_pixels=window_pixels,
        min_height=min_height,
        valid_mask=canopy_mask,
    )
    peak_rows_raw, peak_cols_raw = np.where(peak_mask_raw)
    peak_values_raw = filtered[peak_rows_raw, peak_cols_raw]

    peak_rows, peak_cols, peak_values = prune_peaks_by_distance(
        peak_rows_raw,
        peak_cols_raw,
        peak_values_raw,
        transform=transform,
        min_separation_m=min_peak_sep_m,
    )

    peak_rows, peak_cols, peak_values = screen_false_peaks(
        filtered,
        peak_rows,
        peak_cols,
        peak_values,
        transform=transform,
        angle_threshold_deg=angle_threshold_deg,
        band_width_px=banded_neighborhood_px,
        max_pair_distance_m=screen_max_pair_distance_m,
    )

    markers = build_marker_raster(filtered.shape, peak_rows, peak_cols)
    labels = watershed_labels(filtered, markers, canopy_mask)
    labels = remove_small_segments(labels, transform=transform, min_area_m2=min_crown_area_m2)
    labels = relabel_sequential(labels)

    peak_counts = label_peak_counts(labels, peak_rows_raw, peak_cols_raw)

    labels_fp = out_root / "labels" / f"{stem}_watershed_labels.tif"
    treetops_fc = peaks_to_geojson(
        peak_rows,
        peak_cols,
        peak_values,
        transform=transform,
        labels=labels,
        peak_counts=peak_counts,
    )
    crowns_fc = labels_to_geojson(
        labels,
        transform=transform,
        height_surface=arr,
        peak_counts=peak_counts,
    )

    treetops_fp = out_root / "treetops" / f"{stem}_watershed_treetops.shp"
    crowns_fp = out_root / "crowns" / f"{stem}_watershed_crowns.shp"

    outputs: dict[str, str] = {
        "labels_raster": write_label_raster(labels_fp, labels, profile),
        "treetops_shp": write_shapefile(treetops_fp, treetops_fc, crs=crs),
        "crowns_shp": write_shapefile(crowns_fp, crowns_fc, crs=crs),
    }

    if write_filtered:
        outputs["filtered_chm_raster"] = write_float_raster(
            out_root / "qa" / f"{stem}_watershed_filtered_chm.tif",
            filtered,
            profile,
        )

    multi_apex_segments = int(sum(1 for v in peak_counts.values() if v > 1))

    return {
        "status": "ok",
        "method": "watershed",
        "source_raster": str(chm_raster),
        "outputs": outputs,
        "tree_count": int(np.max(labels)) if labels.size else 0,
        "seed_count_raw": int(len(peak_rows_raw)),
        "seed_count_screened": int(len(peak_rows)),
        "multi_apex_segments": multi_apex_segments,
        "notes": [
            "Treetops are written as ESRI Shapefile points.",
            "Crowns are written as ESRI Shapefile polygons.",
            "Crown attributes include area_m2, rad_m, max_h_m, and mean_h_m.",
            "Segments containing more than one raw local maximum are flagged using multi_apx and pk_count.",
        ],
        "parameters": {
            "itd_min_height": min_height,
            "itd_crown_window_m": crown_window_m,
            "itd_min_peak_separation_m": min_peak_sep_m,
            "itd_angle_threshold_deg": angle_threshold_deg,
            "itd_screen_max_pair_distance_m": screen_max_pair_distance_m,
            "itd_banded_neighborhood_px": banded_neighborhood_px,
            "itd_min_crown_area_m2": min_crown_area_m2,
        },
    }