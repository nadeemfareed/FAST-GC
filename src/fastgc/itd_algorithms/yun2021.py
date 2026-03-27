from __future__ import annotations

import heapq
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import Delaunay

from .common import (
    build_marker_raster,
    dual_gaussian_filter,
    label_peak_counts,
    labels_to_feature_collection,
    local_maxima_mask,
    map_point,
    morphological_hole_fill,
    peaks_to_feature_collection,
    pixel_size,
    prune_peaks_by_distance,
    read_surface_raster,
    relabel_sequential,
    remove_small_segments,
    screen_false_peaks,
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
    "itd_write_filtered_surface": True,
    "itd_hj": 0.50,
    "itd_alpha": 0.65,
    "itd_beta": 1.00,
    "itd_gamma": 0.85,
    "itd_max_crown_radius_m": 12.0,
    "itd_polygon_smooth_distance": 0.20,
    "itd_polygon_simplify_tol": 0.10,
}


def _build_peak_neighbors(
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    transform,
) -> dict[int, set[int]]:
    n = len(rows)
    if n <= 1:
        return {i: set() for i in range(n)}

    xy = np.array([map_point(transform, int(r), int(c)) for r, c in zip(rows, cols)], dtype=np.float64)
    neighbors = {i: set() for i in range(n)}

    if n == 2:
        neighbors[0].add(1)
        neighbors[1].add(0)
        return neighbors

    tri = Delaunay(xy)
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                a = int(simplex[i])
                b = int(simplex[j])
                neighbors[a].add(b)
                neighbors[b].add(a)

    return neighbors


def _nearest_peak_distance_map(
    peak_rows: np.ndarray,
    peak_cols: np.ndarray,
    *,
    transform,
    shape_: tuple[int, int],
) -> np.ndarray:
    """
    Approximate local competition scale from nearest peak spacing.
    """
    h, w = shape_
    out = np.full((h, w), np.nan, dtype=np.float32)

    if len(peak_rows) == 0:
        return out

    xy = np.array([map_point(transform, int(r), int(c)) for r, c in zip(peak_rows, peak_cols)], dtype=np.float64)

    nn_d = np.full(len(xy), np.inf, dtype=np.float32)
    if len(xy) > 1:
        for i in range(len(xy)):
            d = np.hypot(xy[:, 0] - xy[i, 0], xy[:, 1] - xy[i, 1])
            d[i] = np.inf
            nn_d[i] = float(np.min(d))
    else:
        nn_d[:] = 8.0

    for i, (r, c) in enumerate(zip(peak_rows, peak_cols)):
        out[int(r), int(c)] = nn_d[i]

    valid = np.isfinite(out)
    if np.any(valid):
        inds = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        out = out[tuple(inds)]

    return out.astype(np.float32)


def _yun2021_water_expansion(
    surface: np.ndarray,
    peak_rows: np.ndarray,
    peak_cols: np.ndarray,
    peak_values: np.ndarray,
    *,
    transform,
    canopy_mask: np.ndarray,
    hj: float,
    alpha: float,
    beta: float,
    gamma: float,
    max_crown_radius_m: float,
) -> np.ndarray:
    """
    Practical Yun-style layered expansion:
    - local maxima as seeds
    - expansion in descending surface order
    - energy penalizes steep gradient, loss of height relative to seed, and
      excessive radial spread
    - synchronous competition via global priority queue
    """
    h, w = surface.shape
    labels = np.zeros((h, w), dtype=np.int32)
    if len(peak_rows) == 0:
        return labels

    filled = surface.astype(np.float32, copy=True)
    filled[~np.isfinite(filled)] = -np.inf

    gx = ndi.sobel(np.where(np.isfinite(surface), surface, 0.0), axis=1, mode="nearest")
    gy = ndi.sobel(np.where(np.isfinite(surface), surface, 0.0), axis=0, mode="nearest")
    grad = np.hypot(gx, gy).astype(np.float32)
    grad_scale = float(np.nanpercentile(grad[np.isfinite(grad)], 95)) if np.any(np.isfinite(grad)) else 1.0
    grad_scale = max(grad_scale, 1e-6)

    dx, dy = pixel_size(transform)
    mean_res = max((dx + dy) / 2.0, 1e-6)

    seed_rc = [(int(r), int(c)) for r, c in zip(peak_rows, peak_cols)]
    seed_h = np.asarray(peak_values, dtype=np.float32)
    local_scale = _nearest_peak_distance_map(peak_rows, peak_cols, transform=transform, shape_=surface.shape)
    local_scale = np.clip(local_scale, 2.0 * mean_res, max_crown_radius_m)

    pq: list[tuple[float, int, int, int]] = []

    for seed_id, (r, c) in enumerate(seed_rc, start=1):
        if not canopy_mask[r, c]:
            continue
        labels[r, c] = seed_id
        heapq.heappush(pq, (-float(filled[r, c]), seed_id, r, c))

    nbh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while pq:
        neg_priority, seed_id, r, c = heapq.heappop(pq)
        current_h = -neg_priority
        seed_idx = seed_id - 1
        sr, sc = seed_rc[seed_idx]
        sh = float(seed_h[seed_idx])

        for dr, dc in nbh:
            rr = r + dr
            cc = c + dc

            if rr < 0 or rr >= h or cc < 0 or cc >= w:
                continue
            if not canopy_mask[rr, cc]:
                continue
            if labels[rr, cc] != 0:
                continue
            if not np.isfinite(surface[rr, cc]):
                continue

            nh = float(surface[rr, cc])

            # layered expansion control
            if nh > current_h + hj:
                continue

            # energy terms
            dh_seed = max(sh - nh, 0.0)
            g = float(grad[rr, cc] / grad_scale)

            # radial term
            x0, y0 = map_point(transform, sr, sc)
            x1, y1 = map_point(transform, rr, cc)
            radial = math.hypot(x1 - x0, y1 - y0)

            local_r = float(local_scale[sr, sc]) if np.isfinite(local_scale[sr, sc]) else max_crown_radius_m
            local_r = min(max(local_r, 2.0 * mean_res), max_crown_radius_m)
            radial_norm = radial / max(local_r, 1e-6)

            # Yun-style control: prefer low energy while expanding downhill
            energy = alpha * (dh_seed / max(sh, 1e-6)) + beta * g + gamma * radial_norm

            # hard cut against unrealistic expansion
            if radial > max_crown_radius_m:
                continue
            if radial_norm > 1.35:
                continue
            if energy > 2.0:
                continue

            labels[rr, cc] = seed_id
            priority = nh - energy
            heapq.heappush(pq, (-priority, seed_id, rr, cc))

    return relabel_sequential(labels)


def run_itd_on_surface(
    surface_raster: str | Path,
    out_root: str | Path,
    *,
    surface_type: str = "CHM",
    surface_variant: str = "unknown",
    **kwargs: Any,
) -> dict[str, Any]:
    surface_raster = Path(surface_raster)
    out_root = Path(out_root)
    stem = surface_raster.stem

    min_height = float(kwargs.get("itd_min_height", DEFAULTS["itd_min_height"]))
    crown_window_m = float(kwargs.get("itd_crown_window_m", DEFAULTS["itd_crown_window_m"]))
    min_peak_sep_m = float(kwargs.get("itd_min_peak_separation_m", DEFAULTS["itd_min_peak_separation_m"]))
    angle_threshold_deg = float(kwargs.get("itd_angle_threshold_deg", DEFAULTS["itd_angle_threshold_deg"]))
    screen_max_pair_distance_m = float(kwargs.get("itd_screen_max_pair_distance_m", DEFAULTS["itd_screen_max_pair_distance_m"]))
    banded_neighborhood_px = int(kwargs.get("itd_banded_neighborhood_px", DEFAULTS["itd_banded_neighborhood_px"]))
    min_crown_area_m2 = float(kwargs.get("itd_min_crown_area_m2", DEFAULTS["itd_min_crown_area_m2"]))
    write_filtered = bool(kwargs.get("itd_write_filtered_surface", DEFAULTS["itd_write_filtered_surface"]))

    hj = float(kwargs.get("itd_hj", DEFAULTS["itd_hj"]))
    alpha = float(kwargs.get("itd_alpha", DEFAULTS["itd_alpha"]))
    beta = float(kwargs.get("itd_beta", DEFAULTS["itd_beta"]))
    gamma = float(kwargs.get("itd_gamma", DEFAULTS["itd_gamma"]))
    max_crown_radius_m = float(kwargs.get("itd_max_crown_radius_m", DEFAULTS["itd_max_crown_radius_m"]))
    poly_smooth = float(kwargs.get("itd_polygon_smooth_distance", DEFAULTS["itd_polygon_smooth_distance"]))
    poly_simplify = float(kwargs.get("itd_polygon_simplify_tol", DEFAULTS["itd_polygon_simplify_tol"]))

    arr, profile, transform, crs = read_surface_raster(surface_raster)

    # Yun paper emphasizes DSM and uses filtering + morphology before peak screening.
    base = morphological_hole_fill(arr, structure_size=3, min_valid_value=min_height)
    filtered = dual_gaussian_filter(base, sigma_dist=0.9, sigma_height=1.8)
    filtered[~np.isfinite(arr)] = np.nan

    canopy_mask = np.isfinite(filtered) & (filtered >= min_height)

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

    labels = _yun2021_water_expansion(
        filtered,
        peak_rows,
        peak_cols,
        peak_values,
        transform=transform,
        canopy_mask=canopy_mask,
        hj=hj,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        max_crown_radius_m=max_crown_radius_m,
    )

    labels = remove_small_segments(labels, transform=transform, min_area_m2=min_crown_area_m2)
    labels = relabel_sequential(labels)

    peak_counts = label_peak_counts(labels, peak_rows_raw, peak_cols_raw)

    labels_fp = out_root / "labels" / f"{stem}_yun2021_labels.tif"

    treetops_fc = peaks_to_feature_collection(
        peak_rows,
        peak_cols,
        peak_values,
        transform=transform,
        labels=labels,
        peak_counts=peak_counts,
    )
    crowns_fc = labels_to_feature_collection(
        labels,
        transform=transform,
        height_surface=arr,
        peak_counts=peak_counts,
        smooth_distance=poly_smooth,
        simplify_tol=poly_simplify,
    )

    treetops_fp = out_root / "treetops" / f"{stem}_yun2021_treetops.shp"
    crowns_fp = out_root / "crowns" / f"{stem}_yun2021_crowns.shp"

    outputs: dict[str, str] = {
        "labels_raster": write_label_raster(labels_fp, labels, profile),
        "treetops_shp": write_shapefile(treetops_fp, treetops_fc, crs=crs),
        "crowns_shp": write_shapefile(crowns_fp, crowns_fc, crs=crs),
    }

    if write_filtered:
        outputs["filtered_surface_raster"] = write_float_raster(
            out_root / "qa" / f"{stem}_yun2021_filtered_{surface_type.lower()}.tif",
            filtered,
            profile,
        )

    multi_apex_segments = int(sum(1 for v in peak_counts.values() if v > 1))

    return {
        "status": "ok",
        "method": "yun2021",
        "surface_type": surface_type,
        "surface_variant": surface_variant,
        "source_raster": str(surface_raster),
        "outputs": outputs,
        "tree_count": int(np.max(labels)) if labels.size else 0,
        "seed_count_raw": int(len(peak_rows_raw)),
        "seed_count_screened": int(len(peak_rows)),
        "multi_apex_segments": multi_apex_segments,
        "notes": [
            "Paper-inspired ITD using dual filtering, morphological preprocessing, angle-threshold peak screening, and constrained water expansion.",
            "Works on DSM or CHM surfaces.",
            "Treetops are ESRI Shapefile points; crowns are ESRI Shapefile polygons.",
        ],
        "parameters": {
            "itd_min_height": min_height,
            "itd_crown_window_m": crown_window_m,
            "itd_min_peak_separation_m": min_peak_sep_m,
            "itd_angle_threshold_deg": angle_threshold_deg,
            "itd_screen_max_pair_distance_m": screen_max_pair_distance_m,
            "itd_banded_neighborhood_px": banded_neighborhood_px,
            "itd_min_crown_area_m2": min_crown_area_m2,
            "itd_hj": hj,
            "itd_alpha": alpha,
            "itd_beta": beta,
            "itd_gamma": gamma,
            "itd_max_crown_radius_m": max_crown_radius_m,
            "itd_polygon_smooth_distance": poly_smooth,
            "itd_polygon_simplify_tol": poly_simplify,
        },
    }


# backward-compatible hook for older CHM-only router
def run_itd_on_chm(
    chm_raster: str | Path,
    out_root: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    return run_itd_on_surface(
        surface_raster=chm_raster,
        out_root=out_root,
        surface_type="CHM",
        surface_variant=str(kwargs.get("source_chm_variant", "unknown")),
        **kwargs,
    )