from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any

import laspy
import numpy as np
from tqdm import tqdm


MANIFEST_VERSION = 6


def _iter_las_files(in_path: str, recursive: bool) -> list[str]:
    p = Path(in_path)
    if p.is_file():
        return [str(p)]
    if recursive:
        files = [str(q) for q in p.rglob("*") if q.is_file() and q.suffix.lower() in {".las", ".laz"}]
    else:
        files = [str(q) for q in p.iterdir() if q.is_file() and q.suffix.lower() in {".las", ".laz"}]
    files.sort()
    return files


def _dataset_label(in_path: str) -> str:
    p = Path(in_path)
    label = p.stem if p.is_file() else p.name
    label = label.strip() or "dataset"
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)


def get_workspace_root(in_path: str, out_dir: str | None, sensor_mode: str) -> Path:
    p = Path(in_path)
    base_root = Path(out_dir) if out_dir else (p.parent if p.is_file() else p.parent)
    return base_root / f"{sensor_mode.upper()}_tiles"


def _read_xy_bounds(src_fp: str) -> tuple[tuple[float, float, float, float] | None, int]:
    with laspy.open(src_fp) as reader:
        hdr = reader.header
        point_count = int(hdr.point_count)
        if point_count <= 0:
            return None, 0
        mins = np.asarray(hdr.mins, dtype=np.float64)
        maxs = np.asarray(hdr.maxs, dtype=np.float64)
        return (float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1])), point_count


def _read_header(src_fp: str) -> laspy.LasHeader:
    with laspy.open(src_fp) as reader:
        return reader.header


def _read_crs_wkt(src_fp: str) -> str | None:
    try:
        with laspy.open(src_fp) as reader:
            crs = reader.header.parse_crs()
            if crs is None:
                return None
            if hasattr(crs, "to_wkt"):
                return crs.to_wkt()
            return str(crs)
    except Exception:
        return None



def _safe_parse_header_crs(header: laspy.LasHeader):
    try:
        crs = header.parse_crs()
    except Exception:
        crs = None
    return crs


def _crs_from_wkt(crs_wkt: str | None):
    if not crs_wkt:
        return None
    try:
        import pyproj
        return pyproj.CRS.from_wkt(crs_wkt)
    except Exception:
        return None


def _attach_crs_if_possible(header: laspy.LasHeader, *, fallback_wkt: str | None = None) -> laspy.LasHeader:
    """
    Keep CRS/VLR information attached when survey-retiled LAS tiles are written.
    """
    crs = _safe_parse_header_crs(header)
    if crs is None:
        crs = _crs_from_wkt(fallback_wkt)

    if crs is not None:
        try:
            header.add_crs(crs)
        except Exception:
            pass
    return header


def _core_area(bounds: list[float] | tuple[float, float, float, float]) -> float:
    xmin, ymin, xmax, ymax = [float(v) for v in bounds]
    return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)


def _union_bounds(a: list[float], b: list[float]) -> list[float]:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    return [min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)]


def _buffer_from_core(core_bounds: list[float], buffer_m: float) -> list[float]:
    xmin, ymin, xmax, ymax = [float(v) for v in core_bounds]
    b = float(buffer_m)
    return [xmin - b, ymin - b, xmax + b, ymax + b]


def _bounds_intersect(
    a: list[float] | tuple[float, float, float, float],
    b: list[float] | tuple[float, float, float, float],
) -> bool:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def _intersection_area(
    a: list[float] | tuple[float, float, float, float],
    b: list[float] | tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    dx = min(ax1, bx1) - max(ax0, bx0)
    dy = min(ay1, by1) - max(ay0, by0)
    if dx <= 0.0 or dy <= 0.0:
        return 0.0
    return dx * dy


def _build_source_catalog(in_path: str, recursive: bool) -> list[dict[str, Any]]:
    files = _iter_las_files(in_path, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No LAS/LAZ files found in: {in_path}")

    catalog: list[dict[str, Any]] = []
    pbar = tqdm(files, desc="PLAN catalog", unit="file", dynamic_ncols=True)
    try:
        for src_fp in pbar:
            src_name = Path(src_fp).name
            pbar.set_postfix_str(src_name)
            bounds, point_count = _read_xy_bounds(src_fp)
            if bounds is None:
                continue
            catalog.append(
                {
                    "source_path": str(src_fp),
                    "source_name": src_name,
                    "source_bounds": [float(v) for v in bounds],
                    "source_point_count": int(point_count),
                    "source_crs_wkt": _read_crs_wkt(src_fp),
                }
            )
    finally:
        pbar.close()

    if not catalog:
        raise FileNotFoundError(f"No non-empty LAS/LAZ files found in: {in_path}")

    return catalog


def _catalog_union_bounds(catalog: list[dict[str, Any]]) -> list[float]:
    ub = None
    for rec in catalog:
        b = rec["source_bounds"]
        ub = list(b) if ub is None else _union_bounds(ub, b)
    if ub is None:
        raise RuntimeError("Unable to compute union bounds from empty source catalog.")
    return ub


def _plan_tiles_for_union_bounds(
    union_bounds: list[float],
    dataset_label: str,
    tile_size_m: float,
    buffer_m: float,
    small_tile_merge_frac: float,
) -> tuple[list[dict[str, Any]], int]:
    xmin, ymin, xmax, ymax = [float(v) for v in union_bounds]
    width = max(0.0, xmax - xmin)
    height = max(0.0, ymax - ymin)
    nx = max(1, int(math.ceil(width / tile_size_m)))
    ny = max(1, int(math.ceil(height / tile_size_m)))

    nominal_area = float(tile_size_m) * float(tile_size_m)
    area_threshold = float(max(0.0, small_tile_merge_frac)) * nominal_area

    raw_tiles: list[dict[str, Any]] = []
    for iy in range(ny):
        core_ymin = ymin + iy * tile_size_m
        core_ymax = min(core_ymin + tile_size_m, ymax)
        for ix in range(nx):
            core_xmin = xmin + ix * tile_size_m
            core_xmax = min(core_xmin + tile_size_m, xmax)
            core_bounds = [float(core_xmin), float(core_ymin), float(core_xmax), float(core_ymax)]
            tile_id = f"{dataset_label}_x{ix:04d}_y{iy:04d}"
            raw_tiles.append(
                {
                    "tile_id": tile_id,
                    "tile_name": f"{tile_id}.las",
                    "core_bounds": core_bounds,
                    "buffer_bounds": _buffer_from_core(core_bounds, buffer_m),
                    "tile_ix": int(ix),
                    "tile_iy": int(iy),
                    "merged_tile_count": 1,
                    "union_bounds": [float(v) for v in union_bounds],
                }
            )

    merged_small_count = 0
    if area_threshold > 0.0 and len(raw_tiles) > 1:
        by_idx: dict[tuple[int, int], dict[str, Any]] = {(int(t["tile_ix"]), int(t["tile_iy"])): t for t in raw_tiles}
        consumed: set[tuple[int, int]] = set()
        final_tiles: list[dict[str, Any]] = []

        for iy in range(ny):
            for ix in range(nx):
                key = (ix, iy)
                if key in consumed or key not in by_idx:
                    continue

                tile = by_idx[key]
                area = _core_area(tile["core_bounds"])

                if area < area_threshold:
                    neighbor_key = None
                    for cand in ((ix - 1, iy), (ix, iy - 1), (ix + 1, iy), (ix, iy + 1)):
                        if cand in by_idx and cand not in consumed:
                            neighbor_key = cand
                            break

                    if neighbor_key is not None:
                        nb = by_idx[neighbor_key]
                        nb["core_bounds"] = _union_bounds(nb["core_bounds"], tile["core_bounds"])
                        nb["buffer_bounds"] = _buffer_from_core(nb["core_bounds"], buffer_m)
                        nb["merged_tile_count"] = int(nb.get("merged_tile_count", 1)) + 1
                        consumed.add(key)
                        merged_small_count += 1
                        continue

                final_tiles.append(tile)
                consumed.add(key)

        seen_ids = {t["tile_id"] for t in final_tiles}
        for tile in by_idx.values():
            if tile["tile_id"] not in seen_ids and (int(tile["tile_ix"]), int(tile["tile_iy"])) not in consumed:
                final_tiles.append(tile)

        planned_tiles = []
        for t in sorted(final_tiles, key=lambda d: (d["tile_iy"], d["tile_ix"])):
            core = [float(v) for v in t["core_bounds"]]
            t = dict(t)
            t["core_bounds"] = core
            t["buffer_bounds"] = _buffer_from_core(core, buffer_m)
            planned_tiles.append(t)
        return planned_tiles, merged_small_count

    return raw_tiles, merged_small_count



def _quantile_or_nan(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def _grid_point_count_support_stats(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bounds: list[float],
    cell_m: float,
) -> dict[str, float]:
    xmin, ymin, xmax, ymax = [float(v) for v in bounds]
    width = max(0.0, xmax - xmin)
    height = max(0.0, ymax - ymin)
    nx = max(1, int(math.ceil(width / float(cell_m))))
    ny = max(1, int(math.ceil(height / float(cell_m))))
    total_cells = int(nx * ny)

    if x.size == 0:
        return {
            "cell_m": float(cell_m),
            "occupied_cells": 0.0,
            "total_cells": float(total_cells),
            "occupancy_ratio": 0.0,
            "pointcount_median": float("nan"),
            "pointcount_mean": float("nan"),
            "pointcount_q25": float("nan"),
            "pointcount_q75": float("nan"),
        }

    ix = np.floor((x - xmin) / float(cell_m)).astype(np.int64)
    iy = np.floor((y - ymin) / float(cell_m)).astype(np.int64)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    if not np.any(valid):
        return {
            "cell_m": float(cell_m),
            "occupied_cells": 0.0,
            "total_cells": float(total_cells),
            "occupancy_ratio": 0.0,
            "pointcount_median": float("nan"),
            "pointcount_mean": float("nan"),
            "pointcount_q25": float("nan"),
            "pointcount_q75": float("nan"),
        }

    key = iy[valid] * nx + ix[valid]
    _, counts = np.unique(key, return_counts=True)
    occupied = int(counts.size)
    occupancy = occupied / max(total_cells, 1)
    return {
        "cell_m": float(cell_m),
        "occupied_cells": float(occupied),
        "total_cells": float(total_cells),
        "occupancy_ratio": float(occupancy),
        "pointcount_median": float(np.median(counts)),
        "pointcount_mean": float(np.mean(counts)),
        "pointcount_q25": _quantile_or_nan(counts.astype(np.float64), 0.25),
        "pointcount_q75": _quantile_or_nan(counts.astype(np.float64), 0.75),
    }


def _compute_tile_support_stats(out_las: laspy.LasData, tile: dict[str, Any]) -> dict[str, float]:
    """
    FIXED: robust coordinate extraction for all LAS formats
    """

    # ---- SAFE XYZ EXTRACTION (CRITICAL FIX) ----
    try:
        x = np.asarray(out_las.x, dtype=np.float64)
        y = np.asarray(out_las.y, dtype=np.float64)
    except Exception:
        # fallback (works for all LAS)
        X = np.asarray(out_las.X, dtype=np.float64)
        Y = np.asarray(out_las.Y, dtype=np.float64)

        scale = out_las.header.scales
        offset = out_las.header.offsets

        x = X * scale[0] + offset[0]
        y = Y * scale[1] + offset[1]

    # ------------------------------------------

    point_count = int(x.size)

    bounds = [float(v) for v in tile.get("buffer_bounds", tile.get("core_bounds"))]
    area_m2 = _core_area(bounds)
    density = point_count / area_m2 if area_m2 > 0 else float("nan")

    g2 = _grid_point_count_support_stats(x, y, bounds=bounds, cell_m=2.0)
    g4 = _grid_point_count_support_stats(x, y, bounds=bounds, cell_m=4.0)

    return {
        "tile_area_m2": float(area_m2),
        "density_pts_m2": float(density),
        "grid_2m_pointcount_median": float(g2["pointcount_median"]),
        "grid_2m_occupancy_ratio": float(g2["occupancy_ratio"]),
        "grid_4m_pointcount_median": float(g4["pointcount_median"]),
    }

def _manifest_support_summary(tiles: list[dict[str, Any]]) -> dict[str, float]:
    densities = [float(t["density_pts_m2"]) for t in tiles if math.isfinite(float(t.get("density_pts_m2", float("nan")))) and int(t.get("point_count", 0)) > 0]
    pc2 = [float(t["grid_2m_pointcount_median"]) for t in tiles if math.isfinite(float(t.get("grid_2m_pointcount_median", float("nan")))) and int(t.get("point_count", 0)) > 0]
    occ2 = [float(t["grid_2m_occupancy_ratio"]) for t in tiles if math.isfinite(float(t.get("grid_2m_occupancy_ratio", float("nan")))) and int(t.get("point_count", 0)) > 0]
    pc4 = [float(t["grid_4m_pointcount_median"]) for t in tiles if math.isfinite(float(t.get("grid_4m_pointcount_median", float("nan")))) and int(t.get("point_count", 0)) > 0]

    out: dict[str, float] = {}
    if densities:
        out["density_pts_m2_median"] = float(np.median(np.asarray(densities, dtype=np.float64)))
    if pc2:
        out["grid_2m_pointcount_median_median"] = float(np.median(np.asarray(pc2, dtype=np.float64)))
    if occ2:
        out["grid_2m_occupancy_ratio_median"] = float(np.median(np.asarray(occ2, dtype=np.float64)))
    if pc4:
        out["grid_4m_pointcount_median_median"] = float(np.median(np.asarray(pc4, dtype=np.float64)))
    return out

def _build_manifest(
    in_path: str,
    out_dir: str | None,
    sensor_mode: str,
    *,
    tile_size_m: float,
    buffer_m: float,
    recursive: bool,
    small_tile_merge_frac: float,
) -> dict[str, Any]:
    dataset_label = _dataset_label(in_path)
    workspace_root = get_workspace_root(in_path, out_dir, sensor_mode)
    tiles_dir = workspace_root / "tiles"

    t0 = perf_counter()

    catalog = _build_source_catalog(in_path, recursive=recursive)
    union_bounds = _catalog_union_bounds(catalog)

    tiles, merged_small_count = _plan_tiles_for_union_bounds(
        union_bounds=union_bounds,
        dataset_label=dataset_label,
        tile_size_m=tile_size_m,
        buffer_m=buffer_m,
        small_tile_merge_frac=small_tile_merge_frac,
    )

    for tile in tiles:
        bb = tile["buffer_bounds"]
        sources = [rec for rec in catalog if _bounds_intersect(bb, rec["source_bounds"])]
        tile["source_paths"] = [rec["source_path"] for rec in sources]
        tile["source_names"] = [rec["source_name"] for rec in sources]
        tile["source_count"] = len(sources)
        tile["source_path"] = tile["source_paths"][0] if len(tile["source_paths"]) == 1 else None

    planning_time_s = perf_counter() - t0

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_label": dataset_label,
        "sensor_mode": sensor_mode.upper(),
        "input_path": str(in_path),
        "workspace_root": str(workspace_root),
        "tiles_dir": str(tiles_dir),
        "tile_size_m": float(tile_size_m),
        "buffer_m": float(buffer_m),
        "small_tile_merge_frac": float(small_tile_merge_frac),
        "recursive": bool(recursive),
        "existing_tiles_mode": False,
        "input_file_count": len(catalog),
        "raw_tile_count": int(len(tiles) + merged_small_count),
        "merged_small_tiles": int(merged_small_count),
        "tile_count": len(tiles),
        "planning_time_s": float(planning_time_s),
        "union_bounds": [float(v) for v in union_bounds],
        "files": catalog,
        "tiles": tiles,
    }
    return manifest


def _diagnose_existing_tiles(catalog: list[dict[str, Any]]) -> dict[str, Any]:
    if not catalog:
        return {
            "overlap_detected": False,
            "likely_gap_detected": False,
            "coverage_ratio_vs_envelope": 0.0,
            "warnings": [],
        }

    bounds_all = [rec["source_bounds"] for rec in catalog]
    union_bounds = _catalog_union_bounds(catalog)
    envelope_area = _core_area(union_bounds)
    total_area = float(sum(_core_area(b) for b in bounds_all))

    overlap_detected = False
    for i in range(len(bounds_all)):
        bi = bounds_all[i]
        for j in range(i + 1, len(bounds_all)):
            bj = bounds_all[j]
            if _intersection_area(bi, bj) > 0.0:
                overlap_detected = True
                break
        if overlap_detected:
            break

    coverage_ratio = total_area / envelope_area if envelope_area > 0 else 0.0

    likely_gap_detected = False
    if not overlap_detected and coverage_ratio < 0.98:
        likely_gap_detected = True

    warnings = []
    if not overlap_detected:
        warnings.append(
            "Caution! No tile boundary overlap was detected among existing tiles. "
            "Tile-edge artifacts may appear in merged raster products."
        )
    if likely_gap_detected:
        warnings.append(
            "Warning: possible gaps were detected between existing tile footprints "
            "(or the footprint is highly irregular relative to the bounding envelope). "
            "Merged outputs may contain uncovered areas or discontinuities."
        )

    return {
        "overlap_detected": overlap_detected,
        "likely_gap_detected": likely_gap_detected,
        "coverage_ratio_vs_envelope": float(coverage_ratio),
        "warnings": warnings,
    }


def _build_manifest_from_existing_tiles(
    in_path: str,
    out_dir: str | None,
    sensor_mode: str,
    *,
    recursive: bool,
) -> dict[str, Any]:
    dataset_label = _dataset_label(in_path)
    workspace_root = get_workspace_root(in_path, out_dir, sensor_mode)
    tiles_dir = workspace_root / "tiles"

    t0 = perf_counter()
    catalog = _build_source_catalog(in_path, recursive=recursive)
    union_bounds = _catalog_union_bounds(catalog)
    diagnostics = _diagnose_existing_tiles(catalog)

    tiles = []
    for i, rec in enumerate(catalog):
        src_name = rec["source_name"]
        src_bounds = [float(v) for v in rec["source_bounds"]]
        tiles.append(
            {
                "tile_id": f"{dataset_label}_existing_{i:04d}",
                "tile_name": src_name,
                "tile_ix": None,
                "tile_iy": None,
                "core_bounds": src_bounds,
                "buffer_bounds": src_bounds,
                "union_bounds": [float(v) for v in union_bounds],
                "source_paths": [rec["source_path"]],
                "source_names": [rec["source_name"]],
                "source_count": 1,
                "source_path": rec["source_path"],
                "point_count": int(rec["source_point_count"]),
                "tile_area_m2": float(_core_area(src_bounds)),
                "density_pts_m2": float(int(rec["source_point_count"]) / max(_core_area(src_bounds), 1e-9)),
                "grid_2m_pointcount_median": float("nan"),
                "grid_2m_occupancy_ratio": float("nan"),
                "grid_4m_pointcount_median": float("nan"),
                "merged_tile_count": 1,
                "existing_tile_mode": True,
            }
        )

    planning_time_s = perf_counter() - t0

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_label": dataset_label,
        "sensor_mode": sensor_mode.upper(),
        "input_path": str(in_path),
        "workspace_root": str(workspace_root),
        "tiles_dir": str(tiles_dir),
        "tile_size_m": None,
        "buffer_m": 0.0,
        "small_tile_merge_frac": 0.0,
        "recursive": bool(recursive),
        "existing_tiles_mode": True,
        "input_file_count": len(catalog),
        "raw_tile_count": len(tiles),
        "merged_small_tiles": 0,
        "tile_count": len(tiles),
        "planning_time_s": float(planning_time_s),
        "union_bounds": [float(v) for v in union_bounds],
        "files": catalog,
        "tiles": tiles,
        "tile_diagnostics": diagnostics,
        "adaptive_support_summary": _manifest_support_summary(tiles),
    }
    return manifest


def _print_plan_summary(manifest: dict[str, Any], manifest_fp: Path):
    print(f"[PREP] Sensor mode      : {manifest['sensor_mode']}")
    print(f"[PREP] Input files      : {manifest['input_file_count']}")
    print(f"[PREP] Tile size (m)    : {manifest['tile_size_m']}")
    print(f"[PREP] Buffer (m)       : {manifest['buffer_m']}")
    print(f"[PREP] Raw tiles        : {manifest.get('raw_tile_count', manifest['tile_count'])}")
    print(f"[PREP] Small merged     : {manifest.get('merged_small_tiles', 0)}")
    print(f"[PREP] Planned tiles    : {manifest['tile_count']}")
    print(f"[PREP] Manifest         : {manifest_fp}")
    print(f"[TIME] PLAN             : {manifest['planning_time_s']:.2f}s")

    if manifest.get("existing_tiles_mode"):
        diag = manifest.get("tile_diagnostics", {})
        warnings = diag.get("warnings", [])
        for w in warnings:
            print(f"[WARN] {w}")


def _check_header_compatibility(headers: list[laspy.LasHeader], source_paths: list[str]):
    if not headers:
        return

    first = headers[0]
    first_scales = tuple(float(v) for v in first.scales)
    first_offsets = tuple(float(v) for v in first.offsets)

    def _crs_wkt(hdr: laspy.LasHeader) -> str | None:
        try:
            crs = hdr.parse_crs()
            if crs is None:
                return None
            if hasattr(crs, "to_wkt"):
                return crs.to_wkt()
            return str(crs)
        except Exception:
            return None

    first_crs = _crs_wkt(first)

    for hdr, src in zip(headers[1:], source_paths[1:]):
        if hdr.point_format.id != first.point_format.id:
            raise RuntimeError(
                f"Incompatible point formats across source tiles: "
                f"{hdr.point_format.id} vs {first.point_format.id} | file={src}"
            )
        if int(hdr.version.major) != int(first.version.major) or int(hdr.version.minor) != int(first.version.minor):
            raise RuntimeError(
                f"Incompatible LAS versions across source tiles: "
                f"{hdr.version} vs {first.version} | file={src}"
            )

        scales = tuple(float(v) for v in hdr.scales)
        offsets = tuple(float(v) for v in hdr.offsets)
        if scales != first_scales:
            raise RuntimeError(
                f"Incompatible LAS scales across source tiles: {scales} vs {first_scales} | file={src}"
            )
        if offsets != first_offsets:
            raise RuntimeError(
                f"Incompatible LAS offsets across source tiles: {offsets} vs {first_offsets} | file={src}"
            )

        crs = _crs_wkt(hdr)
        if crs != first_crs:
            raise RuntimeError(f"Incompatible CRS across source tiles | file={src}")


def _merge_point_records(records: list[Any], template_header: laspy.LasHeader):
    if len(records) == 1:
        return records[0]

    merged_array = np.concatenate([rec.array for rec in records])
    return laspy.PackedPointRecord(merged_array, template_header.point_format)


def _collect_points_for_tile(tile: dict[str, Any]) -> tuple[laspy.LasData | None, np.ndarray, list[str]]:
    bxmin, bymin, bxmax, bymax = [float(v) for v in tile["buffer_bounds"]]
    source_paths = list(tile.get("source_paths", []))

    if not source_paths:
        return None, np.zeros(0, dtype=bool), []

    kept_records = []
    kept_headers = []
    kept_sources = []
    template_header = None
    template_crs_wkt = None

    for src_fp in source_paths:
        las = laspy.read(src_fp)

        x = np.asarray(las.x, dtype=np.float64)
        y = np.asarray(las.y, dtype=np.float64)
        mask = (x >= bxmin) & (x <= bxmax) & (y >= bymin) & (y <= bymax)

        if np.any(mask):
            if template_header is None:
                try:
                    template_header = las.header.copy()
                except Exception:
                    template_header = las.header
                try:
                    crs = las.header.parse_crs()
                    if crs is not None and hasattr(crs, "to_wkt"):
                        template_crs_wkt = crs.to_wkt()
                    elif crs is not None:
                        template_crs_wkt = str(crs)
                except Exception:
                    template_crs_wkt = None
            kept_headers.append(las.header)
            kept_records.append(las.points[mask].copy())
            kept_sources.append(str(src_fp))

    if not kept_records or template_header is None:
        return None, np.zeros(0, dtype=bool), []

    _check_header_compatibility(kept_headers, kept_sources)

    template_header = _attach_crs_if_possible(template_header, fallback_wkt=template_crs_wkt)
    out = laspy.LasData(template_header)
    out.points = _merge_point_records(kept_records, template_header)

    try:
        out.header = _attach_crs_if_possible(out.header, fallback_wkt=template_crs_wkt)
    except Exception:
        pass

    return out, np.ones(len(out.points), dtype=bool), kept_sources


def _link_or_copy(src: Path, dst: Path):
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def _register_existing_tiles_to_workspace(manifest: dict[str, Any], tiles_dir: Path):
    pbar = tqdm(manifest["tiles"], desc="REGISTER existing tiles", unit="tile", dynamic_ncols=True)
    try:
        for tile in pbar:
            src_fp = Path(tile["source_path"])
            dst_fp = tiles_dir / tile["tile_name"]
            _link_or_copy(src_fp, dst_fp)
            tile["tile_path"] = str(dst_fp)
            tile["kept_source_paths"] = [str(src_fp)]
            tile["kept_source_count"] = 1
    finally:
        pbar.close()


def tile_las_dataset(
    in_path: str,
    out_dir: str | None,
    sensor_mode: str,
    *,
    tile_size_m: float,
    buffer_m: float,
    recursive: bool = False,
    overwrite_tiles: bool = False,
    small_tile_merge_frac: float = 0.25,
    use_existing_tiles: bool = False,
) -> dict[str, Any]:
    workspace_root = get_workspace_root(in_path, out_dir, sensor_mode)
    tiles_dir = workspace_root / "tiles"
    manifest_fp = workspace_root / "tile_manifest.json"
    workspace_root.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    if manifest_fp.exists() and not overwrite_tiles:
        with manifest_fp.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        _print_plan_summary(manifest, manifest_fp)
        print("[INFO] Existing tile manifest found. Reusing planned tiles. Use --overwrite_tiles to rebuild.")
        return manifest

    for old in tiles_dir.glob("*.las"):
        old.unlink()
    for old in tiles_dir.glob("*.laz"):
        old.unlink()

    in_path_obj = Path(in_path)

    if use_existing_tiles:
        if in_path_obj.is_file():
            raise RuntimeError("--use_existing_tiles expects a folder of existing LAS/LAZ tiles, not a single file.")

        manifest = _build_manifest_from_existing_tiles(
            in_path=in_path,
            out_dir=out_dir,
            sensor_mode=sensor_mode,
            recursive=recursive,
        )
        _print_plan_summary(manifest, manifest_fp)

        _register_existing_tiles_to_workspace(manifest, tiles_dir)
        manifest["tiling_time_s"] = 0.0
        manifest["tiles_written"] = int(len(manifest["tiles"]))
        manifest["adaptive_support_summary"] = _manifest_support_summary(manifest["tiles"])

        with manifest_fp.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(
            f"[TIME] TOTAL TILE REGISTRATION {sensor_mode.upper()}: 0.00s | "
            f"{manifest['tile_count']} existing tiles registered"
        )
        return manifest

    manifest = _build_manifest(
        in_path=in_path,
        out_dir=out_dir,
        sensor_mode=sensor_mode,
        tile_size_m=tile_size_m,
        buffer_m=buffer_m,
        recursive=recursive,
        small_tile_merge_frac=small_tile_merge_frac,
    )
    _print_plan_summary(manifest, manifest_fp)

    if manifest["tile_count"] == 0:
        with manifest_fp.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return manifest

    grand_t0 = perf_counter()
    global_tiles_done = 0

    pbar_tiles = tqdm(
        manifest["tiles"],
        desc=f"TILE {sensor_mode.upper()} total",
        unit="tile",
        dynamic_ncols=True,
        leave=True,
    )

    try:
        for tile in pbar_tiles:
            tile["tile_path"] = str(tiles_dir / tile["tile_name"])
            tile_t0 = perf_counter()

            out_las, _dummy_mask, kept_sources = _collect_points_for_tile(tile)
            tile_fp = Path(tile["tile_path"])

            if out_las is not None and len(out_las.points) > 0:
                out_las.write(tile_fp)
                tile["point_count"] = int(len(out_las.points))
                tile["kept_source_paths"] = kept_sources
                tile["kept_source_count"] = len(kept_sources)
                tile.update(_compute_tile_support_stats(out_las, tile))
            else:
                tile["point_count"] = 0
                tile["kept_source_paths"] = []
                tile["kept_source_count"] = 0
                tile["tile_area_m2"] = float(_core_area(tile.get("buffer_bounds", tile.get("core_bounds", [0,0,0,0]))))
                tile["density_pts_m2"] = float("nan")
                tile["grid_2m_pointcount_median"] = float("nan")
                tile["grid_2m_occupancy_ratio"] = float("nan")
                tile["grid_4m_pointcount_median"] = float("nan")
                if tile_fp.exists():
                    tile_fp.unlink()

            global_tiles_done += 1
            elapsed = perf_counter() - grand_t0
            sec_per_tile = elapsed / max(global_tiles_done, 1)
            tile_elapsed = perf_counter() - tile_t0
            pbar_tiles.set_postfix_str(
                f"{global_tiles_done}/{manifest['tile_count']} | "
                f"{sec_per_tile:.2f}s/tile | current={tile['tile_name']} | "
                f"tile={tile_elapsed:.2f}s | src={tile.get('kept_source_count', 0)}"
            )
    finally:
        pbar_tiles.close()

    total_elapsed = perf_counter() - grand_t0
    manifest["tiling_time_s"] = float(total_elapsed)
    manifest["tiles_written"] = int(sum(1 for t in manifest["tiles"] if int(t.get("point_count", 0)) > 0))
    manifest["adaptive_support_summary"] = _manifest_support_summary(manifest["tiles"])

    with manifest_fp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    sec_per_tile = total_elapsed / max(manifest["tile_count"], 1)
    print(
        f"[TIME] TOTAL TILING {sensor_mode.upper()}: {total_elapsed:.2f}s | "
        f"{manifest['tile_count']} tiles | {sec_per_tile:.2f}s/tile"
    )
    return manifest


__all__ = ["get_workspace_root", "tile_las_dataset"]