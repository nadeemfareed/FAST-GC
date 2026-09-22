from __future__ import annotations

def _fastgc_debug_enabled() -> bool:
    """Return True only when detailed internal diagnostics are requested."""
    import os
    return os.environ.get(
        "FASTGC_DEBUG", ""
    ).strip().lower() in {
        "1", "true", "yes", "on"
    }


import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import laspy
import numpy as np
from tqdm import tqdm

from .monster import ProgressDashboard, log_info
from .vertical_support import build_vertical_support


MANIFEST_VERSION = 6



def _grid_support_stats_xy(x: np.ndarray, y: np.ndarray, cell_m: float) -> dict[str, float]:
    if x.size == 0:
        return {"pointcount_median": float("nan"), "occupancy_ratio": 0.0}
    cell = float(cell_m)
    x0 = math.floor(float(np.min(x)) / cell) * cell
    y0 = math.floor(float(np.min(y)) / cell) * cell
    ix = np.floor((x - x0) / cell).astype(np.int64)
    iy = np.floor((y - y0) / cell).astype(np.int64)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1
    key = iy * nx + ix
    _, cnt = np.unique(key, return_counts=True)
    total = max(1, nx * ny)
    return {
        "pointcount_median": float(np.median(cnt)) if cnt.size else float("nan"),
        "occupancy_ratio": float(cnt.size / total),
    }


def _support_stats_xy(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if x.size == 0:
        return {
            "density_pts_m2": float("nan"),
            "grid_2m_pointcount_median": float("nan"),
            "grid_2m_occupancy_ratio": 0.0,
            "grid_4m_pointcount_median": float("nan"),
        }
    width = max(float(np.max(x) - np.min(x)), 1.0)
    height = max(float(np.max(y) - np.min(y)), 1.0)
    area = width * height
    g2 = _grid_support_stats_xy(x, y, 2.0)
    g4 = _grid_support_stats_xy(x, y, 4.0)
    return {
        "density_pts_m2": float(x.size / area),
        "grid_2m_pointcount_median": float(g2["pointcount_median"]),
        "grid_2m_occupancy_ratio": float(g2["occupancy_ratio"]),
        "grid_4m_pointcount_median": float(g4["pointcount_median"]),
    }


def _vertical_support_summary(tiles: list[dict[str, Any]]) -> dict[str, float]:
    fields = (
        "median_total_density_pts_m2",
        "median_lower2_density_pts_m2",
        "median_lower2_fraction",
        "median_vertical_span_m",
        "weak_lower_support_fraction",
        "very_weak_lower_support_fraction",
        "canopy_dominated_fraction",
    )
    out: dict[str, float] = {
        "xy_cell_m": 5.0,
        "z_bin_m": 1.0,
        "classification_effect": "none_v1_diagnostics_only",
    }
    valid = [t.get("vertical_support") for t in tiles if isinstance(t.get("vertical_support"), dict)]
    for field in fields:
        vals = np.asarray([v.get(field, np.nan) for v in valid], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out[f"{field}_median"] = float(np.median(vals)) if vals.size else float("nan")
        out[f"{field}_p25"] = float(np.quantile(vals, 0.25)) if vals.size else float("nan")
        out[f"{field}_p75"] = float(np.quantile(vals, 0.75)) if vals.size else float("nan")
    out["tile_count_with_vertical_support"] = int(len(valid))
    return out


def _adaptive_support_summary(tiles: list[dict[str, Any]]) -> dict[str, float]:
    fields = (
        "density_pts_m2",
        "grid_2m_pointcount_median",
        "grid_2m_occupancy_ratio",
        "grid_4m_pointcount_median",
    )
    out: dict[str, float] = {}
    for field in fields:
        vals = np.asarray([t.get(field, np.nan) for t in tiles], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out[f"{field}_median"] = float(np.median(vals)) if vals.size else float("nan")
        out[f"{field}_p25"] = float(np.quantile(vals, 0.25)) if vals.size else float("nan")
        out[f"{field}_p75"] = float(np.quantile(vals, 0.75)) if vals.size else float("nan")
    out["tile_count_with_support_stats"] = int(sum(
        1 for t in tiles if np.isfinite(float(t.get("density_pts_m2", np.nan)))
    ))
    return out

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


def _core_area(bounds: list[float] | tuple[float, float, float, float]) -> float:
    xmin, ymin, xmax, ymax = [float(v) for v in bounds]
    return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)


def _core_width_height(
    bounds: list[float] | tuple[float, float, float, float],
) -> tuple[float, float]:
    xmin, ymin, xmax, ymax = [float(v) for v in bounds]
    return max(0.0, xmax - xmin), max(0.0, ymax - ymin)


def _shape_ratio(bounds: list[float] | tuple[float, float, float, float]) -> float:
    """Return short-axis / long-axis ratio in [0, 1]."""
    width, height = _core_width_height(bounds)
    long_axis = max(width, height)
    if long_axis <= 0.0:
        return 0.0
    return min(width, height) / long_axis


def _update_tile_geometry(tile: dict[str, Any], tile_size_m: float, buffer_m: float) -> None:
    width, height = _core_width_height(tile["core_bounds"])
    nominal_area = float(tile_size_m) * float(tile_size_m)
    area = float(width * height)
    tile["core_width_m"] = float(width)
    tile["core_height_m"] = float(height)
    tile["core_area_m2"] = area
    tile["area_fraction"] = float(area / nominal_area) if nominal_area > 0 else 0.0
    tile["shape_ratio"] = float(_shape_ratio(tile["core_bounds"]))
    tile["buffer_bounds"] = _buffer_from_core(tile["core_bounds"], buffer_m)


def _union_bounds(a: list[float], b: list[float]) -> list[float]:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    return [min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)]


def _buffer_from_core(core_bounds: list[float], buffer_m: float) -> list[float]:
    xmin, ymin, xmax, ymax = [float(v) for v in core_bounds]
    b = float(buffer_m)
    return [xmin - b, ymin - b, xmax + b, ymax + b]


def _plan_tiles_for_bounds(
    src_fp: str,
    tile_size_m: float,
    buffer_m: float,
    small_tile_merge_frac: float,
) -> tuple[list[dict[str, Any]], int]:
    bounds, point_count = _read_xy_bounds(src_fp)
    if bounds is None:
        return [], 0

    xmin, ymin, xmax, ymax = bounds
    width = max(0.0, xmax - xmin)
    height = max(0.0, ymax - ymin)
    nx = max(1, int(math.ceil(width / tile_size_m)))
    ny = max(1, int(math.ceil(height / tile_size_m)))

    src_stem = Path(src_fp).stem
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
            tile_id = f"{src_stem}_x{ix:04d}_y{iy:04d}"
            raw_tiles.append(
                {
                    "tile_id": tile_id,
                    "tile_name": f"{tile_id}.las",
                    "source_path": str(src_fp),
                    "source_name": Path(src_fp).name,
                    "source_point_count": int(point_count),
                    "source_bounds": [float(xmin), float(ymin), float(xmax), float(ymax)],
                    "core_bounds": core_bounds,
                    "buffer_bounds": _buffer_from_core(core_bounds, buffer_m),
                    "tile_ix": int(ix),
                    "tile_iy": int(iy),
                    "file_tile_ordinal": int(len(raw_tiles)),
                    "core_area_m2": float(_core_area(core_bounds)),
                    "core_width_m": float(core_xmax - core_xmin),
                    "core_height_m": float(core_ymax - core_ymin),
                    "area_fraction": float(_core_area(core_bounds) / nominal_area) if nominal_area > 0 else 0.0,
                    "shape_ratio": float(_shape_ratio(core_bounds)),
                    "merged_from": [tile_id],
                    "merge_reasons": [],
                    "merge_directions": [],
                    "is_small_exception_tile": False,
                }
            )

    by_idx: dict[tuple[int, int], dict[str, Any]] = {
        (int(t["tile_ix"]), int(t["tile_iy"])): t for t in raw_tiles
    }
    active_keys = set(by_idx.keys())
    merged_small_count = 0
    merged_sliver_count = 0

    # IMPORTANT:
    # Keep one fixed requested tiling topology.  There is no recursive/sub-tiling.
    # Only terminal edge fragments are consolidated into their immediate spatial
    # neighbor before any LAS tile is written.
    axis_threshold = float(max(0.0, small_tile_merge_frac)) * float(tile_size_m)
    sliver_ratio_threshold = 0.30

    def _touches_source_edge(tile: dict[str, Any], axis: str) -> bool:
        tx0, ty0, tx1, ty1 = [float(v) for v in tile["core_bounds"]]
        eps = max(1.0e-8, float(tile_size_m) * 1.0e-9)
        if axis == "x":
            return abs(tx0 - xmin) <= eps or abs(tx1 - xmax) <= eps
        return abs(ty0 - ymin) <= eps or abs(ty1 - ymax) <= eps

    def _is_axis_fragment(tile: dict[str, Any], axis: str) -> bool:
        width, height = _core_width_height(tile["core_bounds"])
        if axis == "x":
            if nx <= 1 or not _touches_source_edge(tile, "x"):
                return False
            return (
                (axis_threshold > 0.0 and width < axis_threshold)
                or (width < height and _shape_ratio(tile["core_bounds"]) < sliver_ratio_threshold)
            )
        if ny <= 1 or not _touches_source_edge(tile, "y"):
            return False
        return (
            (axis_threshold > 0.0 and height < axis_threshold)
            or (height < width and _shape_ratio(tile["core_bounds"]) < sliver_ratio_threshold)
        )

    def _merge_into(
        donor_key: tuple[int, int],
        receiver_key: tuple[int, int],
        *,
        reason: str,
        direction: str,
    ) -> None:
        nonlocal merged_small_count, merged_sliver_count
        if donor_key not in active_keys or receiver_key not in active_keys:
            return
        if donor_key == receiver_key:
            return

        donor = by_idx[donor_key]
        receiver = by_idx[receiver_key]
        receiver["core_bounds"] = _union_bounds(receiver["core_bounds"], donor["core_bounds"])
        receiver.setdefault("merged_from", []).extend(
            donor.get("merged_from", [donor["tile_id"]])
        )
        receiver.setdefault("merge_reasons", []).extend(
            donor.get("merge_reasons", [])
        )
        receiver.setdefault("merge_directions", []).extend(
            donor.get("merge_directions", [])
        )
        receiver["merge_reasons"].append(reason)
        receiver["merge_directions"].append(direction)
        receiver["file_tile_ordinal"] = min(
            int(receiver["file_tile_ordinal"]),
            int(donor["file_tile_ordinal"]),
        )
        _update_tile_geometry(receiver, tile_size_m, buffer_m)
        active_keys.remove(donor_key)

        if reason == "small_edge_fragment":
            merged_small_count += 1
        else:
            merged_sliver_count += 1

    def _receiver_for_axis(ix: int, iy: int, axis: str) -> tuple[tuple[int, int] | None, str]:
        # Prefer the previous immediate neighbor.  For a leading-edge fragment,
        # use the next neighbor.  Never jump across the grid.
        if axis == "x":
            previous = (ix - 1, iy)
            following = (ix + 1, iy)
            prev_dir, next_dir = "left", "right"
        else:
            previous = (ix, iy - 1)
            following = (ix, iy + 1)
            prev_dir, next_dir = "down", "up"

        if previous in active_keys:
            return previous, prev_dir
        if following in active_keys:
            return following, next_dir
        return None, ""

    # Pass 1: consolidate thin terminal columns horizontally.
    # This prevents a narrow final column from surviving as a long strip.
    for iy in range(ny):
        for ix in range(nx):
            key = (ix, iy)
            if key not in active_keys:
                continue
            tile = by_idx[key]
            if not _is_axis_fragment(tile, "x"):
                continue
            receiver_key, direction = _receiver_for_axis(ix, iy, "x")
            if receiver_key is None:
                continue
            area_small = (
                area_threshold > 0.0
                and float(tile["core_area_m2"]) < area_threshold
            )
            reason = "small_edge_fragment" if area_small else "thin_edge_sliver"
            _merge_into(key, receiver_key, reason=reason, direction=direction)

    # Pass 2: consolidate thin terminal rows vertically.
    # Running this after the X pass also absorbs a small corner cleanly into the
    # already widened edge tile, yielding one contiguous rectangular core.
    for iy in range(ny):
        for ix in range(nx):
            key = (ix, iy)
            if key not in active_keys:
                continue
            tile = by_idx[key]
            if not _is_axis_fragment(tile, "y"):
                continue
            receiver_key, direction = _receiver_for_axis(ix, iy, "y")
            if receiver_key is None:
                continue
            area_small = (
                area_threshold > 0.0
                and float(tile["core_area_m2"]) < area_threshold
            )
            reason = "small_edge_fragment" if area_small else "thin_edge_sliver"
            _merge_into(key, receiver_key, reason=reason, direction=direction)

    planned: list[dict[str, Any]] = []
    for key in sorted(active_keys, key=lambda kk: (int(by_idx[kk]["file_tile_ordinal"]), kk[1], kk[0])):
        rec = by_idx[key]
        _update_tile_geometry(rec, tile_size_m, buffer_m)
        rec["merged_from"] = sorted(set(rec.get("merged_from", [rec["tile_id"]])))
        rec["merged_tile_count"] = int(len(rec["merged_from"]))
        rec["merge_reasons"] = list(rec.get("merge_reasons", []))
        rec["merge_directions"] = list(rec.get("merge_directions", []))
        rec["is_small_exception_tile"] = bool(rec["merged_tile_count"] > 1)
        rec["is_sliver_after_merge"] = bool(
            rec["shape_ratio"] < sliver_ratio_threshold
            and (nx > 1 or ny > 1)
        )
        planned.append(rec)

    for ordinal, rec in enumerate(planned):
        rec["file_tile_ordinal"] = int(ordinal)

    return planned, (merged_small_count + merged_sliver_count)


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
    if tile_size_m <= 0:
        raise ValueError("tile_size_m must be > 0")
    if buffer_m < 0:
        raise ValueError("buffer_m must be >= 0")
    if small_tile_merge_frac < 0:
        raise ValueError("small_tile_merge_frac must be >= 0")

    files = _iter_las_files(in_path, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No LAS/LAZ files found in: {in_path}")

    workspace_root = get_workspace_root(in_path, out_dir, sensor_mode)
    tiles_dir = workspace_root / "tiles"

    dataset_label = _dataset_label(in_path)
    file_entries: list[dict[str, Any]] = []
    tiles: list[dict[str, Any]] = []
    raw_tile_count = 0
    merged_small_tiles_total = 0

    t0 = perf_counter()
    # Planning is intentionally quiet; the public dynamic stage begins when tiles are written.
    for src_fp in files:
        planned, merged_small_count = _plan_tiles_for_bounds(src_fp, tile_size_m, buffer_m, small_tile_merge_frac)
        source_raw_tile_count = 0
        if planned:
            source_raw_tile_count = int(sum(int(rec.get("merged_tile_count", 1)) for rec in planned))
        bounds, point_count = _read_xy_bounds(src_fp)
        file_entry = {
            "source_path": str(src_fp),
            "source_name": Path(src_fp).name,
            "tile_count": int(len(planned)),
            "raw_tile_count": int(source_raw_tile_count),
            "merged_small_tiles": int(merged_small_count),
            "source_bounds": list(bounds) if bounds is not None else None,
            "source_point_count": int(point_count),
        }
        file_entries.append(file_entry)
        tiles.extend(planned)
        raw_tile_count += int(source_raw_tile_count)
        merged_small_tiles_total += int(merged_small_count)

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
        "input_file_count": len(files),
        "raw_tile_count": int(raw_tile_count),
        "merged_small_tiles": int(merged_small_tiles_total),
        "sliver_ratio_threshold": 0.30,
        "remaining_sliver_tiles": int(
            sum(1 for t in tiles if bool(t.get("is_sliver_after_merge", False)))
        ),
        "tile_count": len(tiles),
        "planning_time_s": float(planning_time_s),
        "files": file_entries,
        "tiles": tiles,
    }
    return manifest


def _print_plan_summary(manifest: dict[str, Any], manifest_fp: Path):
    if not _fastgc_debug_enabled():
        return
    print(f"[PREP] Sensor mode      : {manifest['sensor_mode']}")
    print(f"[PREP] Input files      : {manifest['input_file_count']}")
    tile_size_value = manifest.get("tile_size_m")
    print(f"[PREP] Tile size (m)    : {tile_size_value if tile_size_value is not None else 'existing'}")
    print(f"[PREP] Buffer (m)       : {manifest.get('buffer_m', 0.0)}")
    print(f"[PREP] Raw tiles        : {manifest.get('raw_tile_count', manifest['tile_count'])}")
    print(f"[PREP] Edge merged      : {manifest.get('merged_small_tiles', 0)}")
    print(f"[PREP] Remaining sliver : {manifest.get('remaining_sliver_tiles', 0)}")
    print(f"[PREP] Planned tiles    : {manifest['tile_count']}")
    print(f"[PREP] Manifest         : {manifest_fp}")
    print(f"[TIME] PLAN             : {manifest['planning_time_s']:.2f}s")



def _manifest_from_existing_tiles(
    in_path: str,
    out_dir: str | None,
    sensor_mode: str,
    *,
    recursive: bool,
) -> dict[str, Any]:
    """Build a lightweight manifest around an already-tiled LAS/LAZ collection.

    No source data are copied, split, re-tiled, or rewritten.  The existing
    LAS/LAZ files become the processing tiles directly.  Their physical bounds
    are used as core bounds because no FAST-GC-created external buffer exists.
    """
    files = _iter_las_files(in_path, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No LAS/LAZ files found in existing-tile input: {in_path}")

    workspace_root = get_workspace_root(in_path, out_dir, sensor_mode)
    workspace_root.mkdir(parents=True, exist_ok=True)

    t0 = perf_counter()
    tiles: list[dict[str, Any]] = []
    file_entries: list[dict[str, Any]] = []

    # Existing-tile cataloging is intentionally quiet; no synthetic progress bar is shown.
    for ordinal, src_fp in enumerate(files):
        bounds, point_count = _read_xy_bounds(src_fp)
        if bounds is None:
            continue
        xmin, ymin, xmax, ymax = [float(v) for v in bounds]
        tile_id = Path(src_fp).stem
        width, height = _core_width_height(bounds)
        tile = {
            "tile_id": tile_id,
            "tile_name": Path(src_fp).name,
            "tile_path": str(src_fp),
            "source_path": str(src_fp),
            "source_name": Path(src_fp).name,
            "source_point_count": int(point_count),
            "source_bounds": [xmin, ymin, xmax, ymax],
            "core_bounds": [xmin, ymin, xmax, ymax],
            "buffer_bounds": [xmin, ymin, xmax, ymax],
            "tile_ix": int(ordinal),
            "tile_iy": 0,
            "file_tile_ordinal": int(ordinal),
            "core_width_m": float(width),
            "core_height_m": float(height),
            "core_area_m2": float(width * height),
            "area_fraction": None,
            "shape_ratio": float(_shape_ratio(bounds)),
            "point_count": int(point_count),
            "merged_from": [tile_id],
            "merge_reasons": [],
            "merge_directions": [],
            "merged_tile_count": 1,
            "is_small_exception_tile": False,
            "is_sliver_after_merge": False,
            "existing_tile": True,
        }
        tiles.append(tile)
        file_entries.append(
            {
                "source_path": str(src_fp),
                "source_name": Path(src_fp).name,
                "tile_count": 1,
                "raw_tile_count": 1,
                "merged_small_tiles": 0,
                "source_bounds": [xmin, ymin, xmax, ymax],
                "source_point_count": int(point_count),
            }
        )

    planning_time_s = perf_counter() - t0
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_label": _dataset_label(in_path),
        "sensor_mode": sensor_mode.upper(),
        "input_path": str(in_path),
        "workspace_root": str(workspace_root),
        # Deliberately points to the user's existing tile folder.  No new
        # physical tile folder is created or populated.
        "tiles_dir": str(Path(in_path)),
        "tile_size_m": None,
        "buffer_m": 0.0,
        "small_tile_merge_frac": 0.0,
        "recursive": bool(recursive),
        "use_existing_tiles": True,
        "input_file_count": len(files),
        "raw_tile_count": len(tiles),
        "merged_small_tiles": 0,
        "sliver_ratio_threshold": 0.30,
        "remaining_sliver_tiles": 0,
        "tile_count": len(tiles),
        "planning_time_s": float(planning_time_s),
        "files": file_entries,
        "tiles": tiles,
        "tiles_written": len(tiles),
    }
    return manifest


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

    # Compatibility with the public --use_existing_tiles CLI option.
    # In this mode the input LAS/LAZ collection is already tiled; FAST-GC only
    # catalogs it and never retiles or rewrites the source files.
    if use_existing_tiles:
        manifest = _manifest_from_existing_tiles(
            in_path=in_path,
            out_dir=out_dir,
            sensor_mode=sensor_mode,
            recursive=recursive,
        )
        with manifest_fp.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        _print_plan_summary(manifest, manifest_fp)
        log_info("Existing LAS/LAZ tiles cataloged directly; no retiling performed.")
        return manifest

    tiles_dir.mkdir(parents=True, exist_ok=True)

    if manifest_fp.exists() and not overwrite_tiles:
        with manifest_fp.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        _print_plan_summary(manifest, manifest_fp)
        log_info("Existing tile manifest found. Reusing planned tiles. Use --overwrite_tiles to rebuild.")
        return manifest

    for old in tiles_dir.glob("*.las"):
        old.unlink()

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

    tiles_by_source: dict[str, list[dict[str, Any]]] = {}
    for tile in manifest["tiles"]:
        tile["tile_path"] = str(tiles_dir / tile["tile_name"])
        tiles_by_source.setdefault(tile["source_path"], []).append(tile)

    grand_t0 = perf_counter()
    global_tiles_done = 0
    global_tiles_written = 0
    global_tiles_skipped = 0
    dashboard = ProgressDashboard(
        f"TILING {sensor_mode.upper()}",
        int(manifest["tile_count"]),
        unit="tile",
        enabled=True,
    )

    try:
        for file_entry in manifest["files"]:
            src_fp = file_entry["source_path"]
            src_name = file_entry["source_name"]
            file_tiles = tiles_by_source.get(src_fp, [])

            las = laspy.read(src_fp)
            x = np.asarray(las.x, dtype=np.float64)
            y = np.asarray(las.y, dtype=np.float64)
            z = np.asarray(las.z, dtype=np.float64)
            vertical_support_dir = workspace_root / "vertical_support"
            vertical_support_dir.mkdir(parents=True, exist_ok=True)

            file_written = 0
            for local_idx, tile in enumerate(file_tiles, start=1):
                tile_t0 = perf_counter()
                bxmin, bymin, bxmax, bymax = [float(v) for v in tile["buffer_bounds"]]
                mask = (x >= bxmin) & (x <= bxmax) & (y >= bymin) & (y <= bymax)
                npts = int(np.count_nonzero(mask))
                tile["point_count"] = npts
                if npts > 0:
                    stats = _support_stats_xy(x[mask], y[mask])
                    tile.update(stats)
                    support_fp = vertical_support_dir / f"{tile['tile_id']}_vertical_support.npz"
                    tile["vertical_support"] = build_vertical_support(
                        x[mask], y[mask], z[mask], support_fp,
                        xy_cell_m=5.0,
                        z_bin_m=1.0,
                    )
                tile_fp = Path(tile["tile_path"])
                if npts > 0:
                    out = laspy.LasData(las.header)
                    out.points = las.points[mask].copy()
                    out.write(tile_fp)
                    file_written += 1
                    global_tiles_written += 1
                else:
                    global_tiles_skipped += 1
                    if tile_fp.exists():
                        tile_fp.unlink()

                global_tiles_done += 1
                dashboard.update(
                    1,
                    current_file=src_name,
                    current_item=f"{tile['tile_name']} ({local_idx}/{max(1, len(file_tiles))})",
                    elapsed_item_sec=perf_counter() - tile_t0,
                    ok_count=global_tiles_written,
                    skipped_count=global_tiles_skipped,
                    failed_count=0,
                )
    finally:
        dashboard.close()

    total_elapsed = perf_counter() - grand_t0
    manifest["tiling_time_s"] = float(total_elapsed)
    manifest["tiles_written"] = int(sum(1 for t in manifest["tiles"] if int(t.get("point_count", 0)) > 0))
    manifest["adaptive_support_summary"] = _adaptive_support_summary(manifest["tiles"])
    manifest["vertical_support_summary"] = _vertical_support_summary(manifest["tiles"])
    with manifest_fp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    sec_per_tile = total_elapsed / max(manifest["tile_count"], 1)
    if _fastgc_debug_enabled():
        print(
            f"[TIME] TOTAL TILING {sensor_mode.upper()}: {total_elapsed:.2f}s | "
            f"{manifest['tile_count']} tiles | {sec_per_tile:.2f}s/tile"
        )
    ss = manifest.get("adaptive_support_summary", {})
    if ss:
        if _fastgc_debug_enabled():
            print(
                f"density={ss.get('density_pts_m2_median', float('nan')):.2f} pts/m2 | "
                f"2m_count={ss.get('grid_2m_pointcount_median_median', float('nan')):.1f} | "
                f"2m_occ={ss.get('grid_2m_occupancy_ratio_median', float('nan')):.3f}"
            )
    vsum = manifest.get("vertical_support_summary", {})
    if vsum:
        if _fastgc_debug_enabled():
            print(
                f"5mXY/1mZ | lower2_density="
                f"{vsum.get('median_lower2_density_pts_m2_median', float('nan')):.3f} pts/m2 | "
                f"weak={100.0 * vsum.get('weak_lower_support_fraction_median', float('nan')):.2f}% | "
                f"canopy_dom={100.0 * vsum.get('canopy_dominated_fraction_median', float('nan')):.2f}%"
            )
    return manifest


__all__ = ["get_workspace_root", "tile_las_dataset"]
