from __future__ import annotations

import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import laspy
import numpy as np
from tqdm import tqdm


MANIFEST_VERSION = 3


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
                    "merged_from": [tile_id],
                    "is_small_exception_tile": False,
                }
            )

    by_idx: dict[tuple[int, int], dict[str, Any]] = {(int(t["tile_ix"]), int(t["tile_iy"])): t for t in raw_tiles}
    active_keys = set(by_idx.keys())
    merged_small_count = 0

    def choose_receiver(ix: int, iy: int) -> tuple[int, int] | None:
        prefs = [(ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)]
        valid = [k for k in prefs if k in active_keys]
        if not valid:
            return None

        non_small = [
            k for k in valid
            if by_idx[k]["core_area_m2"] >= area_threshold or area_threshold <= 0.0
        ]
        candidates = non_small or valid

        def score(k: tuple[int, int]) -> tuple[float, float]:
            rec = by_idx[k]
            cx0, cy0, cx1, cy1 = [float(v) for v in by_idx[(ix, iy)]["core_bounds"]]
            rx0, ry0, rx1, ry1 = [float(v) for v in rec["core_bounds"]]
            shared_x = max(0.0, min(cx1, rx1) - max(cx0, rx0))
            shared_y = max(0.0, min(cy1, ry1) - max(cy0, ry0))
            boundary = max(shared_x, shared_y)
            return (boundary, float(rec["core_area_m2"]))

        return max(candidates, key=score)

    if nx > 1 or ny > 1:
        for iy in range(ny):
            for ix in range(nx):
                key = (ix, iy)
                if key not in active_keys:
                    continue
                tile = by_idx[key]
                if float(tile["core_area_m2"]) >= area_threshold or area_threshold <= 0.0:
                    continue
                receiver_key = choose_receiver(ix, iy)
                if receiver_key is None or receiver_key == key:
                    continue

                receiver = by_idx[receiver_key]
                receiver["core_bounds"] = _union_bounds(receiver["core_bounds"], tile["core_bounds"])
                receiver["buffer_bounds"] = _buffer_from_core(receiver["core_bounds"], buffer_m)
                receiver["core_area_m2"] = float(_core_area(receiver["core_bounds"]))
                receiver.setdefault("merged_from", []).extend(tile.get("merged_from", [tile["tile_id"]]))
                receiver["file_tile_ordinal"] = min(int(receiver["file_tile_ordinal"]), int(tile["file_tile_ordinal"]))
                active_keys.remove(key)
                merged_small_count += 1

    planned: list[dict[str, Any]] = []
    for key in sorted(active_keys, key=lambda kk: (int(by_idx[kk]["file_tile_ordinal"]), kk[1], kk[0])):
        rec = by_idx[key]
        rec["buffer_bounds"] = _buffer_from_core(rec["core_bounds"], buffer_m)
        rec["core_area_m2"] = float(_core_area(rec["core_bounds"]))
        rec["merged_from"] = sorted(set(rec.get("merged_from", [rec["tile_id"]])))
        rec["merged_tile_count"] = int(len(rec["merged_from"]))
        rec["is_small_exception_tile"] = bool(rec["merged_tile_count"] > 1)
        planned.append(rec)

    for ordinal, rec in enumerate(planned):
        rec["file_tile_ordinal"] = int(ordinal)

    return planned, merged_small_count


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
    pbar = tqdm(files, desc=f"PLAN {sensor_mode.upper()}", unit="file", dynamic_ncols=True)
    for src_fp in pbar:
        pbar.set_postfix_str(Path(src_fp).name)
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
        "tile_count": len(tiles),
        "planning_time_s": float(planning_time_s),
        "files": file_entries,
        "tiles": tiles,
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
    global_tile_bar = tqdm(
        total=manifest["tile_count"],
        desc=f"TILE {sensor_mode.upper()} total",
        unit="tile",
        dynamic_ncols=True,
    )
    file_bar = tqdm(
        manifest["files"],
        desc=f"TILE {sensor_mode.upper()} files",
        unit="file",
        dynamic_ncols=True,
    )
    try:
        for file_entry in file_bar:
            src_fp = file_entry["source_path"]
            src_name = file_entry["source_name"]
            file_tiles = tiles_by_source.get(src_fp, [])
            file_bar.set_postfix_str(src_name)

            file_t0 = perf_counter()
            las = laspy.read(src_fp)
            x = np.asarray(las.x, dtype=np.float64)
            y = np.asarray(las.y, dtype=np.float64)

            tile_bar = tqdm(
                file_tiles,
                desc=f"TILE current {src_name}",
                unit="tile",
                dynamic_ncols=True,
                leave=False,
            )
            file_written = 0
            for tile in tile_bar:
                bxmin, bymin, bxmax, bymax = [float(v) for v in tile["buffer_bounds"]]
                mask = (x >= bxmin) & (x <= bxmax) & (y >= bymin) & (y <= bymax)
                npts = int(np.count_nonzero(mask))
                tile["point_count"] = npts
                tile_fp = Path(tile["tile_path"])
                if npts > 0:
                    out = laspy.LasData(las.header)
                    out.points = las.points[mask].copy()
                    out.write(tile_fp)
                    file_written += 1
                else:
                    if tile_fp.exists():
                        tile_fp.unlink()
                global_tiles_done += 1
                elapsed = perf_counter() - grand_t0
                sec_per_tile = elapsed / max(global_tiles_done, 1)
                global_tile_bar.update(1)
                global_tile_bar.set_postfix_str(
                    f"{global_tiles_done}/{manifest['tile_count']} | {sec_per_tile:.2f}s/tile | current={src_name}"
                )
                tile_bar.set_postfix_str(f"{global_tiles_done}/{manifest['tile_count']} | {sec_per_tile:.2f}s/tile")
            tile_bar.close()

            file_elapsed = perf_counter() - file_t0
            denom = max(len(file_tiles), 1)
            print(
                f"[TIME] TILING {src_name}: {file_elapsed:.2f}s | "
                f"{len(file_tiles)} tiles | {file_elapsed / denom:.2f}s/tile | written={file_written}"
            )
    finally:
        file_bar.close()
        global_tile_bar.close()

    total_elapsed = perf_counter() - grand_t0
    manifest["tiling_time_s"] = float(total_elapsed)
    manifest["tiles_written"] = int(sum(1 for t in manifest["tiles"] if int(t.get("point_count", 0)) > 0))
    with manifest_fp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    sec_per_tile = total_elapsed / max(manifest["tile_count"], 1)
    print(
        f"[TIME] TOTAL TILING {sensor_mode.upper()}: {total_elapsed:.2f}s | "
        f"{manifest['tile_count']} tiles | {sec_per_tile:.2f}s/tile"
    )
    return manifest


__all__ = ["get_workspace_root", "tile_las_dataset"]