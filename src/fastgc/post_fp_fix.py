from __future__ import annotations

import json
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any

import laspy
import numpy as np
from tqdm import tqdm

from .io_las import (
    PRODUCT_DEM,
    PRODUCT_GC,
    PRODUCT_NORMALIZED,
    _build_dem_bundle,
    _load_product_context,
    _make_product_dirs,
    _sample_bilinear_from_grid,
    _write_tif,
    list_classified_files,
)


def _write_temp_normalized_from_residual(ctx: dict[str, Any], residual: np.ndarray, out_fp: Path) -> None:
    out = laspy.LasData(ctx["las"].header)
    out.points = ctx["las"].points.copy()
    z_norm = np.maximum(residual, 0.0)

    z_scale = float(out.header.scales[2])
    z_offset = float(out.header.offsets[2])
    raw_Z = np.round((z_norm - z_offset) / z_scale).astype(out.points.array["Z"].dtype, copy=False)
    out.points.array["Z"] = raw_Z
    try:
        out.classification = ctx["cls"].astype(np.uint8)
    except Exception:
        pass
    out.write(out_fp)


def _fix_one_classified_tile(
    classified_fp: str,
    temp_product_dirs: dict[str, str],
    dem_res: float,
    nonground_to_ground_max_z: float,
    ground_to_nonground_min_z: float,
) -> dict[str, Any]:
    ctx = _load_product_context(classified_fp)

    # Try to build provisional DEM. If not enough ground points exist, skip tile.
    try:
        dem_pack = _build_dem_bundle(ctx, dem_res)
    except RuntimeError as e:
        msg = str(e)
        if "Too few ground points" in msg:
            return {
                "tile": Path(classified_fp).name,
                "changed_points": 0,
                "promoted_to_ground": 0,
                "demoted_to_nonground": 0,
                "status": "skipped_too_few_ground",
                "message": msg,
            }
        raise

    # provisional temp DEM
    temp_dem_fp = Path(temp_product_dirs[PRODUCT_DEM]) / f"{ctx['base']}.tif"
    _write_tif(dem_pack["dem"], str(temp_dem_fp), dem_pack["xmin"], dem_pack["ymax"], dem_res, crs=ctx["crs"])

    z_dem = _sample_bilinear_from_grid(
        dem_pack["dem"], ctx["x"], ctx["y"], dem_pack["xmin"], dem_pack["ymax"], dem_res
    )
    residual = ctx["z"] - z_dem

    # provisional temp normalized for inspection/debug
    temp_norm_fp = Path(temp_product_dirs[PRODUCT_NORMALIZED]) / f"{ctx['base']}.las"
    _write_temp_normalized_from_residual(ctx, residual, temp_norm_fp)

    cls = ctx["cls"].copy()

    nonground = cls != 2
    ground = cls == 2

    promote = nonground & np.isfinite(residual) & (residual <= float(nonground_to_ground_max_z))
    demote = ground & np.isfinite(residual) & (residual > float(ground_to_nonground_min_z))

    changed = int(np.count_nonzero(promote) + np.count_nonzero(demote))
    if np.any(promote):
        cls[promote] = 2
    if np.any(demote):
        cls[demote] = 1

    if changed:
        out = laspy.LasData(ctx["las"].header)
        out.points = ctx["las"].points.copy()
        try:
            out.classification = cls.astype(np.uint8, copy=False)
        except Exception:
            out.classification = np.asarray(cls, dtype=np.uint8)
        out.write(classified_fp)

    return {
        "tile": Path(classified_fp).name,
        "changed_points": changed,
        "promoted_to_ground": int(np.count_nonzero(promote)),
        "demoted_to_nonground": int(np.count_nonzero(demote)),
        "status": "ok",
    }


def apply_fp_fix_to_output_root(
    out_root: str | Path,
    sensor_mode: str,
    *,
    dem_res: float = 0.25,
    nonground_to_ground_max_z: float = 0.0,
    ground_to_nonground_min_z: float = 0.06,
    keep_temp: bool = False,
) -> dict[str, Any]:
    out_root = Path(out_root)
    classified_root = out_root / PRODUCT_GC
    if not classified_root.exists():
        raise FileNotFoundError(f"FAST_GC folder not found: {classified_root}")

    classified_files = list_classified_files(classified_root)
    if not classified_files:
        raise FileNotFoundError(f"No classified LAS files found in: {classified_root}")

    temp_root = out_root / "_temp_fp_fix"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_product_dirs = _make_product_dirs(str(temp_root))

    total_t0 = perf_counter()
    summary_rows: list[dict[str, Any]] = []

    with tqdm(classified_files, desc=f"FP-FIX {sensor_mode.upper()}", unit="tile", dynamic_ncols=True) as pbar:
        ema_dt: float | None = None
        for idx, gc_fp in enumerate(pbar, start=1):
            t0 = perf_counter()

            row = _fix_one_classified_tile(
                gc_fp,
                temp_product_dirs=temp_product_dirs,
                dem_res=dem_res,
                nonground_to_ground_max_z=nonground_to_ground_max_z,
                ground_to_nonground_min_z=ground_to_nonground_min_z,
            )

            summary_rows.append(row)

            dt = perf_counter() - t0
            ema_dt = dt if ema_dt is None else (0.85 * ema_dt + 0.15 * dt)

            status = row.get("status", "ok")
            if status == "skipped_too_few_ground":
                pbar.set_postfix_str(
                    f"{idx}/{len(classified_files)} | {ema_dt:.2f}s/tile | skipped={Path(gc_fp).name}"
                )
            else:
                pbar.set_postfix_str(
                    f"{idx}/{len(classified_files)} | {ema_dt:.2f}s/tile | current={Path(gc_fp).name}"
                )

    total_elapsed = perf_counter() - total_t0
    total_changed = int(sum(int(r.get("changed_points", 0)) for r in summary_rows))
    total_promoted = int(sum(int(r.get("promoted_to_ground", 0)) for r in summary_rows))
    total_demoted = int(sum(int(r.get("demoted_to_nonground", 0)) for r in summary_rows))
    skipped_tiles = [r["tile"] for r in summary_rows if r.get("status") == "skipped_too_few_ground"]

    summary = {
        "sensor_mode": sensor_mode.upper(),
        "dem_res": float(dem_res),
        "nonground_to_ground_max_z": float(nonground_to_ground_max_z),
        "ground_to_nonground_min_z": float(ground_to_nonground_min_z),
        "tile_count": len(classified_files),
        "skipped_tile_count": len(skipped_tiles),
        "skipped_tiles": skipped_tiles,
        "total_changed_points": total_changed,
        "total_promoted_to_ground": total_promoted,
        "total_demoted_to_nonground": total_demoted,
        "time_s": float(total_elapsed),
        "tiles": summary_rows,
    }

    summary_fp = temp_root / "fp_fix_summary.json"
    with summary_fp.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[TIME] FP-FIX {sensor_mode.upper()}: {len(classified_files)} tiles | "
        f"total={total_elapsed:.2f}s | avg={total_elapsed / max(len(classified_files), 1):.2f}s/tile | "
        f"changed={total_changed} | skipped={len(skipped_tiles)}"
    )

    if skipped_tiles:
        print("[INFO] FP-FIX skipped tiles with too few ground points:")
        for name in skipped_tiles:
            print(f"  - {name}")

    if not keep_temp:
        shutil.rmtree(temp_root, ignore_errors=True)

    return summary