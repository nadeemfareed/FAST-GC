from __future__ import annotations

import csv
import math
import os
import re
from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
except Exception:  # pragma: no cover
    rasterio = None
    reproject = None
    Resampling = None

from .monster import DEFAULT_BACKEND, log_info, run_stage


PRODUCT_CHANGE = "FAST_CHANGE"
CHANGE_INPUT_CHOICES = {"FAST_DEM", "FAST_DSM", "FAST_CHM", "FAST_TERRAIN"}
CHANGE_MODE_CHOICES = {"pairwise", "sequential", "baseline"}
LOD_MODE_CHOICES = {"threshold_only", "rss", "max"}

DIRECT_RASTER_EXTS = {".tif", ".tiff"}


def _require_rasterio():
    if rasterio is None or reproject is None or Resampling is None:
        raise RuntimeError("rasterio is required for FAST_CHANGE products.")


def _existing_output(path: Path) -> bool:
    return path.exists() and path.is_file()


def _safe_token(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def _extract_sort_key(fp: Path):
    nums = re.findall(r"\d+", fp.stem)
    if nums:
        return tuple(int(n) for n in nums)
    return (fp.stem.lower(),)


def _list_rasters(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    rasters = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in DIRECT_RASTER_EXTS]
    rasters.sort(key=_extract_sort_key)
    return rasters


def _read_raster(fp: str):
    _require_rasterio()
    with rasterio.open(fp) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        bounds = src.bounds
        shape = arr.shape
    return arr, profile, transform, crs, nodata, bounds, shape


def _write_raster(arr: np.ndarray, profile: dict, out_fp: str, nodata=None):
    _require_rasterio()
    os.makedirs(os.path.dirname(out_fp), exist_ok=True)

    profile_out = profile.copy()
    profile_out.update(
        dtype="float32",
        count=1,
        compress="lzw",
    )
    if nodata is not None:
        profile_out["nodata"] = nodata

    with rasterio.open(out_fp, "w", **profile_out) as dst:
        dst.write(arr.astype(np.float32), 1)


def _pixel_area(transform) -> float:
    dx = float(transform.a)
    dy = float(abs(transform.e))
    return dx * dy


def _reproject_to_match(
    src_arr: np.ndarray,
    src_transform,
    src_crs,
    src_nodata,
    dst_shape: tuple[int, int],
    dst_transform,
    dst_crs,
    dst_nodata=np.nan,
):
    out = np.full(dst_shape, dst_nodata, dtype=np.float32)

    reproject(
        source=src_arr,
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=dst_nodata,
        resampling=Resampling.bilinear,
    )
    return out


def _align_pair(ref_fp: str, cmp_fp: str):
    ref_arr, ref_profile, ref_transform, ref_crs, ref_nodata, _, ref_shape = _read_raster(ref_fp)
    cmp_arr, _, cmp_transform, cmp_crs, cmp_nodata, _, cmp_shape = _read_raster(cmp_fp)

    same_shape = ref_shape == cmp_shape
    same_transform = ref_transform == cmp_transform
    same_crs = str(ref_crs) == str(cmp_crs)

    if same_shape and same_transform and same_crs:
        return ref_arr, cmp_arr, ref_profile, ref_nodata

    cmp_aligned = _reproject_to_match(
        src_arr=cmp_arr,
        src_transform=cmp_transform,
        src_crs=cmp_crs,
        src_nodata=cmp_nodata,
        dst_shape=ref_shape,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        dst_nodata=np.nan,
    )
    return ref_arr, cmp_aligned, ref_profile, ref_nodata


def _valid_mask(a: np.ndarray, b: np.ndarray, nodata_a, nodata_b=np.nan):
    mask = np.isfinite(a) & np.isfinite(b)

    if nodata_a is not None and not (isinstance(nodata_a, float) and np.isnan(nodata_a)):
        mask &= a != nodata_a

    if nodata_b is not None and not (isinstance(nodata_b, float) and np.isnan(nodata_b)):
        mask &= b != nodata_b

    return mask


def _compute_lod(
    *,
    change_threshold: float,
    sigma1: float = 0.0,
    sigma2: float = 0.0,
    lod_mode: str = "rss",
) -> float:
    lod_mode = str(lod_mode).strip().lower()
    if lod_mode not in LOD_MODE_CHOICES:
        raise ValueError(f"Unsupported lod_mode: {lod_mode}")

    thr = float(change_threshold)
    s1 = max(0.0, float(sigma1))
    s2 = max(0.0, float(sigma2))

    if lod_mode == "threshold_only":
        return thr
    if lod_mode == "rss":
        return max(thr, math.sqrt(s1 * s1 + s2 * s2))
    if lod_mode == "max":
        return max(thr, s1, s2)
    return thr


def _compute_change_products(
    a: np.ndarray,
    b: np.ndarray,
    valid: np.ndarray,
    lod_value: float,
):
    delta = np.full_like(a, np.nan, dtype=np.float32)
    absolute = np.full_like(a, np.nan, dtype=np.float32)
    gain = np.zeros_like(a, dtype=np.float32)
    loss = np.zeros_like(a, dtype=np.float32)
    stable = np.zeros_like(a, dtype=np.float32)
    lod_mask = np.zeros_like(a, dtype=np.float32)

    d = b - a

    delta[valid] = d[valid]
    absolute[valid] = np.abs(d[valid])

    changed = valid & (np.abs(d) > lod_value)
    stable_mask = valid & ~changed

    gain[valid] = (d[valid] > lod_value).astype(np.float32)
    loss[valid] = (d[valid] < -lod_value).astype(np.float32)
    stable[valid] = stable_mask[valid].astype(np.float32)
    lod_mask[valid] = changed[valid].astype(np.float32)

    return delta, absolute, gain, loss, stable, lod_mask


def _pairwise_indices(n: int) -> list[tuple[int, int]]:
    if n < 2:
        return []
    return [(i, j) for i in range(n - 1) for j in range(i + 1, n)]


def _sequential_indices(n: int) -> list[tuple[int, int]]:
    if n < 2:
        return []
    return [(i, i + 1) for i in range(n - 1)]


def _baseline_indices(n: int, baseline_index: int) -> list[tuple[int, int]]:
    if n < 2:
        return []
    b = int(max(0, min(n - 1, baseline_index)))
    return [(b, i) for i in range(n) if i != b]


def _series_pairs(n: int, mode: str, baseline_index: int) -> list[tuple[int, int]]:
    mode = str(mode).strip().lower()
    if mode not in CHANGE_MODE_CHOICES:
        raise ValueError(f"Unsupported change mode: {mode}")

    if mode == "pairwise":
        return _pairwise_indices(n)
    if mode == "sequential":
        return _sequential_indices(n)
    if mode == "baseline":
        return _baseline_indices(n, baseline_index)
    return []


def _resolve_input_groups(
    processed_root: Path,
    change_input_type: str,
    source_subdir: str | None = None,
) -> dict[str, list[Path]]:
    change_input_type = str(change_input_type).strip()

    direct_rasters = _list_rasters(processed_root)
    if direct_rasters:
        group_name = source_subdir if source_subdir else change_input_type
        return {group_name: direct_rasters}

    if change_input_type == "FAST_DEM":
        root = processed_root / "FAST_DEM"
        if not root.exists():
            raise FileNotFoundError(f"FAST_DEM folder not found: {root}")
        rasters = _list_rasters(root)
        if not rasters:
            raise FileNotFoundError(f"No DEM rasters found in: {root}")
        return {"FAST_DEM": rasters}

    if change_input_type == "FAST_DSM":
        root = processed_root / "FAST_DSM"
        if not root.exists():
            raise FileNotFoundError(f"FAST_DSM folder not found: {root}")
        rasters = _list_rasters(root)
        if not rasters:
            raise FileNotFoundError(f"No DSM rasters found in: {root}")
        return {"FAST_DSM": rasters}

    if change_input_type == "FAST_CHM":
        root = processed_root / "FAST_CHM"
        if root.exists():
            if source_subdir:
                sub = root / source_subdir
                if not sub.exists():
                    raise FileNotFoundError(f"Requested CHM subfolder not found: {sub}")
                rasters = _list_rasters(sub)
                if not rasters:
                    raise FileNotFoundError(f"No CHM rasters found in: {sub}")
                return {source_subdir: rasters}

            groups: dict[str, list[Path]] = {}
            for method_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
                rasters = _list_rasters(method_dir)
                if rasters:
                    groups[method_dir.name] = rasters

            if groups:
                return groups

        direct_rasters = _list_rasters(processed_root)
        if direct_rasters:
            group_name = source_subdir if source_subdir else "FAST_CHM"
            return {group_name: direct_rasters}

        raise FileNotFoundError(f"No CHM rasters found under: {root if root.exists() else processed_root}")

    if change_input_type == "FAST_TERRAIN":
        root = processed_root / "FAST_TERRAIN"
        if not root.exists():
            raise FileNotFoundError(f"FAST_TERRAIN folder not found: {root}")

        if source_subdir:
            sub = root / source_subdir
            if not sub.exists():
                raise FileNotFoundError(f"Requested terrain subfolder not found: {sub}")
            rasters = _list_rasters(sub)
            if not rasters:
                raise FileNotFoundError(f"No terrain rasters found in: {sub}")
            return {source_subdir: rasters}

        groups: dict[str, list[Path]] = {}
        for terr_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            rasters = _list_rasters(terr_dir)
            if rasters:
                groups[terr_dir.name] = rasters

        if not groups:
            raise FileNotFoundError(f"No terrain rasters found under: {root}")
        return groups

    raise ValueError(f"Unsupported change input type: {change_input_type}")


def _output_root(
    processed_root: Path,
    family: str,
    group_name: str,
    mode: str,
) -> Path:
    return processed_root / PRODUCT_CHANGE / family / group_name / mode


def _write_change_bundle(
    out_root: Path,
    profile: dict,
    pair_name: str,
    delta: np.ndarray,
    absolute: np.ndarray,
    gain: np.ndarray,
    loss: np.ndarray,
    stable: np.ndarray,
    lod_mask: np.ndarray,
):
    delta_fp = out_root / "delta" / f"{pair_name}.tif"
    abs_fp = out_root / "absolute" / f"{pair_name}.tif"
    gain_fp = out_root / "gain_mask" / f"{pair_name}.tif"
    loss_fp = out_root / "loss_mask" / f"{pair_name}.tif"
    stable_fp = out_root / "stable_mask" / f"{pair_name}.tif"
    lod_fp = out_root / "lod_mask" / f"{pair_name}.tif"

    _write_raster(delta, profile, str(delta_fp), nodata=np.nan)
    _write_raster(absolute, profile, str(abs_fp), nodata=np.nan)
    _write_raster(gain, profile, str(gain_fp), nodata=0.0)
    _write_raster(loss, profile, str(loss_fp), nodata=0.0)
    _write_raster(stable, profile, str(stable_fp), nodata=0.0)
    _write_raster(lod_mask, profile, str(lod_fp), nodata=0.0)


def _pair_stats(
    *,
    delta: np.ndarray,
    valid: np.ndarray,
    lod_value: float,
    transform,
) -> dict:
    pix_area = _pixel_area(transform)

    valid_count = int(np.count_nonzero(valid))
    changed_mask = valid & (np.abs(delta) > lod_value)
    gain_mask = valid & (delta > lod_value)
    loss_mask = valid & (delta < -lod_value)
    stable_mask = valid & ~changed_mask

    gain_area = float(np.count_nonzero(gain_mask) * pix_area)
    loss_area = float(np.count_nonzero(loss_mask) * pix_area)
    stable_area = float(np.count_nonzero(stable_mask) * pix_area)
    changed_area = float(np.count_nonzero(changed_mask) * pix_area)

    gain_volume = float(np.nansum(np.where(gain_mask, delta, 0.0)) * pix_area)
    loss_volume = float(np.nansum(np.where(loss_mask, delta, 0.0)) * pix_area)
    net_volume = float(np.nansum(np.where(valid, delta, 0.0)) * pix_area)

    stats = {
        "valid_cells": valid_count,
        "lod_value": float(lod_value),
        "delta_min": float(np.nanmin(delta[valid])) if valid_count else np.nan,
        "delta_max": float(np.nanmax(delta[valid])) if valid_count else np.nan,
        "delta_mean": float(np.nanmean(delta[valid])) if valid_count else np.nan,
        "delta_std": float(np.nanstd(delta[valid])) if valid_count else np.nan,
        "changed_area": changed_area,
        "stable_area": stable_area,
        "gain_area": gain_area,
        "loss_area": loss_area,
        "gain_volume": gain_volume,
        "loss_volume": loss_volume,
        "net_volume": net_volume,
    }
    return stats


def _write_summary_csv(out_root: Path, rows: list[dict]):
    if not rows:
        return

    csv_fp = out_root / "change_summary.csv"
    os.makedirs(csv_fp.parent, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with csv_fp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _pair_task_name(pair: tuple[int, int], series: list[Path]) -> str:
    i, j = pair
    fp1 = series[i]
    fp2 = series[j]
    token1 = _safe_token(fp1.stem)
    token2 = _safe_token(fp2.stem)
    return f"{token1}__to__{token2}"


def _extract_ok_records(summary) -> list:
    """
    Handle multiple possible StageSummary layouts.
    Some repo versions expose:
      - summary.results_ok -> iterable
      - summary.ok_results -> iterable
      - summary.records_ok -> iterable
      - summary.ok -> count integer
    """
    for attr in ("results_ok", "ok_results", "records_ok", "ok_records", "successes"):
        value = getattr(summary, attr, None)
        if isinstance(value, (list, tuple)):
            return list(value)
    return []


def _payload_from_record(rec):
    if isinstance(rec, dict):
        payload = rec.get("payload")
        if isinstance(payload, dict):
            return payload
        result_obj = rec.get("result")
        if isinstance(result_obj, dict):
            payload = result_obj.get("payload")
            if isinstance(payload, dict):
                return payload
        return None

    payload = getattr(rec, "payload", None)
    if isinstance(payload, dict):
        return payload

    result_obj = getattr(rec, "result", None)
    if isinstance(result_obj, dict):
        payload = result_obj.get("payload")
        if isinstance(payload, dict):
            return payload

    return None


def _process_change_pair(
    task_name: str,
    *,
    out_root: Path,
    fp1: Path,
    fp2: Path,
    change_threshold: float,
    sigma1: float,
    sigma2: float,
    lod_mode: str,
    skip_existing: bool,
    overwrite: bool,
):
    delta_fp = out_root / "delta" / f"{task_name}.tif"
    if skip_existing and _existing_output(delta_fp) and not overwrite:
        return {
            "status": "skipped",
            "item": task_name,
            "reason": f"Existing FAST_CHANGE output found: {delta_fp}",
            "output": str(delta_fp),
        }

    a, b, profile, nodata = _align_pair(str(fp1), str(fp2))
    valid = _valid_mask(a, b, nodata, np.nan)

    lod_value = _compute_lod(
        change_threshold=change_threshold,
        sigma1=sigma1,
        sigma2=sigma2,
        lod_mode=lod_mode,
    )

    delta, absolute, gain, loss, stable, lod_mask = _compute_change_products(
        a=a,
        b=b,
        valid=valid,
        lod_value=lod_value,
    )

    _write_change_bundle(
        out_root=out_root,
        profile=profile,
        pair_name=task_name,
        delta=delta,
        absolute=absolute,
        gain=gain,
        loss=loss,
        stable=stable,
        lod_mask=lod_mask,
    )

    stats = _pair_stats(
        delta=delta,
        valid=valid,
        lod_value=lod_value,
        transform=profile["transform"],
    )
    stats.update(
        {
            "source_1": str(fp1),
            "source_2": str(fp2),
            "pair_name": task_name,
        }
    )
    return {
        "status": "ok",
        "item": task_name,
        "output": str(delta_fp),
        "payload": stats,
    }


def run_change_from_processed_root(
    processed_root: str | os.PathLike[str],
    *,
    change_input_type: str = "FAST_DEM",
    change_mode: str = "pairwise",
    change_threshold: float = 0.0,
    baseline_index: int = 0,
    skip_existing: bool = False,
    overwrite: bool = False,
    source_subdir: str | None = None,
    sigma1: float = 0.0,
    sigma2: float = 0.0,
    lod_mode: str = "rss",
    n_jobs: int = 1,
    joblib_backend: str = DEFAULT_BACKEND,
    joblib_batch_size: int | str = "auto",
    joblib_pre_dispatch: str = "2*n_jobs",
) -> str:
    _require_rasterio()

    processed_root = Path(processed_root)
    if not processed_root.exists():
        raise FileNotFoundError(f"Processed root not found: {processed_root}")

    groups = _resolve_input_groups(
        processed_root=processed_root,
        change_input_type=change_input_type,
        source_subdir=source_subdir,
    )

    family = str(change_input_type).strip()
    mode = str(change_mode).strip().lower()
    if mode not in CHANGE_MODE_CHOICES:
        raise ValueError(f"Unsupported change mode: {mode}")

    log_info(f"FAST_CHANGE input type: {family}")
    log_info(f"FAST_CHANGE mode: {mode}")
    if source_subdir:
        log_info(f"FAST_CHANGE source subdir: {source_subdir}")

    for group_name, series in groups.items():
        n = len(series)
        if n < 2:
            print(f"[SKIP] Need at least 2 rasters for FAST_CHANGE in group '{group_name}'. Found {n}.")
            continue

        pairs = _series_pairs(n, mode=mode, baseline_index=baseline_index)
        out_root = _output_root(processed_root, family, group_name, mode)
        out_root.mkdir(parents=True, exist_ok=True)

        log_info(f"FAST_CHANGE group '{group_name}': {n} raster(s), {len(pairs)} comparison pair(s)")

        task_map: dict[str, tuple[Path, Path]] = {}
        task_names: list[str] = []
        for i, j in pairs:
            name = _pair_task_name((i, j), series)
            task_names.append(name)
            task_map[name] = (series[i], series[j])

        def _worker(task_name: str):
            fp1, fp2 = task_map[task_name]
            return _process_change_pair(
                task_name,
                out_root=out_root,
                fp1=fp1,
                fp2=fp2,
                change_threshold=change_threshold,
                sigma1=sigma1,
                sigma2=sigma2,
                lod_mode=lod_mode,
                skip_existing=skip_existing,
                overwrite=overwrite,
            )

        summary = run_stage(
            stage_name=f"FAST-GC derive CHANGE [{family}/{group_name}/{mode}]",
            source=str(out_root),
            items=task_names,
            worker=_worker,
            n_jobs=n_jobs,
            unit="pair",
            joblib_backend=joblib_backend,
            joblib_batch_size=joblib_batch_size,
            joblib_pre_dispatch=joblib_pre_dispatch,
        )

        summary_rows: list[dict] = []
        ok_records = _extract_ok_records(summary)

        for rec in ok_records:
            payload = _payload_from_record(rec)
            if isinstance(payload, dict):
                row = payload.copy()
                row.update({"group": group_name, "mode": mode})
                summary_rows.append(row)

        if not summary_rows and len(task_names) == 1:
            # Fallback for repo variants where StageSummary stores only counts.
            fp1, fp2 = task_map[task_names[0]]
            a, b, profile, nodata = _align_pair(str(fp1), str(fp2))
            valid = _valid_mask(a, b, nodata, np.nan)
            lod_value = _compute_lod(
                change_threshold=change_threshold,
                sigma1=sigma1,
                sigma2=sigma2,
                lod_mode=lod_mode,
            )
            delta = np.full_like(a, np.nan, dtype=np.float32)
            d = b - a
            delta[valid] = d[valid]
            row = _pair_stats(
                delta=delta,
                valid=valid,
                lod_value=lod_value,
                transform=profile["transform"],
            )
            row.update(
                {
                    "source_1": str(fp1),
                    "source_2": str(fp2),
                    "pair_name": task_names[0],
                    "group": group_name,
                    "mode": mode,
                }
            )
            summary_rows.append(row)

        _write_summary_csv(out_root, summary_rows)

    return str(processed_root / PRODUCT_CHANGE)


__all__ = [
    "PRODUCT_CHANGE",
    "CHANGE_INPUT_CHOICES",
    "CHANGE_MODE_CHOICES",
    "LOD_MODE_CHOICES",
    "run_change_from_processed_root",
]