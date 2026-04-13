from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import generic_filter

from .sensors import sensor_defaults
from .monster import log_info, run_stage
import laspy


PRODUCT_STRUCTURE = "FAST_STRUCTURE"
STRUCTURE_PRODUCT_CHOICES = [
    "all",
    "canopy_cover",
    "z_mean",
    "z_max",
    "z_sd",
    "FHD",
    "VCI",
    "n_points",
]


# =========================================================
# Sensor-aware defaults for normalized-point-cloud metrics
# =========================================================
# Note:
# - FAST-GC currently validates sensor_mode as ALS | ULS | TLS.
# - MLS and PLS should be routed under TLS for now at the CLI / caller level.
# =========================================================


@dataclass(frozen=True)
class StructureDefaults:
    res: float
    min_h: float
    bin_size: float
    canopy_threshold: float
    canopy_mode: str = "all_points"
    na_fill: str = "none"  # none | 3x3_mean


_STRUCTURE_DEFAULTS: Dict[str, StructureDefaults] = {
    "ALS": StructureDefaults(
        res=1.0,
        min_h=2.0,
        bin_size=1.0,
        canopy_threshold=2.0,
        canopy_mode="all_points",
        na_fill="none",
    ),
    "ULS": StructureDefaults(
        res=0.5,
        min_h=1.5,
        bin_size=0.5,
        canopy_threshold=2.0,
        canopy_mode="all_points",
        na_fill="none",
    ),
    "TLS": StructureDefaults(
        res=0.25,
        min_h=0.5,
        bin_size=0.25,
        canopy_threshold=1.0,
        canopy_mode="all_points",
        na_fill="none",
    ),
}



def _find_tile_manifest_for_path(src_fp: str) -> Path | None:
    p = Path(src_fp).resolve()
    for parent in [p.parent, *p.parents]:
        cand = parent / "tile_manifest.json"
        if cand.exists():
            return cand
    return None


def _load_manifest_json(manifest_fp: Path) -> dict | None:
    try:
        with manifest_fp.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _match_tile_record(src_fp: str, manifest: dict) -> dict | None:
    src_path = str(Path(src_fp).resolve())
    src_name = Path(src_fp).name
    for tile in manifest.get("tiles", []):
        tile_path = tile.get("tile_path")
        if tile_path:
            try:
                if str(Path(tile_path).resolve()) == src_path:
                    return tile
            except Exception:
                if str(tile_path) == src_fp:
                    return tile
        if str(tile.get("tile_name", "")) == src_name:
            return tile
    return None


def _safe_parse_crs_from_las_or_manifest(las: laspy.LasData, fp: str | Path):
    try:
        crs = las.header.parse_crs()
    except Exception:
        crs = None

    if crs is None:
        try:
            manifest_fp = _find_tile_manifest_for_path(str(fp))
            manifest = _load_manifest_json(manifest_fp) if manifest_fp is not None else None
            tile = _match_tile_record(str(fp), manifest) if manifest is not None else None

            src_candidates: list[str] = []
            if tile is not None:
                for key in ("kept_source_paths", "source_paths"):
                    vals = tile.get(key)
                    if isinstance(vals, list):
                        src_candidates.extend([str(v) for v in vals if v])
                if tile.get("source_path"):
                    src_candidates.append(str(tile["source_path"]))

            seen = set()
            for src_fp in src_candidates:
                if src_fp in seen:
                    continue
                seen.add(src_fp)
                try:
                    with laspy.open(src_fp) as reader:
                        crs = reader.header.parse_crs()
                    if crs is not None:
                        break
                except Exception:
                    continue
        except Exception:
            crs = None

    if crs is None:
        return None

    try:
        from rasterio.crs import CRS as RioCRS
        return RioCRS.from_user_input(crs)
    except Exception:
        try:
            if hasattr(crs, "to_wkt"):
                from rasterio.crs import CRS as RioCRS
                return RioCRS.from_wkt(crs.to_wkt())
        except Exception:
            pass
    return crs

# =========================================================
# Public helpers
# =========================================================


def structure_defaults(sensor_mode: str) -> Dict[str, object]:
    """Return sensor-aware defaults for FAST_STRUCTURE.

    Notes
    -----
    The repo currently recognizes ALS, ULS, and TLS in sensors.py.
    MLS and PLS should be mapped to TLS upstream for now.
    """
    sm = (sensor_mode or "").upper().strip()
    if sm not in _STRUCTURE_DEFAULTS:
        raise ValueError(f"sensor_mode must be one of ALS|ULS|TLS (got {sensor_mode!r})")

    # Pull the existing repo defaults too so this module stays aligned with sensor_defaults().
    base = dict(sensor_defaults(sm))
    sdef = _STRUCTURE_DEFAULTS[sm]
    base.update(
        {
            "structure_res_default": sdef.res,
            "structure_min_h_default": sdef.min_h,
            "structure_bin_size_default": sdef.bin_size,
            "structure_canopy_threshold_default": sdef.canopy_threshold,
            "structure_canopy_mode_default": sdef.canopy_mode,
            "structure_na_fill_default": sdef.na_fill,
        }
    )
    return base


# =========================================================
# Core utilities
# =========================================================


def _validate_positive(name: str, value: float) -> None:
    if value is None or float(value) <= 0:
        raise ValueError(f"{name} must be > 0 (got {value!r})")


def _cell_slices(ix: np.ndarray, iy: np.ndarray, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort point-to-cell mapping and return row/col arrays + cell start offsets.

    Returns
    -------
    cell_ids_sorted : np.ndarray
        Sorted flattened cell ids.
    order : np.ndarray
        Point order used for sorting.
    starts : np.ndarray
        Start offsets per unique cell id in cell_ids_sorted.
    """
    cell_ids = iy.astype(np.int64) * np.int64(nx) + ix.astype(np.int64)
    order = np.argsort(cell_ids, kind="mergesort")
    cell_ids_sorted = cell_ids[order]
    unique_ids, starts = np.unique(cell_ids_sorted, return_index=True)
    return unique_ids, order, starts


def _nanmean_filter(values: np.ndarray) -> float:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return np.nan
    return float(np.mean(vals))


def _fill_na(grid: np.ndarray, mode: str) -> np.ndarray:
    mode = (mode or "none").lower().strip()
    if mode in {"none", "off", "false"}:
        return grid
    if mode not in {"3x3", "3x3_mean", "mean3x3"}:
        raise ValueError(f"Unsupported structure_na_fill mode: {mode!r}")
    return generic_filter(grid, _nanmean_filter, size=3, mode="nearest")


def _fix_pits_and_voids(grid: np.ndarray, pit_threshold: float = 1.0, size: int = 3) -> np.ndarray:
    """Fill NaNs with 0 and replace only local pits using a neighborhood mean."""
    out = np.asarray(grid, dtype=np.float32).copy()
    if out.size == 0:
        return out

    out[~np.isfinite(out)] = 0.0
    size = int(max(1, size))
    if size % 2 == 0:
        size += 1

    local = generic_filter(out, _nanmean_filter, size=size, mode="nearest")
    thr = float(max(0.0, pit_threshold))
    pit_mask = np.isfinite(out) & (out < (local - thr))
    out[pit_mask] = local[pit_mask]
    return out.astype(np.float32, copy=False)


# =========================================================
# Metric math
# =========================================================


def _compute_entropy_metrics(vals: np.ndarray, bin_size: float) -> Tuple[float, float]:
    """Return (FHD, VCI) from height values in one cell.

    FHD = Shannon entropy across height bins.
    VCI = normalized entropy (0..1), using the occupied-bin count.
    """
    if vals.size == 0:
        return np.nan, np.nan

    zmax = float(np.max(vals))
    if zmax <= 0:
        return np.nan, np.nan

    # Ensure at least one valid interval.
    upper = max(bin_size, zmax + bin_size)
    bins = np.arange(0.0, upper + 1e-9, bin_size, dtype=np.float64)
    if bins.size < 2:
        return np.nan, np.nan

    hist, _ = np.histogram(vals, bins=bins)
    total = int(hist.sum())
    if total == 0:
        return np.nan, np.nan

    p = hist.astype(np.float64) / float(total)
    p = p[p > 0]
    if p.size == 0:
        return np.nan, np.nan

    fhd = -float(np.sum(p * np.log(p)))
    max_entropy = float(np.log(p.size)) if p.size > 1 else 0.0
    vci = (fhd / max_entropy) if max_entropy > 0 else 0.0
    return fhd, vci


# =========================================================
# Main computation
# =========================================================


def compute_structure_metrics(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    sensor_mode: str,
    res: float | None = None,
    min_h: float | None = None,
    bin_size: float | None = None,
    canopy_threshold: float | None = None,
    canopy_mode: str | None = None,
    na_fill: str | None = None,
    bounds: Tuple[float, float, float, float] | None = None,
) -> Dict[str, object]:
    """Compute FAST_STRUCTURE metrics on a normalized point cloud.

    Parameters
    ----------
    x, y, z
        Normalized point cloud coordinates. z must already be height-above-ground.
    sensor_mode
        ALS, ULS, or TLS. MLS/PLS should be mapped to TLS upstream for now.
    res
        Horizontal raster resolution for structure metrics.
    min_h
        Minimum normalized height to include in vegetation metrics.
    bin_size
        Vertical bin size for FHD and VCI.
    canopy_threshold
        Threshold for canopy cover.
    canopy_mode
        Currently supports "all_points". Kept as an explicit switch for future extension.
    na_fill
        Optional NA fill mode: none | 3x3_mean.
    bounds
        Optional (xmin, ymin, xmax, ymax). If omitted, point bounds are used.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if not (x.size == y.size == z.size):
        raise ValueError("x, y, z must have the same length")
    if x.size == 0:
        raise ValueError("Point cloud is empty")

    sm = (sensor_mode or "").upper().strip()
    sdef = _STRUCTURE_DEFAULTS.get(sm)
    if sdef is None:
        raise ValueError(f"sensor_mode must be one of ALS|ULS|TLS (got {sensor_mode!r})")

    res = float(sdef.res if res is None else res)
    min_h = float(sdef.min_h if min_h is None else min_h)
    bin_size = float(sdef.bin_size if bin_size is None else bin_size)
    canopy_threshold = float(sdef.canopy_threshold if canopy_threshold is None else canopy_threshold)
    canopy_mode = str(sdef.canopy_mode if canopy_mode is None else canopy_mode).lower().strip()
    na_fill = str(sdef.na_fill if na_fill is None else na_fill).lower().strip()

    _validate_positive("structure_res", res)
    _validate_positive("structure_min_h", min_h)
    _validate_positive("structure_bin_size", bin_size)
    _validate_positive("canopy_threshold", canopy_threshold)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[valid]
    y = y[valid]
    z = z[valid]
    if x.size == 0:
        raise ValueError("No finite points available after filtering")

    veg = z >= min_h
    x = x[veg]
    y = y[veg]
    z = z[veg]
    if x.size == 0:
        raise ValueError("No points remain after applying structure_min_h filter")

    if bounds is None:
        xmin, ymin = float(np.min(x)), float(np.min(y))
        xmax, ymax = float(np.max(x)), float(np.max(y))
    else:
        xmin, ymin, xmax, ymax = map(float, bounds)

    width = xmax - xmin
    height = ymax - ymin
    if width <= 0 or height <= 0:
        raise ValueError("Invalid bounds for structure metric rasterization")

    nx = max(1, int(np.ceil(width / res)))
    ny = max(1, int(np.ceil(height / res)))

    ix = np.floor((x - xmin) / res).astype(np.int64)
    iy = np.floor((ymax - y) / res).astype(np.int64)

    in_bounds = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    x = x[in_bounds]
    y = y[in_bounds]
    z = z[in_bounds]
    ix = ix[in_bounds]
    iy = iy[in_bounds]
    if x.size == 0:
        raise ValueError("No points fall inside the requested structure metric bounds")

    z_mean = np.full((ny, nx), np.nan, dtype=np.float32)
    z_max = np.full((ny, nx), np.nan, dtype=np.float32)
    z_sd = np.full((ny, nx), np.nan, dtype=np.float32)
    canopy_cover = np.full((ny, nx), np.nan, dtype=np.float32)
    fhd = np.full((ny, nx), np.nan, dtype=np.float32)
    vci = np.full((ny, nx), np.nan, dtype=np.float32)
    n_points = np.zeros((ny, nx), dtype=np.int32)

    cell_ids, order, starts = _cell_slices(ix, iy, nx, ny)
    z_sorted = z[order]

    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = z_sorted.size

    for cell_id, start, end in zip(cell_ids, starts, ends):
        vals = z_sorted[start:end]
        row = int(cell_id // nx)
        col = int(cell_id % nx)

        if vals.size == 0:
            continue

        n_points[row, col] = int(vals.size)
        z_mean[row, col] = np.float32(np.mean(vals))
        z_max[row, col] = np.float32(np.max(vals))
        z_sd[row, col] = np.float32(np.std(vals, ddof=0))

        if canopy_mode not in {"all_points", "all"}:
            raise ValueError(
                f"Unsupported canopy_mode {canopy_mode!r}; currently only 'all_points' is implemented"
            )
        canopy_cover[row, col] = np.float32(np.mean(vals >= canopy_threshold))

        fhd_val, vci_val = _compute_entropy_metrics(vals, bin_size=bin_size)
        fhd[row, col] = np.float32(fhd_val) if np.isfinite(fhd_val) else np.nan
        vci[row, col] = np.float32(vci_val) if np.isfinite(vci_val) else np.nan

    z_mean = _fix_pits_and_voids(_fill_na(z_mean, na_fill), pit_threshold=1.0, size=3)
    z_max = _fix_pits_and_voids(_fill_na(z_max, na_fill), pit_threshold=1.0, size=3)
    z_sd = _fix_pits_and_voids(_fill_na(z_sd, na_fill), pit_threshold=0.5, size=3)
    canopy_cover = _fix_pits_and_voids(_fill_na(canopy_cover, na_fill), pit_threshold=0.2, size=3)
    fhd = _fix_pits_and_voids(_fill_na(fhd, na_fill), pit_threshold=0.5, size=3)
    vci = _fix_pits_and_voids(_fill_na(vci, na_fill), pit_threshold=0.2, size=3)

    transform = from_origin(xmin, ymax, res, res)

    return {
        "sensor_mode": sm,
        "res": res,
        "min_h": min_h,
        "bin_size": bin_size,
        "canopy_threshold": canopy_threshold,
        "canopy_mode": canopy_mode,
        "na_fill": na_fill,
        "bounds": (xmin, ymin, xmax, ymax),
        "transform": transform,
        "shape": (ny, nx),
        "metrics": {
            "canopy_cover": canopy_cover,
            "z_mean": z_mean,
            "z_max": z_max,
            "z_sd": z_sd,
            "FHD": fhd,
            "VCI": vci,
            "n_points": n_points,
        },
    }


# =========================================================
# Raster export helpers
# =========================================================


def write_structure_rasters(
    result: Mapping[str, object],
    out_dir: str | Path,
    *,
    crs=None,
    nodata: float = np.nan,
    prefix: str | None = None,
) -> Dict[str, Path]:
    """Write metric rasters from compute_structure_metrics() output.

    Returns a mapping of metric name -> written path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = result["transform"]
    metrics: Mapping[str, np.ndarray] = result["metrics"]

    written: Dict[str, Path] = {}
    for name, arr in metrics.items():
        fname = f"{prefix}_{name}.tif" if prefix else f"{name}.tif"
        path = out_dir / fname

        arr_write = np.asarray(arr)
        dtype = arr_write.dtype
        if arr_write.dtype.kind not in {"f", "i", "u"}:
            arr_write = arr_write.astype(np.float32)
            dtype = arr_write.dtype

        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=arr_write.shape[0],
            width=arr_write.shape[1],
            count=1,
            dtype=str(dtype),
            transform=transform,
            crs=crs,
            nodata=nodata,
            compress="deflate",
        ) as dst:
            dst.write(arr_write, 1)

        written[name] = path

    return written



def _existing_structure_output(path: Path) -> bool:
    return path.exists() and any(path.rglob('*.tif'))


def _resolve_structure_inputs(source_root: str | Path):
    p = Path(source_root)
    if p.is_file() and p.suffix.lower() in {'.las', '.laz'} and 'FAST_NORMALIZED' in p.stem:
        return [p], p.parent / PRODUCT_STRUCTURE, 'direct_normalized'
    if p.is_dir() and p.name == 'FAST_NORMALIZED':
        files = sorted([q for q in p.iterdir() if q.is_file() and q.suffix.lower() in {'.las','.laz'}])
        return files, p.parent / PRODUCT_STRUCTURE, 'normalized_dir'
    if p.is_dir() and p.name.startswith('Merged_'):
        files = sorted([q for q in p.iterdir() if q.is_file() and q.suffix.lower() in {'.las','.laz'} and 'FAST_NORMALIZED' in q.stem])
        return files, p / PRODUCT_STRUCTURE, 'merged_root'
    if p.is_dir() and (p / 'FAST_NORMALIZED').exists():
        nr = p / 'FAST_NORMALIZED'
        files = sorted([q for q in nr.iterdir() if q.is_file() and q.suffix.lower() in {'.las','.laz'}])
        return files, p / PRODUCT_STRUCTURE, 'processed_root'
    raise FileNotFoundError(f'Could not resolve FAST_NORMALIZED source from: {p}')


def _filter_metrics(metrics: dict, wanted: list[str] | None):
    requested = list(wanted or ['all'])
    if 'all' in requested:
        return metrics
    keep = set(requested)
    return {k:v for k,v in metrics.items() if k in keep}


def run_structure_from_root(
    source_root: str | Path,
    *,
    sensor_mode: str,
    structure_products: list[str] | None = None,
    structure_res: float | None = None,
    structure_min_h: float | None = None,
    structure_bin_size: float | None = None,
    canopy_thr: float | None = None,
    canopy_mode: str | None = None,
    structure_na_fill: str | None = None,
    n_jobs: int = 1,
    joblib_backend: str = 'loky',
    joblib_batch_size: int | str = 'auto',
    joblib_pre_dispatch: str | int = '2*n_jobs',
    skip_existing: bool = False,
    overwrite: bool = False,
):
    src_files, out_base_root, input_mode = _resolve_structure_inputs(source_root)
    if not src_files:
        raise FileNotFoundError(f'No FAST_NORMALIZED LAS files found under: {source_root}')
    out_base_root.mkdir(parents=True, exist_ok=True)

    log_info(f'FAST_STRUCTURE input mode: {input_mode}')
    log_info(f'FAST_STRUCTURE source files: {len(src_files)}')

    def _task(fp: Path):
        dataset_label = fp.stem
        out_dir = out_base_root / dataset_label
        if skip_existing and _existing_structure_output(out_dir) and not overwrite:
            return {'status':'skipped','output':str(out_dir)}
        out_dir.mkdir(parents=True, exist_ok=True)
        las = laspy.read(fp)
        result = compute_structure_metrics(
            np.asarray(las.x), np.asarray(las.y), np.asarray(las.z),
            sensor_mode=sensor_mode,
            res=structure_res,
            min_h=structure_min_h,
            bin_size=structure_bin_size,
            canopy_threshold=canopy_thr,
            canopy_mode=canopy_mode,
            na_fill=structure_na_fill,
        )
        result['metrics'] = _filter_metrics(result['metrics'], structure_products)
        crs = _safe_parse_crs_from_las_or_manifest(las, fp)
        written = write_structure_rasters(result, out_dir, crs=crs, prefix=dataset_label)
        return {'status':'ok','output':str(out_dir),'written':{k:str(v) for k,v in written.items()}}

    summary = run_stage(
        stage_name='FAST-GC derive STRUCTURE',
        items=src_files,
        func=_task,
        item_name_fn=lambda p: Path(p).name,
        n_jobs=n_jobs,
        backend=joblib_backend,
        batch_size=joblib_batch_size,
        pre_dispatch=joblib_pre_dispatch,
        source=str(source_root),
        unit='dataset',
    )

    manifest = {
        'module': PRODUCT_STRUCTURE,
        'source_root': str(source_root),
        'input_mode': input_mode,
        'sensor_mode': sensor_mode,
        'products': structure_products or ['all'],
        'res': structure_res,
        'min_h': structure_min_h,
        'bin_size': structure_bin_size,
        'canopy_thr': canopy_thr,
        'canopy_mode': canopy_mode,
        'na_fill': structure_na_fill,
        'outputs': [r.result for r in summary.records if getattr(r, 'result', None) is not None],
    }
    (out_base_root / 'structure_manifest.json').write_text(__import__('json').dumps(manifest, indent=2), encoding='utf-8')
    return str(out_base_root)
