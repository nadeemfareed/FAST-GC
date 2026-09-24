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
    "all", "canopy_cover", "n_points", "n_points_all", "point_density",
    "vegetation_point_density", "point_density_all",
    "z_min", "z_mean", "z_median", "z_max", "z_range", "z_sd", "z_variance", "z_cv",
    "z_skewness", "z_kurtosis", "z_excess_kurtosis", "z_p01", "z_p05", "z_p10", "z_p20", "z_p25",
    "z_p30", "z_p40", "z_p50", "z_p60", "z_p70", "z_p75", "z_p80", "z_p90",
    "z_p95", "z_p99", "z_iqr", "density_above_mean", "FHD", "VCI", "vertical_entropy",
    "vertical_richness", "vertical_evenness", "vertical_effective_layers", "vertical_dominance",
    "vertical_simpson", "vertical_occupancy_fraction", "vertical_gap_count", "vertical_gap_fraction",
    "max_vertical_gap_m", "sigma_z", "eigenvalue_1", "eigenvalue_2", "eigenvalue_3",
    "linearity", "planarity", "sphericity", "anisotropy", "surface_variation", "eigenentropy",
    "omnivariance", "robust_scale", "axis_x", "axis_y", "axis_z", "normal_x", "normal_y",
    "normal_z", "slope", "normal_inclination_deg",
    "intensity_mean", "intensity_sd", "intensity_p50", "intensity_p95",
    "first_return_fraction", "multi_return_fraction", "mean_number_of_returns",
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



def _vertical_profile_metrics(vals: np.ndarray, bin_size: float) -> Dict[str, float]:
    if vals.size == 0:
        return {}
    lo = max(0.0, float(np.min(vals)))
    hi = float(np.max(vals))
    if hi <= lo:
        return {"vertical_entropy": 0.0, "vertical_richness": 1.0, "vertical_evenness": 0.0,
                "vertical_effective_layers": 1.0, "vertical_dominance": 1.0, "vertical_simpson": 0.0,
                "vertical_occupancy_fraction": 1.0, "vertical_gap_count": 0.0,
                "vertical_gap_fraction": 0.0, "max_vertical_gap_m": 0.0}
    edges = np.arange(0.0, hi + bin_size + 1e-12, bin_size)
    hist, _ = np.histogram(vals, bins=edges)
    occ = hist > 0
    p = hist[occ].astype(float)
    p /= p.sum()
    H = -float(np.sum(p * np.log(p))) if p.size else np.nan
    richness = int(occ.sum())
    even = H / np.log(richness) if richness > 1 else 0.0
    simpson = 1.0 - float(np.sum(p*p)) if p.size else np.nan
    dominance = float(np.max(p)) if p.size else np.nan
    # Internal empty runs between first/last occupied layers only.
    idx = np.flatnonzero(occ)
    gaps=[]
    if idx.size > 1:
        run=0
        for flag in occ[idx[0]:idx[-1]+1]:
            if not flag: run += 1
            elif run: gaps.append(run); run=0
    span = int(idx[-1]-idx[0]+1) if idx.size else 0
    return {"vertical_entropy": H, "vertical_richness": float(richness), "vertical_evenness": float(even),
            "vertical_effective_layers": float(np.exp(H)) if np.isfinite(H) else np.nan,
            "vertical_dominance": dominance, "vertical_simpson": simpson,
            "vertical_occupancy_fraction": float(richness/span) if span else np.nan,
            "vertical_gap_count": float(len(gaps)),
            "vertical_gap_fraction": float(sum(gaps)/span) if span else 0.0,
            "max_vertical_gap_m": float(max(gaps)*bin_size) if gaps else 0.0}


def _geometry_metrics(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict[str, float]:
    if z.size < 3:
        return {}
    pts=np.column_stack((x,y,z)).astype(np.float64)
    center=np.median(pts,axis=0)
    d=np.linalg.norm(pts-center,axis=1)
    med=float(np.median(d)); mad=float(np.median(np.abs(d-med)))
    scale=max(1.4826*mad, 1e-9); c=1.345*scale
    w=np.ones_like(d); mask=d>c; w[mask]=c/d[mask]
    sw=float(w.sum())
    if sw <= 0: return {}
    center=(pts*w[:,None]).sum(axis=0)/sw
    q=pts-center
    cov=(q*w[:,None]).T@q/sw
    try: vals, vecs=np.linalg.eigh(cov)
    except np.linalg.LinAlgError: return {}
    order=np.argsort(vals)[::-1]; vals=np.maximum(vals[order],0.0); vecs=vecs[:,order]
    l1,l2,l3=map(float,vals); total=l1+l2+l3
    axis=vecs[:,0].copy(); normal=vecs[:,-1].copy()
    if axis[2] < 0: axis=-axis
    if normal[2] < 0: normal=-normal
    p=vals/total if total>0 else np.zeros(3)
    pe=p[p>0]
    normal_inclination_deg=float(
        np.degrees(np.arccos(np.clip(normal[2],-1,1)))
    )
    return {"eigenvalue_1":l1,"eigenvalue_2":l2,"eigenvalue_3":l3,
            "linearity":(l1-l2)/l1 if l1>0 else np.nan,"planarity":(l2-l3)/l1 if l1>0 else np.nan,
            "sphericity":l3/l1 if l1>0 else np.nan,"anisotropy":(l1-l3)/l1 if l1>0 else np.nan,
            "surface_variation":l3/total if total>0 else np.nan,
            "eigenentropy":-float(np.sum(pe*np.log(pe))) if pe.size else np.nan,
            "omnivariance":float(np.cbrt(l1*l2*l3)),"robust_scale":scale,
            "axis_x":float(axis[0]),"axis_y":float(axis[1]),"axis_z":float(axis[2]),
            "normal_x":float(normal[0]),"normal_y":float(normal[1]),"normal_z":float(normal[2]),
            "slope":normal_inclination_deg,
            "normal_inclination_deg":normal_inclination_deg}


def _sigma_z(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Residual vertical dispersion around a locally fitted XY plane.

    XY coordinates are centered before least-squares fitting.  This keeps
    the fit numerically stable for projected coordinate systems such as UTM
    while leaving the fitted residuals invariant to coordinate translation.
    """
    if z.size < 3:
        return np.nan

    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    zz = np.asarray(z, dtype=np.float64)

    xc = xx - np.mean(xx)
    yc = yy - np.mean(yy)

    A = np.column_stack((xc, yc, np.ones(zz.size)))

    try:
        coef, _, rank, _ = np.linalg.lstsq(A, zz, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan

    if rank < 3:
        return np.nan

    residual = zz - A @ coef

    return (
        float(np.std(residual, ddof=1))
        if residual.size > 1
        else 0.0
    )



# =========================================================
# Main computation
# =========================================================


def compute_structure_metrics(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, *, sensor_mode: str,
    res: float | None = None, min_h: float | None = None, bin_size: float | None = None,
    canopy_threshold: float | None = None, canopy_mode: str | None = None,
    na_fill: str | None = None, bounds: Tuple[float,float,float,float] | None = None,
    intensity: np.ndarray | None = None, return_number: np.ndarray | None = None,
    number_of_returns: np.ndarray | None = None,
) -> Dict[str, object]:
    """Compute resolution-controlled FAST_STRUCTURE metrics from normalized XYZ.

    All-point metrics (n_points_all/canopy_cover) use every finite point. Height,
    vertical-profile and geometry metrics use points >= min_h. Optional LAS
    attributes are emitted only when supplied. No plot/tile assumptions live here.
    """
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float); z=np.asarray(z,dtype=float)
    if not (x.size==y.size==z.size) or x.size==0: raise ValueError("x, y, z must be non-empty and have the same length")
    sm=(sensor_mode or '').upper().strip(); sdef=_STRUCTURE_DEFAULTS.get(sm)
    if sdef is None: raise ValueError(f"sensor_mode must be one of ALS|ULS|TLS (got {sensor_mode!r})")
    res=float(sdef.res if res is None else res); min_h=float(sdef.min_h if min_h is None else min_h)
    bin_size=float(sdef.bin_size if bin_size is None else bin_size)
    canopy_threshold=float(sdef.canopy_threshold if canopy_threshold is None else canopy_threshold)
    canopy_mode=str(sdef.canopy_mode if canopy_mode is None else canopy_mode).lower().strip()
    na_fill=str(sdef.na_fill if na_fill is None else na_fill).lower().strip()
    for n,v in [('structure_res',res),('structure_min_h',min_h),('structure_bin_size',bin_size),('canopy_threshold',canopy_threshold)]: _validate_positive(n,v)
    if canopy_mode not in {'all_points','all'}: raise ValueError(f"Unsupported canopy_mode {canopy_mode!r}; currently only 'all_points' is implemented")
    attrs={}
    for name,a in [('intensity',intensity),('return_number',return_number),('number_of_returns',number_of_returns)]:
        if a is not None:
            a=np.asarray(a);
            if a.size!=x.size: raise ValueError(f"{name} must have the same length as x, y, z")
            attrs[name]=a
    valid=np.isfinite(x)&np.isfinite(y)&np.isfinite(z)
    x=x[valid]; y=y[valid]; z=z[valid]; attrs={k:v[valid] for k,v in attrs.items()}
    if x.size==0: raise ValueError("No finite points available after filtering")
    if bounds is None: xmin,ymin,xmax,ymax=float(x.min()),float(y.min()),float(x.max()),float(y.max())
    else: xmin,ymin,xmax,ymax=map(float,bounds)
    if xmax<=xmin or ymax<=ymin: raise ValueError("Invalid bounds for structure metric rasterization")
    nx=max(1,int(np.ceil((xmax-xmin)/res))); ny=max(1,int(np.ceil((ymax-ymin)/res)))
    ix=np.floor((x-xmin)/res).astype(np.int64); iy=np.floor((ymax-y)/res).astype(np.int64)
    inside=(ix>=0)&(ix<nx)&(iy>=0)&(iy<ny)
    x=x[inside]; y=y[inside]; z=z[inside]; ix=ix[inside]; iy=iy[inside]; attrs={k:v[inside] for k,v in attrs.items()}
    if x.size==0: raise ValueError("No points fall inside the requested structure metric bounds")
    metric_names=list(STRUCTURE_PRODUCT_CHOICES[1:])

    optional_metric_requirements = {
        "intensity_mean": "intensity",
        "intensity_sd": "intensity",
        "intensity_p50": "intensity",
        "intensity_p95": "intensity",
        "first_return_fraction": "return_number",
        "multi_return_fraction": "return_number",
        "mean_number_of_returns": "number_of_returns",
    }
    metric_names = [
        k for k in metric_names
        if (
            k not in optional_metric_requirements
            or optional_metric_requirements[k] in attrs
        )
    ]

    metrics={k:np.full((ny,nx),np.nan,dtype=np.float32) for k in metric_names if k not in {'n_points','n_points_all'}}
    metrics['n_points']=np.zeros((ny,nx),dtype=np.int32); metrics['n_points_all']=np.zeros((ny,nx),dtype=np.int32)
    ids,order,starts=_cell_slices(ix,iy,nx,ny); ends=np.r_[starts[1:],order.size]
    xs=x[order]; ys=y[order]; zs=z[order]; sorted_attrs={k:v[order] for k,v in attrs.items()}
    qs=[1,5,10,20,25,30,40,50,60,70,75,80,90,95,99]
    for cid,a,b in zip(ids,starts,ends):
        row=int(cid//nx); col=int(cid%nx); za=zs[a:b]; xa=xs[a:b]; ya=ys[a:b]
        metrics['n_points_all'][row,col]=za.size
        metrics['canopy_cover'][row,col]=np.mean(za>=canopy_threshold)
        metrics['point_density_all'][row,col]=za.size/(res*res)
        veg=za>=min_h; zv=za[veg]; xv=xa[veg]; yv=ya[veg]; metrics['n_points'][row,col]=zv.size
        if zv.size==0: continue
        mean=float(zv.mean()); sd=float(zv.std(ddof=0)); centered=zv-mean
        base={'z_min':zv.min(),'z_mean':mean,'z_median':np.median(zv),'z_max':zv.max(),'z_range':np.ptp(zv),'z_sd':sd,
              'z_variance':sd*sd,'z_cv':sd/mean if abs(mean)>1e-12 else np.nan,
              'z_skewness':np.mean(centered**3)/(sd**3) if sd>0 else 0.0,
              'z_kurtosis':np.mean(centered**4)/(sd**4)-3.0 if sd>0 else 0.0,
              'z_excess_kurtosis':np.mean(centered**4)/(sd**4)-3.0 if sd>0 else 0.0,
              'point_density':zv.size/(res*res),
              'vegetation_point_density':zv.size/(res*res),
              'density_above_mean':100.0*np.mean(zv>mean),
              'sigma_z':_sigma_z(xv,yv,zv)}
        pct=np.percentile(zv,qs)
        for q,v in zip(qs,pct): base[f'z_p{q:02d}']=v
        base['z_iqr']=base['z_p75']-base['z_p25']
        vertical = _vertical_profile_metrics(zv, bin_size)
        base.update(vertical)
        # Backward-compatible aliases. FHD is Shannon vertical entropy and
        # VCI is Shannon evenness normalized by occupied vertical layers.
        base['FHD'] = vertical['vertical_entropy']
        base['VCI'] = vertical['vertical_evenness']
        base.update(_geometry_metrics(xv, yv, zv))
        if 'intensity' in sorted_attrs:
            iv=np.asarray(sorted_attrs['intensity'][a:b])[veg].astype(float)
            if iv.size: base.update(intensity_mean=iv.mean(),intensity_sd=iv.std(),intensity_p50=np.percentile(iv,50),intensity_p95=np.percentile(iv,95))
        if 'return_number' in sorted_attrs:
            rv=np.asarray(sorted_attrs['return_number'][a:b])[veg].astype(float)
            if rv.size: base.update(first_return_fraction=np.mean(rv==1),multi_return_fraction=np.mean(rv>1))
        if 'number_of_returns' in sorted_attrs:
            nv=np.asarray(sorted_attrs['number_of_returns'][a:b])[veg].astype(float)
            if nv.size: base['mean_number_of_returns']=nv.mean()
        for k,v in base.items():
            if k in metrics and np.isfinite(v): metrics[k][row,col]=np.float32(v)
    # Structural metrics represent observed point-cloud statistics.
    # Preserve NaN in unobserved cells by default. Spatial interpolation is
    # applied only when the caller explicitly requests structure_na_fill.
    if str(na_fill).lower().strip() not in {"none", "off", "false"}:
        for k in ("z_mean", "z_max", "z_sd", "canopy_cover", "FHD", "VCI"):
            metrics[k] = _fill_na(metrics[k], na_fill)
    return {'sensor_mode':sm,'res':res,'min_h':min_h,'bin_size':bin_size,'canopy_threshold':canopy_threshold,
            'canopy_mode':canopy_mode,'na_fill':na_fill,'bounds':(xmin,ymin,xmax,ymax),'transform':from_origin(xmin,ymax,res,res),
            'shape':(ny,nx),'metrics':metrics}


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

        # Floating-point structure metrics use NaN for missing cells.
        # Integer metrics such as n_points cannot represent NaN; zero is
        # already the meaningful value for cells containing no points, so
        # do not assign an integer nodata sentinel here.
        raster_nodata = nodata if arr_write.dtype.kind == "f" else None

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
            nodata=raster_nodata,
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
        dim_names = set(las.point_format.dimension_names)
        optional = {}
        if "intensity" in dim_names:
            optional["intensity"] = np.asarray(las.intensity)
        if "return_number" in dim_names:
            optional["return_number"] = np.asarray(las.return_number)
        if "number_of_returns" in dim_names:
            optional["number_of_returns"] = np.asarray(las.number_of_returns)
        result = compute_structure_metrics(
            np.asarray(las.x), np.asarray(las.y), np.asarray(las.z),
            sensor_mode=sensor_mode,
            **optional,
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
