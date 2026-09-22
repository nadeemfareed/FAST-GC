from __future__ import annotations

import json
import math
import copy
import os
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

import laspy
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, distance_transform_edt
from scipy.spatial import Delaunay, QhullError, cKDTree
from .monster import ProgressDashboard, log_info, progress_bar, run_stage
from .invert_vote import InvertVoteConfig, build_surface_invert_vote, classify_by_surface
from .sensors import sensor_defaults
from .tls_vote import TlsInvertDsmVoteConfig, build_tls_surface_invert_dsm_vote, classify_tls_by_surface, get_tls_surface_diagnostics
from .tls_final_ground_vote import refine_tls_final_ground_vote
from .tls_voxel_reduce import reduce_tls_terrain_domain, select_tls_active_evidence
from .tls_membrane_refine import TlsMembraneConfig, refine_tls_ground_membrane
from .tls_terrain_recover import TlsTerrainRecoveryConfig, recover_tls_microtopography
from .utils import as_f64
from .void_recover import recover_ground_in_voids
from .als_statistical_sheet_cleaner import clean_statistical_ground_sheet
from .als_facet_support_guard import clean_als_facet_support_consistency
from .als_ground_vertical_profile_guard import refine_als_ground_vertical_profile
from .als_detached_ground_guard import apply_detached_ground_guard
from .als_surface_impurity_guard import apply_surface_impurity_guard
from .als_multiscale_canopy_blob_guard import apply_multiscale_canopy_blob_guard
from .als_terrain_blob_guard import apply_terrain_blob_guard, apply_hard_airborne_ground_guard
from .als_final_surface_vote import apply_final_surface_vote
from .als_d2_reverse_context_vote import apply_d2_reverse_context_vote
from .vertical_support_final_guard import apply_vertical_support_final_guard
from .depression_sweeper import sweep_ground_depressions
from .als_postfinal_surface import recover_postfinal_surface
from .surface_consensus_guard import build_surface_consensus_workspace, recover_surface_false_negatives, two_way_surface_classification_swipe
from .prederive_seam import finalize_fastgc_seams_in_place
try:
    import rasterio
    from rasterio.transform import from_origin
except Exception:  # pragma: no cover
    rasterio = None
    from_origin = None

PRODUCT_ALL = "all"
PRODUCT_GC = "FAST_GC"
PRODUCT_DEM = "FAST_DEM"
PRODUCT_NORMALIZED = "FAST_NORMALIZED"
PRODUCT_DSM = "FAST_DSM"
PRODUCT_CHM = "FAST_CHM"
PRODUCT_TERRAIN = "FAST_TERRAIN"
PRODUCT_CHANGE = "FAST_CHANGE"
PRODUCT_ITD = "FAST_ITD"
PRODUCT_STRUCTURE = "FAST_STRUCTURE"

ALL_PRODUCTS = {
    PRODUCT_GC,
    PRODUCT_DEM,
    PRODUCT_NORMALIZED,
    PRODUCT_DSM,
    PRODUCT_CHM,
    PRODUCT_TERRAIN,
    PRODUCT_CHANGE,
    PRODUCT_ITD,
    PRODUCT_STRUCTURE,
}

RASTER_METHOD_CHOICES = {"min", "max", "mean", "nearest", "idw", "spikefree"}


class _StepDashboard:
    def __init__(self, total: int, desc: str, enabled: bool):
        self._dash = ProgressDashboard(desc, total, unit="step", enabled=enabled)
        self._enabled = enabled
        self._step = 0

    def update(self, n: int = 1):
        if not self._enabled:
            return
        self._step += int(n)
        self._dash.update(
            n,
            current_file="-",
            current_item=f"step {min(self._step, self._dash.total)}/{self._dash.total}",
        )

    def set_description(self, desc: str):
        if self._enabled:
            self._dash.stage_name = str(desc)
            self._dash.set_context(current_item=f"step {self._step}/{self._dash.total}")

    def close(self):
        self._dash.close()


def _stage_bar(total: int, desc: str, enabled: bool):
    if 'ProgressDashboard' in globals():
        class _StepDashboard:
            def __init__(self, total: int, desc: str, enabled: bool):
                self._dash = ProgressDashboard(desc, total, unit='step', enabled=enabled)
                self._enabled = bool(enabled)
                self._step = 0
                self.total = int(total)

            def update(self, n: int = 1):
                if not self._enabled:
                    return
                self._step += int(n)
                self._dash.update(int(n), current_file='-', current_item=f'step {min(self._step, self.total)}/{self.total}')

            def set_description(self, value: str, refresh: bool = True):
                if self._enabled:
                    self._dash.stage_name = str(value)

            def set_description_str(self, value: str, refresh: bool = True):
                self.set_description(value, refresh=refresh)

            def set_postfix_str(self, value: str, refresh: bool = True):
                if self._enabled and refresh:
                    self._dash.set_context(current_item=str(value))

            def refresh(self):
                if self._enabled:
                    self._dash.set_context(current_item=f'step {min(self._step, self.total)}/{self.total}')

            def close(self):
                self._dash.close()

        return _StepDashboard(total, desc, enabled)

    return progress_bar(total=total, desc=desc, unit='step', dynamic_ncols=True, leave=False, disable=not enabled)


def _sample_points_xy_from_las(src_fp: str, max_points: int = 250000) -> tuple[np.ndarray, np.ndarray]:
    las = laspy.read(src_fp)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    n = int(x.size)
    if n == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    if n > max_points:
        idx = np.random.default_rng(42).choice(n, size=max_points, replace=False)
        x = x[idx]
        y = y[idx]
    return x, y


def _grid_point_count_support_stats_xy(x: np.ndarray, y: np.ndarray, cell_m: float) -> dict[str, float]:
    if x.size == 0:
        return {
            "cell_m": float(cell_m),
            "occupied_cells": 0.0,
            "total_cells": 0.0,
            "occupancy_ratio": 0.0,
            "pointcount_median": float("nan"),
        }

    xmin = float(np.min(x))
    ymin = float(np.min(y))
    ix = np.floor((x - xmin) / float(cell_m)).astype(np.int64)
    iy = np.floor((y - ymin) / float(cell_m)).astype(np.int64)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1
    total_cells = float(nx * ny)
    key = iy * nx + ix
    _, counts = np.unique(key, return_counts=True)
    occupied = float(counts.size)
    return {
        "cell_m": float(cell_m),
        "occupied_cells": occupied,
        "total_cells": total_cells,
        "occupancy_ratio": occupied / total_cells if total_cells > 0 else 0.0,
        "pointcount_median": float(np.median(counts.astype(np.float64))) if counts.size else float("nan"),
    }


def _compute_support_stats_for_path(src_fp: str) -> dict[str, float]:
    x, y = _sample_points_xy_from_las(src_fp)
    n = int(x.size)
    if n == 0:
        return {
            "density_pts_m2": float("nan"),
            "grid_2m_pointcount_median": float("nan"),
            "grid_2m_occupancy_ratio": float("nan"),
            "grid_4m_pointcount_median": float("nan"),
        }

    xmin = float(np.min(x)); ymin = float(np.min(y)); xmax = float(np.max(x)); ymax = float(np.max(y))
    area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
    g2 = _grid_point_count_support_stats_xy(x, y, 2.0)
    g4 = _grid_point_count_support_stats_xy(x, y, 4.0)
    return {
        "density_pts_m2": float(n / area) if area > 0 else float("nan"),
        "grid_2m_pointcount_median": float(g2["pointcount_median"]),
        "grid_2m_occupancy_ratio": float(g2["occupancy_ratio"]),
        "grid_4m_pointcount_median": float(g4["pointcount_median"]),
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


def _adaptive_support_context_for_input(src_fp: str) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    manifest_fp = _find_tile_manifest_for_path(src_fp)
    if manifest_fp is not None:
        manifest = _load_manifest_json(manifest_fp)
        if manifest is not None:
            tile = _match_tile_record(src_fp, manifest)
            dataset_stats = manifest.get("adaptive_support_summary") or {}
            if tile is not None:
                tile_stats = {
                    "density_pts_m2": tile.get("density_pts_m2"),
                    "grid_2m_pointcount_median": tile.get("grid_2m_pointcount_median"),
                    "grid_2m_occupancy_ratio": tile.get("grid_2m_occupancy_ratio"),
                    "grid_4m_pointcount_median": tile.get("grid_4m_pointcount_median"),
                }
                return tile_stats, dataset_stats
            if dataset_stats:
                return None, dataset_stats

    support = _compute_support_stats_for_path(src_fp)
    dataset_stats = {
        "density_pts_m2_median": support.get("density_pts_m2"),
        "grid_2m_pointcount_median_median": support.get("grid_2m_pointcount_median"),
        "grid_2m_occupancy_ratio_median": support.get("grid_2m_occupancy_ratio"),
        "grid_4m_pointcount_median_median": support.get("grid_4m_pointcount_median"),
    }
    return support, dataset_stats


def _nearest_fill_grid(a: np.ndarray) -> np.ndarray:
    """Nearest-value fill used only to estimate local terrain orientation."""
    a = np.asarray(a, dtype=np.float64)
    finite = np.isfinite(a)
    if not np.any(finite):
        return a.copy()
    if np.all(finite):
        return a.copy()
    # distance_transform_edt returns indices of the closest zero in the mask;
    # use ~finite so zeros correspond to valid cells.
    _, inds = distance_transform_edt(~finite, return_indices=True)
    return a[tuple(inds)]


def _candidate_mask_one_origin(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    cell: float,
    dz: float,
    x0: float,
    y0: float,
) -> np.ndarray:
    """
    Terrain-oriented lower-layer candidate mask for one XY grid origin.

    The published rule uses z <= zmin(cell) + dz, i.e. a horizontal slab.
    Here zmin is retained as the lower-envelope anchor, but the slab is
    rotated using the local gradient of the lower-envelope grid.  For a
    planar terrain patch, zmin is interpreted as the minimum of a local
    plane over the cell, which gives the plane elevation at cell centre as

        zc = zmin + 0.5*cell*(|gx| + |gy|).

    This removes the orientation-dependent clipping that creates rectangular
    omissions on slopes while preserving the original flat-terrain rule.
    """
    ix = np.floor((x - x0) / cell).astype(np.int64)
    iy = np.floor((y - y0) / cell).astype(np.int64)
    valid = (ix >= 0) & (iy >= 0)
    if not np.any(valid):
        return np.zeros(x.shape, dtype=bool)

    nx = int(ix[valid].max()) + 1
    ny = int(iy[valid].max()) + 1
    key = iy[valid] * nx + ix[valid]

    zmin_flat = np.full(nx * ny, np.inf, dtype=np.float64)
    np.minimum.at(zmin_flat, key, z[valid])
    zmin = zmin_flat.reshape(ny, nx)
    zmin[~np.isfinite(zmin)] = np.nan

    finite = np.isfinite(zmin)
    # 3x3 support count without an additional scipy filter dependency.
    pad = np.pad(finite.astype(np.uint8), 1, mode="constant")
    support = np.zeros_like(zmin, dtype=np.uint8)
    for dj in range(3):
        for di in range(3):
            support += pad[dj:dj + ny, di:di + nx]

    zfill = _nearest_fill_grid(zmin)
    if ny > 1 and nx > 1:
        gy, gx = np.gradient(zfill, float(cell), float(cell))
    elif ny > 1:
        gy = np.gradient(zfill, float(cell), axis=0)
        gx = np.zeros_like(zfill)
    elif nx > 1:
        gx = np.gradient(zfill, float(cell), axis=1)
        gy = np.zeros_like(zfill)
    else:
        gx = np.zeros_like(zfill)
        gy = np.zeros_like(zfill)

    out = np.zeros(x.shape, dtype=bool)
    ids = np.flatnonzero(valid)
    i = ix[valid]
    j = iy[valid]
    z0 = zmin[j, i]

    # Published horizontal rule is always available as a conservative fallback.
    base_keep = z[valid] <= (z0 + float(dz))

    well_supported = support[j, i] >= 4
    if np.any(well_supported):
        xc = x0 + (i.astype(np.float64) + 0.5) * float(cell)
        yc = y0 + (j.astype(np.float64) + 0.5) * float(cell)
        gxi = gx[j, i]
        gyi = gy[j, i]

        # Reconstruct the local planar sheet from its within-cell minimum.
        zc = z0 + 0.5 * float(cell) * (np.abs(gxi) + np.abs(gyi))
        zpred = zc + gxi * (x[valid] - xc) + gyi * (y[valid] - yc)

        # The residual is converted approximately to normal distance so the
        # physical candidate thickness does not inflate with terrain slope.
        norm = np.sqrt(1.0 + gxi * gxi + gyi * gyi)
        dperp = (z[valid] - zpred) / norm
        oriented_keep = dperp <= float(dz)
        base_keep = np.where(well_supported, oriented_keep, base_keep)

    out[ids] = base_keep
    return out


def _candidate_layer_by_cellmin(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cell: float,
    dz: float,
) -> np.ndarray:
    """High-recall, slope-aware ALS/ULS lower-layer candidate mask.

    The older horizontal ``z <= zmin(cell) + dz`` slab is orientation dependent:
    on a steep cell, legitimate terrain at the uphill side can sit more than
    ``dz`` above the cell minimum and is discarded before FAST-GC ever sees it.

    Two half-cell-offset, locally oriented lower-envelope masks are unioned.
    This preserves a thin terrain sheet through slopes, rounded crests, and
    concave/convex breaks while retaining the original cell-minimum anchor.
    """
    if x.size == 0:
        return np.zeros(0, dtype=bool)

    cell = float(cell)
    dz = float(dz)
    x0 = float(np.min(x))
    y0 = float(np.min(y))

    m0 = _candidate_mask_one_origin(
        x, y, z, cell=cell, dz=dz, x0=x0, y0=y0
    )
    m1 = _candidate_mask_one_origin(
        x, y, z, cell=cell, dz=dz,
        x0=x0 - 0.5 * cell, y0=y0 - 0.5 * cell,
    )
    return m0 | m1

def _safe_parse_crs(las, *, fallback_fp: str | None = None):
    try:
        crs = las.header.parse_crs()
    except Exception:
        crs = None

    if crs is None and fallback_fp:
        try:
            manifest_fp = _find_tile_manifest_for_path(fallback_fp)
            manifest = _load_manifest_json(manifest_fp) if manifest_fp is not None else None
            tile = _match_tile_record(fallback_fp, manifest) if manifest is not None else None

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

    if rasterio is None:
        return crs

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


def _read_las(fp: str):
    las = laspy.read(fp)
    x = as_f64(las.x)
    y = as_f64(las.y)
    z = as_f64(las.z)

    cls = np.zeros_like(x, dtype=np.uint8)
    try:
        cls = np.asarray(las.classification, dtype=np.uint8)
    except Exception:
        pass

    crs = _safe_parse_crs(las, fallback_fp=fp)
    return las, x, y, z, cls, crs


def _write_full_cloud_with_classification(template: laspy.LasData, fp_out: str, classification: np.ndarray):
    # Never share the template header with an output LAS.  Diagnostic
    # Extra Bytes dimensions must not mutate the production header.
    out = laspy.LasData(copy.deepcopy(template.header))
    out.points = template.points.copy()
    out.classification = classification.astype(np.uint8, copy=False)
    os.makedirs(os.path.dirname(fp_out), exist_ok=True)
    out.write(fp_out)


def _ensure_tls_diag_extra_dim(
    las: laspy.LasData,
    name: str,
    dtype,
    description: str,
) -> None:
    """Ensure a TLS diagnostic Extra Bytes dimension exists."""
    names = {dim.name for dim in las.point_format.extra_dimensions}

    if name not in names:
        las.add_extra_dim(
            laspy.ExtraBytesParams(
                name=name,
                type=np.dtype(dtype),
                description=description[:32],
            )
        )


def _write_tls_diagnostic_cloud(
    template: laspy.LasData,
    fp_out: str,
    classification: np.ndarray,
    *,
    work_indices: np.ndarray,
    xw: np.ndarray,
    yw: np.ndarray,
    zw: np.ndarray,
    surf_z: np.ndarray,
    sx0: float,
    sy0: float,
    cell: float,
    diagnostics: dict[str, np.ndarray],
) -> str:
    """Write a separate TLS diagnostic LAS.

    Classification is copied from the final FAST-GC result.
    Diagnostic values do not alter classification.
    """
    # Diagnostics add Extra Bytes dimensions.  Work on an independent
    # header so those dimensions cannot leak back into the source LAS
    # or the subsequent production output.
    out = laspy.LasData(copy.deepcopy(template.header))
    out.points = template.points.copy()
    out.classification = np.asarray(
        classification,
        dtype=np.uint8,
    )

    work_indices = np.asarray(
        work_indices,
        dtype=np.int64,
    )

    xw = np.asarray(xw, dtype=np.float64)
    yw = np.asarray(yw, dtype=np.float64)
    zw = np.asarray(zw, dtype=np.float64)

    if not (
        work_indices.size
        == xw.size
        == yw.size
        == zw.size
    ):
        raise ValueError(
            "TLS diagnostic point mapping mismatch: "
            f"indices={work_indices.size}, "
            f"x={xw.size}, "
            f"y={yw.size}, "
            f"z={zw.size}"
        )

    n = len(out.points)
    ny, nx = surf_z.shape

    # --------------------------------------------------------
    # Grid-cell lookup for cell diagnostics.
    # --------------------------------------------------------

    gx = np.floor(
        (xw - float(sx0)) / float(cell)
    ).astype(np.int64)

    gy = np.floor(
        (yw - float(sy0)) / float(cell)
    ).astype(np.int64)

    valid = (
        (gx >= 0)
        & (gx < nx)
        & (gy >= 0)
        & (gy < ny)
    )

    def sample_grid(name: str, default=np.nan):
        result = np.full(
            xw.size,
            default,
            dtype=np.float64,
        )

        arr = diagnostics.get(name)

        if arr is None:
            return result

        arr = np.asarray(arr)

        if arr.shape != surf_z.shape:
            return result

        result[valid] = arr[
            gy[valid],
            gx[valid],
        ]

        return result

    # --------------------------------------------------------
    # Surface residual.
    #
    # Use bilinear interpolation where all four surrounding
    # surface cells are finite.
    # --------------------------------------------------------

    fx = (xw - float(sx0)) / float(cell)
    fy = (yw - float(sy0)) / float(cell)

    ix = np.floor(fx).astype(np.int64)
    iy = np.floor(fy).astype(np.int64)

    tx = fx - ix
    ty = fy - iy

    surf_point = np.full(
        xw.size,
        np.nan,
        dtype=np.float64,
    )

    bvalid = (
        (ix >= 0)
        & (iy >= 0)
        & (ix + 1 < nx)
        & (iy + 1 < ny)
    )

    ids = np.flatnonzero(bvalid)

    if ids.size:
        bx = ix[ids]
        by = iy[ids]

        z00 = surf_z[by, bx]
        z10 = surf_z[by, bx + 1]
        z01 = surf_z[by + 1, bx]
        z11 = surf_z[by + 1, bx + 1]

        finite = (
            np.isfinite(z00)
            & np.isfinite(z10)
            & np.isfinite(z01)
            & np.isfinite(z11)
        )

        good = ids[finite]

        if good.size:
            gx0 = ix[good]
            gy0 = iy[good]

            txx = tx[good]
            tyy = ty[good]

            zz00 = surf_z[gy0, gx0]
            zz10 = surf_z[gy0, gx0 + 1]
            zz01 = surf_z[gy0 + 1, gx0]
            zz11 = surf_z[gy0 + 1, gx0 + 1]

            surf_point[good] = (
                (1.0 - txx) * (1.0 - tyy) * zz00
                + txx * (1.0 - tyy) * zz10
                + (1.0 - txx) * tyy * zz01
                + txx * tyy * zz11
            )

    surf_dz = zw - surf_point

    # --------------------------------------------------------
    # Extract grid diagnostics.
    # --------------------------------------------------------

    seed = sample_grid(
        "seed_mask",
        default=0.0,
    )

    prop = sample_grid(
        "propagation_distance"
    )

    support = sample_grid(
        "support_count"
    )

    candconf = sample_grid(
        "candidate_confidence"
    )

    colspan = sample_grid(
        "column_span"
    )

    surfconf = sample_grid(
        "confidence"
    )

    # --------------------------------------------------------
    # Allocate full-cloud arrays.
    # --------------------------------------------------------

    tls_seed = np.zeros(
        n,
        dtype=np.uint8,
    )

    tls_prop = np.full(
        n,
        np.nan,
        dtype=np.float32,
    )

    tls_supp = np.full(
        n,
        np.nan,
        dtype=np.float32,
    )

    tls_cconf = np.full(
        n,
        np.nan,
        dtype=np.float32,
    )

    tls_cspan = np.full(
        n,
        np.nan,
        dtype=np.float32,
    )

    tls_sconf = np.full(
        n,
        np.nan,
        dtype=np.float32,
    )

    tls_surfdz = np.full(
        n,
        np.nan,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Map TLS working-set values to original point order.
    # --------------------------------------------------------

    tls_seed[work_indices] = (
        np.isfinite(seed)
        & (seed > 0.5)
    ).astype(np.uint8)

    tls_prop[work_indices] = prop.astype(
        np.float32
    )

    tls_supp[work_indices] = support.astype(
        np.float32
    )

    tls_cconf[work_indices] = candconf.astype(
        np.float32
    )

    tls_cspan[work_indices] = colspan.astype(
        np.float32
    )

    tls_sconf[work_indices] = surfconf.astype(
        np.float32
    )

    tls_surfdz[work_indices] = surf_dz.astype(
        np.float32
    )

    fields = {
        "tls_seed": (
            tls_seed,
            np.uint8,
            "TLS initial terrain seed",
        ),
        "tls_prop": (
            tls_prop,
            np.float32,
            "TLS propagation distance",
        ),
        "tls_supp": (
            tls_supp,
            np.float32,
            "TLS candidate support count",
        ),
        "tls_cconf": (
            tls_cconf,
            np.float32,
            "TLS candidate confidence",
        ),
        "tls_cspan": (
            tls_cspan,
            np.float32,
            "TLS robust column span",
        ),
        "tls_sconf": (
            tls_sconf,
            np.float32,
            "TLS surface confidence",
        ),
        "tls_surfdz": (
            tls_surfdz,
            np.float32,
            "Z minus TLS terrain surface",
        ),
    }

    for name, (
        values,
        dtype,
        description,
    ) in fields.items():

        _ensure_tls_diag_extra_dim(
            out,
            name,
            dtype,
            description,
        )

        setattr(
            out,
            name,
            values,
        )

    out_dir = os.path.dirname(fp_out)

    if out_dir:
        os.makedirs(
            out_dir,
            exist_ok=True,
        )

    out.write(fp_out)

    return fp_out


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


def _get_output_root(in_path: str, out_dir: str | None, sensor_mode: str) -> str:
    p = Path(in_path)

    if p.is_file():
        root = Path(out_dir) if out_dir else p.parent
        root.mkdir(parents=True, exist_ok=True)
        return str(root)

    base_root = Path(out_dir) if out_dir else p.parent
    out_root = base_root / f"Processed_{sensor_mode.upper()}"
    out_root.mkdir(parents=True, exist_ok=True)
    return str(out_root)


def _make_product_dirs(root: str) -> dict[str, str]:
    dirs = {
        PRODUCT_GC: os.path.join(root, PRODUCT_GC),
        PRODUCT_DEM: os.path.join(root, PRODUCT_DEM),
        PRODUCT_NORMALIZED: os.path.join(root, PRODUCT_NORMALIZED),
        PRODUCT_DSM: os.path.join(root, PRODUCT_DSM),
        PRODUCT_CHM: os.path.join(root, PRODUCT_CHM),
        PRODUCT_TERRAIN: os.path.join(root, PRODUCT_TERRAIN),
        PRODUCT_CHANGE: os.path.join(root, PRODUCT_CHANGE),
        PRODUCT_ITD: os.path.join(root, PRODUCT_ITD),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def _grid_support(points_x: np.ndarray, points_y: np.ndarray, points_z: np.ndarray, cell: float, mode: str):
    x = as_f64(points_x)
    y = as_f64(points_y)
    z = as_f64(points_z)

    if mode not in {"min", "max", "mean"}:
        raise ValueError("mode must be 'min', 'max', or 'mean'")

    x0 = float(np.min(x))
    y0 = float(np.min(y))
    ix = np.floor((x - x0) / cell).astype(np.int64)
    iy = np.floor((y - y0) / cell).astype(np.int64)

    nx = int(ix.max()) + 1
    key = iy * nx + ix

    order = np.lexsort((z, key))
    key_s = key[order]
    z_s = z[order]
    ix_s = ix[order]
    iy_s = iy[order]

    starts = np.r_[0, 1 + np.flatnonzero(key_s[1:] != key_s[:-1])]
    ends = np.r_[starts[1:], key_s.size]

    xs, ys, zs = [], [], []
    for a, b in zip(starts, ends):
        gx = int(ix_s[a])
        gy = int(iy_s[a])
        vals = z_s[a:b]
        if mode == "min":
            zv = float(np.min(vals))
        elif mode == "max":
            zv = float(np.max(vals))
        else:
            zv = float(np.mean(vals))
        xs.append(x0 + (gx + 0.5) * cell)
        ys.append(y0 + (gy + 0.5) * cell)
        zs.append(zv)

    return np.asarray(xs), np.asarray(ys), np.asarray(zs)


def _idw_grid(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    bounds: tuple[float, float, float, float],
    res: float,
    k: int = 8,
    power: float = 2.0,
):
    xs, ys, xx, yy = _grid_xy(bounds, res)
    gx = xx.ravel()
    gy = yy.ravel()
    pts = np.column_stack([px, py]).astype(np.float64, copy=False)
    q = np.column_stack([gx, gy]).astype(np.float64, copy=False)

    if pts.shape[0] == 0:
        raise ValueError("Need at least 1 support point for IDW interpolation.")

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(pts)
        kk = max(1, min(int(k), pts.shape[0]))
        d, idx = tree.query(q, k=kk)
        if kk == 1:
            d = d[:, None]
            idx = idx[:, None]
        vals = pz[idx]
        w = 1.0 / np.maximum(d, 1e-6) ** power
        z = np.sum(w * vals, axis=1) / np.sum(w, axis=1)
    except Exception:
        nn = NearestNDInterpolator(pts, pz)
        z = nn(gx, gy)

    return z.reshape(xx.shape).astype(np.float32), xs, ys


def _grid_xy(bounds: tuple[float, float, float, float], res: float):
    xmin, ymin, xmax, ymax = bounds
    xs = np.arange(xmin + 0.5 * res, xmax, res, dtype=np.float64)
    ys = np.arange(ymax - 0.5 * res, ymin, -res, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    return xs, ys, xx, yy


def _support_mask_from_points(
    px: np.ndarray,
    py: np.ndarray,
    bounds: tuple[float, float, float, float],
    res: float,
    *,
    grow_cells: int = 1,
) -> np.ndarray:
    xs, ys, xx, _yy = _grid_xy(bounds, res)
    ny, nx = xx.shape
    mask = np.zeros((ny, nx), dtype=bool)

    if px.size == 0:
        return mask

    xmin, ymin, xmax, ymax = bounds
    ix = np.floor((px - xmin) / res).astype(np.int32)
    iy = np.floor((ymax - py) / res).astype(np.int32)

    valid = np.isfinite(px) & np.isfinite(py)
    ix = ix[valid]
    iy = iy[valid]

    if ix.size == 0:
        return mask

    np.clip(ix, 0, nx - 1, out=ix)
    np.clip(iy, 0, ny - 1, out=iy)
    mask[iy, ix] = True

    grow = max(0, int(grow_cells))
    if grow > 0 and np.any(mask):
        mask = binary_dilation(mask, iterations=grow)

    return mask


def _ground_footprint_mask_from_points(
    px: np.ndarray,
    py: np.ndarray,
    bounds: tuple[float, float, float, float],
    res: float,
    *,
    grow_cells: int = 2,
) -> np.ndarray:
    """
    DEM footprint mask from ground points.

    Important difference from DSM support:
    - this is only used to clip DEM outside the true ground-supported area
    - it should NOT create internal holes inside the ground-supported region
    """
    xs, ys, xx, _yy = _grid_xy(bounds, res)
    ny, nx = xx.shape
    mask = np.zeros((ny, nx), dtype=bool)

    if px.size == 0:
        return mask

    xmin, ymin, xmax, ymax = bounds
    ix = np.floor((px - xmin) / res).astype(np.int32)
    iy = np.floor((ymax - py) / res).astype(np.int32)

    valid = np.isfinite(px) & np.isfinite(py)
    ix = ix[valid]
    iy = iy[valid]

    if ix.size == 0:
        return mask

    np.clip(ix, 0, nx - 1, out=ix)
    np.clip(iy, 0, ny - 1, out=iy)
    mask[iy, ix] = True

    grow = max(0, int(grow_cells))
    if grow > 0 and np.any(mask):
        mask = binary_dilation(mask, iterations=grow)

    return mask


def _apply_support_mask(grid: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    if grid.shape != support_mask.shape:
        raise ValueError(
            f"Raster/support mask shape mismatch: grid={grid.shape}, mask={support_mask.shape}"
        )
    out = np.array(grid, copy=True, dtype=np.float32)
    out[~support_mask] = np.nan
    return out


def _estimate_dem_edge_limit(
    px: np.ndarray,
    py: np.ndarray,
    res: float,
    *,
    k: int = 6,
) -> float:
    """
    Estimate a conservative maximum TIN edge length for valid DEM support.
    This is used only to define the DEM footprint mask, not the DEM surface itself.
    """
    n = int(px.size)
    if n < 3:
        return float(max(4.0 * res, 1.0))

    pts = np.column_stack([px, py]).astype(np.float64, copy=False)
    kk = max(2, min(int(k) + 1, n))
    try:
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=kk)
        if d.ndim == 1:
            d = d[:, None]
        d = d[:, 1:]
        d = d[np.isfinite(d) & (d > 0)]
        if d.size == 0:
            return float(max(4.0 * res, 1.0))
        edge = float(np.percentile(d, 90.0) * 3.0)
        return float(max(edge, 4.0 * res, 1.0))
    except Exception:
        return float(max(4.0 * res, 1.0))


def _build_valid_dem_mask_from_ground_points(
    px: np.ndarray,
    py: np.ndarray,
    bounds: tuple[float, float, float, float],
    res: float,
    *,
    edge_limit: float | None = None,
) -> np.ndarray:
    """
    Build a DEM-valid footprint from ground-point geometry.

    Strategy:
    - accept grid cells covered by a constrained TIN of ground points
    - reject triangles spanning unrealistically long gaps
    - close/fill small holes so DEM stays continuous inside valid support
    - fall back to the simpler ground footprint mask for very sparse cases
    """
    base_mask = _ground_footprint_mask_from_points(px, py, bounds, res, grow_cells=1)
    if px.size < 3:
        return base_mask

    if edge_limit is None:
        edge_limit = _estimate_dem_edge_limit(px, py, res)

    dummy_z = np.zeros_like(px, dtype=np.float64)
    tin_mask_grid, _xmin, _ymax = _rasterize_constrained_tin(
        px, py, dummy_z, bounds, res, max_edge=edge_limit
    )
    tin_mask = np.isfinite(tin_mask_grid)

    if not np.any(tin_mask):
        return base_mask

    out = np.asarray(tin_mask, dtype=bool)
    out = binary_closing(out, iterations=1)
    out = binary_fill_holes(out)
    out |= base_mask
    return out.astype(bool, copy=False)


def _fill_dem_grid_by_nearest_ground(
    grid: np.ndarray,
    valid_mask: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    bounds: tuple[float, float, float, float],
    res: float,
    *,
    k: int = 8,
    power: float = 2.0,
) -> np.ndarray:
    """
    Fill unresolved DEM cells inside the valid DEM footprint using k-nearest
    ground neighbors. Cells outside valid_mask remain nodata.
    """
    out = np.array(grid, copy=True, dtype=np.float32)
    need = valid_mask & (~np.isfinite(out))
    if not np.any(need):
        out[~valid_mask] = np.nan
        return out

    pts = np.column_stack([px, py]).astype(np.float64, copy=False)
    if pts.shape[0] == 0:
        out[~valid_mask] = np.nan
        return out

    xs, ys, xx, yy = _grid_xy(bounds, res)
    qx = xx[need].astype(np.float64, copy=False)
    qy = yy[need].astype(np.float64, copy=False)
    q = np.column_stack([qx, qy])

    try:
        tree = cKDTree(pts)
        kk = max(1, min(int(k), pts.shape[0]))
        d, idx = tree.query(q, k=kk)
        if kk == 1:
            d = d[:, None]
            idx = idx[:, None]
        vals = pz[idx]
        w = 1.0 / np.maximum(d, 1e-6) ** float(power)
        zfill = np.sum(w * vals, axis=1) / np.sum(w, axis=1)
    except Exception:
        nn = NearestNDInterpolator(pts, pz)
        zfill = nn(qx, qy)

    out[need] = np.asarray(zfill, dtype=np.float32)
    out[~valid_mask] = np.nan
    return out


def _interpolate_tin_grid(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    bounds: tuple[float, float, float, float],
    res: float,
):
    xs, ys, xx, yy = _grid_xy(bounds, res)
    pts = np.column_stack([px, py])

    if pts.shape[0] < 1:
        raise ValueError("Need at least 1 support point for interpolation.")

    if pts.shape[0] < 3:
        nn = NearestNDInterpolator(pts, pz)
        grid = nn(xx, yy)
        return grid.astype(np.float32), xs, ys

    try:
        lin = LinearNDInterpolator(pts, pz, fill_value=np.nan)
        grid = lin(xx, yy)
    except QhullError:
        grid = np.full(xx.shape, np.nan, dtype=np.float64)

    if np.any(~np.isfinite(grid)):
        nn = NearestNDInterpolator(pts, pz)
        fill = nn(xx, yy)
        grid = np.where(np.isfinite(grid), grid, fill)

    return grid.astype(np.float32), xs, ys


def _cell_index_arrays(
    x: np.ndarray,
    y: np.ndarray,
    bounds: tuple[float, float, float, float],
    res: float,
):
    xmin, ymin, xmax, ymax = bounds
    xs = np.arange(xmin + 0.5 * res, xmax, res, dtype=np.float64)
    ys = np.arange(ymax - 0.5 * res, ymin, -res, dtype=np.float64)
    nx = xs.size
    ny = ys.size

    ix = np.floor((x - xmin) / res).astype(np.int32)
    iy = np.floor((ymax - y) / res).astype(np.int32)
    np.clip(ix, 0, nx - 1, out=ix)
    np.clip(iy, 0, ny - 1, out=iy)
    flat = iy.astype(np.int64) * int(nx) + ix.astype(np.int64)
    return ix, iy, flat, nx, ny, xmin, ymax


def _rasterize_stat(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
    res: float,
    mode: str = "max",
    percentile: float = 99.0,
):
    """
    Rasterize point elevations to a regular grid using a per-cell statistic.

    Supported modes:
      - max
      - min
      - mean
      - percentile
    """
    _ix, _iy, flat, nx, ny, xmin, ymax = _cell_index_arrays(x, y, bounds, res)
    grid = np.full((ny, nx), np.nan, dtype=np.float32)

    if z.size == 0:
        return grid, xmin, ymax

    mode = str(mode).strip().lower()

    if mode == "max":
        out = np.full(nx * ny, -np.inf, dtype=np.float64)
        np.maximum.at(out, flat, z)
        valid = np.isfinite(out)
        grid.flat[valid] = out[valid].astype(np.float32)
        return grid, xmin, ymax

    if mode == "min":
        out = np.full(nx * ny, np.inf, dtype=np.float64)
        np.minimum.at(out, flat, z)
        valid = np.isfinite(out)
        grid.flat[valid] = out[valid].astype(np.float32)
        return grid, xmin, ymax

    if mode == "mean":
        sums = np.zeros(nx * ny, dtype=np.float64)
        counts = np.zeros(nx * ny, dtype=np.int64)
        np.add.at(sums, flat, z)
        np.add.at(counts, flat, 1)
        valid = counts > 0
        out = np.full(nx * ny, np.nan, dtype=np.float64)
        out[valid] = sums[valid] / counts[valid]
        grid.flat[valid] = out[valid].astype(np.float32)
        return grid, xmin, ymax

    if mode == "percentile":
        order = np.argsort(flat, kind="mergesort")
        flat_s = flat[order]
        z_s = z[order]
        starts = np.r_[0, 1 + np.flatnonzero(flat_s[1:] != flat_s[:-1])]
        ends = np.r_[starts[1:], flat_s.size]
        unique = flat_s[starts]

        vals = np.empty(unique.size, dtype=np.float32)
        pct = float(np.clip(percentile, 0.0, 100.0))
        for j, (a, b) in enumerate(zip(starts, ends)):
            vals[j] = np.float32(np.percentile(z_s[a:b], pct))

        grid.flat[unique] = vals
        return grid, xmin, ymax

    raise ValueError(f"Unsupported rasterize stat mode: {mode}")


def _rasterize_nearest_in_cell(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
    res: float,
):
    """ArcGIS-like BINNING NEAREST cell assignment."""
    ix, iy, flat, nx, ny, xmin, ymax = _cell_index_arrays(x, y, bounds, res)
    grid = np.full((ny, nx), np.nan, dtype=np.float32)

    if z.size == 0:
        return grid, xmin, ymax

    cx = xmin + (ix.astype(np.float64) + 0.5) * res
    cy = ymax - (iy.astype(np.float64) + 0.5) * res
    d2 = (x - cx) ** 2 + (y - cy) ** 2

    order = np.argsort(flat, kind="mergesort")
    flat_s = flat[order]
    d2_s = d2[order]
    z_s = z[order]

    starts = np.r_[0, 1 + np.flatnonzero(flat_s[1:] != flat_s[:-1])]
    ends = np.r_[starts[1:], flat_s.size]

    for a, b in zip(starts, ends):
        loc = a + int(np.argmin(d2_s[a:b]))
        grid.flat[int(flat_s[a])] = np.float32(z_s[loc])

    return grid, xmin, ymax


def _fill_nan_cells_from_tin(
    grid: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    bounds: tuple[float, float, float, float],
    res: float,
) -> np.ndarray:
    """Fill NaN cells with TIN interpolation across the full tile extent."""
    out = np.array(grid, copy=True, dtype=np.float32)
    need = ~np.isfinite(out)
    if not np.any(need):
        return out

    tin_grid, _xs, _ys = _interpolate_tin_grid(px, py, pz, bounds=bounds, res=res)
    out[need] = tin_grid[need]
    return out


def _cellmax_support_points(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
    res: float,
):
    _ix, _iy, flat, nx, ny, xmin, ymax = _cell_index_arrays(x, y, bounds, res)

    if z.size == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )

    zmax = np.full(nx * ny, -np.inf, dtype=np.float64)
    np.maximum.at(zmax, flat, z)
    valid = np.isfinite(zmax)

    idx = np.flatnonzero(valid)
    iy_cells = idx // nx
    ix_cells = idx % nx

    xs = xmin + (ix_cells.astype(np.float64) + 0.5) * res
    ys = ymax - (iy_cells.astype(np.float64) + 0.5) * res
    zs = zmax[idx]
    return xs, ys, zs


def _triangle_barycentric_mask(
    px: np.ndarray,
    py: np.ndarray,
    tri_xy: np.ndarray,
):
    x1, y1 = tri_xy[0]
    x2, y2 = tri_xy[1]
    x3, y3 = tri_xy[2]

    det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(det) < 1e-12:
        return None, None, None, None

    l1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / det
    l2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / det
    l3 = 1.0 - l1 - l2
    inside = (l1 >= -1e-8) & (l2 >= -1e-8) & (l3 >= -1e-8)
    return inside, l1, l2, l3


def _rasterize_constrained_tin(
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    bounds: tuple[float, float, float, float],
    res: float,
    *,
    max_edge: float | None,
):
    xs, ys, _xx, _yy = _grid_xy(bounds, res)
    xmin, ymin, xmax, ymax = bounds
    nx = xs.size
    ny = ys.size
    grid = np.full((ny, nx), np.nan, dtype=np.float32)

    if sx.size < 1:
        return grid, float(xs[0] - 0.5 * res), float(ys[0] + 0.5 * res)
    if sx.size < 3:
        return _rasterize_stat(sx, sy, sz, bounds, res, mode="max")

    pts = np.column_stack([sx, sy])

    try:
        tri = Delaunay(pts)
    except QhullError:
        return _rasterize_stat(sx, sy, sz, bounds, res, mode="max")

    edge_limit = float(max_edge) if max_edge is not None else math.inf
    if edge_limit <= 0:
        edge_limit = math.inf

    for simplex in tri.simplices:
        tri_xy = pts[simplex]
        tri_z = sz[simplex]

        e01 = np.hypot(*(tri_xy[0] - tri_xy[1]))
        e12 = np.hypot(*(tri_xy[1] - tri_xy[2]))
        e20 = np.hypot(*(tri_xy[2] - tri_xy[0]))
        longest = max(e01, e12, e20)

        if longest > edge_limit:
            continue

        tri_xmin = max(xmin, float(np.min(tri_xy[:, 0])))
        tri_xmax = min(xmax, float(np.max(tri_xy[:, 0])))
        tri_ymin = max(ymin, float(np.min(tri_xy[:, 1])))
        tri_ymax = min(ymax, float(np.max(tri_xy[:, 1])))

        c0 = max(0, int(np.floor((tri_xmin - xmin) / res)))
        c1 = min(nx - 1, int(np.floor((tri_xmax - xmin) / res)))
        r0 = max(0, int(np.floor((ymax - tri_ymax) / res)))
        r1 = min(ny - 1, int(np.floor((ymax - tri_ymin) / res)))

        if c1 < c0 or r1 < r0:
            continue

        sub_x = xs[c0:c1 + 1]
        sub_y = ys[r0:r1 + 1]
        sub_xx, sub_yy = np.meshgrid(sub_x, sub_y)

        inside, l1, l2, l3 = _triangle_barycentric_mask(sub_xx, sub_yy, tri_xy)
        if inside is None or not np.any(inside):
            continue

        zsurf = l1 * tri_z[0] + l2 * tri_z[1] + l3 * tri_z[2]

        sub = grid[r0:r1 + 1, c0:c1 + 1]
        write_mask = inside & (~np.isfinite(sub))
        sub[write_mask] = zsurf[write_mask].astype(np.float32)

        overlap_mask = inside & np.isfinite(sub)
        sub[overlap_mask] = np.maximum(sub[overlap_mask], zsurf[overlap_mask]).astype(np.float32)
        grid[r0:r1 + 1, c0:c1 + 1] = sub

    out_xmin = float(xs[0] - 0.5 * res)
    out_ymax = float(ys[0] + 0.5 * res)
    return grid, out_xmin, out_ymax


def _build_spikefree_surface(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    bounds: tuple[float, float, float, float],
    res: float,
    *,
    freeze_distance: float | None,
    insertion_buffer: float | None,
):
    sx, sy, sz = _cellmax_support_points(px, py, pz, bounds, res)

    if sx.size < 1:
        xs, ys, xx, _yy = _grid_xy(bounds, res)
        return np.full(xx.shape, np.nan, dtype=np.float32), xs, ys

    fd = float(freeze_distance) if freeze_distance is not None else max(4.0 * res, 1.0)
    grid, xmin, ymax = _rasterize_constrained_tin(
        sx, sy, sz, bounds, res, max_edge=fd
    )

    base_max, _bxmin, _bymax = _rasterize_stat(px, py, pz, bounds, res, mode="max")
    buf = float(insertion_buffer) if insertion_buffer is not None else max(0.25, 0.5 * res)

    valid_tin = np.isfinite(grid)
    valid_base = np.isfinite(base_max)
    clip_mask = valid_tin & valid_base & (grid > (base_max + buf))
    grid[clip_mask] = (base_max + buf)[clip_mask].astype(np.float32)

    xs = np.arange(xmin + 0.5 * res, bounds[2], res, dtype=np.float64)
    ys = np.arange(ymax - 0.5 * res, bounds[1], -res, dtype=np.float64)
    return grid.astype(np.float32), xs, ys


def _build_surface_from_points(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    bounds: tuple[float, float, float, float],
    res: float,
    method: str,
    *,
    spikefree_freeze_distance: float | None = None,
    spikefree_insertion_buffer: float | None = None,
    mask_kind: str = "support",
    mask_grow_cells: int | None = None,
):
    method = str(method).lower().strip()
    if method not in RASTER_METHOD_CHOICES:
        raise ValueError(f"Unsupported raster method: {method}")

    if px.size < 1:
        raise ValueError("Need at least 1 point to build raster surface.")
    if res <= 0:
        raise ValueError("res must be > 0")

    mask_kind = str(mask_kind).strip().lower()
    if mask_kind not in {"support", "footprint", "none"}:
        raise ValueError(f"Unsupported mask_kind: {mask_kind}")

    if method in {"min", "max", "mean"}:
        sx, sy, sz = _grid_support(px, py, pz, cell=res, mode=method)
        grid, xs, ys = _interpolate_tin_grid(sx, sy, sz, bounds=bounds, res=res)
    elif method == "nearest":
        xs, ys, xx, yy = _grid_xy(bounds, res)
        pts = np.column_stack([px, py])
        nn = NearestNDInterpolator(pts, pz)
        grid = nn(xx, yy).astype(np.float32)
    elif method == "idw":
        grid, xs, ys = _idw_grid(px, py, pz, bounds=bounds, res=res)
    elif method == "spikefree":
        grid, xs, ys = _build_spikefree_surface(
            px,
            py,
            pz,
            bounds,
            res,
            freeze_distance=spikefree_freeze_distance,
            insertion_buffer=spikefree_insertion_buffer,
        )
    else:
        raise ValueError(f"Unsupported raster method: {method}")

    if mask_kind == "none":
        return grid, xs, ys

    if mask_kind == "support":
        grow = 1 if mask_grow_cells is None else int(mask_grow_cells)
        mask = _support_mask_from_points(px, py, bounds, res, grow_cells=max(0, grow))
    else:
        grow = 1 if mask_grow_cells is None else int(mask_grow_cells)
        mask = _build_valid_dem_mask_from_ground_points(px, py, bounds, res)
        if grow > 0 and np.any(mask):
            mask = binary_dilation(mask, iterations=max(0, grow))

    grid = _apply_support_mask(grid, mask)
    return grid, xs, ys


def _write_tif(arr, out_fp, xmin, ymax, grid_res, crs=None, nodata=np.nan):
    if rasterio is None:
        raise RuntimeError("rasterio is required for DEM/DSM/CHM GeoTIFF outputs.")
    if grid_res <= 0:
        raise ValueError("grid_res must be > 0")

    os.makedirs(os.path.dirname(out_fp), exist_ok=True)

    height, width = arr.shape
    transform = from_origin(float(xmin), float(ymax), float(grid_res), float(grid_res))
    assert abs(float(transform.a) - float(grid_res)) < 1e-12, "Grid resolution mismatch in raster transform"

    profile = {
        "driver": "GTiff",
        "height": int(height),
        "width": int(width),
        "count": 1,
        "dtype": str(arr.dtype),
        "transform": transform,
        "compress": "lzw",
    }

    if np.issubdtype(arr.dtype, np.floating):
        profile["nodata"] = nodata
    elif nodata is not None and not np.isnan(nodata):
        profile["nodata"] = nodata

    if crs is not None:
        try:
            from rasterio.crs import CRS as RioCRS
            profile["crs"] = RioCRS.from_user_input(crs)
        except Exception:
            try:
                if hasattr(crs, "to_wkt"):
                    profile["crs"] = crs.to_wkt()
            except Exception:
                print(f"[WARN] Could not attach CRS to raster: {out_fp}")

    with rasterio.open(out_fp, "w", **profile) as dst:
        dst.write(arr, 1)


def _sample_bilinear_from_grid(
    grid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    xmin: float,
    ymax: float,
    res: float,
) -> np.ndarray:
    h, w = grid.shape
    gx = (x - xmin) / res - 0.5
    gy = (ymax - y) / res - 0.5

    x0 = np.floor(gx).astype(np.int32)
    y0 = np.floor(gy).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = gx - x0
    wy = gy - y0

    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)

    g00 = grid[y0c, x0c]
    g10 = grid[y0c, x1c]
    g01 = grid[y1c, x0c]
    g11 = grid[y1c, x1c]

    return (
        (1 - wx) * (1 - wy) * g00
        + wx * (1 - wy) * g10
        + (1 - wx) * wy * g01
        + wx * wy * g11
    )


def _sample_grid_bilinear_then_nearest(
    grid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    xmin: float,
    ymax: float,
    res: float,
) -> np.ndarray:
    """
    Sample grid first with bilinear interpolation.
    Any non-finite sampled values are backfilled with nearest-neighbor
    from the valid raster cells so FAST_NORMALIZED never writes NaN Z values.
    """
    z = _sample_bilinear_from_grid(grid, x, y, xmin, ymax, res)

    bad = ~np.isfinite(z)
    if not np.any(bad):
        return z

    h, w = grid.shape
    valid = np.isfinite(grid)
    if not np.any(valid):
        return z

    rows, cols = np.where(valid)
    gx = xmin + (cols.astype(np.float64) + 0.5) * res
    gy = ymax - (rows.astype(np.float64) + 0.5) * res
    gv = grid[rows, cols].astype(np.float64)

    nn = NearestNDInterpolator(np.column_stack([gx, gy]), gv)
    z_fill = nn(x[bad], y[bad])
    z[bad] = z_fill
    return z


def _grid_min_count(x: np.ndarray, y: np.ndarray, z: np.ndarray, cell: float):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if x.size == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.int32), 0.0, 0.0

    x0 = float(np.min(x))
    y0 = float(np.min(y))
    ix = np.floor((x - x0) / cell).astype(np.int32)
    iy = np.floor((y - y0) / cell).astype(np.int32)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1

    zmin = np.full((ny, nx), np.nan, dtype=np.float64)
    counts = np.zeros((ny, nx), dtype=np.int32)
    for i in range(z.size):
        cx = int(ix[i])
        cy = int(iy[i])
        zi = float(z[i])
        counts[cy, cx] += 1
        v = zmin[cy, cx]
        if np.isnan(v) or zi < v:
            zmin[cy, cx] = zi
    return zmin.astype(np.float32), counts, x0, y0


def _quick_demote_high_ground_cells(x: np.ndarray, y: np.ndarray, z: np.ndarray, ground: np.ndarray, cell: float) -> np.ndarray:
    g = np.asarray(ground, dtype=bool).copy()
    if np.count_nonzero(g) < 10:
        return g

    zmin, counts, x0, y0 = _grid_min_count(x[g], y[g], z[g], cell=cell)
    ny, nx = zmin.shape
    if ny == 0 or nx == 0:
        return g

    suspicious = np.zeros((ny, nx), dtype=bool)
    support_limit = 2
    hard_jump = 1.0 if cell >= 0.75 else 0.5

    for yy in range(ny):
        y1 = max(0, yy - 1)
        y2 = min(ny, yy + 2)
        for xx in range(nx):
            if not np.isfinite(zmin[yy, xx]):
                continue
            if counts[yy, xx] > support_limit:
                continue
            x1 = max(0, xx - 1)
            x2 = min(nx, xx + 2)
            win = zmin[y1:y2, x1:x2].copy()
            if win.size <= 1:
                continue
            local = win[np.isfinite(win)]
            if local.size < 3:
                continue
            self_val = float(zmin[yy, xx])
            idx = np.flatnonzero(np.isclose(local, self_val))
            if idx.size:
                local = np.delete(local, idx[0])
            if local.size < 2:
                continue
            med = float(np.median(local))
            if self_val - med > hard_jump:
                suspicious[yy, xx] = True

    if not np.any(suspicious):
        return g

    xg = x[g]
    yg = y[g]
    ix = np.floor((xg - x0) / cell).astype(np.int32)
    iy = np.floor((yg - y0) / cell).astype(np.int32)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    bad = suspicious[iy, ix]
    if not np.any(bad):
        return g

    keep_ground = np.flatnonzero(g)
    g[keep_ground[bad]] = False
    return g


def _fallback_ground_mask_from_surface(
    xw: np.ndarray,
    yw: np.ndarray,
    zw: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """Robust fallback for rare feature-shape failures in vote-based classifiers.

    Uses a conservative lower-envelope surface from candidate points and classifies
    points as ground when they lie close to that surface.
    """
    if xw.size == 0:
        return np.zeros(0, dtype=bool)

    cell = float(cfg.get("vote_cell_m", cfg.get("base_cell_m", 1.5)))
    cell = max(cell, 0.25)
    thr = float(cfg.get("vote_ground_threshold_m", cfg.get("cand_dz_m", 0.60)))
    thr = max(thr, 0.10)

    bounds = (float(np.min(xw)), float(np.min(yw)), float(np.max(xw)), float(np.max(yw)))

    try:
        sx, sy, sz = _grid_support(xw, yw, zw, cell=cell, mode="min")
        if sx.size < 3:
            raise ValueError("Too few support points for fallback surface.")

        surf, xs, ys = _build_surface_from_points(
            sx,
            sy,
            sz,
            bounds=bounds,
            res=cell,
            method="nearest",
            mask_kind="none",
        )
        xmin = float(xs[0] - 0.5 * cell)
        ymax = float(ys[0] + 0.5 * cell)
        zsurf = _sample_grid_bilinear_then_nearest(surf, xw, yw, xmin, ymax, cell)
        ground_mask = zw <= (zsurf + thr)
    except Exception:
        q = float(np.quantile(zw, 0.05))
        ground_mask = zw <= q

    if np.count_nonzero(ground_mask) < max(25, int(0.01 * zw.size)):
        q = float(np.quantile(zw, 0.05))
        ground_mask = zw <= q

    return np.asarray(ground_mask, dtype=bool)


def _classify_ground_file_with_fallback(
    las: laspy.LasData,
    out_path: str,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    keep: np.ndarray,
    m1: np.ndarray,
    cfg: dict,
) -> None:
    x1 = x[keep]
    y1 = y[keep]
    z1 = z[keep]

    xw = x1[m1]
    yw = y1[m1]
    zw = z1[m1]

    classification = np.ones(x.size, dtype=np.uint8)
    if xw.size == 0:
        _write_full_cloud_with_classification(las, out_path, classification)
        return

    ground_mask = _fallback_ground_mask_from_surface(xw, yw, zw, cfg)
    classification[np.flatnonzero(keep)[m1][ground_mask]] = 2
    _write_full_cloud_with_classification(las, out_path, classification)


def _resolve_products(products: Iterable[str] | None) -> tuple[set[str], set[str]]:
    requested = set(products or {PRODUCT_GC})
    if not requested:
        requested = {PRODUCT_GC}
    if PRODUCT_ALL in requested:
        requested = set(ALL_PRODUCTS)

    compute = set(requested)

    if PRODUCT_NORMALIZED in requested:
        compute.add(PRODUCT_DEM)

    if PRODUCT_CHM in requested:
        compute.update({PRODUCT_DEM, PRODUCT_NORMALIZED})

    if PRODUCT_TERRAIN in requested:
        compute.update({PRODUCT_DEM})

    if PRODUCT_STRUCTURE in requested:
        compute.update({PRODUCT_DEM, PRODUCT_NORMALIZED})

    return requested, compute


def _require_rasterio_for_products(compute_products: set[str]):
    needs_raster = bool({PRODUCT_DEM, PRODUCT_DSM, PRODUCT_CHM, PRODUCT_TERRAIN} & compute_products)
    if needs_raster and rasterio is None:
        raise RuntimeError(
            "rasterio is required for DEM/DSM/CHM/TERRAIN products. Install FAST-GC with rasterio available."
        )


def _load_product_context(classified_fp: str):
    las, x, y, z, cls, crs = _read_las(classified_fp)
    base = Path(classified_fp).stem
    bounds = (float(np.min(x)), float(np.min(y)), float(np.max(x)), float(np.max(y)))
    return {
        "classified_fp": classified_fp,
        "las": las,
        "x": x,
        "y": y,
        "z": z,
        "cls": cls,
        "crs": crs,
        "base": base,
        "bounds": bounds,
    }


def _build_dem_bundle(ctx: dict, grid_res: float, dem_method: str = "nearest"):
    if grid_res <= 0:
        raise ValueError("grid_res must be > 0")

    x = ctx["x"]
    y = ctx["y"]
    z = ctx["z"]
    cls = ctx["cls"]
    bounds = ctx["bounds"]

    ground = cls == 2
    if np.count_nonzero(ground) < 3:
        raise RuntimeError(f"Too few ground points to build DEM for {ctx['classified_fp']}")

    ground = _quick_demote_high_ground_cells(x, y, z, ground, cell=grid_res)
    if np.count_nonzero(ground) < 3:
        raise RuntimeError(f"Too few ground points remain after DEM cleanup for {ctx['classified_fp']}")

    gx = x[ground]
    gy = y[ground]
    gz = z[ground]

    dem_method = str(dem_method).strip().lower()

    # DEM contract:
    # - use only ground points
    # - interpolate across the full tile extent without interior masking
    # - rely on tile bounds / merge cropping instead of sparse support masks
    if dem_method == "nearest":
        dem, xmin, ymax = _rasterize_nearest_in_cell(gx, gy, gz, bounds, grid_res)
        dem = _fill_nan_cells_from_tin(dem, gx, gy, gz, bounds, grid_res)
    elif dem_method in {"min", "mean"}:
        dem, xmin, ymax = _rasterize_stat(gx, gy, gz, bounds, grid_res, mode=dem_method)
        dem = _fill_nan_cells_from_tin(dem, gx, gy, gz, bounds, grid_res)
    elif dem_method == "idw":
        dem, xs, ys = _idw_grid(gx, gy, gz, bounds=bounds, res=grid_res)
        xmin = float(xs[0] - 0.5 * grid_res)
        ymax = float(ys[0] + 0.5 * grid_res)
    else:
        dem, xs, ys = _interpolate_tin_grid(gx, gy, gz, bounds=bounds, res=grid_res)
        xmin = float(xs[0] - 0.5 * grid_res)
        ymax = float(ys[0] + 0.5 * grid_res)

    dem = np.asarray(dem, dtype=np.float32)
    if not np.all(np.isfinite(dem)):
        dem = _fill_dem_grid_by_nearest_ground(
            dem,
            np.ones_like(dem, dtype=bool),
            gx,
            gy,
            gz,
            bounds,
            grid_res,
            k=8,
            power=2.0,
        )

    return {
        "dem": dem.astype(np.float32),
        "xmin": float(xmin),
        "ymax": float(ymax),
        "ground_mask": ground,
    }


def _build_dsm(

    ctx: dict,
    grid_res: float,
    dsm_method: str = "max",
    *,
    spikefree_freeze_distance: float | None = None,
    spikefree_insertion_buffer: float | None = None,
):
    dsm, xs, ys = _build_surface_from_points(
        ctx["x"],
        ctx["y"],
        ctx["z"],
        bounds=ctx["bounds"],
        res=grid_res,
        method=dsm_method,
        spikefree_freeze_distance=spikefree_freeze_distance,
        spikefree_insertion_buffer=spikefree_insertion_buffer,
    )
    xmin = float(xs[0] - 0.5 * grid_res)
    ymax = float(ys[0] + 0.5 * grid_res)
    return {"dsm": dsm, "xmin": xmin, "ymax": ymax}


def _write_dem_from_context(
    ctx: dict,
    dem_pack: dict,
    product_dirs: dict[str, str],
    grid_res: float,
):
    """Write FAST_DEM from an already-loaded tile context.

    This helper deliberately performs no scientific recomputation.  It exists so
    coupled downstream products can reuse one LAS/LAZ read and one DEM build.
    """
    out_fp = os.path.join(product_dirs[PRODUCT_DEM], f"{ctx['base']}.tif")
    _write_tif(dem_pack["dem"], out_fp, dem_pack["xmin"], dem_pack["ymax"], grid_res, crs=ctx["crs"])
    return {"status": "ok", "product": PRODUCT_DEM, "tile": ctx["classified_fp"], "output": out_fp, "reason": None}


def _write_normalized_from_context(
    ctx: dict,
    dem_pack: dict,
    product_dirs: dict[str, str],
    grid_res: float,
):
    """Write FAST_NORMALIZED from a shared tile context and DEM bundle."""
    z_dem = _sample_grid_bilinear_then_nearest(
        dem_pack["dem"],
        ctx["x"],
        ctx["y"],
        dem_pack["xmin"],
        dem_pack["ymax"],
        grid_res,
    )

    # Final safety: if anything is still non-finite, replace with original z
    # so we never write invalid integer Z values.
    bad_dem = ~np.isfinite(z_dem)
    if np.any(bad_dem):
        z_dem = np.array(z_dem, copy=True)
        z_dem[bad_dem] = ctx["z"][bad_dem]

    z_norm = ctx["z"] - z_dem
    z_norm = np.where(dem_pack["ground_mask"], 0.0, z_norm)
    z_norm = np.maximum(z_norm, 0.0)

    # Final guard before integer cast
    bad_norm = ~np.isfinite(z_norm)
    if np.any(bad_norm):
        z_norm = np.array(z_norm, copy=True)
        z_norm[bad_norm] = 0.0

    out = laspy.LasData(ctx["las"].header)
    out.points = ctx["las"].points.copy()
    z_scale = float(out.header.scales[2])
    z_offset = float(out.header.offsets[2])

    raw_Z = np.round((z_norm - z_offset) / z_scale)
    raw_Z = np.nan_to_num(raw_Z, nan=0.0, posinf=0.0, neginf=0.0)
    raw_Z = raw_Z.astype(out.points.array["Z"].dtype, copy=False)

    out.points.array["Z"] = raw_Z
    try:
        out.classification = ctx["cls"].astype(np.uint8)
    except Exception:
        pass

    out_fp = os.path.join(product_dirs[PRODUCT_NORMALIZED], f"{ctx['base']}.las")
    out.write(out_fp)
    return {"status": "ok", "product": PRODUCT_NORMALIZED, "tile": ctx["classified_fp"], "output": out_fp, "reason": None}


def _load_and_build_dem(
    classified_fp: str,
    grid_res: float,
    dem_method: str,
):
    ctx = _load_product_context(classified_fp)
    try:
        dem_pack = _build_dem_bundle(ctx, grid_res, dem_method=dem_method)
    except RuntimeError as e:
        msg = str(e)
        if "Too few ground points" in msg:
            return None, None, msg
        raise
    return ctx, dem_pack, None


def _write_dem(classified_fp: str, product_dirs: dict[str, str], grid_res: float, dem_method: str = "min"):
    ctx, dem_pack, reason = _load_and_build_dem(classified_fp, grid_res, dem_method)
    if reason is not None:
        return {"status": "skipped", "product": PRODUCT_DEM, "tile": classified_fp, "output": None, "reason": reason}
    return _write_dem_from_context(ctx, dem_pack, product_dirs, grid_res)


def _write_normalized(classified_fp: str, product_dirs: dict[str, str], grid_res: float, dem_method: str = "min"):
    ctx, dem_pack, reason = _load_and_build_dem(classified_fp, grid_res, dem_method)
    if reason is not None:
        return {"status": "skipped", "product": PRODUCT_NORMALIZED, "tile": classified_fp, "output": None, "reason": reason}
    return _write_normalized_from_context(ctx, dem_pack, product_dirs, grid_res)


def _write_dem_and_normalized_shared(
    classified_fp: str,
    product_dirs: dict[str, str],
    grid_res: float,
    dem_method: str = "min",
):
    """Derive DEM + normalized cloud with one LAS read and one DEM build.

    Output calculations are delegated to the same writers used by the independent
    product paths; only redundant I/O and DEM reconstruction are removed.
    """
    ctx, dem_pack, reason = _load_and_build_dem(classified_fp, grid_res, dem_method)
    if reason is not None:
        return {
            "status": "skipped",
            "product": f"{PRODUCT_DEM}+{PRODUCT_NORMALIZED}",
            "tile": classified_fp,
            "outputs": {},
            "reason": reason,
        }

    dem_result = _write_dem_from_context(ctx, dem_pack, product_dirs, grid_res)
    norm_result = _write_normalized_from_context(ctx, dem_pack, product_dirs, grid_res)
    return {
        "status": "ok",
        "product": f"{PRODUCT_DEM}+{PRODUCT_NORMALIZED}",
        "tile": classified_fp,
        "outputs": {
            PRODUCT_DEM: dem_result["output"],
            PRODUCT_NORMALIZED: norm_result["output"],
        },
        "reason": None,
    }


def _write_dsm_from_raw(
    raw_fp: str,
    product_dirs: dict[str, str],
    grid_res: float,
    dsm_method: str = "max",
    *,
    spikefree_freeze_distance: float | None = None,
    spikefree_insertion_buffer: float | None = None,
):
    _las, x, y, z, _, crs = _read_las(raw_fp)
    bounds = (float(np.min(x)), float(np.min(y)), float(np.max(x)), float(np.max(y)))
    dsm, xs, ys = _build_surface_from_points(
        x,
        y,
        z,
        bounds=bounds,
        res=grid_res,
        method=dsm_method,
        spikefree_freeze_distance=spikefree_freeze_distance,
        spikefree_insertion_buffer=spikefree_insertion_buffer,
    )
    xmin = float(xs[0] - 0.5 * grid_res)
    ymax = float(ys[0] + 0.5 * grid_res)

    out_fp = os.path.join(product_dirs[PRODUCT_DSM], f"{Path(raw_fp).stem}.tif")
    _write_tif(dsm, out_fp, xmin, ymax, grid_res, crs=crs)
    return {"status": "ok", "product": PRODUCT_DSM, "tile": raw_fp, "output": out_fp, "reason": None}



def classify_ground_file(in_path: str, out_path: str, cfg: dict, show_progress: bool = False):
    """
    Restore the original FAST-GC classification contract while preserving the
    newer downstream product pipeline in the rest of this module.

    Notes
    -----
    - zclean / outlier removal is intentionally NOT applied here. FAST-GC begins
      from the input cloud supplied to this function; optional zclean should run
      upstream as preprocessing.
    - The newer fallback classifier is intentionally not used by default here, so
      FAST-GC remains on the original sensor-specific vote-surface path for ALS,
      ULS, and TLS.
    - Downstream DEM / DSM / CHM / terrain / structure / change / ITD helpers in
      this file remain unchanged.
    """
    bar = _stage_bar(6, f"CLASS {Path(in_path).name}", show_progress)

    las, x, y, z, _, _crs = _read_las(in_path)
    N = x.size
    sm = str(cfg.get("sensor_mode", "ALS")).upper().strip()
    bar.update(1)

    # FAST-GC core should not perform zclean internally.
    keep = np.ones(N, dtype=bool)
    bar.update(1)

    if np.count_nonzero(keep) < 1000:
        classification = np.ones(N, dtype=np.uint8)
        _write_full_cloud_with_classification(las, out_path, classification)
        bar.update(4)
        bar.close()
        return

    x1 = x[keep]
    y1 = y[keep]
    z1 = z[keep]

    if sm == "TLS":
        if bool(cfg.get("tls_voxel_reduce_enabled", True)):
            tls_voxel_result = reduce_tls_terrain_domain(
                x1,
                y1,
                z1,
                fine_cell_m=float(cfg.get("tls_voxel_fine_cell_m", 0.50)),
                support_window_m=float(cfg.get("tls_voxel_support_window_m", 5.0)),
                low_quantile=float(cfg.get("tls_voxel_low_quantile", 0.05)),
                min_points_per_cell=int(cfg.get("tls_voxel_min_points_per_cell", 3)),
                min_support_cells=int(cfg.get("tls_voxel_min_support_cells", 5)),
                vertical_voxel_m=float(cfg.get("tls_voxel_vertical_m", 0.25)),
                safe_height_m=float(cfg.get("tls_voxel_safe_height_m", 2.0)),
            )
            # V3 terrain-first TLS contract:
            #
            # domain_mask:
            #     conservative V2 terrain-domain eligibility.
            #
            # active_mask:
            #     density-normalized representatives used ONLY to build
            #     the TLS terrain surface.
            #
            # Points omitted from active_mask are not vegetation and are
            # not removed from final classification.
            tls_domain_mask = np.asarray(
                tls_voxel_result.keep_mask,
                dtype=bool,
            )

            tls_evidence = select_tls_active_evidence(
                x1,
                y1,
                z1,
                tls_domain_mask,
                xy_voxel_m=float(
                    cfg.get(
                        "tls_evidence_xy_voxel_m",
                        0.20,
                    )
                ),
                z_voxel_m=float(
                    cfg.get(
                        "tls_evidence_z_voxel_m",
                        0.10,
                    )
                ),
                max_points_per_voxel=int(
                    cfg.get(
                        "tls_evidence_max_points_per_voxel",
                        8,
                    )
                ),
            )

            m1 = np.asarray(
                tls_evidence.active_mask,
                dtype=bool,
            )

            # Safety fallback:
            # if evidence normalization becomes unexpectedly aggressive,
            # use the complete V2 domain rather than risking terrain loss.
            if np.count_nonzero(m1) < max(
                1000,
                int(0.05 * np.count_nonzero(tls_domain_mask)),
            ):
                m1 = tls_domain_mask.copy()

            # Second safety fallback:
            # V2 itself must never collapse the TLS working domain.
            if np.count_nonzero(m1) < max(
                1000,
                int(0.05 * z1.size),
            ):
                tls_domain_mask = np.ones_like(
                    z1,
                    dtype=bool,
                )
                m1 = tls_domain_mask.copy()
        else:
            tls_domain_mask = np.ones_like(
                z1,
                dtype=bool,
            )
            m1 = tls_domain_mask.copy()
    else:
        cand_cell = float(cfg.get("cand_cell_m", cfg.get("base_cell_m", 1.5)))
        cand_dz = float(cfg.get("cand_dz_m", 0.60))
        m1 = _candidate_layer_by_cellmin(x1, y1, z1, cell=cand_cell, dz=cand_dz)
        if np.count_nonzero(m1) < max(1000, int(0.01 * z1.size)):
            m1 = np.ones_like(z1, dtype=bool)

    xw = x1[m1]
    yw = y1[m1]
    zw = z1[m1]
    bar.update(1)

    if xw.size < 1000:
        classification = np.ones(N, dtype=np.uint8)
        _write_full_cloud_with_classification(las, out_path, classification)
        bar.update(3)
        bar.close()
        return

    if sm == "TLS":
        vcfg = TlsInvertDsmVoteConfig(
            cell=float(cfg["vote_cell_m"]),
            top_m=int(cfg["vote_top_m"]),
            neighbor_radius_cells=int(cfg["vote_neighbor_radius_cells"]),
            min_neighbor_cells=int(cfg["vote_min_neighbor_cells"]),
            max_robust_z=float(cfg["vote_max_robust_z"]),
            mad_floor=float(cfg["vote_mad_floor"]),
            fill_iters=int(cfg["vote_fill_iters"]),
            smooth_sigma_cells=float(cfg["vote_smooth_sigma_cells"]),
            ground_threshold=float(cfg["vote_ground_threshold_m"]),
            slope_adapt_k=float(cfg["vote_slope_adapt_k"]),
            candidate_cluster_gap=float(cfg.get("tls_candidate_cluster_gap_m", 0.08)),
            candidate_cluster_span=float(cfg.get("tls_candidate_cluster_span_m", 0.14)),
            candidate_min_cluster_points=int(cfg.get("tls_candidate_min_cluster_points", 3)),
            below_ground_gap_threshold=float(cfg.get("tls_below_ground_gap_m", 0.16)),
            below_ground_max_cluster_points=int(cfg.get("tls_below_ground_max_cluster_points", 2)),
            use_offset_grid=bool(cfg.get("tls_use_offset_grid", True)),
            radius_min_cells=int(cfg.get("tls_radius_min_cells", 2)),
            radius_max_cells=int(cfg.get("tls_radius_max_cells", 12)),
            min_support_sectors=int(cfg.get("tls_min_support_sectors", 3)),
            seed_confidence=float(cfg.get("tls_seed_confidence", 0.68)),
            seed_max_vertical_span=float(cfg.get("tls_seed_max_vertical_span_m", 0.35)),
            propagation_iters=int(cfg.get("tls_propagation_iters", 24)),
            max_extrapolation_cells=int(cfg.get("tls_max_extrapolation_cells", 5)),
            plane_min_support=int(cfg.get("tls_plane_min_support", 6)),
            plane_max_residual=float(cfg.get("tls_plane_max_residual_m", 0.14)),
            fill_min_bank_fraction=float(cfg.get("tls_fill_min_bank_fraction", 0.50)),
            fill_max_distance_cells=int(cfg.get("tls_fill_max_distance_cells", 5)),
            reject_edge_connected_voids=bool(cfg.get("tls_reject_edge_connected_voids", True)),
            smooth_iters=int(cfg.get("tls_smooth_iters", 2)),
            bilateral_height_sigma=float(cfg.get("tls_bilateral_height_sigma_m", 0.18)),
            bilateral_slope_sigma=float(cfg.get("tls_bilateral_slope_sigma", 0.60)),
            curvature_adapt_k=float(cfg.get("tls_curvature_adapt_k", 0.05)),
            roughness_adapt_k=float(cfg.get("tls_roughness_adapt_k", 0.30)),
            threshold_min=float(cfg.get("tls_threshold_min_m", 0.03)),
            threshold_max=float(cfg.get("tls_threshold_max_m", 0.28)),
            lower_threshold_factor=float(cfg.get("tls_lower_threshold_factor", 1.75)),
            min_surface_confidence=float(cfg.get("tls_min_surface_confidence", 0.28)),
            min_classification_confidence=float(cfg.get("tls_min_classification_confidence", 0.34)),
            weak_confidence_tighten=float(cfg.get("tls_weak_confidence_tighten", 0.45)),
        )
        # ACTIVE evidence constructs the TLS terrain surface.
        surf_z, sx0, sy0 = build_tls_surface_invert_dsm_vote(
            xw,
            yw,
            zw,
            vcfg,
        )
        tls_surface_diagnostics = get_tls_surface_diagnostics(
            surf_z
        )

        # --------------------------------------------------------
        # TERRAIN-FIRST TLS CONTRACT
        # --------------------------------------------------------
        # ACTIVE points are only support for surface construction.
        #
        # Once the surface exists, classify the complete conservative
        # V2 terrain domain against it.  A point is therefore never
        # non-ground merely because V3 marked it redundant.
        #
        # Clearly V2-deferred points remain outside the TLS terrain
        # domain by design.
        domain_indices = np.flatnonzero(tls_domain_mask)

        xd = x1[tls_domain_mask]
        yd = y1[tls_domain_mask]
        zd = z1[tls_domain_mask]

        ground_mask = np.asarray(
            classify_tls_by_surface(
                xd,
                yd,
                zd,
                surf_z,
                sx0,
                sy0,
                vcfg,
            ),
            dtype=bool,
        )

        tls_candidate_ground = ground_mask.copy()

        # From this point through membrane/recovery/final TLS QC,
        # xw/yw/zw intentionally represent the complete V2 terrain
        # domain, not only the ACTIVE surface-building evidence.
        xw, yw, zw = xd, yd, zd

        # m1 is the mapping from x1/y1/z1 into the working TLS domain.
        m1 = tls_domain_mask.copy()

        # Preserve the TLS-v2 high-recall candidate layer, then purify it into
        # a geometrically coherent, paper-thin terrain membrane.
        if bool(cfg.get("tls_membrane_enabled", False)):
            membrane_cfg = TlsMembraneConfig(
                enabled=True,
                cell=float(cfg.get("tls_membrane_cell_m", 0.15)),
                offset_fraction=float(cfg.get("tls_membrane_offset_fraction", 0.50)),
                cluster_gap_m=float(cfg.get("tls_membrane_cluster_gap_m", 0.035)),
                cluster_span_max_m=float(cfg.get("tls_membrane_cluster_span_max_m", 0.040)),
                cluster_min_points=int(cfg.get("tls_membrane_cluster_min_points", 2)),
                cluster_max_points_examined=int(cfg.get("tls_membrane_cluster_max_points_examined", 32)),
                below_noise_gap_m=float(cfg.get("tls_membrane_below_noise_gap_m", 0.12)),
                below_noise_max_points=int(cfg.get("tls_membrane_below_noise_max_points", 2)),
                offset_agreement_m=float(cfg.get("tls_membrane_offset_agreement_m", 0.035)),
                anchor_min_offset_votes=int(cfg.get("tls_membrane_anchor_min_offset_votes", 3)),
                anchor_min_score=float(cfg.get("tls_membrane_anchor_min_score", 0.58)),
                anchor_min_local_occupancy=float(cfg.get("tls_membrane_anchor_min_local_occupancy", 0.22)),
                anchor_density_quantile=float(cfg.get("tls_membrane_anchor_density_quantile", 0.45)),
                anchor_max_cluster_thickness_m=float(cfg.get("tls_membrane_anchor_max_cluster_thickness_m", 0.035)),
                propagation_iters=int(cfg.get("tls_membrane_propagation_iters", 60)),
                propagation_radius_cells=int(cfg.get("tls_membrane_propagation_radius_cells", 2)),
                max_anchor_distance_cells=int(cfg.get("tls_membrane_max_anchor_distance_cells", 20)),
                weak_observation_gate_m=float(cfg.get("tls_membrane_weak_observation_gate_m", 0.080)),
                anchor_lock_weight=float(cfg.get("tls_membrane_anchor_lock_weight", 12.0)),
                weak_observation_weight=float(cfg.get("tls_membrane_weak_observation_weight", 0.20)),
                relaxation=float(cfg.get("tls_membrane_relaxation", 0.65)),
                sheet_thickness_min_m=float(cfg.get("tls_membrane_sheet_thickness_min_m", 0.010)),
                sheet_thickness_base_m=float(cfg.get("tls_membrane_sheet_thickness_base_m", 0.018)),
                sheet_thickness_max_m=float(cfg.get("tls_membrane_sheet_thickness_max_m", 0.035)),
                upper_factor=float(cfg.get("tls_membrane_upper_factor", 1.00)),
                lower_factor=float(cfg.get("tls_membrane_lower_factor", 1.35)),
                thickness_regularize_iters=int(cfg.get("tls_membrane_thickness_regularize_iters", 6)),
                min_keep_confidence=float(cfg.get("tls_membrane_min_keep_confidence", 0.22)),
                recover_ground=False,
            )
            membrane = refine_tls_ground_membrane(
                x=xw, y=yw, z=zw, candidate_ground=ground_mask, config=membrane_cfg
            )
            ground_mask = np.asarray(membrane.final_ground, dtype=bool)

        # Conservative TLS micro-topography recovery.  The membrane remains the
        # trusted precision layer.  This stage may restore only points that were
        # already in the pre-membrane TLS-v2 candidate set; raw non-ground points
        # are never recruited.  No stdout logging is emitted here so the parent
        # CLASSIFY progress dashboard remains a single dynamic line.
        if bool(cfg.get("tls_terrain_recovery_enabled", False)):
            terrain_cfg = TlsTerrainRecoveryConfig(
                enabled=True,
                cell=float(cfg.get("tls_terrain_recovery_cell_m", cfg.get("tls_membrane_cell_m", 0.15))),
                max_passes=int(cfg.get("tls_terrain_recovery_passes", 3)),
                low_cluster_gap_m=float(cfg.get("tls_terrain_recovery_cluster_gap_m", 0.040)),
                low_cluster_span_max_m=float(cfg.get("tls_terrain_recovery_cluster_span_m", 0.060)),
                min_cluster_points=int(cfg.get("tls_terrain_recovery_min_cluster_points", 2)),
                max_points_examined=int(cfg.get("tls_terrain_recovery_max_points_examined", 48)),
                detached_low_noise_gap_m=float(cfg.get("tls_terrain_recovery_noise_gap_m", 0.12)),
                detached_low_noise_max_points=int(cfg.get("tls_terrain_recovery_noise_max_points", 2)),
                min_directional_support=int(cfg.get("tls_terrain_recovery_min_support_dirs", 2)),
                bilateral_radius_cells=int(cfg.get("tls_terrain_recovery_bilateral_radius_cells", 4)),
                base_residual_m=float(cfg.get("tls_terrain_recovery_base_residual_m", 0.050)),
                slope_residual_k=float(cfg.get("tls_terrain_recovery_slope_k", 0.42)),
                bilateral_base_residual_m=float(cfg.get("tls_terrain_recovery_bilateral_residual_m", 0.070)),
                max_residual_m=float(cfg.get("tls_terrain_recovery_max_residual_m", 0.135)),
                max_neighbor_step_m=float(cfg.get("tls_terrain_recovery_max_step_m", 0.18)),
                point_lower_m=float(cfg.get("tls_terrain_recovery_point_lower_m", 0.030)),
                point_upper_m=float(cfg.get("tls_terrain_recovery_point_upper_m", 0.050)),
            )
            terrain_recovery = recover_tls_microtopography(
                x=xw,
                y=yw,
                z=zw,
                trusted_ground=ground_mask,
                candidate_pool=tls_candidate_ground,
                config=terrain_cfg,
            )
            ground_mask = np.asarray(terrain_recovery.final_ground, dtype=bool)
    elif sm in {"ULS", "ALS"}:
        vcfg = InvertVoteConfig(
            cell=float(cfg["vote_cell_m"]),
            top_m=int(cfg["vote_top_m"]),
            neighbor_radius_cells=int(cfg["vote_neighbor_radius_cells"]),
            min_neighbor_cells=int(cfg["vote_min_neighbor_cells"]),
            max_robust_z=float(cfg["vote_max_robust_z"]),
            mad_floor=float(cfg["vote_mad_floor"]),
            fill_iters=int(cfg["vote_fill_iters"]),
            smooth_sigma_cells=float(cfg["vote_smooth_sigma_cells"]),
            ground_threshold=float(cfg["vote_ground_threshold_m"]),
            slope_adapt_k=float(cfg["vote_slope_adapt_k"]),
        )

        # TERRAIN-FIRST CONTRACT
        # ----------------------
        # The lower-layer candidate mask is only a robust SUPPORT selector for
        # construction of the voted terrain surface.  It is NOT an eligibility
        # gate for the final classification.  On steep/curved terrain an XY cell
        # can legitimately span a large Z range, so permanently discarding points
        # outside the candidate band creates unrecoverable holes at ridges,
        # convex/concave bulges and slope transitions.
        surf_z, sx0, sy0 = build_surface_invert_vote(xw, yw, zw, vcfg)

        # Classify the complete cleaned ALS/ULS cloud against the support surface.
        # From this point onward xw/yw/zw intentionally refer to the complete
        # cleaned cloud, allowing every observed return to participate in final
        # terrain recovery/QC.  No point is invented.
        ground_mask = np.asarray(
            classify_by_surface(x1, y1, z1, surf_z, sx0, sy0, vcfg),
            dtype=bool,
        )
        xw, yw, zw = x1, y1, z1
        m1 = np.ones_like(z1, dtype=bool)
    else:
        bar.close()
        raise ValueError(f"Unsupported sensor mode: {sm}")

    if np.count_nonzero(ground_mask) < 500:
        q = np.quantile(zw, 0.05)
        ground_mask = zw <= q

    bar.update(1)

    ground_mask = recover_ground_in_voids(
        xw=xw,
        yw=yw,
        zw=zw,
        ground_mask=ground_mask,
        surf_z=surf_z,
        sx0=sx0,
        sy0=sy0,
        surf_cell=float(vcfg.cell),
        sensor_mode=sm,
        cfg=cfg,
    )

    # Post-classification depression/ditch validation.  This does not alter the
    # primary FAST-GC classifier: it only inspects holes in the trusted ground
    # support and selectively promotes a coherent low layer when surrounding
    # ground validates a real concavity.  TLS is a no-op here.
    ground_mask, ditch_qc = sweep_ground_depressions(
        x=xw,
        y=yw,
        z=zw,
        ground_mask=ground_mask,
        sensor_mode=sm,
        cfg=cfg,
        return_report=True,
    )
    # Keep parallel tile workers silent while the parent process owns the
    # single live progress dashboard.  Per-tile DITCH-QC messages from loky
    # workers were interleaving with and breaking the CLASSIFY progress line.
    # The message remains available for explicit single-file progress runs.
    if show_progress and int(ditch_qc.get("recovered_points", 0)) > 0:
        log_info(
            "DITCH-QC "
            f"candidates={ditch_qc.get('candidate_components', 0)} | "
            f"validated={ditch_qc.get('validated_components', 0)} | "
            f"recovered={ditch_qc.get('recovered_points', 0)}"
        )

    bar.update(1)


    # TERRAIN-FIRST FINAL QC
    # ----------------------
    # Building/roof removal is deliberately NOT part of core FAST_GC anymore.
    # A coherent roof may remain class 2 if removing it would risk deleting a
    # real ridge, convex/concave landform, embankment or other terrain feature.
    # The core classifier is responsible for terrain-vs-vegetation separation;
    # anthropogenic-object cleanup belongs in an optional downstream DEM module.

    # Final multi-pass terrain swipe.  The recovery routine is curvature-aware,
    # uses slope-normal residuals, and promotes only actually observed low-layer
    # returns supported by nearby trusted terrain.  Multiple conservative passes
    # let support propagate inward through a large systematic void instead of
    # repairing only its outermost cell ring.
    _sc_cell=float(cfg.get('surface_consensus_cell_m', 0.60 if sm == 'ULS' else 1.00))
    _sc_workspace=build_surface_consensus_workspace(xw,yw,zw,_sc_cell) if sm in {'ALS','ULS'} else None
    ground_mask, final_swipe_qc = recover_surface_false_negatives(
        x=xw, y=yw, z=zw, ground_mask=ground_mask,
        sensor_mode=sm, cfg=cfg, workspace=_sc_workspace,
    )
    if show_progress and int(final_swipe_qc.get('recovered_points', 0)) > 0:
        log_info(
            'FINAL-TERRAIN-SWIPE '
            f"passes={final_swipe_qc.get('passes_run',0)} | "
            f"candidates={final_swipe_qc.get('candidate_cells',0)} | "
            f"validated={final_swipe_qc.get('validated_cells',0)} | "
            f"recovered={final_swipe_qc.get('recovered_points',0)}"
        )

    # TWO-WAY NEAR-SURFACE SWIPE
    # --------------------------
    # The mature core terrain result is now treated as the baseline.  This final
    # tile-local pass does not alter the vote classifier, slope logic or seam QC.
    # It only reconciles local label inconsistencies around the fitted terrain:
    #   * class-2 -> non-ground when the point is above the local terrain and its
    #     same-elevation neighbourhood is dominated by non-ground returns;
    #   * non-ground -> class-2 only when it lies on the local terrain and its
    #     same-elevation neighbourhood is strongly ground dominated.
    # This catches flying ground in high canopy, understory and low shrubs while
    # keeping steep/curved terrain protected by surface-normal residuals.
    ground_mask, two_way_qc = two_way_surface_classification_swipe(
        x=xw, y=yw, z=zw, ground_mask=ground_mask, sensor_mode=sm, cfg=cfg,
        workspace=_sc_workspace,
    )
    # Release the tile-local QC index before LAS serialization.
    _sc_workspace=None
    if show_progress and (int(two_way_qc.get('demoted_points', 0)) > 0 or int(two_way_qc.get('promoted_points', 0)) > 0):
        log_info(
            'TWO-WAY-SURFACE-SWIPE '
            f"ground_candidates={two_way_qc.get('candidate_ground_points',0)} | "
            f"nonground_candidates={two_way_qc.get('candidate_nonground_points',0)} | "
            f"demoted={two_way_qc.get('demoted_points',0)} | "
            f"promoted={two_way_qc.get('promoted_points',0)} | "
            f"surface_impurity_cells={two_way_qc.get('surface_impurity_candidate_cells',0)} | "
            f"surface_impurity_confirmed={two_way_qc.get('surface_impurity_confirmed_cells',0)} | "
            f"surface_impurity_demoted={two_way_qc.get('surface_impurity_demoted_points',0)}"
        )

    # ALS POST-FINAL SURFACE COMPLETION V5
    # ------------------------------------
    # Runs AFTER all existing final terrain/two-way QC and immediately
    # BEFORE Classification is serialized. Promotion only.
    ground_mask, postfinal_qc = recover_postfinal_surface(
        x=xw,
        y=yw,
        z=zw,
        ground_mask=ground_mask,
        sensor_mode=sm,
        cfg=cfg,
        return_report=True,
    )
    if show_progress and sm != "TLS":
        log_info(
            'POSTFINAL-SURFACE '
            f"before={postfinal_qc.get('ground_before',0)} | "
            f"candidates={postfinal_qc.get('candidate_points',0)} | "
            f"recovered={postfinal_qc.get('recovered_points',0)} | "
            f"cells={postfinal_qc.get('recovered_cells',0)} | "
            f"after={postfinal_qc.get('ground_after',0)}"
        )

    # ALS STATISTICAL GROUND-SHEET CLEANER V9
    # DEMOTION ONLY: 1 m raw XYZ statistics + lowest current-ground support.
    ground_mask, statsheet_qc = clean_statistical_ground_sheet(
        x=xw, y=yw, z=zw, ground_mask=ground_mask, sensor_mode=sm, cfg=cfg, return_report=True,
    )
    if show_progress and sm != "TLS":
        log_info(
            'POSTFINAL-STATSHEET '
            f"before={statsheet_qc.get('ground_before',0)} | "
            f"demoted={statsheet_qc.get('demoted_points',0)} | "
            f"per_iter={statsheet_qc.get('demoted_per_iteration',[])} | "
            f"thin={statsheet_qc.get('thin_cells',0)} | "
            f"moderate={statsheet_qc.get('moderate_cells',0)} | "
            f"complex={statsheet_qc.get('complex_cells',0)} | "
            f"after={statsheet_qc.get('ground_after',0)}"
        )


    # ALS FACET + 3-D SUPPORT CONSISTENCY GUARD V11
    # -------------------------------------------------
    # Point-cloud native, ALS-only, DEMOTION-only.
    # 1) current-ground local facet
    # 2) coherent lower-ground facet and slope-invariant plane separation
    # 3) counterpart non-ground facet + non-ground dominance
    # 4) weak downward ground connectivity
    if sm != "TLS":
        ground_mask, facet3d_qc = clean_als_facet_support_consistency(
            x=xw,
            y=yw,
            z=zw,
            ground_mask=ground_mask,
            sensor_mode=sm,
            cfg=cfg,
            return_report=True,
        )
        if show_progress:
            _iters = facet3d_qc.get("iterations", [])
            _txt = "; ".join(
                (
                    f"iter={r.get('iteration', 0)} "
                    f"zrange={r.get('zrange_threshold_m', 0.0):.3f} "
                    f"flagged_cells={r.get('flagged_cells', 0)} "
                    f"candidates={r.get('candidate_ground_points', 0)} "
                    f"residual={r.get('sheet_residual_candidates', 0)} "
                    f"isolated={r.get('spatially_isolated', 0)} "
                    f"lower_supported={r.get('lower_sheet_supported', 0)} "
                    f"forced={r.get('forced_sparse_residuals', 0)} "
                    f"demoted={r.get('reclassified_to_nonground', 0)}"
                )
                for r in _iters
            )
            log_info(
                "POSTFINAL-FACET3D "
                f"before={facet3d_qc.get('ground_before',0)} | "
                f"demoted={facet3d_qc.get('reclassified_to_nonground',0)} | "
                f"after={facet3d_qc.get('ground_after',0)} | "
                f"{_txt}"
            )

    # ALS GROUND-ONLY VERTICAL PROFILE GUARD V1
    # ------------------------------------------
    # Final ALS-only, G->NG-only contamination cleanup.
    # This stage uses ONLY current ground-labelled points.
    # It builds vertical point-frequency distributions inside 5/7.5/10 m XY
    # footprints, protects the supported lowest ground mode, and flags elevated
    # secondary ground modes/spikes. Where the true low mode is absent because
    # of canopy occlusion, terrain elevation is propagated from neighboring
    # trusted ground cells using a small local plane/IDW prediction.
    ground_mask, ground_profile_qc = refine_als_ground_vertical_profile(
        x=xw,
        y=yw,
        z=zw,
        ground_mask=ground_mask,
        sensor_mode=sm,
        cfg=cfg,
        return_report=True,
    )
    if show_progress and sm != "TLS":
        log_info(
            "POSTFINAL-GROUND-PROFILE "
            f"before={ground_profile_qc.get('ground_before',0)} | "
            f"scales={ground_profile_qc.get('scales_m',[])} | "
            f"trusted={ground_profile_qc.get('trusted_cells_per_scale',[])} | "
            f"unsupported={ground_profile_qc.get('unsupported_cells_per_scale',[])} | "
            f"spikes={ground_profile_qc.get('spike_candidates_per_scale',[])} | "
            f"consensus={ground_profile_qc.get('consensus_candidates',0)} | "
            f"hard={ground_profile_qc.get('hard_candidates',0)} | "
            f"demoted={ground_profile_qc.get('demoted_points',0)} | "
            f"after={ground_profile_qc.get('ground_after',0)}"
        )

    # ========================================================
    # ALS POINT-FIRST DETACHED GROUND GUARD V1
    # ========================================================
    # Runs before the raster/radial V4.1 guard.
    # Detects suspended class-2 points directly and therefore
    # does not require a slope-anomaly seed.
    ground_mask, detached_ground_qc = apply_detached_ground_guard(
        x=xw,
        y=yw,
        z=zw,
        ground_mask=ground_mask,
        sensor_mode=sm,
        cfg=cfg,
        return_report=True,
    )

    if show_progress and sm != "TLS":
        log_info(
            "POSTFINAL-DETACHED-GROUND "
            f"before={detached_ground_qc.get('ground_before', 0)} | "
            f"screen_cells={detached_ground_qc.get('screen_cells', 0)} | "
            f"modeled_cells={detached_ground_qc.get('modeled_cells', 0)} | "
            f"candidates={detached_ground_qc.get('candidate_points', 0)} | "
            f"demoted={detached_ground_qc.get('demoted_points', 0)} | "
            f"after={detached_ground_qc.get('ground_after', 0)}"
        )

    # ========================================================
    # ALS FINAL SURFACE IMPURITY GUARD V4.1
    # ========================================================
    # Final demotion-only surface contamination cleanup.
    # This is deliberately the last classification refinement
    # before Classification is serialized.
    ground_mask, surface_impurity_qc = apply_surface_impurity_guard(
        x=xw,
        y=yw,
        z=zw,
        ground_mask=ground_mask,
        sensor_mode=sm,
        cfg=cfg,
        return_report=True,
    )

    if show_progress and sm != "TLS":
        log_info(
            "POSTFINAL-SURFACE-IMPURITY "
            f"before={surface_impurity_qc.get('ground_before', 0)} | "
            f"components={surface_impurity_qc.get('candidate_components', 0)} | "
            f"modeled={surface_impurity_qc.get('modeled_components', 0)} | "
            f"candidates={surface_impurity_qc.get('candidate_points', 0)} | "
            f"demoted={surface_impurity_qc.get('demoted_points', 0)} | "
            f"after={surface_impurity_qc.get('ground_after', 0)}"
        )

    classification = np.ones(N, dtype=np.uint8)
    # ========================================================
    # ========================================================
    # ALS TERRAIN BLOB / CANOPY IMPOSTOR GUARD V1
    # ========================================================
    # Final raster-first / point-confirmed cleanup.
    #
    # Independent candidate evidence:
    #   * 0.5 m MAX-Z DEM positive prominence
    #   * 1.5 / 3 / 5 m multiscale response
    #   * slope EXCESS relative to surrounding terrain
    #   * robust local Z outlier
    #   * weak ground support / void
    #   * connected-component morphology
    #
    # Final exact-point confirmation:
    #   * leave-component-out 5 m plane/quadratic terrain
    #   * terrain-normal residual
    #   * radial return to terrain
    #   * neighbor slope/Z disagreement
    #   * nearby non-ground canopy-layer consistency
    #
    # DEMOTION ONLY.
    ground_mask, terrain_blob_qc = apply_terrain_blob_guard(
        x=xw,
        y=yw,
        z=zw,
        ground_mask=ground_mask,
        sensor_mode=sm,
        cfg=cfg,
        return_report=True,
    )

    # ========================================================
    # VERTICAL SUPPORT FINAL GUARD V2.1
    # ========================================================
    # The raw 5 m x 5 m x 1 m profile proposes suspicious upper-ground
    # locations. Surrounding 3-D terrain geometry independently confirms
    # each demotion. Handles both smaller patches and larger upper blobs.
    ground_mask, vertical_support_qc = apply_vertical_support_final_guard(
        x=xw,
        y=yw,
        z=zw,
        ground_mask=ground_mask,
        sensor_mode=sm,
        cfg=cfg,
        return_report=True,
    )

    if show_progress and sm != "TLS":
        log_info(
            "POSTFINAL-VERTICAL-SUPPORT "
            f"enabled={vertical_support_qc.get('enabled',False)} | "
            f"cells={vertical_support_qc.get('support_cells',0)} | "
            f"weak_cells={vertical_support_qc.get('weak_cells',0)} | "
            f"canopy_cells={vertical_support_qc.get('canopy_cells',0)} | "
            f"flagged_cells={vertical_support_qc.get('flagged_cells',0)} | "
            f"candidates={vertical_support_qc.get('candidate_points',0)} | "
            f"small={vertical_support_qc.get('small_candidates',0)} | "
            f"large={vertical_support_qc.get('large_candidates',0)} | "
            f"modeled={vertical_support_qc.get('modeled_points',0)} | "
            f"demoted={vertical_support_qc.get('demoted_points',0)} | "
            f"demoted_small={vertical_support_qc.get('demoted_small',0)} | "
            f"demoted_large={vertical_support_qc.get('demoted_large',0)} | "
            f"median_gap={vertical_support_qc.get('median_gap_m',0.0):.3f} | "
            f"max_gap={vertical_support_qc.get('max_gap_m',0.0):.3f} | "
            f"after={vertical_support_qc.get('ground_after',0)}"
        )

    # ========================================================
    # TLS HARD AIRBORNE / CANOPY-LEAK GUARD
    # ========================================================
    # Reuses only the proven point-first ALS airborne detector.
    # The broader ALS/ULS raster/blob pipeline remains unchanged.
    # Strictly demotion-only.
    tls_airborne_qc = {
        "enabled": False,
        "ground_before": int(ground_mask.sum()),
        "ground_after": int(ground_mask.sum()),
        "screen_candidates": 0,
        "modeled_candidates": 0,
        "confirmed_candidates": 0,
        "demoted_points": 0,
    }

    if (
        sm == "TLS"
        and bool(
            cfg.get(
                "tls_hard_airborne_guard_enabled",
                False,
            )
        )
    ):
        _tls_before_airborne = ground_mask.copy()

        ground_mask, tls_airborne_qc = (
            apply_hard_airborne_ground_guard(
                x=xw,
                y=yw,
                z=zw,
                ground_mask=ground_mask,
                sensor_mode="TLS",
                cfg=cfg,
                return_report=True,
            )
        )

        if np.any(
            ground_mask
            & ~_tls_before_airborne
        ):
            raise RuntimeError(
                "TLS airborne guard recruited new ground."
            )

        if show_progress:
            log_info(
                "TLS-FINAL-AIRBORNE "
                f"before={int(_tls_before_airborne.sum())} | "
                f"screen={tls_airborne_qc.get('screen_candidates',0)} | "
                f"modeled={tls_airborne_qc.get('modeled_candidates',0)} | "
                f"confirmed={tls_airborne_qc.get('confirmed_candidates',0)} | "
                f"demoted={tls_airborne_qc.get('demoted_points',0)} | "
                f"after={int(ground_mask.sum())}"
            )

    if show_progress and sm != "TLS":
        log_info(
            "POSTFINAL-TERRAIN-BLOB "
            f"before={terrain_blob_qc.get('ground_before',0)} | "
            f"components={terrain_blob_qc.get('candidate_components',0)} | "
            f"modeled={terrain_blob_qc.get('modeled_components',0)} | "
            f"validated={terrain_blob_qc.get('validated_components',0)} | "
            f"candidates={terrain_blob_qc.get('candidate_points',0)} | "
            f"demoted={terrain_blob_qc.get('demoted_points',0)} | "
            f"after={terrain_blob_qc.get('ground_after',0)}"
        )

        _nout = terrain_blob_qc.get(
            "neighbor_outlier",
            {},
        )

        log_info(
            "POSTFINAL-NEIGHBOR-OUTLIER "
            f"iterations={_nout.get('iterations_run',0)} | "
            f"tested={_nout.get('tested',0)} | "
            f"candidates={_nout.get('candidates',0)} | "
            f"modeled={_nout.get('modeled',0)} | "
            f"confirmed={_nout.get('confirmed',0)} | "
            f"demoted={_nout.get('demoted_points',0)} | "
            f"max_ratio={_nout.get('max_ratio',0.0):.3f} | "
            f"max_gap={_nout.get('max_gap_m',0.0):.3f} | "
            f"max_rn={_nout.get('max_rn',0.0):.3f}"
        )

        for _nop in _nout.get("passes", []):
            log_info(
                "POSTFINAL-NEIGHBOR-OUTLIER-PASS "
                f"pass={_nop.get('iteration',0)} | "
                f"before={_nop.get('before',0)} | "
                f"cand={_nop.get('candidates',0)} | "
                f"model={_nop.get('modeled',0)} | "
                f"confirm={_nop.get('confirmed',0)} | "
                f"demote={_nop.get('demoted',0)} | "
                f"ratio={_nop.get('max_ratio',0.0):.3f} | "
                f"gap={_nop.get('max_gap_m',0.0):.3f} | "
                f"rn={_nop.get('max_rn',0.0):.3f} | "
                f"after={_nop.get('after',0)}"
            )

        _crg = terrain_blob_qc.get(
            "canopy_ridge_authorization",
            {},
        )

        if _crg:
            log_info(
                "POSTFINAL-CANOPY-RIDGE-GATE "
                f"attempted={_crg.get('attempted_demotions',0)} | "
                f"components={_crg.get('components',0)} | "
                f"ridge_keep={_crg.get('ridge_protected_components',0)} | "
                f"no_canopy_keep={_crg.get('no_canopy_components',0)} | "
                f"canopy_confirmed={_crg.get('canopy_confirmed_components',0)} | "
                f"restored={_crg.get('restored_points',0)} | "
                f"authorized={_crg.get('authorized_demotions',0)}"
            )


        _airborne = terrain_blob_qc.get(
            "airborne",
            {},
        )

        log_info(
            "POSTFINAL-AIRBORNE "
            f"screen={_airborne.get('screen_candidates',0)} | "
            f"modeled={_airborne.get('modeled_candidates',0)} | "
            f"confirmed={_airborne.get('confirmed_candidates',0)} | "
            f"demoted={_airborne.get('demoted_points',0)} | "
            f"max_rn={_airborne.get('max_residual_m',0.0):.3f} | "
            f"median_rn={_airborne.get('median_residual_m',0.0):.3f} | "
            f"max_screen_gap={_airborne.get('max_screen_gap_m',0.0):.3f}"
        )

        for _tb in terrain_blob_qc.get("passes", []):
            log_info(
                "POSTFINAL-TERRAIN-BLOB-PASS "
                f"pass={_tb.get('pass',0)} | "
                f"occupied={_tb.get('occupied_cells',0)} | "
                f"p1={_tb.get('prominence_1',0)} | "
                f"p2={_tb.get('prominence_2',0)} | "
                f"p3={_tb.get('prominence_3',0)} | "
                f"strong={_tb.get('strong_prominence',0)} | "
                f"slope={_tb.get('slope_excess',0)} | "
                f"outlier={_tb.get('local_outliers',0)} | "
                f"void={_tb.get('void_support',0)} | "
                f"seed={_tb.get('seed_cells',0)} | "
                f"components={_tb.get('components',0)} | "
                f"modeled={_tb.get('modeled_components',0)} | "
                f"validated={_tb.get('validated_components',0)} | "
                f"candidates={_tb.get('candidate_points',0)} | "
                f"demoted={_tb.get('demoted_points',0)}"
            )

    # ================================================================
    # FINAL NORMAL-SPACE OFFSET SURFACE VOTING
    # ================================================================
    # Hidden low-density channel: automatically active only below 5 pts/m2.
    # Existing FAST-GC is completed first; this stage is DEMOTION ONLY.
    ground_mask, final_surface_vote_qc = apply_final_surface_vote(
        x=xw, y=yw, z=zw, ground_mask=ground_mask,
        sensor_mode=sm, cfg=cfg, return_report=True,
    )
    if show_progress and sm != "TLS":
        _ld = bool(final_surface_vote_qc.get('activated', False))
        log_info(
            'POSTFINAL-SURFACE-VOTE '
            f"density={final_surface_vote_qc.get('density_pts_m2',0.0):.3f} | "
            f"low_density={_ld} | "
            f"trigger_lt={final_surface_vote_qc.get('density_trigger_pts_m2',5.0):.3f} | "
            f"passes={final_surface_vote_qc.get('passes',0)} | "
            f"patches={final_surface_vote_qc.get('patches_tested',0)} | "
            f"modeled={final_surface_vote_qc.get('patches_modeled',0)} | "
            f"voted_points={final_surface_vote_qc.get('points_with_votes',0)} | "
            f"required_votes={final_surface_vote_qc.get('required_votes',0)} | "
            f"max_votes={final_surface_vote_qc.get('max_votes',0)} | "
            f"candidate_cells={final_surface_vote_qc.get('candidate_cells',0)} | "
            f"candidates={final_surface_vote_qc.get('candidate_points',0)} | "
            f"demoted={final_surface_vote_qc.get('demoted_points',0)} | "
            f"median_r={final_surface_vote_qc.get('median_candidate_residual_m',0.0):.3f}"
        )
    # ================================================================
    # D2 REVERSE MULTISCALE CONTEXT VOTING
    # ================================================================
    # VERY_LOW (<3 pts/m2) only. Existing forward Z voting is retained.
    # Reverse audit uses external Z prominence + slope distribution + normals.
    # Genuine coarse terrain complexity can veto removal. DEMOTION ONLY.
    ground_mask, d2_reverse_qc = apply_d2_reverse_context_vote(
        x=xw, y=yw, z=zw, ground_mask=ground_mask,
        sensor_mode=sm, cfg=cfg, return_report=True,
    )
    if show_progress and sm != "TLS":
        log_info(
            'D2-REVERSE-CONTEXT '
            f"density={d2_reverse_qc.get('density_pts_m2',0.0):.3f} | "
            f"activated={d2_reverse_qc.get('activated',False)} | "
            f"components={d2_reverse_qc.get('candidate_components',0)} | "
            f"audited={d2_reverse_qc.get('audited_components',0)} | "
            f"approved={d2_reverse_qc.get('approved_components',0)} | "
            f"terrain_veto={d2_reverse_qc.get('terrain_veto_components',0)} | "
            f"z_votes={d2_reverse_qc.get('z_votes',0)} | "
            f"slope_votes={d2_reverse_qc.get('slope_votes',0)} | "
            f"normal_votes={d2_reverse_qc.get('normal_votes',0)} | "
            f"demoted={d2_reverse_qc.get('demoted_points',0)}"
        )
    # BEGIN D2 POST-REVERSE NORMAL/D6 POLISH V2.6B
    # V2.5 VERY_LOW data only. Reuse the installed NORMAL/D6 one-pass route
    # after reverse Z+slope+normal cleanup. The real cfg is never modified.
    d2_polish_real_density = float(d2_reverse_qc.get("density_pts_m2", 999.0))
    if bool(d2_reverse_qc.get("activated", False)) and d2_polish_real_density < 3.0:
        d2_polish_cfg = dict(cfg)
        d2_polish_cfg["adaptive_support_dataset_density_pts_m2"] = 6.0
        d2_polish_cfg["dataset_density_pts_m2"] = 6.0
        d2_polish_cfg["density_pts_m2"] = 6.0

        ground_mask, d2_polish_qc = apply_final_surface_vote(
            x=xw, y=yw, z=zw, ground_mask=ground_mask,
            sensor_mode=sm, cfg=d2_polish_cfg, return_report=True,
        )

        if show_progress and sm != "TLS":
            log_info(
                "D2-POSTREVERSE-POLISH "
                f"real_density={d2_polish_real_density:.3f} | "
                f"forced_density={d2_polish_qc.get('density_pts_m2',0.0):.3f} | "
                f"regime={d2_polish_qc.get('density_regime','?')} | "
                f"iterations={d2_polish_qc.get('iterations_completed',0)} | "
                f"demoted={d2_polish_qc.get('demoted_points',0)} | "
                f"after={d2_polish_qc.get('ground_after',0)}"
            )
    elif show_progress and sm != "TLS":
        log_info(
            "D2-POSTREVERSE-POLISH "
            f"real_density={d2_polish_real_density:.3f} | skipped=True"
        )
    # END D2 POST-REVERSE NORMAL/D6 POLISH V2.6B
    # ================================================================
    # TLS FINAL HIGH-CONFIDENCE TERRAIN VOTE
    # ================================================================
    # This is intentionally the LAST TLS geometry operation.
    #
    # It builds a second robust TLS terrain surface using ONLY points
    # that remain classed as ground after all previous TLS/common QC.
    # Classification against that surface is intersected with the
    # incoming mask:
    #
    #     final = incoming_ground & voted_ground
    #
    # Therefore this stage is strictly DEMOTION ONLY. No subsequent
    # recovery stage is allowed to add class-2 points back.
    tls_final_vote_qc = {
        "enabled": False,
        "reason": "not_tls",
        "ground_before": int(np.count_nonzero(ground_mask)),
        "ground_after": int(np.count_nonzero(ground_mask)),
        "demoted_points": 0,
        "support_points": int(np.count_nonzero(ground_mask)),
    }

    if sm == "TLS":
        _tls_final_before = np.asarray(
            ground_mask,
            dtype=bool,
        ).copy()

        _tls_final_result = refine_tls_final_ground_vote(
            x=xw,
            y=yw,
            z=zw,
            ground_mask=_tls_final_before,
            cfg=cfg,
        )

        ground_mask = np.asarray(
            _tls_final_result.final_ground,
            dtype=bool,
        )

        if np.any(
            ground_mask
            & ~_tls_final_before
        ):
            raise RuntimeError(
                "TLS final vote recruited new ground."
            )

        if int(np.count_nonzero(ground_mask)) > int(
            np.count_nonzero(_tls_final_before)
        ):
            raise RuntimeError(
                "TLS final vote increased ground count."
            )

        tls_final_vote_qc = {
            "enabled": bool(_tls_final_result.enabled),
            "reason": str(_tls_final_result.reason),
            "ground_before": int(_tls_final_result.ground_before),
            "ground_after": int(_tls_final_result.ground_after),
            "demoted_points": int(_tls_final_result.demoted_points),
            "support_points": int(_tls_final_result.support_points),
        }

        if show_progress:
            log_info(
                "TLS-FINAL-VOTE "
                f"enabled={tls_final_vote_qc['enabled']} | "
                f"reason={tls_final_vote_qc['reason']} | "
                f"support={tls_final_vote_qc['support_points']} | "
                f"before={tls_final_vote_qc['ground_before']} | "
                f"demoted={tls_final_vote_qc['demoted_points']} | "
                f"after={tls_final_vote_qc['ground_after']}"
            )

    classification[np.flatnonzero(keep)[m1][ground_mask]] = 2

    # TLS diagnostic output is opt-in and separate from production output.
    if sm == "TLS" and bool(cfg.get("tls_write_diagnostics", False)):
        tls_diag_path = os.path.splitext(out_path)[0] + "_TLS_DIAG.las"

        _write_tls_diagnostic_cloud(
            las,
            tls_diag_path,
            classification,
            work_indices=np.flatnonzero(keep)[m1],
            xw=xw,
            yw=yw,
            zw=zw,
            surf_z=surf_z,
            sx0=sx0,
            sy0=sy0,
            cell=float(vcfg.cell),
            diagnostics=tls_surface_diagnostics,
        )

        if show_progress:
            log_info(
                f"TLS-DIAG {tls_diag_path}"
            )

    _write_full_cloud_with_classification(
        las,
        out_path,
        classification,
    )
    bar.update(1)
    bar.close()


def classify_ground_path(
    in_path: str,
    out_dir: str,
    sensor_mode: str,
    show_progress: bool = False,
    *,
    tile_support_stats: dict[str, float] | None = None,
    dataset_support_stats: dict[str, float] | None = None,
    adaptive: bool = True,
) -> str:
    if adaptive and (tile_support_stats is None and dataset_support_stats is None):
        tile_support_stats, dataset_support_stats = _adaptive_support_context_for_input(in_path)

    cfg = sensor_defaults(
        sensor_mode,
        tile_support_stats=tile_support_stats,
        dataset_support_stats=dataset_support_stats,
        adaptive=adaptive,
    )

    # Reuse the immutable 5 m XY x 1 m Z support product created during
    # preprocessing.  This lookup is tile-size independent.
    _vs_manifest_fp = _find_tile_manifest_for_path(in_path)
    if _vs_manifest_fp is not None:
        _vs_manifest = _load_manifest_json(_vs_manifest_fp)
        if _vs_manifest is not None:
            _vs_tile = _match_tile_record(in_path, _vs_manifest)
            if _vs_tile is not None:
                _vs_meta = _vs_tile.get("vertical_support") or {}
                _vs_file = _vs_meta.get("support_file")
                if _vs_file:
                    cfg["vertical_support_file"] = str(_vs_file)
                    cfg["vertical_support_final_guard_enabled"] = True

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(in_path))[0]
    out_path = os.path.join(out_dir, f"{base}.las")
    classify_ground_file(in_path, out_path, cfg, show_progress=show_progress)
    return out_path


def _classify_ground_path_resume(
    in_path: str,
    out_dir: str,
    sensor_mode: str,
    *,
    skip_existing: bool = False,
    overwrite: bool = False,
    show_progress: bool = False,
):
    """Restart-safe wrapper used by folder/tile processing.

    Existing classified LAS outputs are skipped only when ``skip_existing`` is
    explicitly requested and ``overwrite`` is false.  The public
    ``classify_ground_path`` return contract remains unchanged.
    """
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(in_path))[0]
    out_path = os.path.join(out_dir, f"{base}.las")

    if skip_existing and not overwrite and os.path.isfile(out_path):
        return {
            "status": "skipped",
            "tile": in_path,
            "output": out_path,
            "reason": "classified output exists",
        }

    return classify_ground_path(
        in_path,
        out_dir,
        sensor_mode,
        show_progress=show_progress,
    )


def list_classified_files(classified_root: str | os.PathLike[str]) -> list[str]:
    root = Path(classified_root)
    if root.is_file():
        return [str(root)]
    files = [str(p) for p in root.rglob("*.las")]
    files.sort()
    return files


def _run_phase(
    items: list[str],
    desc: str,
    sensor_mode: str,
    worker: Callable[[str], object],
    *,
    n_jobs: int = 1,
    joblib_backend: str = "loky",
    joblib_batch_size: int | str = "auto",
    joblib_pre_dispatch: str | int = "2*n_jobs",
    source: str | None = None,
) -> dict[str, list[object]]:
    if not items:
        log_info(f"{desc}: no input tiles found.")
        return {"ok": [], "skipped": [], "failed": []}

    summary = run_stage(
        stage_name=desc,
        items=items,
        func=worker,
        n_jobs=n_jobs,
        backend=joblib_backend,
        batch_size=joblib_batch_size,
        pre_dispatch=joblib_pre_dispatch,
        source=source,
        unit="tile",
        show_banner=True,
        show_progress=True,
    )

    out: dict[str, list[object]] = {"ok": [], "skipped": [], "failed": []}
    for rec in summary.records:
        if rec.status == "ok":
            out["ok"].append(rec.result)
        elif rec.status == "skipped":
            if isinstance(rec.result, dict):
                out["skipped"].append(rec.result)
            else:
                out["skipped"].append(
                    {"status": "skipped", "tile": rec.name, "reason": "worker returned skipped"}
                )
        else:
            reason = rec.error
            if isinstance(rec.result, dict):
                reason = rec.result.get("reason", reason)
            out["failed"].append(
                {
                    "status": "failed",
                    "tile": rec.name,
                    "reason": reason or "worker failed",
                }
            )


    if out["failed"]:
        print(f"[ERROR] {desc} failed tiles:")
        for rec in out["failed"]:
            tile_name = Path(str(rec.get("tile", ""))).name
            reason = str(rec.get("reason", "failed")).splitlines()[0]
            print(f"  - {tile_name}: {reason}")
        raise RuntimeError(f"{desc} failed for {len(out['failed'])} tile(s).")

    return out


def derive_products_from_classified_root(
    classified_root: str | os.PathLike[str],
    out_root: str | os.PathLike[str],
    products: Iterable[str] | None,
    grid_res: float = 0.5,
    dem_method: str = "min",
    *,
    n_jobs: int = 1,
    joblib_backend: str = "loky",
    joblib_batch_size: int | str = "auto",
    joblib_pre_dispatch: str | int = "2*n_jobs",
) -> str:
    requested_products, compute_products = _resolve_products(products)
    requested_products.discard(PRODUCT_GC)
    compute_products.discard(PRODUCT_GC)
    requested_products.discard(PRODUCT_DSM)
    compute_products.discard(PRODUCT_DSM)
    requested_products.discard(PRODUCT_CHM)
    compute_products.discard(PRODUCT_CHM)
    requested_products.discard(PRODUCT_TERRAIN)
    compute_products.discard(PRODUCT_TERRAIN)
    requested_products.discard(PRODUCT_CHANGE)
    compute_products.discard(PRODUCT_CHANGE)
    requested_products.discard(PRODUCT_ITD)
    compute_products.discard(PRODUCT_ITD)

    if not requested_products:
        return str(out_root)

    _require_rasterio_for_products(compute_products)

    classified_files = list_classified_files(classified_root)
    if not classified_files:
        raise FileNotFoundError(f"No classified LAS files found in: {classified_root}")

    product_dirs = _make_product_dirs(str(out_root))

    derive_dem = PRODUCT_DEM in requested_products
    derive_norm = PRODUCT_NORMALIZED in requested_products

    if derive_dem and derive_norm:
        shared_summary = _run_phase(
            classified_files,
            desc="FAST-GC derive DEM+NORMALIZED",
            sensor_mode="POST",
            worker=lambda gc_fp: _write_dem_and_normalized_shared(
                gc_fp, product_dirs, grid_res, dem_method=dem_method
            ),
            n_jobs=n_jobs,
            joblib_backend=joblib_backend,
            joblib_batch_size=joblib_batch_size,
            joblib_pre_dispatch=joblib_pre_dispatch,
            source=str(classified_root),
        )
        if not shared_summary["ok"] and shared_summary["skipped"]:
            raise RuntimeError(
                "FAST-GC derive DEM+NORMALIZED skipped all tiles because no valid ground points were available."
            )
    else:
        if derive_dem:
            dem_summary = _run_phase(
                classified_files,
                desc="FAST-GC derive DEM",
                sensor_mode="POST",
                worker=lambda gc_fp: _write_dem(gc_fp, product_dirs, grid_res, dem_method=dem_method),
                n_jobs=n_jobs,
                joblib_backend=joblib_backend,
                joblib_batch_size=joblib_batch_size,
                joblib_pre_dispatch=joblib_pre_dispatch,
                source=str(classified_root),
            )
            if not dem_summary["ok"] and dem_summary["skipped"]:
                raise RuntimeError("FAST-GC derive DEM skipped all tiles because no valid ground points were available.")

        if derive_norm:
            norm_summary = _run_phase(
                classified_files,
                desc="FAST-GC derive NORMALIZED",
                sensor_mode="POST",
                worker=lambda gc_fp: _write_normalized(gc_fp, product_dirs, grid_res, dem_method=dem_method),
                n_jobs=n_jobs,
                joblib_backend=joblib_backend,
                joblib_batch_size=joblib_batch_size,
                joblib_pre_dispatch=joblib_pre_dispatch,
                source=str(classified_root),
            )
            if not norm_summary["ok"] and norm_summary["skipped"]:
                raise RuntimeError("FAST-GC derive NORMALIZED skipped all tiles because no valid DEM could be built.")

    return str(out_root)


def derive_products_from_raw_root(
    raw_root: str | os.PathLike[str],
    out_root: str | os.PathLike[str],
    sensor_mode: str,
    products: Iterable[str] | None,
    grid_res: float = 0.5,
    dsm_method: str = "max",
    recursive: bool = False,
    *,
    n_jobs: int = 1,
    joblib_backend: str = "loky",
    joblib_batch_size: int | str = "auto",
    joblib_pre_dispatch: str | int = "2*n_jobs",
    spikefree_freeze_distance: float | None = None,
    spikefree_insertion_buffer: float | None = None,
) -> str:
    requested_products, compute_products = _resolve_products(products)
    requested_products.intersection_update({PRODUCT_DSM})
    compute_products.intersection_update({PRODUCT_DSM})

    if not requested_products:
        return str(out_root)

    _require_rasterio_for_products(compute_products)

    raw_files = _iter_las_files(str(raw_root), recursive=recursive)
    if not raw_files:
        raise FileNotFoundError(f"No LAS/LAZ files found in: {raw_root}")

    product_dirs = _make_product_dirs(str(out_root))

    if PRODUCT_DSM in requested_products:
        _run_phase(
            raw_files,
            desc=f"FAST-GC raw DSM {sensor_mode}",
            sensor_mode=sensor_mode,
            worker=lambda raw_fp: _write_dsm_from_raw(
                raw_fp,
                product_dirs,
                grid_res,
                dsm_method=dsm_method,
                spikefree_freeze_distance=spikefree_freeze_distance,
                spikefree_insertion_buffer=spikefree_insertion_buffer,
            ),
            n_jobs=n_jobs,
            joblib_backend=joblib_backend,
            joblib_batch_size=joblib_batch_size,
            joblib_pre_dispatch=joblib_pre_dispatch,
            source=str(raw_root),
        )

    return str(out_root)


def process_fastgc_path(
    in_path: str,
    out_dir: str | None,
    sensor_mode: str,
    products: list[str] | None = None,
    grid_res: float = 0.5,
    recursive: bool = False,
    dem_method: str = "min",
    dsm_method: str = "max",
    *,
    n_jobs: int = 1,
    joblib_backend: str = "loky",
    joblib_batch_size: int | str = "auto",
    joblib_pre_dispatch: str | int = "2*n_jobs",
    spikefree_freeze_distance: float | None = None,
    spikefree_insertion_buffer: float | None = None,
    skip_existing: bool = False,
    overwrite: bool = False,
) -> str:
    requested_products, compute_products = _resolve_products(products)
    _require_rasterio_for_products(compute_products)

    files = _iter_las_files(in_path, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No LAS/LAZ files found in: {in_path}")

    out_root = _get_output_root(in_path, out_dir, sensor_mode)
    product_dirs = _make_product_dirs(out_root)

    total_t0 = perf_counter()

    need_classified = bool({PRODUCT_GC, PRODUCT_DEM, PRODUCT_NORMALIZED, PRODUCT_CHM, PRODUCT_TERRAIN} & requested_products)
    need_dsm = PRODUCT_DSM in requested_products

    if Path(in_path).is_file():
        stage_t0 = perf_counter()

        gc_fp = None
        if need_classified:
            gc_result = _classify_ground_path_resume(
                in_path,
                product_dirs[PRODUCT_GC],
                sensor_mode,
                skip_existing=skip_existing,
                overwrite=overwrite,
                show_progress=True,
            )
            if isinstance(gc_result, dict):
                gc_fp = str(gc_result["output"])
                print(f"[SKIP] {Path(in_path).name}: {gc_result['reason']}")
            else:
                gc_fp = str(gc_result)

        need_dem = PRODUCT_DEM in requested_products
        need_norm = PRODUCT_NORMALIZED in requested_products
        if gc_fp is not None and need_dem and need_norm:
            _write_dem_and_normalized_shared(gc_fp, product_dirs, grid_res, dem_method=dem_method)
        else:
            if need_dem and gc_fp is not None:
                _write_dem(gc_fp, product_dirs, grid_res, dem_method=dem_method)
            if need_norm and gc_fp is not None:
                _write_normalized(gc_fp, product_dirs, grid_res, dem_method=dem_method)

        if need_dsm:
            _write_dsm_from_raw(
                in_path,
                product_dirs,
                grid_res,
                dsm_method=dsm_method,
                spikefree_freeze_distance=spikefree_freeze_distance,
                spikefree_insertion_buffer=spikefree_insertion_buffer,
            )

        total_dt = perf_counter() - stage_t0
        log_info(f"[TIME] FAST-GC {sensor_mode} {Path(in_path).name}: total={total_dt:.2f}s")
        return out_root

    classified_files: list[str] = []
    classified_root = product_dirs[PRODUCT_GC]

    if need_classified:
        classified_phase = _run_phase(
            files,
            desc=f"FAST-GC phase 1/3 CLASSIFY {sensor_mode}",
            sensor_mode=sensor_mode,
            worker=lambda fp: _classify_ground_path_resume(
                fp,
                product_dirs[PRODUCT_GC],
                sensor_mode,
                skip_existing=skip_existing,
                overwrite=overwrite,
                show_progress=False,
            ),
            n_jobs=n_jobs,
            joblib_backend=joblib_backend,
            joblib_batch_size=joblib_batch_size,
            joblib_pre_dispatch=joblib_pre_dispatch,
            source=in_path,
        )

        # IMPORTANT FOR RESTARTS:
        # downstream products must see both newly completed and previously
        # completed FAST_GC tiles.  Using only classified_phase["ok"] would
        # discard the skipped-existing tiles from DEM/NORMALIZED processing.
        classified_files = list_classified_files(classified_root)
        if not classified_files:
            raise RuntimeError(
                f"FAST-GC classification produced no usable outputs in: {classified_root}"
            )

        # Finalize buffered-neighbor seam decisions IN THE CLASSIFICATION STAGE.
        # This intentionally occurs before DEM/NORMALIZED/CHM derivation.  Merge
        # is subsequently a pure core-trim + concatenate operation and never
        # changes a class label.
        seam_qc = finalize_fastgc_seams_in_place(out_root)
        if seam_qc.get("applied"):
            log_info(
                "FAST_GC tile-final seam QC: "
                f"tiles_changed={seam_qc.get('tiles_changed', 0)} | "
                f"points_changed={seam_qc.get('points_changed', 0)} | "
                f"report={seam_qc.get('report_path', '')}"
            )
        # Re-list after in-place finalization so all downstream phases consume
        # the finalized tile set, including restart/skipped tiles.
        classified_files = list_classified_files(classified_root)

    need_dem_product = PRODUCT_DEM in requested_products
    need_norm_product = PRODUCT_NORMALIZED in requested_products

    if need_dem_product and need_norm_product:
        shared_phase = _run_phase(
            classified_files,
            desc=f"FAST-GC phase 2/3 DEM+NORMALIZED {sensor_mode}",
            sensor_mode=sensor_mode,
            worker=lambda gc_fp: _write_dem_and_normalized_shared(
                gc_fp, product_dirs, grid_res, dem_method=dem_method
            ),
            n_jobs=n_jobs,
            joblib_backend=joblib_backend,
            joblib_batch_size=joblib_batch_size,
            joblib_pre_dispatch=joblib_pre_dispatch,
            source=str(classified_root),
        )
        if not shared_phase["ok"] and shared_phase["skipped"]:
            raise RuntimeError(f"FAST-GC phase 2/3 DEM+NORMALIZED {sensor_mode} skipped all tiles.")
    else:
        if need_dem_product:
            dem_phase = _run_phase(
                classified_files,
                desc=f"FAST-GC phase 2/3 DEM {sensor_mode}",
                sensor_mode=sensor_mode,
                worker=lambda gc_fp: _write_dem(gc_fp, product_dirs, grid_res, dem_method=dem_method),
                n_jobs=n_jobs,
                joblib_backend=joblib_backend,
                joblib_batch_size=joblib_batch_size,
                joblib_pre_dispatch=joblib_pre_dispatch,
                source=str(classified_root),
            )
            if not dem_phase["ok"] and dem_phase["skipped"]:
                raise RuntimeError(f"FAST-GC phase 2/3 DEM {sensor_mode} skipped all tiles.")

        if need_norm_product:
            norm_phase = _run_phase(
                classified_files,
                desc=f"FAST-GC phase 3/3 NORMALIZED {sensor_mode}",
                sensor_mode=sensor_mode,
                worker=lambda gc_fp: _write_normalized(gc_fp, product_dirs, grid_res, dem_method=dem_method),
                n_jobs=n_jobs,
                joblib_backend=joblib_backend,
                joblib_batch_size=joblib_batch_size,
                joblib_pre_dispatch=joblib_pre_dispatch,
                source=str(classified_root),
            )
            if not norm_phase["ok"] and norm_phase["skipped"]:
                raise RuntimeError(f"FAST-GC phase 3/3 NORMALIZED {sensor_mode} skipped all tiles.")

    if need_dsm:
        _run_phase(
            files,
            desc=f"FAST-GC raw DSM {sensor_mode}",
            sensor_mode=sensor_mode,
            worker=lambda raw_fp: _write_dsm_from_raw(
                raw_fp,
                product_dirs,
                grid_res,
                dsm_method=dsm_method,
                spikefree_freeze_distance=spikefree_freeze_distance,
                spikefree_insertion_buffer=spikefree_insertion_buffer,
            ),
            n_jobs=n_jobs,
            joblib_backend=joblib_backend,
            joblib_batch_size=joblib_batch_size,
            joblib_pre_dispatch=joblib_pre_dispatch,
            source=in_path,
        )

    total_dt = perf_counter() - total_t0
    print(
        f"[TIME] FAST-GC TOTAL {sensor_mode}: {len(files)} input tiles | "
        f"total={total_dt:.2f}s | avg={total_dt / max(len(files), 1):.2f}s/tile"
    )
    return out_root


__all__ = [
    "PRODUCT_ALL",
    "PRODUCT_GC",
    "PRODUCT_DEM",
    "PRODUCT_NORMALIZED",
    "PRODUCT_DSM",
    "PRODUCT_CHM",
    "PRODUCT_TERRAIN",
    "PRODUCT_CHANGE",
    "PRODUCT_ITD",
    "PRODUCT_STRUCTURE",
    "ALL_PRODUCTS",
    "RASTER_METHOD_CHOICES",
    "classify_ground_file",
    "classify_ground_path",
    "derive_products_from_classified_root",
    "derive_products_from_raw_root",
    "list_classified_files",
    "process_fastgc_path",
]







