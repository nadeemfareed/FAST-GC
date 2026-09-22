from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import laspy
import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - scipy is a FAST-GC dependency
    cKDTree = None


GROUND_CLASS = 2


@dataclass(frozen=True)
class SeamConfig:
    """Conservative merge-time seam reconciliation settings."""

    seam_width_m: float = 1.25
    support_radius_m: float = 2.75
    min_support_total: int = 12
    min_support_each_side: int = 3
    min_side_offset_m: float = 0.20
    base_vertical_tol_m: float = 0.18
    mad_scale: float = 3.0
    max_vertical_tol_m: float = 0.45
    slope_tol_gain: float = 0.08
    recommended_buffer_m: float = 5.0


def _raw_signature(las: laspy.LasData) -> np.ndarray:
    """Return a compact, exact identity signature for duplicated buffered points."""
    names = set(las.point_format.dimension_names)
    dtype_fields: list[tuple[str, Any]] = [
        ("X", np.int64),
        ("Y", np.int64),
        ("Z", np.int64),
    ]
    optional = []
    for name in ("return_number", "number_of_returns", "point_source_id"):
        if name in names:
            optional.append(name)
            dtype_fields.append((name, np.int64))

    sig = np.empty(len(las.points), dtype=np.dtype(dtype_fields))
    sig["X"] = np.asarray(las.X, dtype=np.int64)
    sig["Y"] = np.asarray(las.Y, dtype=np.int64)
    sig["Z"] = np.asarray(las.Z, dtype=np.int64)
    for name in optional:
        sig[name] = np.asarray(getattr(las, name), dtype=np.int64)
    return sig


def _adjacent_pair(a: dict[str, Any], b: dict[str, Any], eps: float = 1e-7):
    """Return seam metadata for tiles sharing a core edge, else None."""
    ax0, ay0, ax1, ay1 = map(float, a["core_bounds"])
    bx0, by0, bx1, by1 = map(float, b["core_bounds"])

    # vertical seam x = const
    if abs(ax1 - bx0) <= eps or abs(bx1 - ax0) <= eps:
        x = ax1 if abs(ax1 - bx0) <= eps else bx1
        lo = max(ay0, by0)
        hi = min(ay1, by1)
        if hi - lo > eps:
            return {"axis": "x", "coord": float(x), "tmin": float(lo), "tmax": float(hi)}

    # horizontal seam y = const
    if abs(ay1 - by0) <= eps or abs(by1 - ay0) <= eps:
        y = ay1 if abs(ay1 - by0) <= eps else by1
        lo = max(ax0, bx0)
        hi = min(ax1, bx1)
        if hi - lo > eps:
            return {"axis": "y", "coord": float(y), "tmin": float(lo), "tmax": float(hi)}

    return None


def _candidate_mask(x: np.ndarray, y: np.ndarray, seam: dict[str, Any], width: float) -> np.ndarray:
    if seam["axis"] == "x":
        return (
            (np.abs(x - seam["coord"]) <= width)
            & (y >= seam["tmin"] - 1e-9)
            & (y <= seam["tmax"] + 1e-9)
        )
    return (
        (np.abs(y - seam["coord"]) <= width)
        & (x >= seam["tmin"] - 1e-9)
        & (x <= seam["tmax"] + 1e-9)
    )


def _fit_local_plane(
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    qx: float,
    qy: float,
    seam: dict[str, Any],
    cfg: SeamConfig,
    tree: Any,
) -> tuple[float, float] | None:
    """Predict terrain z from bilaterally supported, agreed-ground neighbors."""
    if tree is None:
        return None
    ids = tree.query_ball_point([qx, qy], r=cfg.support_radius_m)
    if len(ids) < cfg.min_support_total:
        return None

    ids = np.asarray(ids, dtype=np.int64)
    sx = gx[ids]
    sy = gy[ids]
    sz = gz[ids]

    # Require original agreed ground on both sides of the seam. This is the
    # key safety condition that prevents the merger from inventing terrain.
    if seam["axis"] == "x":
        left = np.count_nonzero(sx < seam["coord"] - cfg.min_side_offset_m)
        right = np.count_nonzero(sx > seam["coord"] + cfg.min_side_offset_m)
    else:
        left = np.count_nonzero(sy < seam["coord"] - cfg.min_side_offset_m)
        right = np.count_nonzero(sy > seam["coord"] + cfg.min_side_offset_m)
    if left < cfg.min_support_each_side or right < cfg.min_support_each_side:
        return None

    # Center coordinates to keep the least-squares system well conditioned for
    # large map coordinates.
    dx = sx - qx
    dy = sy - qy
    A = np.column_stack((dx, dy, np.ones_like(dx)))
    try:
        coef, *_ = np.linalg.lstsq(A, sz, rcond=None)
    except np.linalg.LinAlgError:
        return None

    pred = float(coef[2])
    residual = sz - (A @ coef)
    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med)))
    robust_sigma = 1.4826 * mad
    slope = float(np.hypot(coef[0], coef[1]))

    tol = max(cfg.base_vertical_tol_m, cfg.mad_scale * robust_sigma + cfg.slope_tol_gain * slope)
    tol = min(cfg.max_vertical_tol_m, tol)
    return pred, float(tol)


def _tile_path(processed_root: Path, product: str, tile_name: str) -> Path:
    return processed_root / product / tile_name



def _raw_signature_subset(las: laspy.LasData, idx: np.ndarray) -> np.ndarray:
    """Exact point signatures for only the requested points.

    The old implementation built signatures for an entire dense TLS tile for
    every neighboring pair and then sliced them.  This version constructs
    signatures only for the narrow seam/support subset.
    """
    idx = np.asarray(idx, dtype=np.int64)
    names = set(las.point_format.dimension_names)
    dtype_fields: list[tuple[str, Any]] = [
        ("X", np.int64),
        ("Y", np.int64),
        ("Z", np.int64),
    ]
    optional: list[str] = []
    for name in ("return_number", "number_of_returns", "point_source_id"):
        if name in names:
            optional.append(name)
            dtype_fields.append((name, np.int64))

    sig = np.empty(idx.size, dtype=np.dtype(dtype_fields))
    sig["X"] = np.asarray(las.X, dtype=np.int64)[idx]
    sig["Y"] = np.asarray(las.Y, dtype=np.int64)[idx]
    sig["Z"] = np.asarray(las.Z, dtype=np.int64)[idx]
    for name in optional:
        sig[name] = np.asarray(getattr(las, name), dtype=np.int64)[idx]
    return sig


def _build_neighbor_pairs(tiles: list[dict[str, Any]]):
    pairs: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    tile_seams: dict[str, list[dict[str, Any]]] = {t["tile_name"]: [] for t in tiles}
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            seam = _adjacent_pair(tiles[i], tiles[j])
            if seam is None:
                continue
            a = tiles[i]
            b = tiles[j]
            pairs.append((a, b, seam))
            tile_seams[a["tile_name"]].append(seam)
            tile_seams[b["tile_name"]].append(seam)
    return pairs, tile_seams


def _load_seam_cache(
    tile: dict[str, Any],
    seams: list[dict[str, Any]],
    processed_root: Path,
    product: str,
    width: float,
) -> dict[str, Any]:
    """Read one processed tile once and retain only seam/support strips in RAM."""
    fp = _tile_path(processed_root, product, tile["tile_name"])
    las = laspy.read(fp)

    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)

    union = np.zeros(x.size, dtype=bool)
    for seam in seams:
        union |= _candidate_mask(x, y, seam, width)

    idx = np.flatnonzero(union)
    if idx.size == 0:
        return {
            "x": np.empty(0, dtype=np.float64),
            "y": np.empty(0, dtype=np.float64),
            "z": np.empty(0, dtype=np.float64),
            "cls": np.empty(0, dtype=np.uint8),
            "sig": np.empty(0, dtype=np.dtype([("X", np.int64), ("Y", np.int64), ("Z", np.int64)])),
        }

    return {
        "x": x[idx].copy(),
        "y": y[idx].copy(),
        "z": np.asarray(las.z, dtype=np.float64)[idx].copy(),
        "cls": np.asarray(las.classification, dtype=np.uint8)[idx].copy(),
        "sig": _raw_signature_subset(las, idx),
    }


def build_fastgc_seam_overrides(
    manifest: dict[str, Any],
    processed_root: str | Path,
    *,
    product: str = "FAST_GC",
    cfg: SeamConfig | None = None,
    report_path: str | Path | None = None,
) -> dict[str, dict[bytes, int]]:
    """
    RAM-cached merge-time classification reconciliation for FAST_GC.

    Scientific decisions are identical to the previous seam reconciler.  The
    speed change is architectural only:

      * neighboring pairs are discovered first;
      * every processed tile is read ONCE for seam QC;
      * only narrow seam/support strips are retained in RAM;
      * exact point signatures are built only for those strips;
      * pair comparisons reuse the cached arrays.

    This removes repeated full LAS reads and repeated full-tile signature
    construction, which are especially expensive for dense TLS.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from time import perf_counter
    import os

    from .monster import ProgressDashboard

    cfg = cfg or SeamConfig()
    processed_root = Path(processed_root)
    tiles = [
        t for t in manifest.get("tiles", [])
        if _tile_path(processed_root, product, t["tile_name"]).exists()
    ]
    overrides: dict[str, dict[bytes, int]] = {t["tile_name"]: {} for t in tiles}

    report: dict[str, Any] = {
        "product": product,
        "buffer_m": float(manifest.get("buffer_m") or 0.0),
        "recommended_buffer_m": cfg.recommended_buffer_m,
        "seam_width_m": cfg.seam_width_m,
        "support_radius_m": cfg.support_radius_m,
        "cache_mode": "ram_seam_strips",
        "cache_workers": 0,
        "cached_tiles": 0,
        "cached_points": 0,
        "pairs_checked": 0,
        "duplicate_points_checked": 0,
        "classification_disagreements": 0,
        "overrides_to_ground": 0,
        "overrides_to_nonground": 0,
        "unresolved_disagreements": 0,
        "cache_seconds": 0.0,
        "pair_seconds": 0.0,
        "pairs": [],
    }

    if cKDTree is None or len(tiles) < 2:
        if report_path is not None:
            Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return overrides

    pairs, tile_seams = _build_neighbor_pairs(tiles)
    if not pairs:
        if report_path is not None:
            Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return overrides

    active_tiles = [t for t in tiles if tile_seams.get(t["tile_name"])]
    support_width = max(cfg.seam_width_m, cfg.support_radius_m)

    sensor_mode = str(manifest.get("sensor_mode", "")).upper().strip()
    cpu = os.cpu_count() or 1
    # TLS benefits most from concurrent disk reads / LAS decode because the
    # workstation has abundant RAM.  Keep this modest enough not to thrash disk.
    workers = min(len(active_tiles), 8 if sensor_mode == "TLS" else 4, max(1, cpu))
    workers = max(1, workers)
    report["cache_workers"] = workers

    cache: dict[str, dict[str, Any]] = {}
    total_steps = len(active_tiles) + len(pairs)
    dashboard = ProgressDashboard(
        f"SEAM QC {product}",
        total_steps,
        unit="step",
        enabled=True,
    )

    t_cache = perf_counter()
    try:
        if workers == 1:
            for tile in active_tiles:
                entry = _load_seam_cache(
                    tile,
                    tile_seams[tile["tile_name"]],
                    processed_root,
                    product,
                    support_width,
                )
                cache[tile["tile_name"]] = entry
                report["cached_tiles"] += 1
                report["cached_points"] += int(entry["x"].size)
                dashboard.update(
                    1,
                    current_item=f"cache {tile['tile_name']}",
                    ok_count=report["cached_tiles"],
                    skipped_count=0,
                    failed_count=0,
                )
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {
                    ex.submit(
                        _load_seam_cache,
                        tile,
                        tile_seams[tile["tile_name"]],
                        processed_root,
                        product,
                        support_width,
                    ): tile
                    for tile in active_tiles
                }
                for fut in as_completed(futs):
                    tile = futs[fut]
                    entry = fut.result()
                    cache[tile["tile_name"]] = entry
                    report["cached_tiles"] += 1
                    report["cached_points"] += int(entry["x"].size)
                    dashboard.update(
                        1,
                        current_item=f"cache {tile['tile_name']}",
                        ok_count=report["cached_tiles"],
                        skipped_count=0,
                        failed_count=0,
                    )

        report["cache_seconds"] = float(perf_counter() - t_cache)
        t_pairs = perf_counter()

        for pair_idx, (a, b, seam) in enumerate(pairs, start=1):
            ca_cache = cache.get(a["tile_name"])
            cb_cache = cache.get(b["tile_name"])
            if ca_cache is None or cb_cache is None:
                dashboard.update(
                    1,
                    current_item=f"{a['tile_name']} <> {b['tile_name']}",
                    ok_count=report["pairs_checked"],
                    skipped_count=pair_idx - report["pairs_checked"],
                    failed_count=0,
                )
                continue

            xa = ca_cache["x"]
            ya = ca_cache["y"]
            za = ca_cache["z"]
            xb = cb_cache["x"]
            yb = cb_cache["y"]
            zb = cb_cache["z"]

            ma = _candidate_mask(xa, ya, seam, support_width)
            mb = _candidate_mask(xb, yb, seam, support_width)
            ia = np.flatnonzero(ma)
            ib = np.flatnonzero(mb)
            if ia.size == 0 or ib.size == 0:
                dashboard.update(
                    1,
                    current_item=f"{a['tile_name']} <> {b['tile_name']}",
                    ok_count=report["pairs_checked"],
                    skipped_count=pair_idx - report["pairs_checked"],
                    failed_count=0,
                )
                continue

            sa = ca_cache["sig"][ia]
            sb = cb_cache["sig"][ib]
            common, posa, posb = np.intersect1d(
                sa,
                sb,
                assume_unique=False,
                return_indices=True,
            )
            if common.size == 0:
                dashboard.update(
                    1,
                    current_item=f"{a['tile_name']} <> {b['tile_name']}",
                    ok_count=report["pairs_checked"],
                    skipped_count=pair_idx - report["pairs_checked"],
                    failed_count=0,
                )
                continue

            ia_common = ia[posa]
            ib_common = ib[posb]
            cla = ca_cache["cls"][ia_common]
            clb = cb_cache["cls"][ib_common]

            seam_a = _candidate_mask(
                xa[ia_common],
                ya[ia_common],
                seam,
                cfg.seam_width_m,
            )
            disagreement = (cla != clb) & seam_a

            report["pairs_checked"] += 1
            report["duplicate_points_checked"] += int(common.size)
            report["classification_disagreements"] += int(np.count_nonzero(disagreement))

            pair_report = {
                "tile_a": a["tile_name"],
                "tile_b": b["tile_name"],
                "axis": seam["axis"],
                "coord": seam["coord"],
                "duplicates": int(common.size),
                "disagreements": int(np.count_nonzero(disagreement)),
                "to_ground": 0,
                "to_nonground": 0,
                "unresolved": 0,
            }

            if np.any(disagreement):
                agreed_ground = (cla == GROUND_CLASS) & (clb == GROUND_CLASS)
                if np.count_nonzero(agreed_ground) < cfg.min_support_total:
                    n = int(np.count_nonzero(disagreement))
                    pair_report["unresolved"] = n
                    report["unresolved_disagreements"] += n
                else:
                    gx = xa[ia_common][agreed_ground]
                    gy = ya[ia_common][agreed_ground]
                    gz = za[ia_common][agreed_ground]
                    tree = cKDTree(np.column_stack((gx, gy)))

                    for p in np.flatnonzero(disagreement):
                        qx = float(xa[ia_common[p]])
                        qy = float(ya[ia_common[p]])
                        qz = float(za[ia_common[p]])
                        fit = _fit_local_plane(gx, gy, gz, qx, qy, seam, cfg, tree)
                        if fit is None:
                            pair_report["unresolved"] += 1
                            report["unresolved_disagreements"] += 1
                            continue

                        pred_z, tol = fit
                        new_class = GROUND_CLASS if abs(qz - pred_z) <= tol else 1
                        key = common[p].tobytes()
                        overrides[a["tile_name"]][key] = int(new_class)
                        overrides[b["tile_name"]][key] = int(new_class)

                        if new_class == GROUND_CLASS:
                            pair_report["to_ground"] += 1
                            report["overrides_to_ground"] += 1
                        else:
                            pair_report["to_nonground"] += 1
                            report["overrides_to_nonground"] += 1

            report["pairs"].append(pair_report)
            dashboard.update(
                1,
                current_item=f"{a['tile_name']} <> {b['tile_name']}",
                ok_count=report["pairs_checked"],
                skipped_count=pair_idx - report["pairs_checked"],
                failed_count=0,
            )

        report["pair_seconds"] = float(perf_counter() - t_pairs)
    finally:
        dashboard.close()

    if report_path is not None:
        rp = Path(report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return overrides


def apply_overrides_to_las(
    las: laspy.LasData,
    tile_overrides: dict[bytes, int] | None,
) -> int:
    """Apply precomputed overrides in-place; return number of changed labels."""
    if not tile_overrides:
        return 0

    sig = _raw_signature(las)
    if sig.size == 0:
        return 0

    key_bytes = b"".join(tile_overrides.keys())
    if not key_bytes:
        return 0
    override_sig = np.frombuffer(key_bytes, dtype=sig.dtype)
    override_vals = np.fromiter(
        tile_overrides.values(),
        dtype=np.uint8,
        count=len(tile_overrides),
    )

    _common, idx_las, idx_override = np.intersect1d(
        sig,
        override_sig,
        assume_unique=False,
        return_indices=True,
    )
    if idx_las.size == 0:
        return 0

    cls = np.asarray(las.classification, dtype=np.uint8).copy()
    new_vals = override_vals[idx_override]
    changed_mask = cls[idx_las] != new_vals
    changed = int(np.count_nonzero(changed_mask))
    if changed:
        cls[idx_las[changed_mask]] = new_vals[changed_mask]
        las.classification = cls
    return changed


__all__ = [
    "SeamConfig",
    "apply_overrides_to_las",
    "build_fastgc_seam_overrides",
]
