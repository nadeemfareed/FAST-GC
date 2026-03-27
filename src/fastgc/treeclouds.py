from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import laspy
import numpy as np

from .monster import DEFAULT_BACKEND, log_info, run_stage

PRODUCT_TREECLOUDS = "FAST_TREECLOUDS"

_VECTOR_SUFFIXES = {".shp", ".geojson", ".json"}
_LAS_SUFFIXES = {".las", ".laz"}

try:  # pragma: no cover
    import fiona
except Exception:  # pragma: no cover
    fiona = None

try:  # pragma: no cover
    from shapely.geometry import shape
    from shapely import contains_xy
except Exception:  # pragma: no cover
    shape = None
    contains_xy = None

try:  # pragma: no cover
    from scipy.spatial import cKDTree
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    cKDTree = None
    ndi = None


def _sanitize_name(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in {"_", "-", "."} else "_" for c in str(name))
    return safe.strip("._") or "unnamed"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _existing_output(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return True
    return any(path.iterdir())


def _find_single_merged_las(root: Path, source_name: str) -> Path:
    stems = [f"*_{source_name}.las", f"*_{source_name}.laz"]
    matches: list[Path] = []
    for pat in stems:
        matches.extend(sorted([p for p in root.glob(pat) if p.is_file()]))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Merged {source_name} LAS not found under: {root}")
    raise RuntimeError(f"Multiple merged {source_name} LAS files found under {root}: {[str(p) for p in matches]}")


def _find_crown_vector(root: Path, method: str, variant: str | None) -> tuple[Path, str]:
    itd_root = root / "FAST_ITD" / method / "chm"
    if variant:
        candidate_dir = itd_root / variant / "crowns"
        if candidate_dir.exists():
            files = []
            for suf in ("*.shp", "*.geojson", "*.json"):
                files.extend(sorted(candidate_dir.glob(suf)))
            if len(files) == 1:
                return files[0], variant
            if len(files) > 1:
                shp = [p for p in files if p.suffix.lower() == ".shp"]
                return (shp[0] if shp else files[0]), variant
        candidate_dir = itd_root / variant
        if candidate_dir.exists():
            files = []
            for suf in ("*_crowns.shp", "*_crowns.geojson", "*_crowns.json"):
                files.extend(sorted(candidate_dir.rglob(suf)))
            if files:
                shp = [p for p in files if p.suffix.lower() == ".shp"]
                return (shp[0] if shp else files[0]), variant
        raise FileNotFoundError(f"Crowns vector not found for method={method}, variant={variant} under {root}")

    variants = [p for p in itd_root.iterdir() if p.is_dir()] if itd_root.exists() else []
    found: list[tuple[Path, str]] = []
    for v in variants:
        crown_dir = v / "crowns"
        files = []
        if crown_dir.exists():
            for suf in ("*.shp", "*.geojson", "*.json"):
                files.extend(sorted(crown_dir.glob(suf)))
        if not files:
            for suf in ("*_crowns.shp", "*_crowns.geojson", "*_crowns.json"):
                files.extend(sorted(v.rglob(suf)))
        if files:
            shp = [p for p in files if p.suffix.lower() == ".shp"]
            found.append(((shp[0] if shp else files[0]), v.name))
    if len(found) == 1:
        return found[0]
    if not found:
        raise FileNotFoundError(f"No crowns vectors found for method={method} under {root}")
    raise RuntimeError(f"Multiple crown variants found for method={method} under {root}; please specify source_chm/variant.")


def _load_polygons(vector_path: Path) -> list[dict[str, Any]]:
    if fiona is None or shape is None:
        raise RuntimeError("fiona and shapely are required for FAST_TREECLOUDS crown vector reading.")

    records: list[dict[str, Any]] = []
    with fiona.open(vector_path) as src:
        for idx, feat in enumerate(src, start=1):
            geom = feat.get("geometry")
            if not geom:
                continue
            poly = shape(geom)
            if poly.is_empty:
                continue
            props = dict(feat.get("properties") or {})
            source_crown_id = props.get("crown_id")
            if source_crown_id is None:
                source_crown_id = props.get("tree_id")
            if source_crown_id is None:
                source_crown_id = idx

            centroid = poly.representative_point()
            records.append({
                # guaranteed-unique feature id for internal export bookkeeping
                "feature_uid": int(idx),
                # preserve original crown/tree id from vector if present
                "crown_id": int(source_crown_id),
                "geometry": poly,
                "properties": props,
                "area": float(poly.area),
                "centroid": (float(centroid.x), float(centroid.y)),
            })
    # smaller crowns first to reduce swallowing of suppressed crowns
    records.sort(key=lambda r: (r["area"], r["feature_uid"]))
    return records


def _contains_mask(poly, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    minx, miny, maxx, maxy = poly.bounds
    bbox = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
    if not np.any(bbox):
        return bbox
    out = np.zeros(x.shape[0], dtype=bool)
    idx = np.flatnonzero(bbox)
    if contains_xy is not None:
        out[idx] = contains_xy(poly, x[idx], y[idx])
        return out
    from shapely.geometry import Point  # pragma: no cover
    out[idx] = [poly.contains(Point(float(xx), float(yy))) for xx, yy in zip(x[idx], y[idx])]
    return out


def _ensure_extra_dim(las: laspy.LasData, name: str, dtype) -> None:
    names = {dim.name for dim in las.point_format.extra_dimensions}
    if name not in names:
        las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))


def _copy_subset_las(template: laspy.LasData, point_indices: np.ndarray, attrs: dict[str, np.ndarray] | None = None) -> laspy.LasData:
    point_indices = np.asarray(point_indices, dtype=np.int64)
    out = laspy.LasData(template.header)
    out.points = template.points[point_indices].copy()
    attrs = attrs or {}
    for name, arr in attrs.items():
        _ensure_extra_dim(out, name, arr.dtype)
        setattr(out, name, np.asarray(arr))
    return out


def _infer_dataset_label(las_fp: Path) -> str:
    stem = las_fp.stem
    for suf in ("_FAST_NORMALIZED", "_FAST_GC"):
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def _assign_residual_segments(
    x: np.ndarray,
    y: np.ndarray,
    unassigned_idx: np.ndarray,
    parent_tree_ids: np.ndarray,
    *,
    cell_size: float = 0.5,
) -> tuple[np.ndarray, dict[int, int]]:
    seg_ids = np.zeros(unassigned_idx.size, dtype=np.uint32)
    seg_counts: dict[int, int] = {}
    if unassigned_idx.size == 0:
        return seg_ids, seg_counts
    if ndi is None:
        for i, ptid in enumerate(parent_tree_ids.tolist()):
            seg_counts[ptid] = seg_counts.get(ptid, 0) + 1
            seg_ids[i] = np.uint32(seg_counts[ptid])
        return seg_ids, seg_counts

    ux = x[unassigned_idx]
    uy = y[unassigned_idx]
    gx = np.floor((ux - ux.min()) / max(cell_size, 1e-6)).astype(int)
    gy = np.floor((uy - uy.min()) / max(cell_size, 1e-6)).astype(int)
    shape = (gx.max() + 1, gy.max() + 1)
    parent_grid: dict[tuple[int, int, int], list[int]] = {}
    for local_i, (ix, iy, ptid) in enumerate(zip(gx, gy, parent_tree_ids.tolist())):
        parent_grid.setdefault((int(ptid), int(ix), int(iy)), []).append(local_i)

    for ptid in sorted(set(parent_tree_ids.tolist())):
        mask = np.zeros(shape, dtype=bool)
        for (pid, ix, iy), vals in parent_grid.items():
            if pid == int(ptid):
                mask[ix, iy] = True
        labels, nlab = ndi.label(mask)
        if nlab == 0:
            continue
        local_count = 0
        for lab in range(1, nlab + 1):
            cells = np.argwhere(labels == lab)
            pts_local: list[int] = []
            for ix, iy in cells:
                pts_local.extend(parent_grid.get((int(ptid), int(ix), int(iy)), []))
            if not pts_local:
                continue
            local_count += 1
            seg_ids[np.asarray(pts_local, dtype=int)] = np.uint32(local_count)
        seg_counts[int(ptid)] = local_count

    zero = np.flatnonzero(seg_ids == 0)
    for zi in zero:
        ptid = int(parent_tree_ids[zi])
        seg_counts[ptid] = seg_counts.get(ptid, 0) + 1
        seg_ids[zi] = np.uint32(seg_counts[ptid])
    return seg_ids, seg_counts


def _segment_all_points_by_crowns(
    las_path: Path,
    crowns: list[dict[str, Any]],
    *,
    min_height: float = 0.5,
) -> tuple[
    laspy.LasData,
    dict[int, dict[str, Any]],
    dict[tuple[int, int], np.ndarray],
    dict[str, Any],
]:
    las = laspy.read(las_path)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    n = x.shape[0]

    tree_id = np.zeros(n, dtype=np.uint32)
    crown_id = np.zeros(n, dtype=np.uint32)
    crown_uid = np.zeros(n, dtype=np.uint32)
    parent_tree_id = np.zeros(n, dtype=np.uint32)
    segment_id = np.zeros(n, dtype=np.uint32)
    obj_type = np.zeros(n, dtype=np.uint8)

    per_tree: dict[int, dict[str, Any]] = {}
    tree_centroids: list[tuple[float, float]] = []
    assigned_mask = np.zeros(n, dtype=bool)

    next_tree_id = 1
    for rec in crowns:
        cid = int(rec["crown_id"])
        cuid = int(rec["feature_uid"])
        inside = _contains_mask(rec["geometry"], x, y)
        inside &= ~assigned_mask
        inside &= z >= float(min_height)
        idx = np.flatnonzero(inside)
        if idx.size == 0:
            continue

        tid = next_tree_id
        next_tree_id += 1

        assigned_mask[idx] = True
        tree_id[idx] = np.uint32(tid)
        crown_id[idx] = np.uint32(cid)
        crown_uid[idx] = np.uint32(cuid)
        parent_tree_id[idx] = np.uint32(tid)
        obj_type[idx] = np.uint8(1)

        per_tree[tid] = {
            "indices": idx,
            "source_crown_id": cid,
            "feature_uid": cuid,
        }
        tree_centroids.append(rec["centroid"])

    unassigned_idx = np.flatnonzero(~assigned_mask)
    per_segment: dict[tuple[int, int], np.ndarray] = {}
    if unassigned_idx.size > 0 and tree_centroids:
        if cKDTree is None:
            cent = np.asarray(tree_centroids, dtype=np.float64)
            dx = x[unassigned_idx][:, None] - cent[:, 0][None, :]
            dy = y[unassigned_idx][:, None] - cent[:, 1][None, :]
            nn = np.argmin(dx * dx + dy * dy, axis=1)
        else:
            tree = cKDTree(np.asarray(tree_centroids, dtype=np.float64))
            _, nn = tree.query(np.column_stack([x[unassigned_idx], y[unassigned_idx]]), k=1)
        parent_ids = (nn.astype(np.uint32) + 1)
        seg_local_ids, _ = _assign_residual_segments(x, y, unassigned_idx, parent_ids)
        tree_id[unassigned_idx] = parent_ids
        parent_tree_id[unassigned_idx] = parent_ids
        segment_id[unassigned_idx] = seg_local_ids
        obj_type[unassigned_idx] = np.uint8(2)
        for pid in np.unique(parent_ids):
            local = unassigned_idx[parent_ids == pid]
            loc_seg = seg_local_ids[parent_ids == pid]
            for sid in np.unique(loc_seg):
                if sid == 0:
                    continue
                per_segment[(int(pid), int(sid))] = local[loc_seg == sid]
    elif unassigned_idx.size > 0:
        tree_id[unassigned_idx] = np.uint32(0)
        parent_tree_id[unassigned_idx] = np.uint32(0)
        segment_id[unassigned_idx] = np.uint32(1)
        obj_type[unassigned_idx] = np.uint8(2)
        per_segment[(0, 1)] = unassigned_idx

    combined = _copy_subset_las(
        las,
        np.arange(n, dtype=np.int64),
        attrs={
            "tree_id": tree_id,
            "crown_id": crown_id,
            "crown_uid": crown_uid,
            "parent_tree_id": parent_tree_id,
            "segment_id": segment_id,
            "obj_type": obj_type,
        },
    )
    stats = {
        "n_input_points": int(n),
        "n_crowns": int(len(crowns)),
        "n_tree_points": int(np.count_nonzero(obj_type == 1)),
        "n_segment_points": int(np.count_nonzero(obj_type == 2)),
        "n_trees": int(len(per_tree)),
        "n_segments": int(len(per_segment)),
        "points_preserved": int(n),
    }
    return combined, per_tree, per_segment, stats


def run_treeclouds_from_root(
    source_root: str | Path,
    *,
    method: str,
    source_chm: str | None = None,
    las_source: str = "FAST_NORMALIZED",
    normalized_las: str | Path | None = None,
    crowns_path: str | Path | None = None,
    min_height: float = 0.5,
    write_individual: bool = False,
    skip_existing: bool = False,
    overwrite: bool = False,
    n_jobs: int = 1,
    joblib_backend: str = DEFAULT_BACKEND,
    joblib_batch_size: int | str = "auto",
    joblib_pre_dispatch: str | int = "2*n_jobs",
) -> str:
    root = Path(source_root)
    if not root.exists():
        raise FileNotFoundError(f"FAST_TREECLOUDS source root not found: {root}")

    method = str(method).strip().lower()
    variant = str(source_chm).strip() if source_chm is not None and str(source_chm).strip() else None
    las_source = str(las_source).strip().upper() or "FAST_NORMALIZED"
    if las_source not in {"FAST_NORMALIZED", "FAST_GC"}:
        raise ValueError("las_source must be FAST_NORMALIZED or FAST_GC")

    if root.is_file() and root.suffix.lower() in _LAS_SUFFIXES:
        source_base = root.parent
        las_fp = root
    else:
        source_base = root
        las_fp = Path(normalized_las) if normalized_las is not None else _find_single_merged_las(source_base, las_source)

    crowns_fp, resolved_variant = (Path(crowns_path), variant or "unknown") if crowns_path is not None else _find_crown_vector(source_base, method, variant)

    out_root = source_base / PRODUCT_TREECLOUDS / method / "chm" / resolved_variant / las_source
    if skip_existing and _existing_output(out_root) and not overwrite:
        log_info(f"[SKIP] Existing FAST_TREECLOUDS output found: {out_root}")
        return str(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    indiv_tree_root = out_root / "individual_trees"
    indiv_seg_root = out_root / "individual_segments"
    if write_individual:
        indiv_tree_root.mkdir(parents=True, exist_ok=True)
        indiv_seg_root.mkdir(parents=True, exist_ok=True)

    crowns = _load_polygons(crowns_fp)
    dataset_label = _infer_dataset_label(las_fp)

    log_info(f"FAST_TREECLOUDS method: {method}")
    log_info(f"FAST_TREECLOUDS source variant: {resolved_variant}")
    log_info(f"FAST_TREECLOUDS crowns: {crowns_fp}")
    log_info(f"FAST_TREECLOUDS LAS source: {las_source}")
    log_info(f"FAST_TREECLOUDS input LAS: {las_fp}")
    log_info(f"FAST_TREECLOUDS crowns loaded: {len(crowns)}")

    def _task(_dummy: int = 0):
        combined, per_tree, per_segment, stats = _segment_all_points_by_crowns(las_fp, crowns, min_height=min_height)
        combined_fp = out_root / f"{dataset_label}_{method}_{resolved_variant}_{las_source}_segmented_allpoints.las"
        combined.write(combined_fp)

        tree_outputs: list[str] = []
        segment_outputs: list[str] = []
        if write_individual:
            template = laspy.read(las_fp)

            total_tree = len(per_tree)
            for i, (tid, rec) in enumerate(per_tree.items(), start=1):
                idx = rec["indices"]
                cid = int(rec["source_crown_id"])
                cuid = int(rec["feature_uid"])
                attrs = {
                    "tree_id": np.full(idx.size, tid, dtype=np.uint32),
                    "crown_id": np.full(idx.size, cid, dtype=np.uint32),
                    "crown_uid": np.full(idx.size, cuid, dtype=np.uint32),
                    "parent_tree_id": np.full(idx.size, tid, dtype=np.uint32),
                    "segment_id": np.zeros(idx.size, dtype=np.uint32),
                    "obj_type": np.full(idx.size, 1, dtype=np.uint8),
                }
                sub = _copy_subset_las(template, idx, attrs=attrs)
                out_fp = indiv_tree_root / f"tree_{int(tid):06d}_crown_{int(cuid):06d}.las"
                sub.write(out_fp)
                tree_outputs.append(str(out_fp))
                if i == 1 or i % 250 == 0 or i == total_tree:
                    log_info(f"FAST_TREECLOUDS trees written: {i}/{total_tree}")

            total_seg = len(per_segment)
            for i, ((ptid, sid), idx) in enumerate(per_segment.items(), start=1):
                attrs = {
                    "tree_id": np.full(idx.size, ptid, dtype=np.uint32),
                    "crown_id": np.zeros(idx.size, dtype=np.uint32),
                    "crown_uid": np.zeros(idx.size, dtype=np.uint32),
                    "parent_tree_id": np.full(idx.size, ptid, dtype=np.uint32),
                    "segment_id": np.full(idx.size, sid, dtype=np.uint32),
                    "obj_type": np.full(idx.size, 2, dtype=np.uint8),
                }
                sub = _copy_subset_las(template, idx, attrs=attrs)
                out_fp = indiv_seg_root / f"segment_{int(ptid):06d}_{int(sid):02d}.las"
                sub.write(out_fp)
                segment_outputs.append(str(out_fp))
                if i == 1 or i % 250 == 0 or i == total_seg:
                    log_info(f"FAST_TREECLOUDS segments written: {i}/{total_seg}")

        payload = {
            "module": PRODUCT_TREECLOUDS,
            "method": method,
            "surface_type": "CHM",
            "surface_variant": resolved_variant,
            "source_root": str(source_base),
            "las_source": las_source,
            "input_las": str(las_fp),
            "crowns": str(crowns_fp),
            "combined_output": str(combined_fp),
            "individual_tree_outputs": tree_outputs,
            "individual_segment_outputs": segment_outputs,
            "stats": stats,
            "message": "FAST_TREECLOUDS preserves all points. Crown-covered points are obj_type=1 trees; uncrowned residual patches are obj_type=2 segments attached to nearest tree neighborhoods.",
        }
        _write_json(out_root / "treeclouds_manifest.json", payload)
        return str(combined_fp)

    summary = run_stage(
        stage_name=f"FAST-GC derive TREECLOUDS [{method}]",
        items=[0],
        func=_task,
        n_jobs=n_jobs,
        backend=joblib_backend,
        batch_size=joblib_batch_size,
        pre_dispatch=joblib_pre_dispatch,
        source=str(las_fp),
        unit="dataset",
    )

    outputs: list[str] = []
    for rec in summary.records:
        if rec.result is not None:
            outputs.append(str(rec.result))

    _write_json(
        out_root / "treeclouds_run.json",
        {
            "module": PRODUCT_TREECLOUDS,
            "method": method,
            "surface_type": "CHM",
            "surface_variant": resolved_variant,
            "source_root": str(source_base),
            "las_source": las_source,
            "input_las": str(las_fp),
            "crowns": str(crowns_fp),
            "outputs": outputs,
        },
    )
    return str(out_root)


__all__ = ["PRODUCT_TREECLOUDS", "run_treeclouds_from_root"]
