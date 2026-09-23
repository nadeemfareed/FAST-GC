"""Transactional multi-source LAS/LAZ clipping for FAST-GIS.

Only source tiles intersecting the requested plot are scanned.  Points from
source-tile overlaps are de-duplicated on the exact XYZ grid representable by
the output LAS header.  Source files are never modified.
"""
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import laspy
import numpy as np
from pyproj import CRS
from shapely import intersects_xy

from .las_query import query_las_chunks


def _compatible(reference, candidate, path):
    if reference.point_format.id != candidate.point_format.id:
        raise ValueError(f"Point format differs across survey tiles: {path}")
    if list(reference.point_format.dimension_names) != list(candidate.point_format.dimension_names):
        raise ValueError(f"Point dimensions differ across survey tiles: {path}")
    a, b = reference.parse_crs(), candidate.parse_crs()
    if a is None or b is None or not CRS.from_user_input(a).equals(CRS.from_user_input(b)):
        raise ValueError(f"CRS differs across survey tiles: {path}")


def _xyz_keys(points, header):
    """Integer XYZ keys on the output header's representable coordinate grid."""
    xyz = np.column_stack((np.asarray(points.x), np.asarray(points.y), np.asarray(points.z)))
    scales = np.asarray(header.scales, dtype=np.float64)
    offsets = np.asarray(header.offsets, dtype=np.float64)
    return np.rint((xyz - offsets) / scales).astype(np.int64)


def clip_las_sources(source_paths, output_path, *, geometry, chunk_size=500_000, deduplicate_xyz=True):
    sources = [Path(p).resolve() for p in source_paths]
    destination = Path(output_path).resolve()
    if not sources:
        raise ValueError("No intersecting source LAS/LAZ tiles")
    if destination.exists():
        raise FileExistsError(destination)
    for p in sources:
        if not p.is_file():
            raise FileNotFoundError(p)

    with laspy.open(sources[0]) as r:
        header = r.header.copy()
    for p in sources[1:]:
        with laspy.open(p) as r:
            _compatible(header, r.header, p)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.stem}_", suffix=destination.suffix, delete=False) as h:
        temporary = Path(h.name)

    selected = 0
    duplicates = 0
    per_source = []
    seen = set()
    started = time.perf_counter()
    bounds = geometry.bounds
    try:
        with laspy.open(temporary, mode="w", header=header) as writer:
            for source in sources:
                source_selected = 0
                source_duplicates = 0
                for result in query_las_chunks(source, bounds, chunk_size=chunk_size):
                    pts = result.points
                    inside = intersects_xy(geometry, np.asarray(pts.x), np.asarray(pts.y))
                    pts = pts[np.flatnonzero(inside)]
                    if not len(pts):
                        continue
                    if deduplicate_xyz:
                        keys = _xyz_keys(pts, header)
                        keep = np.ones(len(pts), dtype=bool)
                        for i, key in enumerate(keys):
                            item = (int(key[0]), int(key[1]), int(key[2]))
                            if item in seen:
                                keep[i] = False
                                source_duplicates += 1
                            else:
                                seen.add(item)
                        pts = pts[np.flatnonzero(keep)]
                    if len(pts):
                        writer.write_points(pts)
                        source_selected += len(pts)
                selected += source_selected
                duplicates += source_duplicates
                per_source.append({"path": str(source), "selected_points": source_selected,
                                   "duplicates_removed": source_duplicates})
        with laspy.open(temporary) as r:
            if int(r.header.point_count) != selected:
                raise RuntimeError("Multi-source output point count mismatch")
        with temporary.open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
    finally:
        temporary.unlink(missing_ok=True)

    return {"output": str(destination), "selected_points": selected,
            "duplicates_removed": duplicates, "source_count": len(sources),
            "sources": per_source, "deduplication": "output_grid_exact_xyz" if deduplicate_xyz else "none",
            "elapsed_seconds": round(time.perf_counter() - started, 4)}
