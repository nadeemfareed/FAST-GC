from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree


def get_src_idx(las):
    names = set(las.point_format.extra_dimension_names)
    if "fastgc_src_idx" not in names:
        raise RuntimeError(
            "Classified thinned cloud has no 'fastgc_src_idx' Extra Byte. "
            "Create the thinned cloud with fastgc_thin.py so exact retained-point "
            "identity survives FAST-GC and merge."
        )
    return np.asarray(las["fastgc_src_idx"], dtype=np.uint64)


def restore(original_path, classified_path, output_path, xy_radius, z_tolerance,
            ground_class, non_ground_mode, overwrite):
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}")

    t0 = time.perf_counter()
    original = laspy.read(original_path)
    classified = laspy.read(classified_path)

    ox = np.asarray(original.x, dtype=np.float64)
    oy = np.asarray(original.y, dtype=np.float64)
    oz = np.asarray(original.z, dtype=np.float64)

    cx = np.asarray(classified.x, dtype=np.float64)
    cy = np.asarray(classified.y, dtype=np.float64)
    cz = np.asarray(classified.z, dtype=np.float64)
    cc = np.asarray(classified.classification, dtype=np.uint8)
    src = get_src_idx(classified)

    if len(src) != len(classified.points):
        raise RuntimeError("fastgc_src_idx length mismatch.")
    if len(src) and int(src.max()) >= len(original.points):
        raise RuntimeError(
            "fastgc_src_idx references points outside the supplied original cloud. "
            "Check that --original matches the source used for thinning."
        )
    if len(np.unique(src)) != len(src):
        raise RuntimeError("Duplicate fastgc_src_idx values detected after processing/merge.")

    # Work on a complete copy of the original cloud.
    out = laspy.LasData(original.header.copy())
    out.points = original.points.copy()
    out_cls = np.asarray(out.classification, dtype=np.uint8).copy()

    # Exact retained-point labels: no spatial inference required.
    out_cls[src.astype(np.int64)] = cc
    exact_ground = src[cc == ground_class].astype(np.int64)

    # Spatial restoration is intentionally ground-only. Non-ground labels are
    # not propagated through space because that could destroy valid original
    # semantic classes. Removed points are promoted to ground only when they
    # lie close in XY AND Z to classified ground support.
    ground_mask = cc == ground_class
    projected = np.zeros(len(original.points), dtype=bool)

    if np.any(ground_mask):
        gxy = np.column_stack((cx[ground_mask], cy[ground_mask]))
        gz = cz[ground_mask]
        tree = cKDTree(gxy)

        # Query nearest classified ground in XY.
        dist, nn = tree.query(
            np.column_stack((ox, oy)),
            k=1,
            distance_upper_bound=xy_radius,
            workers=-1,
        )
        valid = np.isfinite(dist) & (nn < len(gz))
        candidate = np.flatnonzero(valid)

        if len(candidate):
            dz = np.abs(oz[candidate] - gz[nn[candidate]])
            candidate = candidate[dz <= z_tolerance]
            projected[candidate] = True
            out_cls[candidate] = np.uint8(ground_class)

    # Exact classified retained non-ground points remain exact. For removed
    # non-ground points, default behavior is preserve original Classification.
    if non_ground_mode == "unclassified":
        retained = np.zeros(len(original.points), dtype=bool)
        retained[src.astype(np.int64)] = True
        removed_not_ground = (~retained) & (~projected)
        out_cls[removed_not_ground] = np.uint8(1)

    out.classification = out_cls
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.write(output_path)

    summary = {
        "original": str(original_path),
        "classified_thinned": str(classified_path),
        "output": str(output_path),
        "original_points": len(original.points),
        "classified_thinned_points": len(classified.points),
        "exact_retained_ground_points": int(len(exact_ground)),
        "projected_ground_points_total": int(projected.sum()),
        "final_ground_points": int(np.sum(out_cls == ground_class)),
        "ground_class": int(ground_class),
        "xy_radius_m": float(xy_radius),
        "z_tolerance_m": float(z_tolerance),
        "non_ground_mode": non_ground_mode,
        "elapsed_s": time.perf_counter() - t0,
    }
    output_path.with_suffix(output_path.suffix + ".restore.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"[DONE] RESTORE | original={len(original.points):,} | "
          f"thin={len(classified.points):,} | final ground={summary['final_ground_points']:,} | "
          f"elapsed={summary['elapsed_s']:.2f}s")


def main():
    p = argparse.ArgumentParser(
        description="Project FAST-GC ground classification from a thinned cloud back to the original cloud."
    )
    p.add_argument("--original", required=True, type=Path)
    p.add_argument("--classified", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--xy_radius_m", type=float, default=0.25)
    p.add_argument("--z_tolerance_m", type=float, default=0.15)
    p.add_argument("--ground_class", type=int, default=2)
    p.add_argument("--non_ground_mode", choices=["preserve","unclassified"], default="preserve")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    restore(args.original, args.classified, args.output,
            args.xy_radius_m, args.z_tolerance_m, args.ground_class,
            args.non_ground_mode, args.overwrite)


if __name__ == "__main__":
    main()
