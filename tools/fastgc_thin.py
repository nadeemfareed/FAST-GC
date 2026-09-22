from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import laspy
import numpy as np


def stable_uniform(n: int, seed: int) -> np.ndarray:
    """Stable deterministic pseudo-random value per original point index."""
    idx = np.arange(n, dtype=np.uint64)
    x = idx + np.uint64(seed)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) * (1.0 / float(1 << 53))


def bbox_area_xy(x, y):
    if len(x) == 0:
        return 0.0
    return max(float(np.ptp(x)) * float(np.ptp(y)), 1e-12)


def voxel_groups(x, y, z, voxel_m):
    if voxel_m <= 0:
        raise ValueError("--voxel_m must be > 0")
    ix = np.floor(x / voxel_m).astype(np.int64)
    iy = np.floor(y / voxel_m).astype(np.int64)
    iz = np.floor(z / voxel_m).astype(np.int64)
    order = np.lexsort((iz, iy, ix))
    sx, sy, sz = ix[order], iy[order], iz[order]
    changed = np.empty(len(order), dtype=bool)
    if len(order):
        changed[0] = True
        changed[1:] = (sx[1:] != sx[:-1]) | (sy[1:] != sy[:-1]) | (sz[1:] != sz[:-1])
    starts = np.flatnonzero(changed) if len(order) else np.empty(0, dtype=np.int64)
    ends = np.r_[starts[1:], len(order)] if len(starts) else np.empty(0, dtype=np.int64)
    return order, starts, ends


def nearest_centroid(inds, x, y, z):
    xx, yy, zz = x[inds], y[inds], z[inds]
    cx, cy, cz = xx.mean(), yy.mean(), zz.mean()
    d2 = (xx-cx)**2 + (yy-cy)**2 + (zz-cz)**2
    return int(inds[int(np.argmin(d2))])


def farthest(inds, selected, x, y, z):
    best = np.full(len(inds), np.inf)
    for s in selected:
        d2 = (x[inds]-x[s])**2 + (y[inds]-y[s])**2 + (z[inds]-z[s])**2
        best = np.minimum(best, d2)
    return int(inds[int(np.argmax(best))])


def geometry_indices(x, y, z, voxel_m, diversity_low, diversity_high, max_points):
    order, starts, ends = voxel_groups(x, y, z, voxel_m)
    keep = []
    for start, end in zip(starts, ends):
        inds = order[start:end]
        if len(inds) == 1:
            keep.append(int(inds[0]))
            continue
        dx, dy, dz = float(np.ptp(x[inds])), float(np.ptp(y[inds])), float(np.ptp(z[inds]))
        diversity = math.sqrt(dx*dx + dy*dy + dz*dz) / voxel_m
        budget = 1 if diversity < diversity_low else (2 if diversity < diversity_high else 3)
        budget = min(budget, max_points, len(inds))
        selected = [nearest_centroid(inds, x, y, z)]
        while len(selected) < budget:
            candidate = farthest(inds, selected, x, y, z)
            if candidate in selected:
                break
            selected.append(candidate)
        keep.extend(selected)
    return np.unique(np.asarray(keep, dtype=np.int64))


def voxel_indices(x, y, z, voxel_m):
    order, starts, ends = voxel_groups(x, y, z, voxel_m)
    keep = [nearest_centroid(order[s:e], x, y, z) for s, e in zip(starts, ends)]
    return np.sort(np.asarray(keep, dtype=np.int64))


def write_selected(las, keep, output_path):
    out = laspy.LasData(las.header.copy())
    out.points = las.points[keep].copy()
    # Add source point identity so restoration can map retained points exactly.
    if "fastgc_src_idx" not in out.point_format.extra_dimension_names:
        out.add_extra_dim(laspy.ExtraBytesParams(name="fastgc_src_idx", type=np.uint64))
    out["fastgc_src_idx"] = keep.astype(np.uint64)
    out.write(output_path)


def write_meta(path, data):
    path.with_suffix(path.suffix + ".thin.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


def process(fp, out_dir, args):
    t0 = time.perf_counter()
    las = laspy.read(fp)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    n = len(x)
    area = bbox_area_xy(x, y)
    density = n / area
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[THIN] {fp.name} | {n:,} points | {density:.2f} pts/m2 | method={args.method}")

    if args.method == "density":
        ranks = stable_uniform(n, args.seed)
        order = np.argsort(ranks, kind="stable")
        for target in sorted(set(args.targets), reverse=True):
            nkeep = n if density <= target else max(1, min(n, int(round(target * area))))
            keep = np.sort(order[:nkeep])
            label = f"D{target:g}"
            out = out_dir / f"{fp.stem}_{label}{fp.suffix}"
            if out.exists() and not args.overwrite:
                print(f"[SKIP] {out.name}")
                continue
            write_selected(las, keep, out)
            achieved = len(keep) / area
            write_meta(out, {
                "method": "density", "source": str(fp), "source_points": n,
                "target_density_pts_m2": target, "achieved_density_pts_m2": achieved,
                "retained_points": len(keep), "seed": args.seed,
                "nested": True, "class_aware": False,
                "source_index_extra_byte": "fastgc_src_idx"
            })
            print(f"[DONE] {label} | {len(keep):,}/{n:,} | {achieved:.2f} pts/m2")

    elif args.method == "voxel":
        keep = voxel_indices(x, y, z, args.voxel_m)
        out = out_dir / f"{fp.stem}_VOXEL_{args.voxel_m:g}m{fp.suffix}"
        if out.exists() and not args.overwrite:
            print(f"[SKIP] {out.name}")
            return
        write_selected(las, keep, out)
        write_meta(out, {"method":"voxel","source":str(fp),"source_points":n,
                         "retained_points":len(keep),"voxel_m":args.voxel_m,
                         "source_index_extra_byte":"fastgc_src_idx"})
        print(f"[DONE] VOXEL | {len(keep):,}/{n:,}")

    elif args.method in {"geometry", "hybrid"}:
        geom = geometry_indices(x, y, z, args.voxel_m, args.diversity_low,
                                args.diversity_high, args.max_points_per_voxel)
        if args.method == "geometry":
            out = out_dir / f"{fp.stem}_GEOM_{args.voxel_m:g}m{fp.suffix}"
            if out.exists() and not args.overwrite:
                print(f"[SKIP] {out.name}")
                return
            write_selected(las, geom, out)
            write_meta(out, {"method":"geometry","source":str(fp),"source_points":n,
                             "retained_points":len(geom),"voxel_m":args.voxel_m,
                             "diversity_low":args.diversity_low,
                             "diversity_high":args.diversity_high,
                             "max_points_per_voxel":args.max_points_per_voxel,
                             "source_index_extra_byte":"fastgc_src_idx"})
            print(f"[DONE] GEOMETRY | {len(geom):,}/{n:,}")
        else:
            # Preserve geometry-selected points, then deterministically cap within that set.
            grank = stable_uniform(len(geom), args.seed)
            gorder = np.argsort(grank, kind="stable")
            gdensity = len(geom) / area
            for target in sorted(set(args.targets), reverse=True):
                nkeep = len(geom) if gdensity <= target else max(
                    1, min(len(geom), int(round(target * area)))
                )
                keep = np.sort(geom[gorder[:nkeep]])
                out = out_dir / f"{fp.stem}_HYBRID_D{target:g}{fp.suffix}"
                if out.exists() and not args.overwrite:
                    print(f"[SKIP] {out.name}")
                    continue
                write_selected(las, keep, out)
                write_meta(out, {"method":"hybrid","source":str(fp),"source_points":n,
                                 "geometry_points":len(geom),"retained_points":len(keep),
                                 "target_density_pts_m2":target,"voxel_m":args.voxel_m,
                                 "seed":args.seed,
                                 "source_index_extra_byte":"fastgc_src_idx"})
                print(f"[DONE] HYBRID D{target:g} | {len(keep):,}/{n:,}")

    print(f"[TIME] {time.perf_counter()-t0:.2f}s")


def main():
    p = argparse.ArgumentParser(description="FAST-GC deterministic point-cloud thinning")
    p.add_argument("--in_path", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--method", choices=["density","voxel","geometry","hybrid"], default="density")
    p.add_argument("--targets", nargs="+", type=float, default=[16,8,6,4])
    p.add_argument("--voxel_m", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--diversity_low", type=float, default=0.50)
    p.add_argument("--diversity_high", type=float, default=1.00)
    p.add_argument("--max_points_per_voxel", type=int, choices=[1,2,3], default=3)
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if not args.in_path.exists():
        raise FileNotFoundError(args.in_path)
    files = [args.in_path] if args.in_path.is_file() else sorted(
        f for f in args.in_path.glob("**/*" if args.recursive else "*")
        if f.is_file() and f.suffix.lower() in {".las",".laz"}
    )
    if not files:
        raise RuntimeError("No LAS/LAZ files found.")
    for fp in files:
        process(fp, args.out_dir, args)


if __name__ == "__main__":
    main()
