"""Continuous/systematic FAST-GIS plot generation."""
from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path

import laspy
import numpy as np
from pyproj import CRS
from shapely.geometry import Point, mapping
from shapely.prepared import prep

from .geometry import plot_geometry
from .plot_runner import extract_plots, _metre_crs
from .publish import publish_directory
from .survey_catalog import build_survey_catalog


SHAPES = {"square", "rectangle", "circle", "ellipse", "hexagon"}


def _ready_catalog(source, *, source_crs=None):

    catalog = build_survey_catalog(
        source,
        source_crs=source_crs,
    )

    ready = [
        t for t in catalog["tiles"]
        if t["status"] == "ready"
    ]

    if not ready:
        raise ValueError(
            "No survey LAS/LAZ tile has a resolved CRS"
        )

    crs = CRS.from_wkt(ready[0]["crs_wkt"])

    if not _metre_crs(crs):
        raise ValueError(
            "Continuous plotting requires projected metre coordinates"
        )

    if any(
        not crs.equals(CRS.from_wkt(t["crs_wkt"]))
        for t in ready[1:]
    ):
        raise ValueError(
            "Ready survey tiles must share one CRS"
        )

    return catalog, ready, crs


def survey_bounds(ready):

    return (
        min(float(t["bounds"][0]) for t in ready),
        min(float(t["bounds"][1]) for t in ready),
        max(float(t["bounds"][2]) for t in ready),
        max(float(t["bounds"][3]) for t in ready),
    )


def occupied_xy(
    ready,
    bounds,
    *,
    cell_size=2.0,
    chunk_size=500_000,
):

    if cell_size <= 0:
        raise ValueError(
            "coverage_cell_m must be positive"
        )

    minx, miny, maxx, maxy = bounds

    nx = max(
        1,
        int(math.ceil((maxx - minx) / cell_size)),
    )

    ny = max(
        1,
        int(math.ceil((maxy - miny) / cell_size)),
    )

    occupied = np.zeros(
        (ny, nx),
        dtype=bool,
    )

    total_points = 0

    for tile in ready:

        with laspy.open(tile["path"]) as reader:

            for chunk in reader.chunk_iterator(chunk_size):

                total_points += len(chunk)

                x = np.asarray(
                    chunk.x,
                    dtype=np.float64,
                )

                y = np.asarray(
                    chunk.y,
                    dtype=np.float64,
                )

                good = (
                    np.isfinite(x) &
                    np.isfinite(y)
                )

                if not good.any():
                    continue

                col = np.floor(
                    (x[good] - minx) / cell_size
                ).astype(np.int64)

                row = np.floor(
                    (y[good] - miny) / cell_size
                ).astype(np.int64)

                valid = (
                    (col >= 0) &
                    (col < nx) &
                    (row >= 0) &
                    (row < ny)
                )

                occupied[
                    row[valid],
                    col[valid],
                ] = True

    rows, cols = np.nonzero(occupied)

    xs = (
        minx +
        (cols.astype(np.float64) + 0.5) *
        cell_size
    )

    ys = (
        miny +
        (rows.astype(np.float64) + 0.5) *
        cell_size
    )

    return (
        np.column_stack((xs, ys)),
        int(total_points),
    )


def dimensions(
    shape,
    radius,
    width,
    height,
):

    shape = str(shape).lower()

    if shape in {"circle", "hexagon"}:

        r = float(radius)

        if r <= 0:
            raise ValueError(
                "radius must be positive"
            )

        return 2.0 * r, 2.0 * r

    w = float(
        width
        if width is not None
        else 2.0 * radius
    )

    if shape == "square":
        h = w
    else:
        h = float(
            height
            if height is not None
            else w
        )

    if w <= 0 or h <= 0:
        raise ValueError(
            "width and height must be positive"
        )

    return w, h


def lattice_spacing(
    shape,
    radius,
    width,
    height,
    overlap,
):

    shape = str(shape).lower()
    overlap = float(overlap)

    if overlap < 0:
        raise ValueError(
            "overlap_m must be nonnegative"
        )

    width_m, height_m = dimensions(
        shape,
        radius,
        width,
        height,
    )

    if shape == "hexagon":

        r = float(radius)

        sx = 1.5 * r - overlap
        sy = math.sqrt(3.0) * r - overlap

    else:

        sx = width_m - overlap
        sy = height_m - overlap

    if sx <= 0 or sy <= 0:
        raise ValueError(
            "overlap_m is too large for plot dimensions"
        )

    return float(sx), float(sy)


def lattice_centres(
    bounds,
    sx,
    sy,
    phase_x,
    phase_y,
    *,
    shape,
):

    minx, miny, maxx, maxy = bounds

    x0 = minx - sx + phase_x
    y0 = miny - sy + phase_y

    xs = np.arange(
        x0,
        maxx + sx,
        sx,
    )

    ys = np.arange(
        y0,
        maxy + sy,
        sy,
    )

    centres = []

    for row, y in enumerate(ys):

        offset = (
            0.5 * sx
            if shape == "hexagon" and row % 2
            else 0.0
        )

        for x in xs + offset:

            centres.append(
                (float(x), float(y))
            )

    return centres


def score_layout(
    centres,
    occupied,
    *,
    shape,
    radius,
    width,
    height,
    rotation_deg,
):

    covered = np.zeros(
        len(occupied),
        dtype=bool,
    )

    plots = []

    for x, y in centres:

        geom = plot_geometry(
            x,
            y,
            shape=shape,
            radius=radius,
            width=width,
            height=height,
            rotation_deg=rotation_deg,
        )

        minx, miny, maxx, maxy = geom.bounds

        candidates = np.flatnonzero(
            (occupied[:, 0] >= minx) &
            (occupied[:, 0] <= maxx) &
            (occupied[:, 1] >= miny) &
            (occupied[:, 1] <= maxy)
        )

        if not len(candidates):
            continue

        prepared = prep(geom)

        inside = np.fromiter(
            (
                prepared.covers(
                    Point(
                        float(occupied[i, 0]),
                        float(occupied[i, 1]),
                    )
                )
                for i in candidates
            ),
            dtype=bool,
            count=len(candidates),
        )

        hit = candidates[inside]

        if len(hit):

            covered[hit] = True

            plots.append(
                (x, y, geom)
            )

    return plots, covered


def choose_continuous_layout(
    ready,
    *,
    shape="square",
    radius=15.0,
    width=30.0,
    height=30.0,
    overlap=5.0,
    rotation_deg=0.0,
    coverage_cell_m=2.0,
    phase_steps=10,
    chunk_size=500_000,
):

    shape = str(shape).lower()

    if shape not in SHAPES:
        raise ValueError(
            f"Unsupported shape: {shape}"
        )

    bounds = survey_bounds(ready)

    occupied, total_points = occupied_xy(
        ready,
        bounds,
        cell_size=coverage_cell_m,
        chunk_size=chunk_size,
    )

    if not len(occupied):
        raise ValueError(
            "No finite XY LiDAR coverage found"
        )

    sx, sy = lattice_spacing(
        shape,
        radius,
        width,
        height,
        overlap,
    )

    phase_steps = int(phase_steps)

    if phase_steps < 1:
        raise ValueError(
            "phase_steps must be positive"
        )

    phases_x = np.linspace(
        0.0,
        sx,
        phase_steps,
        endpoint=False,
    )

    phases_y = np.linspace(
        0.0,
        sy,
        phase_steps,
        endpoint=False,
    )

    best = None

    for px in phases_x:

        for py in phases_y:

            centres = lattice_centres(
                bounds,
                sx,
                sy,
                float(px),
                float(py),
                shape=shape,
            )

            plots, covered = score_layout(
                centres,
                occupied,
                shape=shape,
                radius=radius,
                width=width,
                height=height,
                rotation_deg=rotation_deg,
            )

            score = int(
                covered.sum()
            )

            # Primary objective:
            # maximum observed XY coverage.
            #
            # Tie breakers:
            # fewer plots, then deterministic
            # lower phase offsets.
            rank = (
                score,
                -len(plots),
                -float(px),
                -float(py),
            )

            if (
                best is None or
                rank > best["rank"]
            ):

                best = {
                    "rank": rank,
                    "plots": plots,
                    "covered": covered,
                    "phase_x": float(px),
                    "phase_y": float(py),
                }

    occupied_count = int(
        len(occupied)
    )

    covered_count = int(
        best["covered"].sum()
    )

    return {
        "bounds": tuple(
            map(float, bounds)
        ),
        "total_source_points":
            int(total_points),
        "occupied_cell_count":
            occupied_count,
        "covered_cell_count":
            covered_count,
        "excluded_cell_count":
            occupied_count - covered_count,
        "coverage_fraction":
            covered_count / occupied_count,
        "phase_x_m":
            best["phase_x"],
        "phase_y_m":
            best["phase_y"],
        "spacing_x_m":
            sx,
        "spacing_y_m":
            sy,
        "plots":
            best["plots"],
    }


def exact_unique_point_coverage(
    ready,
    plot_geometries,
    *,
    chunk_size=500_000,
):
    """Count source LiDAR points covered by at least one core plot.

    Each source point is counted once even where neighbouring plots overlap.
    """
    if not plot_geometries:
        return {
            "source_point_count": 0,
            "covered_unique_points": 0,
            "excluded_points": 0,
            "coverage_fraction": 0.0,
        }

    prepared = [
        (geom.bounds, prep(geom))
        for geom in plot_geometries
    ]

    total = 0
    covered_total = 0

    for tile in ready:
        with laspy.open(tile["path"]) as reader:
            for chunk in reader.chunk_iterator(chunk_size):

                x = np.asarray(chunk.x, dtype=np.float64)
                y = np.asarray(chunk.y, dtype=np.float64)

                finite = np.isfinite(x) & np.isfinite(y)
                total += len(chunk)

                if not finite.any():
                    continue

                idx = np.flatnonzero(finite)
                hit = np.zeros(len(idx), dtype=bool)

                xx = x[idx]
                yy = y[idx]

                for bounds, prepared_geom in prepared:

                    xmin, ymin, xmax, ymax = bounds

                    candidates = np.flatnonzero(
                        (~hit) &
                        (xx >= xmin) &
                        (xx <= xmax) &
                        (yy >= ymin) &
                        (yy <= ymax)
                    )

                    if not len(candidates):
                        continue

                    inside = np.fromiter(
                        (
                            prepared_geom.covers(
                                Point(
                                    float(xx[j]),
                                    float(yy[j]),
                                )
                            )
                            for j in candidates
                        ),
                        dtype=bool,
                        count=len(candidates),
                    )

                    hit[candidates[inside]] = True

                    if hit.all():
                        break

                covered_total += int(hit.sum())

    return {
        "source_point_count": int(total),
        "covered_unique_points": int(covered_total),
        "excluded_points": int(total - covered_total),
        "coverage_fraction": (
            float(covered_total) / float(total)
            if total
            else 0.0
        ),
    }



def sample_continuous_and_clip(
    source,
    output_dir,
    *,
    radius=15.0,
    width=30.0,
    height=30.0,
    overlap=5.0,
    buffer=5.0,
    shape="square",
    rotation_deg=0.0,
    coverage_cell_m=2.0,
    phase_steps=10,
    chunk_size=500_000,
    sensor_mode="ALS",
    source_crs=None,
):

    source = Path(source).resolve()
    target = Path(output_dir).resolve()

    sensor_mode = str(
        sensor_mode
    ).upper()

    shape = str(
        shape
    ).lower()

    if sensor_mode not in {
        "ALS",
        "ULS",
        "TLS",
    }:
        raise ValueError(
            "sensor_mode must be ALS, ULS, or TLS"
        )

    if not source.exists():
        raise FileNotFoundError(source)

    if target.exists():
        raise FileExistsError(target)

    catalog, ready, crs = _ready_catalog(
        source,
        source_crs=source_crs,
    )

    layout = choose_continuous_layout(
        ready,
        shape=shape,
        radius=radius,
        width=width,
        height=height,
        overlap=overlap,
        rotation_deg=rotation_deg,
        coverage_cell_m=coverage_cell_m,
        phase_steps=phase_steps,
        chunk_size=chunk_size,
    )

    exact_coverage = exact_unique_point_coverage(
        ready,
        [geom for _, _, geom in layout["plots"]],
        chunk_size=chunk_size,
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}_stage_",
            dir=target.parent,
        )
    )

    try:

        features = []
        records = []

        for i, (x, y, geom) in enumerate(
            layout["plots"],
            1,
        ):

            name = f"Plot_{i:04d}"

            props = {
                "plot_name": name,
                "shape": shape,
                "x": float(x),
                "y": float(y),
                "overlap_m":
                    float(overlap),
                "rotation_deg":
                    float(rotation_deg),
                "sampling_method":
                    "continuous_optimized_coverage",
            }

            if shape in {
                "circle",
                "hexagon",
            }:

                props["radius_m"] = (
                    float(radius)
                )

            else:

                w, h = dimensions(
                    shape,
                    radius,
                    width,
                    height,
                )

                props["width_m"] = w
                props["height_m"] = h

            features.append({
                "type": "Feature",
                "id": name,
                "properties": props,
                "geometry": mapping(geom),
            })

            records.append({
                "plot_name": name,
                "x": float(x),
                "y": float(y),
                "bounds": list(
                    map(float, geom.bounds)
                ),
                "geometry": mapping(geom),
            })

        plot_file = (
            stage /
            "sample_plots.geojson"
        )

        plot_file.write_text(
            json.dumps({
                "type":
                    "FeatureCollection",
                "name":
                    "FAST_GIS_continuous",
                "crs": {
                    "type": "name",
                    "properties": {
                        "name":
                            crs.to_string()
                    },
                },
                "features":
                    features,
            }, indent=2),
            encoding="utf-8",
        )

        w, h = dimensions(
            shape,
            radius,
            width,
            height,
        )

        sampling_manifest = {
            "schema":
                "fastgc.gis.continuous_sampling",
            "schema_version": 1,
            "method":
                "continuous_optimized_coverage",
            "source":
                str(source),
            "crs":
                crs.to_string(),
            "sensor_mode":
                sensor_mode,
            "source_bounds":
                list(layout["bounds"]),
            "shape":
                shape,
            "width_m":
                w,
            "height_m":
                h,
            "radius_m":
                float(radius)
                if shape in {
                    "circle",
                    "hexagon",
                }
                else None,
            "overlap_m":
                float(overlap),
            "buffer_m":
                float(buffer),
            "rotation_deg":
                float(rotation_deg),
            "spacing_x_m":
                layout["spacing_x_m"],
            "spacing_y_m":
                layout["spacing_y_m"],
            "optimized_phase_x_m":
                layout["phase_x_m"],
            "optimized_phase_y_m":
                layout["phase_y_m"],
            "coverage_cell_m":
                float(coverage_cell_m),
            "phase_steps":
                int(phase_steps),
            "source_point_count":
                exact_coverage["source_point_count"],
            "covered_unique_source_points":
                exact_coverage["covered_unique_points"],
            "excluded_source_points":
                exact_coverage["excluded_points"],
            "exact_point_coverage_fraction":
                exact_coverage["coverage_fraction"],
            "exact_point_coverage_percent":
                100.0 * exact_coverage["coverage_fraction"],
            "occupied_cell_count":
                layout["occupied_cell_count"],
            "covered_occupied_cell_count":
                layout["covered_cell_count"],
            "excluded_occupied_cell_count":
                layout["excluded_cell_count"],
            "occupied_coverage_fraction":
                layout["coverage_fraction"],
            "occupied_coverage_percent":
                100.0 *
                layout["coverage_fraction"],
            "plot_count":
                len(records),
            "plots":
                records,
        }

        (
            stage /
            "sampling_manifest.json"
        ).write_text(
            json.dumps(
                sampling_manifest,
                indent=2,
            ),
            encoding="utf-8",
        )

        clip_root = (
            stage /
            f"{sensor_mode}_plots"
        )

        extract_plots(
            source,
            plot_file,
            clip_root,
            buffer=buffer,
            chunk_size=chunk_size,
            sensor_mode=sensor_mode,
            sampling_method=
                "continuous_optimized_coverage",
            shape=shape,
            radius=(
                radius
                if shape in {
                    "circle",
                    "hexagon",
                }
                else None
            ),
            source_crs=source_crs,
        )

        workspace = {
            "schema":
                "fastgc.gis.workspace",
            "schema_version": 3,
            "sensor_mode":
                sensor_mode,
            "sampling_method":
                "continuous_optimized_coverage",
            "shape":
                shape,
            "width_m":
                w,
            "height_m":
                h,
            "buffer_m":
                float(buffer),
            "overlap_m":
                float(overlap),
            "rotation_deg":
                float(rotation_deg),
            "plot_count":
                len(records),
            "coverage_percent":
                100.0 * exact_coverage["coverage_fraction"],
            "covered_unique_source_points":
                exact_coverage["covered_unique_points"],
            "excluded_source_points":
                exact_coverage["excluded_points"],
            "collection":
                f"{sensor_mode}_plots",
            "plots_manifest":
                f"{sensor_mode}_plots/plots_manifest.json",
            "sample_plots":
                "sample_plots.geojson",
            "sampling_manifest":
                "sampling_manifest.json",
        }

        (
            stage /
            "workspace_manifest.json"
        ).write_text(
            json.dumps(
                workspace,
                indent=2,
            ),
            encoding="utf-8",
        )

        publish_directory(
            stage,
            target,
        )

        return target

    finally:

        if stage.exists():
            shutil.rmtree(
                stage,
                ignore_errors=True,
            )
