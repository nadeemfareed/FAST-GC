"""Standalone, memory-bounded LAS/LAZ clipping for FAST-GIS."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
from pathlib import Path

import laspy
import numpy as np
from shapely import intersects_xy
from shapely.geometry.base import BaseGeometry

from .las_query import query_las_chunks


def clip_las(
    input_path,
    output_path,
    bounds=None,
    *,
    geometry=None,
    buffer=0.0,
    chunk_size=500_000,
    write_report=True,
):
    """Clip LAS/LAZ points using inclusive XY bounds.

    Bounds must use the source file's coordinate system.
    A positive buffer expands the requested bounds on all sides.

    The source file is never modified. The output is written
    through a temporary file and moved into place only after
    successful validation.
    """

    source = Path(input_path).resolve()
    destination = Path(output_path).resolve()

    if not source.is_file():
        raise FileNotFoundError(source)

    if source == destination:
        raise ValueError("Input and output must differ.")

    if destination.exists():
        raise FileExistsError(destination)

    report_path = destination.with_suffix(".json")

    if write_report and report_path.exists():
        raise FileExistsError(report_path)

    if geometry is not None:
        if bounds is not None:
            raise ValueError(
                "Specify either bounds or geometry, not both."
            )

        if not isinstance(geometry, BaseGeometry):
            raise TypeError("geometry must be a Shapely geometry.")

        if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError(
                "Only Polygon and MultiPolygon geometries are supported."
            )

        if geometry.is_empty or not geometry.is_valid:
            raise ValueError("Polygon must be nonempty and valid.")

        values = np.asarray(
            geometry.bounds,
            dtype=np.float64,
        )

    else:
        if bounds is None:
            raise ValueError(
                "Either bounds or geometry is required."
            )

        values = np.asarray(bounds, dtype=np.float64)

    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("Bounds must contain four finite numbers.")

    xmin, ymin, xmax, ymax = map(float, values)

    if xmin > xmax or ymin > ymax:
        raise ValueError("Invalid spatial bounds.")

    if not isinstance(buffer, (int, float)) or not math.isfinite(buffer):
        raise ValueError("Buffer must be finite.")

    if buffer < 0:
        raise ValueError("Buffer cannot be negative.")

    if isinstance(chunk_size, (bool, np.bool_)) or not isinstance(
        chunk_size, (int, np.integer)
    ) or chunk_size <= 0:
        raise ValueError("Chunk size must be a positive integer.")

    selection_geometry = None

    if geometry is not None:
        selection_geometry = (
            geometry.buffer(buffer)
            if buffer > 0
            else geometry
        )

        expanded = tuple(
            map(float, selection_geometry.bounds)
        )

    else:
        expanded = (
            xmin - buffer,
            ymin - buffer,
            xmax + buffer,
            ymax + buffer,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    with laspy.open(source) as reader:
        source_header = reader.header.copy()
        source_count = int(source_header.point_count)

    started = time.perf_counter()
    selected_count = 0

    # Preserve the requested output extension so laspy selects
    # the correct LAS or LAZ writer.
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.stem}_",
        suffix=destination.suffix,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)

    try:
        with laspy.open(
            temporary,
            mode="w",
            header=source_header,
        ) as writer:

            for result in query_las_chunks(
                source,
                expanded,
                chunk_size=chunk_size,
            ):
                selected_points = result.points

                if selection_geometry is not None:
                    inside = intersects_xy(
                        selection_geometry,
                        np.asarray(selected_points.x),
                        np.asarray(selected_points.y),
                    )

                    selected_points = selected_points[
                        np.flatnonzero(inside)
                    ]

                if len(selected_points):
                    writer.write_points(selected_points)
                    selected_count += len(selected_points)

        with laspy.open(temporary) as reader:
            output_header = reader.header

            if output_header.point_count != selected_count:
                raise RuntimeError("Output point count mismatch.")

            if output_header.point_format.id != source_header.point_format.id:
                raise RuntimeError("Point format changed.")

            if str(output_header.version) != str(source_header.version):
                raise RuntimeError("LAS version changed.")

            if list(output_header.point_format.dimension_names) != list(
                source_header.point_format.dimension_names
            ):
                raise RuntimeError("Point dimensions changed.")

            np.testing.assert_array_equal(
                output_header.scales,
                source_header.scales,
            )

            np.testing.assert_array_equal(
                output_header.offsets,
                source_header.offsets,
            )

            source_crs = source_header.parse_crs()
            output_crs = output_header.parse_crs()

            if (source_crs is None) != (output_crs is None):
                raise RuntimeError("CRS presence changed.")

            if source_crs is not None and not source_crs.equals(output_crs):
                raise RuntimeError("CRS changed.")

        # Publish without overwriting an existing destination.
        # Exclusive creation avoids replacing a file on Windows.
        try:
            with temporary.open("rb") as src:
                with destination.open("xb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
        except FileExistsError:
            raise
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    finally:
        temporary.unlink(missing_ok=True)

    report = {
        "input": str(source),
        "output": str(destination),
        "source_points": source_count,
        "selected_points": selected_count,
        "selection_fraction": (
            selected_count / source_count if source_count else 0.0
        ),
        "selection_type": (
            "polygon" if geometry is not None else "bounds"
        ),
        "requested_bounds": list(map(float, values)),
        "expanded_bounds": list(expanded),
        "buffer": float(buffer),
        "chunk_size": int(chunk_size),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }

    if write_report:
        report_path = destination.with_suffix(".json")

        if report_path.exists():
            raise FileExistsError(
                f"Report already exists: {report_path}"
            )

        report_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

    return report
