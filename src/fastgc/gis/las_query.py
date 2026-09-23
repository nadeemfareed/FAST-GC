"""Memory-bounded LAS/LAZ spatial queries for FAST-GIS.

This module performs sequential chunk filtering, not indexed
random access. Source files are opened read-only.

The returned point records preserve their original LAS dimensions,
scales, offsets and classification values.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import laspy
import numpy as np


@dataclass(frozen=True)
class LASQueryResult:
    """One spatially filtered chunk of LAS/LAZ points."""

    points: laspy.ScaleAwarePointRecord
    indices: np.ndarray
    source: Path

    @property
    def count(self) -> int:
        return len(self.indices)


def _validate_bounds(bounds):
    """Validate an inclusive XY bounding box."""

    values = np.asarray(bounds, dtype=np.float64)

    if values.shape != (4,):
        raise ValueError(
            "Bounds must be (xmin, ymin, xmax, ymax)."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError("Bounds must contain finite values.")

    xmin, ymin, xmax, ymax = values

    if xmin > xmax or ymin > ymax:
        raise ValueError("Invalid bounding-box extent.")

    return xmin, ymin, xmax, ymax


def query_las_chunks(
    path,
    bounds,
    *,
    z_min=None,
    z_max=None,
    chunk_size=1_000_000,
) -> Iterator[LASQueryResult]:
    """Yield filtered LAS/LAZ point chunks.

    Parameters
    ----------
    path:
        Input LAS or LAZ file.

    bounds:
        Inclusive XY bounds in the source file's CRS.

    z_min, z_max:
        Optional inclusive elevation limits.

    chunk_size:
        Maximum number of source points read per chunk.

    Notes
    -----
    This function scans the source file sequentially.

    Returned indices are zero-based positions in the original
    LAS/LAZ file.

    Spatial filtering uses scaled X, Y and Z coordinates.
    """

    path = Path(path)

    xmin, ymin, xmax, ymax = _validate_bounds(bounds)

    if not isinstance(chunk_size, (int, np.integer)):
        raise ValueError("chunk_size must be an integer.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if z_min is not None:
        z_min = float(z_min)

        if not np.isfinite(z_min):
            raise ValueError("z_min must be finite.")

    if z_max is not None:
        z_max = float(z_max)

        if not np.isfinite(z_max):
            raise ValueError("z_max must be finite.")

    if (
        z_min is not None
        and z_max is not None
        and z_min > z_max
    ):
        raise ValueError("z_min cannot exceed z_max.")

    offset = 0

    with laspy.open(path) as reader:

        for points in reader.chunk_iterator(chunk_size):

            count = len(points)

            x = np.asarray(points.x)
            y = np.asarray(points.y)

            selected = (
                (x >= xmin)
                & (x <= xmax)
                & (y >= ymin)
                & (y <= ymax)
            )

            if z_min is not None or z_max is not None:

                z = np.asarray(points.z)

                if z_min is not None:
                    selected &= z >= z_min

                if z_max is not None:
                    selected &= z <= z_max

            local_indices = np.flatnonzero(selected)

            if len(local_indices):

                original_indices = (
                    local_indices.astype(np.int64)
                    + offset
                )

                filtered_points = points[local_indices]

                yield LASQueryResult(
                    points=filtered_points,
                    indices=original_indices,
                    source=path,
                )

            offset += count
