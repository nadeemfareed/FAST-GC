"""Independent 2D and 3D spatial indexing for FAST-GIS."""

import numpy as np
from scipy.spatial import cKDTree


class PointSpatialIndex:
    """Reusable spatial index preserving original point indices."""

    def __init__(self, coordinates):
        points = np.asarray(coordinates, dtype=np.float64)

        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError(
                "Coordinates must have shape (N, 2) or (N, 3)."
            )

        if not np.all(np.isfinite(points)):
            raise ValueError("Coordinates must be finite.")

        self.points = np.array(points, copy=True)
        self.points.flags.writeable = False

        self.dimension = self.points.shape[1]

        self.tree = (
            cKDTree(self.points)
            if len(self.points)
            else None
        )

    def query_radius(self, center, radius):
        """Return original point indices within a radius."""

        center = np.asarray(center, dtype=np.float64)

        if center.shape != (self.dimension,):
            raise ValueError("Invalid query dimensions.")

        if not np.all(np.isfinite(center)):
            raise ValueError("Query coordinates must be finite.")

        if not np.isfinite(radius) or radius < 0:
            raise ValueError("Radius must be finite and nonnegative.")

        if self.tree is None:
            return np.empty(0, dtype=np.int64)

        indices = self.tree.query_ball_point(center, radius)

        return np.asarray(sorted(indices), dtype=np.int64)

    def query_bbox(self, minimum, maximum):
        """Return original indices within an inclusive bounding box."""

        minimum = np.asarray(minimum, dtype=np.float64)
        maximum = np.asarray(maximum, dtype=np.float64)

        if (
            minimum.shape != (self.dimension,)
            or maximum.shape != (self.dimension,)
        ):
            raise ValueError("Invalid bounding-box dimensions.")

        if not (
            np.all(np.isfinite(minimum))
            and np.all(np.isfinite(maximum))
        ):
            raise ValueError("Bounding-box coordinates must be finite.")

        if np.any(minimum > maximum):
            raise ValueError("Minimum exceeds maximum.")

        if self.tree is None:
            return np.empty(0, dtype=np.int64)

        center = (minimum / 2.0) + (maximum / 2.0)
        half_width = (maximum / 2.0) - (minimum / 2.0)

        # An enclosing Euclidean sphere supplies candidates.
        radius = float(np.linalg.norm(half_width))

        candidates = self.tree.query_ball_point(
            center,
            np.nextafter(radius, np.inf),
        )

        candidates = np.asarray(candidates, dtype=np.int64)

        if not len(candidates):
            return candidates

        selected = np.all(
            (self.points[candidates] >= minimum)
            & (self.points[candidates] <= maximum),
            axis=1,
        )

        return np.sort(candidates[selected])

    def query_nearest(self, center, k=1):
        """Return nearest original indices and distances."""

        center = np.asarray(center, dtype=np.float64)

        if center.shape != (self.dimension,):
            raise ValueError("Invalid query dimensions.")

        if not np.all(np.isfinite(center)):
            raise ValueError("Query coordinates must be finite.")

        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer.")

        if self.tree is None:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )

        count = min(k, len(self.points))

        distances, indices = self.tree.query(
            center,
            k=count,
        )

        distances = np.atleast_1d(distances)
        indices = np.atleast_1d(indices)

        # Include all points tied at the kth distance.
        boundary = np.nextafter(float(distances[-1]), np.inf)
        candidates = np.asarray(
            self.tree.query_ball_point(center, boundary),
            dtype=np.int64,
        )

        candidate_distances = np.linalg.norm(
            self.points[candidates] - center,
            axis=1,
        )

        order = np.lexsort((candidates, candidate_distances))
        selected = order[:count]

        return (
            candidates[selected],
            candidate_distances[selected],
        )
