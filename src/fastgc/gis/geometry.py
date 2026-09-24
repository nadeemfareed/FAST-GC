"""Canonical FAST-GIS plot geometries."""
from __future__ import annotations

import math
from shapely.geometry import Point, Polygon


def plot_geometry(x, y, *, shape="hexagon", radius=15.0):
    shape = str(shape).lower()
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be a finite positive number")
    x, y = float(x), float(y)
    if shape == "circle":
        return Point(x, y).buffer(radius, quad_segs=32)
    if shape == "hexagon":
        # radius is circumradius: centre to vertex.
        vertices = [
            (x + radius * math.cos(math.radians(60 * i)),
             y + radius * math.sin(math.radians(60 * i)))
            for i in range(6)
        ]
        return Polygon(vertices)
    raise ValueError("shape must be circle or hexagon")
