"""Canonical FAST-GIS plot geometries."""
from __future__ import annotations

import math

from shapely import affinity
from shapely.geometry import Point, Polygon, box


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def plot_geometry(
    x,
    y,
    *,
    shape="hexagon",
    radius=15.0,
    width=None,
    height=None,
    rotation_deg=0.0,
):
    """Create a canonical metric FAST-GIS plot polygon."""

    shape = str(shape).lower()
    x = float(x)
    y = float(y)
    rotation_deg = float(rotation_deg)

    if not all(math.isfinite(v) for v in (x, y, rotation_deg)):
        raise ValueError("Plot centre and rotation must be finite")

    if shape == "circle":
        radius = _positive(radius, "radius")
        geom = Point(x, y).buffer(radius, quad_segs=32)

    elif shape == "hexagon":
        radius = _positive(radius, "radius")
        geom = Polygon([
            (
                x + radius * math.cos(math.radians(60 * i)),
                y + radius * math.sin(math.radians(60 * i)),
            )
            for i in range(6)
        ])

    elif shape in {"square", "rectangle", "ellipse"}:
        if width is None:
            width = 2.0 * float(radius)

        width = _positive(width, "width")

        if shape == "square":
            height = width
        elif height is None:
            height = width

        height = _positive(height, "height")

        if shape in {"square", "rectangle"}:
            geom = box(
                x - width / 2.0,
                y - height / 2.0,
                x + width / 2.0,
                y + height / 2.0,
            )
        else:
            geom = affinity.scale(
                Point(x, y).buffer(1.0, quad_segs=48),
                xfact=width / 2.0,
                yfact=height / 2.0,
                origin=(x, y),
            )

    else:
        raise ValueError(
            "shape must be circle, hexagon, square, rectangle, or ellipse"
        )

    if rotation_deg:
        geom = affinity.rotate(
            geom,
            rotation_deg,
            origin=(x, y),
            use_radians=False,
        )

    return geom
