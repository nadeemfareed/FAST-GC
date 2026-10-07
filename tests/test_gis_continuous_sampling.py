import numpy as np
from shapely.geometry import Point

from fastgc.gis.geometry import plot_geometry
from fastgc.gis.continuous_sampling import (
    lattice_spacing,
    lattice_centres,
)


def test_square_30m_geometry():
    g = plot_geometry(
        100,
        200,
        shape="square",
        width=30,
    )
    assert np.allclose(
        g.bounds,
        (85, 185, 115, 215),
    )
    assert np.isclose(
        g.area,
        900.0,
    )


def test_rectangle_geometry():
    g = plot_geometry(
        0,
        0,
        shape="rectangle",
        width=40,
        height=20,
    )
    assert np.allclose(
        g.bounds,
        (-20, -10, 20, 10),
    )


def test_ellipse_geometry():
    g = plot_geometry(
        10,
        20,
        shape="ellipse",
        width=40,
        height=20,
    )
    assert g.covers(
        Point(10, 20)
    )


def test_square_30m_5m_overlap():
    sx, sy = lattice_spacing(
        "square",
        15,
        30,
        30,
        5,
    )
    assert sx == 25.0
    assert sy == 25.0


def test_rectangle_spacing():
    sx, sy = lattice_spacing(
        "rectangle",
        15,
        40,
        30,
        5,
    )
    assert sx == 35.0
    assert sy == 25.0


def test_grid_spans_source_bounds():
    centres = lattice_centres(
        (0, 0, 100, 100),
        25,
        25,
        0,
        0,
        shape="square",
    )

    xs = [x for x, _ in centres]
    ys = [y for _, y in centres]

    assert min(xs) <= 0
    assert max(xs) >= 100
    assert min(ys) <= 0
    assert max(ys) >= 100
