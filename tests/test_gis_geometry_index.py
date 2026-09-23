"""FAST-GIS vector spatial-indexing validation."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from shapely.geometry import (
    Point,
    LineString,
    Polygon,
    box,
)

from fastgc.gis.spatial_index import GeometrySpatialIndex


def test_bbox_query():
    geometries = [
        Point(0, 0),
        Point(5, 5),
        Point(10, 10),
    ]

    index = GeometrySpatialIndex(geometries)

    actual = index.query_bbox((-1, -1, 6, 6))

    np.testing.assert_array_equal(actual, [0, 1])


def test_bbox_boundary():
    index = GeometrySpatialIndex([
        Point(0, 0),
        Point(1, 1),
    ])

    actual = index.query_bbox((0, 0, 1, 1))

    np.testing.assert_array_equal(actual, [0, 1])


def test_exact_intersection():
    geometries = [
        LineString([(0, 0), (10, 10)]),
        LineString([(0, 10), (10, 20)]),
    ]

    index = GeometrySpatialIndex(geometries)

    actual = index.query_intersects(
        box(4, 4, 6, 6)
    )

    np.testing.assert_array_equal(actual, [0])


def test_polygon_intersection():
    geometries = [
        box(0, 0, 2, 2),
        box(5, 5, 7, 7),
    ]

    index = GeometrySpatialIndex(geometries)

    actual = index.query_intersects(
        box(1, 1, 3, 3)
    )

    np.testing.assert_array_equal(actual, [0])


def test_nearest_geometry():
    geometries = [
        Point(0, 0),
        Point(10, 0),
        Point(20, 0),
    ]

    index = GeometrySpatialIndex(geometries)

    indices, distances = index.query_nearest(
        Point(9, 0)
    )

    np.testing.assert_array_equal(indices, [1])
    np.testing.assert_allclose(distances, [1.0])


def test_nearest_ties():
    geometries = [
        Point(-1, 0),
        Point(1, 0),
    ]

    index = GeometrySpatialIndex(geometries)

    indices, distances = index.query_nearest(
        Point(0, 0)
    )

    np.testing.assert_array_equal(indices, [0, 1])
    np.testing.assert_allclose(distances, [1, 1])


def test_duplicate_geometries():
    geometries = [
        Point(1, 1),
        Point(1, 1),
        Point(5, 5),
    ]

    index = GeometrySpatialIndex(geometries)

    actual = index.query_intersects(Point(1, 1))

    np.testing.assert_array_equal(actual, [0, 1])


def test_empty_index():
    index = GeometrySpatialIndex([])

    assert len(index.query_bbox((0, 0, 1, 1))) == 0

    assert len(
        index.query_intersects(Point(0, 0))
    ) == 0

    indices, distances = index.query_nearest(
        Point(0, 0)
    )

    assert len(indices) == 0
    assert len(distances) == 0


def test_invalid_geometry():
    with pytest.raises(TypeError):
        GeometrySpatialIndex(["not a geometry"])


def test_empty_geometry():
    with pytest.raises(ValueError):
        GeometrySpatialIndex([Point()])


def test_invalid_bbox():
    index = GeometrySpatialIndex([Point(0, 0)])

    with pytest.raises(ValueError):
        index.query_bbox((10, 10, 0, 0))


def test_3d_geometry_uses_xy():
    geometries = [
        Point(0, 0, 0),
        Point(0, 0, 100),
    ]

    index = GeometrySpatialIndex(geometries)

    actual = index.query_intersects(Point(0, 0, 50))

    np.testing.assert_array_equal(actual, [0, 1])


def test_original_indices_preserved():
    geometries = [
        Point(10, 10),
        Point(0, 0),
        Point(5, 5),
    ]

    index = GeometrySpatialIndex(geometries)

    actual = index.query_bbox((-1, -1, 6, 6))

    np.testing.assert_array_equal(actual, [1, 2])


def test_concurrent_queries():
    rng = np.random.default_rng(123)

    coordinates = rng.uniform(
        -100,
        100,
        size=(1000, 2),
    )

    geometries = [
        Point(x, y)
        for x, y in coordinates
    ]

    index = GeometrySpatialIndex(geometries)

    queries = [
        box(x, y, x + 10, y + 10)
        for x, y in rng.uniform(
            -100,
            90,
            size=(50, 2),
        )
    ]

    expected = [
        index.query_intersects(query)
        for query in queries
    ]

    with ThreadPoolExecutor(max_workers=8) as executor:
        actual = list(
            executor.map(
                index.query_intersects,
                queries,
            )
        )

    for result, reference in zip(actual, expected):
        np.testing.assert_array_equal(result, reference)
