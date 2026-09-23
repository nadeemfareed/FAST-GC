"""Tests for FAST-GIS point spatial indexing."""

import numpy as np
import pytest

from fastgc.gis.spatial_index import PointSpatialIndex


def test_radius_query():
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
    ])

    index = PointSpatialIndex(points)

    result = index.query_radius([0.0, 0.0], 1.0)

    np.testing.assert_array_equal(result, [0, 1])


def test_bbox_query():
    points = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ])

    index = PointSpatialIndex(points)

    result = index.query_bbox(
        [0.0, 0.0],
        [1.0, 1.0],
    )

    np.testing.assert_array_equal(result, [0, 1])


def test_3d_radius():
    points = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 5.0],
    ])

    index = PointSpatialIndex(points)

    result = index.query_radius(
        [0.0, 0.0, 0.0],
        1.0,
    )

    np.testing.assert_array_equal(result, [0])


def test_nearest():
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
    ])

    index = PointSpatialIndex(points)

    indices, distances = index.query_nearest(
        [0.0, 0.0],
        k=2,
    )

    np.testing.assert_array_equal(indices, [0, 1])
    np.testing.assert_allclose(distances, [0.0, 1.0])


def test_empty_index():
    index = PointSpatialIndex(
        np.empty((0, 3))
    )

    result = index.query_radius(
        [0.0, 0.0, 0.0],
        1.0,
    )

    assert len(result) == 0


def test_invalid_coordinates():
    with pytest.raises(ValueError):
        PointSpatialIndex([
            [0.0, np.nan]
        ])


def test_duplicate_points():
    points = np.array([
        [1.0, 1.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ])

    index = PointSpatialIndex(points)

    result = index.query_radius(
        [1.0, 1.0],
        0.0,
    )

    np.testing.assert_array_equal(result, [0, 1])


def test_nearest_tie():
    points = np.array([
        [-1.0, 0.0],
        [1.0, 0.0],
    ])

    index = PointSpatialIndex(points)

    indices, distances = index.query_nearest(
        [0.0, 0.0],
        k=2,
    )

    np.testing.assert_array_equal(indices, [0, 1])
    np.testing.assert_allclose(distances, [1.0, 1.0])
