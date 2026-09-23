"""Additional correctness and concurrency tests for FAST-GIS."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from fastgc.gis.spatial_index import PointSpatialIndex


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("seed", [11, 42, 99])
def test_randomized_radius(dimension, seed):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-100, 100, (2000, dimension))
    index = PointSpatialIndex(points)

    for _ in range(20):
        center = rng.uniform(-100, 100, dimension)
        radius = float(rng.uniform(0, 50))

        expected = np.flatnonzero(
            np.linalg.norm(points - center, axis=1) <= radius
        )

        actual = index.query_radius(center, radius)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dimension", [2, 3])
def test_randomized_bbox(dimension):
    rng = np.random.default_rng(123)
    points = rng.uniform(-100, 100, (2000, dimension))
    index = PointSpatialIndex(points)

    for _ in range(20):
        a = rng.uniform(-100, 100, dimension)
        b = rng.uniform(-100, 100, dimension)
        minimum = np.minimum(a, b)
        maximum = np.maximum(a, b)

        expected = np.flatnonzero(
            np.all(
                (points >= minimum) & (points <= maximum),
                axis=1,
            )
        )

        actual = index.query_bbox(minimum, maximum)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dimension", [2, 3])
def test_randomized_nearest(dimension):
    rng = np.random.default_rng(456)
    points = rng.uniform(-100, 100, (2000, dimension))
    index = PointSpatialIndex(points)

    for _ in range(20):
        center = rng.uniform(-100, 100, dimension)
        k = 5

        distances = np.linalg.norm(points - center, axis=1)
        expected = np.lexsort(
            (np.arange(len(points)), distances)
        )[:k]

        actual, actual_distances = index.query_nearest(center, k)

        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(
            actual_distances,
            distances[expected],
        )


@pytest.mark.parametrize("dimension", [2, 3])
def test_large_projected_coordinates(dimension):
    points = np.full((3, dimension), 5_000_000.0)
    points[1, 0] += 0.25
    points[2, 0] += 2.0

    index = PointSpatialIndex(points)

    actual = index.query_radius(points[0], 0.5)
    np.testing.assert_array_equal(actual, [0, 1])


def test_duplicate_nearest_neighbors():
    points = np.array([
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ])

    index = PointSpatialIndex(points)

    actual, distances = index.query_nearest([1.0, 1.0], k=2)

    np.testing.assert_array_equal(actual, [0, 1])
    np.testing.assert_allclose(distances, [0.0, 0.0])


def test_zero_radius():
    points = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.001],
    ])

    index = PointSpatialIndex(points)

    actual = index.query_radius([0.0, 0.0], 0.0)
    np.testing.assert_array_equal(actual, [0, 1])


def test_concurrent_read_only_queries():
    rng = np.random.default_rng(789)
    points = rng.uniform(-100, 100, (10000, 3))
    centers = rng.uniform(-100, 100, (100, 3))

    index = PointSpatialIndex(points)

    expected = [
        index.query_radius(center, 15.0)
        for center in centers
    ]

    with ThreadPoolExecutor(max_workers=8) as executor:
        actual = list(
            executor.map(
                lambda center: index.query_radius(center, 15.0),
                centers,
            )
        )

    for result, reference in zip(actual, expected):
        np.testing.assert_array_equal(result, reference)


def test_input_coordinates_are_copied():
    points = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
    ])

    index = PointSpatialIndex(points)
    points[0] = [100.0, 100.0]

    actual = index.query_radius([0.0, 0.0], 0.0)
    np.testing.assert_array_equal(actual, [0])
