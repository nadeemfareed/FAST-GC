"""Hardening tests for FAST-GIS chunked LAS/LAZ queries.

Tests:
- CRS preservation
- Extra Bytes preservation
- Empty LAS/LAZ files
- Large point-cloud correctness
- Chunk-size independence
- Original point-index preservation
- Source-file integrity
"""

import numpy as np
import laspy
import pytest

from pyproj import CRS

from fastgc.gis.las_query import query_las_chunks


# ------------------------------------------------------------
# Synthetic LAS/LAZ generator
# ------------------------------------------------------------

def create_synthetic_cloud(path, n=1000, seed=42):

    rng = np.random.default_rng(seed)

    header = laspy.LasHeader(
        point_format=3,
        version="1.2",
    )

    header.scales = np.array([
        0.001,
        0.001,
        0.001,
    ])

    header.offsets = np.array([
        500000.0,
        4000000.0,
        0.0,
    ])

    header.add_crs(CRS.from_epsg(32617))

    las = laspy.LasData(header)

    # Generate coordinates on the LAS quantization grid.
    xi = rng.integers(0, 100000, size=n)
    yi = rng.integers(0, 100000, size=n)
    zi = rng.integers(0, 50000, size=n)

    las.X = xi.astype(np.int32)
    las.Y = yi.astype(np.int32)
    las.Z = zi.astype(np.int32)

    las.classification = rng.integers(
        0, 6, size=n, dtype=np.uint8
    )

    las.intensity = rng.integers(
        0, 65535, size=n, dtype=np.uint16
    )

    las.add_extra_dim(
        laspy.ExtraBytesParams(
            name="pred_leaf_prob",
            type=np.float32,
        )
    )

    las.pred_leaf_prob = rng.random(n).astype(np.float32)

    las.write(path)

    return las


def collect_indices(path, bounds, **kwargs):

    results = list(
        query_las_chunks(
            path,
            bounds,
            **kwargs,
        )
    )

    if not results:
        return np.empty(0, dtype=np.int64)

    return np.concatenate([
        result.indices
        for result in results
    ])


# ------------------------------------------------------------
# CRS metadata
# ------------------------------------------------------------

@pytest.mark.parametrize("extension", ["las", "laz"])
def test_crs_preservation(tmp_path, extension):

    path = tmp_path / f"crs_test.{extension}"

    create_synthetic_cloud(path)

    with laspy.open(path) as reader:
        before = reader.header.parse_crs()

    assert before is not None
    assert before.to_epsg() == 32617

    list(
        query_las_chunks(
            path,
            (500000, 4000000, 500100, 4000100),
            chunk_size=100,
        )
    )

    with laspy.open(path) as reader:
        after = reader.header.parse_crs()

    assert after is not None
    assert after == before


# ------------------------------------------------------------
# Extra Bytes
# ------------------------------------------------------------

@pytest.mark.parametrize("extension", ["las", "laz"])
def test_extra_bytes_preservation(tmp_path, extension):

    path = tmp_path / f"extra_test.{extension}"

    original = create_synthetic_cloud(path)

    bounds = (
        500020.0,
        4000020.0,
        500080.0,
        4000080.0,
    )

    results = list(
        query_las_chunks(
            path,
            bounds,
            chunk_size=100,
        )
    )

    indices = np.concatenate([
        result.indices
        for result in results
    ])

    probabilities = np.concatenate([
        np.asarray(result.points.pred_leaf_prob)
        for result in results
    ])

    expected = np.asarray(
        original.pred_leaf_prob
    )[indices]

    np.testing.assert_array_equal(
        probabilities,
        expected,
    )

    assert probabilities.dtype == np.float32


# ------------------------------------------------------------
# Empty LAS/LAZ files
# ------------------------------------------------------------

@pytest.mark.parametrize("extension", ["las", "laz"])
def test_empty_cloud(tmp_path, extension):

    path = tmp_path / f"empty.{extension}"

    header = laspy.LasHeader(
        point_format=3,
        version="1.2",
    )

    las = laspy.LasData(header)
    las.write(path)

    results = list(
        query_las_chunks(
            path,
            (0, 0, 100, 100),
            chunk_size=100,
        )
    )

    assert results == []


# ------------------------------------------------------------
# Large point-cloud correctness
# ------------------------------------------------------------

@pytest.mark.parametrize("extension", ["las", "laz"])
def test_large_cloud(tmp_path, extension):

    path = tmp_path / f"large.{extension}"

    original = create_synthetic_cloud(
        path,
        n=1_000_000,
    )

    bounds = (
        500025.0,
        4000025.0,
        500075.0,
        4000075.0,
    )

    expected = np.flatnonzero(
        (original.x >= bounds[0])
        & (original.x <= bounds[2])
        & (original.y >= bounds[1])
        & (original.y <= bounds[3])
    )

    actual = collect_indices(
        path,
        bounds,
        chunk_size=100_000,
    )

    np.testing.assert_array_equal(
        actual,
        expected,
    )


# ------------------------------------------------------------
# Chunk-size independence
# ------------------------------------------------------------

@pytest.mark.parametrize("extension", ["las", "laz"])
def test_chunk_size_independence(tmp_path, extension):

    path = tmp_path / f"chunks.{extension}"

    create_synthetic_cloud(
        path,
        n=10000,
    )

    bounds = (
        500010.0,
        4000010.0,
        500090.0,
        4000090.0,
    )

    results = []

    for chunk_size in [1, 7, 100, 1000, 10000]:

        indices = collect_indices(
            path,
            bounds,
            chunk_size=chunk_size,
        )

        results.append(indices)

    for indices in results[1:]:

        np.testing.assert_array_equal(
            indices,
            results[0],
        )


# ------------------------------------------------------------
# Source integrity
# ------------------------------------------------------------

@pytest.mark.parametrize("extension", ["las", "laz"])
def test_source_integrity(tmp_path, extension):

    path = tmp_path / f"integrity.{extension}"

    create_synthetic_cloud(
        path,
        n=10000,
    )

    before = path.read_bytes()

    list(
        query_las_chunks(
            path,
            (500000, 4000000, 500100, 4000100),
            chunk_size=1000,
        )
    )

    after = path.read_bytes()

    assert before == after
