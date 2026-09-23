"""Validation tests for FAST-GIS chunked LAS/LAZ queries."""

import laspy
import numpy as np
import pytest

from fastgc.gis.las_query import query_las_chunks


def create_test_las(path):
    """Create a synthetic LAS/LAZ file with known attributes."""

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

    las = laspy.LasData(header)

    las.x = np.array([
        500000.0,
        500001.0,
        500002.0,
        500003.0,
        500004.0,
    ])

    las.y = np.array([
        4000000.0,
        4000001.0,
        4000002.0,
        4000003.0,
        4000004.0,
    ])

    las.z = np.array([
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
    ])

    las.classification = np.array([
        2,
        1,
        5,
        2,
        5,
    ], dtype=np.uint8)

    las.intensity = np.array([
        100,
        200,
        300,
        400,
        500,
    ], dtype=np.uint16)

    las.write(path)


@pytest.fixture(params=["las", "laz"])
def sample_file(tmp_path, request):
    path = tmp_path / f"sample.{request.param}"
    create_test_las(path)
    return path


def collect_results(path, bounds, **kwargs):
    return list(
        query_las_chunks(
            path,
            bounds,
            **kwargs,
        )
    )


def test_xy_query(sample_file):
    results = collect_results(
        sample_file,
        (
            500001.0,
            4000001.0,
            500003.0,
            4000003.0,
        ),
        chunk_size=2,
    )

    indices = np.concatenate([
        result.indices
        for result in results
    ])

    np.testing.assert_array_equal(
        indices,
        [1, 2, 3],
    )


def test_elevation_filter(sample_file):
    results = collect_results(
        sample_file,
        (
            500000.0,
            4000000.0,
            500004.0,
            4000004.0,
        ),
        z_min=2.0,
        z_max=3.0,
        chunk_size=2,
    )

    indices = np.concatenate([
        result.indices
        for result in results
    ])

    np.testing.assert_array_equal(
        indices,
        [2, 3],
    )


def test_attribute_preservation(sample_file):
    results = collect_results(
        sample_file,
        (
            500001.0,
            4000001.0,
            500003.0,
            4000003.0,
        ),
        chunk_size=2,
    )

    classifications = np.concatenate([
        np.asarray(result.points.classification)
        for result in results
    ])

    intensities = np.concatenate([
        np.asarray(result.points.intensity)
        for result in results
    ])

    np.testing.assert_array_equal(
        classifications,
        [1, 5, 2],
    )

    np.testing.assert_array_equal(
        intensities,
        [200, 300, 400],
    )


def test_original_indices(sample_file):
    results = collect_results(
        sample_file,
        (
            500002.0,
            4000002.0,
            500004.0,
            4000004.0,
        ),
        chunk_size=2,
    )

    indices = np.concatenate([
        result.indices
        for result in results
    ])

    np.testing.assert_array_equal(
        indices,
        [2, 3, 4],
    )


def test_empty_query(sample_file):
    results = collect_results(
        sample_file,
        (0.0, 0.0, 1.0, 1.0),
        chunk_size=2,
    )

    assert results == []


def test_invalid_bounds(sample_file):
    with pytest.raises(ValueError):
        list(
            query_las_chunks(
                sample_file,
                (10.0, 10.0, 0.0, 0.0),
            )
        )


def test_invalid_chunk_size(sample_file):
    with pytest.raises(ValueError):
        list(
            query_las_chunks(
                sample_file,
                (0.0, 0.0, 1.0, 1.0),
                chunk_size=0,
            )
        )


def test_invalid_elevation_range(sample_file):
    with pytest.raises(ValueError):
        list(
            query_las_chunks(
                sample_file,
                (0.0, 0.0, 1.0, 1.0),
                z_min=10.0,
                z_max=0.0,
            )
        )


def test_source_file_unchanged(sample_file):
    before = sample_file.read_bytes()

    collect_results(
        sample_file,
        (
            500000.0,
            4000000.0,
            500004.0,
            4000004.0,
        ),
        chunk_size=2,
    )

    after = sample_file.read_bytes()

    assert before == after
