"""Tests for the FAST-GIS LAS/LAZ clipping module."""

import laspy
import numpy as np
import pytest

from pyproj import CRS

from fastgc.gis.clip import clip_las


@pytest.fixture(params=["las", "laz"])
def sample_cloud(tmp_path, request):

    source = tmp_path / f"source.{request.param}"

    header = laspy.LasHeader(
        point_format=6,
        version="1.4",
    )

    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([500000.0, 5400000.0, 0.0])
    header.add_crs(CRS.from_epsg(25832))

    las = laspy.LasData(header)

    las.x = [500000, 500005, 500010, 500015, 500020]
    las.y = [5400000, 5400005, 5400010, 5400015, 5400020]
    las.z = [100, 101, 102, 103, 104]

    las.classification = [2, 1, 5, 2, 5]
    las.intensity = [100, 200, 300, 400, 500]

    las.add_extra_dim(
        laspy.ExtraBytesParams(
            name="pred_leaf_prob",
            type=np.float32,
        )
    )

    las.pred_leaf_prob = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.5],
        dtype=np.float32,
    )

    las.write(source)

    return source


def test_exact_clip(sample_cloud, tmp_path):

    output = tmp_path / f"exact{sample_cloud.suffix}"

    report = clip_las(
        sample_cloud,
        output,
        (500005, 5400005, 500015, 5400015),
        chunk_size=2,
        write_report=False,
    )

    assert report["selected_points"] == 3

    original = laspy.read(sample_cloud)
    clipped = laspy.read(output)

    np.testing.assert_array_equal(
        clipped.points.array,
        original.points.array[[1, 2, 3]],
    )


def test_buffered_clip(sample_cloud, tmp_path):

    output = tmp_path / f"buffered{sample_cloud.suffix}"

    report = clip_las(
        sample_cloud,
        output,
        (500010, 5400010, 500010, 5400010),
        buffer=5,
        chunk_size=2,
        write_report=False,
    )

    assert report["selected_points"] == 3


def test_metadata_preservation(sample_cloud, tmp_path):

    output = tmp_path / f"metadata{sample_cloud.suffix}"

    clip_las(
        sample_cloud,
        output,
        (500000, 5400000, 500020, 5400020),
        write_report=False,
    )

    source = laspy.read(sample_cloud)
    clipped = laspy.read(output)

    assert source.header.point_format.id == clipped.header.point_format.id
    assert str(source.header.version) == str(clipped.header.version)

    assert source.header.parse_crs().equals(
        clipped.header.parse_crs()
    )

    np.testing.assert_array_equal(
        source.header.scales,
        clipped.header.scales,
    )

    np.testing.assert_array_equal(
        source.header.offsets,
        clipped.header.offsets,
    )

    np.testing.assert_array_equal(
        source.pred_leaf_prob,
        clipped.pred_leaf_prob,
    )


def test_empty_selection(sample_cloud, tmp_path):

    output = tmp_path / f"empty{sample_cloud.suffix}"

    report = clip_las(
        sample_cloud,
        output,
        (0, 0, 10, 10),
        write_report=False,
    )

    assert report["selected_points"] == 0
    assert len(laspy.read(output).points) == 0


def test_existing_output_protected(sample_cloud, tmp_path):

    output = tmp_path / f"existing{sample_cloud.suffix}"
    output.write_bytes(b"existing content")

    with pytest.raises(FileExistsError):
        clip_las(
            sample_cloud,
            output,
            (0, 0, 10, 10),
        )

    assert output.read_bytes() == b"existing content"


def test_invalid_bounds(sample_cloud, tmp_path):

    output = tmp_path / f"invalid{sample_cloud.suffix}"

    with pytest.raises(ValueError):
        clip_las(
            sample_cloud,
            output,
            (10, 10, 0, 0),
        )

    assert not output.exists()


def test_existing_report_protected(sample_cloud, tmp_path):

    output = tmp_path / f"report{sample_cloud.suffix}"
    report_path = output.with_suffix(".json")

    report_path.write_text(
        '{"existing": true}',
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError):
        clip_las(
            sample_cloud,
            output,
            (500000, 5400000, 500020, 5400020),
            write_report=True,
        )

    assert not output.exists()

    assert report_path.read_text(
        encoding="utf-8"
    ) == '{"existing": true}'
