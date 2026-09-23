import laspy
import numpy as np
import pytest

from pyproj import CRS
from shapely.geometry import Polygon, MultiPolygon

from fastgc.gis.clip import clip_las


@pytest.fixture(params=["las", "laz"])
def polygon_cloud(tmp_path, request):

    source = tmp_path / f"source.{request.param}"

    header = laspy.LasHeader(
        point_format=6,
        version="1.4",
    )

    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [500000, 5400000, 0]
    header.add_crs(CRS.from_epsg(25832))

    las = laspy.LasData(header)

    las.x = np.array(
        [500000, 500005, 500010, 500015, 500020],
        dtype=np.float64,
    )

    las.y = np.array(
        [5400000, 5400005, 5400010, 5400015, 5400020],
        dtype=np.float64,
    )

    las.z = np.array(
        [100, 101, 102, 103, 104],
        dtype=np.float64,
    )
    las.classification = [2, 1, 5, 2, 5]

    las.add_extra_dim(
        laspy.ExtraBytesParams(
            name="test_extra",
            type=np.float32,
        )
    )

    las.test_extra = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.5],
        dtype=np.float32,
    )

    las.write(source)

    return source


def test_exact_polygon_clip(polygon_cloud, tmp_path):

    polygon = Polygon([
        (500005, 5400005),
        (500015, 5400005),
        (500015, 5400015),
        (500005, 5400015),
    ])

    output = tmp_path / f"polygon{polygon_cloud.suffix}"

    report = clip_las(
        polygon_cloud,
        output,
        geometry=polygon,
        chunk_size=2,
        write_report=False,
    )

    assert report["selected_points"] == 3
    assert report["selection_type"] == "polygon"

    original = laspy.read(polygon_cloud)
    clipped = laspy.read(output)

    np.testing.assert_array_equal(
        clipped.points.array,
        original.points.array[[1, 2, 3]],
    )


def test_polygon_buffer(polygon_cloud, tmp_path):

    polygon = Polygon([
        (500010, 5400010),
        (500011, 5400010),
        (500011, 5400011),
        (500010, 5400011),
    ])

    original = laspy.read(polygon_cloud)

    # A 5 m polygon buffer includes only the central point.
    output5 = tmp_path / f"buffer5{polygon_cloud.suffix}"

    report5 = clip_las(
        polygon_cloud,
        output5,
        geometry=polygon,
        buffer=5,
        chunk_size=2,
        write_report=False,
    )

    assert report5["selected_points"] == 1

    clipped5 = laspy.read(output5)

    np.testing.assert_array_equal(
        clipped5.points.array,
        original.points.array[[2]],
    )

    # An 8 m buffer also includes the two neighboring points.
    output8 = tmp_path / f"buffer8{polygon_cloud.suffix}"

    report8 = clip_las(
        polygon_cloud,
        output8,
        geometry=polygon,
        buffer=8,
        chunk_size=2,
        write_report=False,
    )

    assert report8["selected_points"] == 3

    clipped8 = laspy.read(output8)

    np.testing.assert_array_equal(
        clipped8.points.array,
        original.points.array[[1, 2, 3]],
    )


def test_polygon_hole(polygon_cloud, tmp_path):

    polygon = Polygon(
        shell=[
            (500000, 5400000),
            (500020, 5400000),
            (500020, 5400020),
            (500000, 5400020),
        ],
        holes=[[
            (500009, 5400009),
            (500011, 5400009),
            (500011, 5400011),
            (500009, 5400011),
        ]],
    )

    output = tmp_path / f"hole{polygon_cloud.suffix}"

    report = clip_las(
        polygon_cloud,
        output,
        geometry=polygon,
        write_report=False,
    )

    assert report["selected_points"] == 4


def test_multipolygon(polygon_cloud, tmp_path):

    first = Polygon([
        (499999, 5399999),
        (500001, 5399999),
        (500001, 5400001),
        (499999, 5400001),
    ])

    second = Polygon([
        (500019, 5400019),
        (500021, 5400019),
        (500021, 5400021),
        (500019, 5400021),
    ])

    output = tmp_path / f"multi{polygon_cloud.suffix}"

    report = clip_las(
        polygon_cloud,
        output,
        geometry=MultiPolygon([first, second]),
        write_report=False,
    )

    assert report["selected_points"] == 2


def test_existing_rectangular_interface(
    polygon_cloud,
    tmp_path,
):

    output = tmp_path / f"rectangle{polygon_cloud.suffix}"

    report = clip_las(
        polygon_cloud,
        output,
        (500005, 5400005, 500015, 5400015),
        write_report=False,
    )

    assert report["selected_points"] == 3
    assert report["selection_type"] == "bounds"


def test_invalid_polygon(polygon_cloud, tmp_path):

    polygon = Polygon([
        (0, 0),
        (2, 2),
        (0, 2),
        (2, 0),
    ])

    output = tmp_path / f"invalid{polygon_cloud.suffix}"

    with pytest.raises(ValueError):
        clip_las(
            polygon_cloud,
            output,
            geometry=polygon,
        )

    assert not output.exists()


def test_polygon_empty_selection(polygon_cloud, tmp_path):

    polygon = Polygon([
        (0, 0),
        (10, 0),
        (10, 10),
        (0, 10),
    ])

    output = tmp_path / f"empty_polygon{polygon_cloud.suffix}"

    report = clip_las(
        polygon_cloud,
        output,
        geometry=polygon,
        write_report=False,
    )

    assert report["selected_points"] == 0
    assert len(laspy.read(output).points) == 0


def test_polygon_metadata_preservation(polygon_cloud, tmp_path):

    polygon = Polygon([
        (500000, 5400000),
        (500020, 5400000),
        (500020, 5400020),
        (500000, 5400020),
    ])

    output = tmp_path / f"metadata_polygon{polygon_cloud.suffix}"

    clip_las(
        polygon_cloud,
        output,
        geometry=polygon,
        chunk_size=2,
        write_report=False,
    )

    original = laspy.read(polygon_cloud)
    clipped = laspy.read(output)

    assert clipped.header.point_format.id == original.header.point_format.id
    assert str(clipped.header.version) == str(original.header.version)

    assert clipped.header.parse_crs().equals(
        original.header.parse_crs()
    )

    np.testing.assert_array_equal(
        clipped.header.scales,
        original.header.scales,
    )

    np.testing.assert_array_equal(
        clipped.header.offsets,
        original.header.offsets,
    )

    np.testing.assert_array_equal(
        clipped.points.array,
        original.points.array,
    )


def test_polygon_existing_output_protected(polygon_cloud, tmp_path):

    polygon = Polygon([
        (500000, 5400000),
        (500020, 5400000),
        (500020, 5400020),
        (500000, 5400020),
    ])

    output = tmp_path / f"existing_polygon{polygon_cloud.suffix}"
    output.write_bytes(b"existing content")

    with pytest.raises(FileExistsError):
        clip_las(
            polygon_cloud,
            output,
            geometry=polygon,
        )

    assert output.read_bytes() == b"existing content"


def test_polygon_existing_report_protected(polygon_cloud, tmp_path):

    polygon = Polygon([
        (500000, 5400000),
        (500020, 5400000),
        (500020, 5400020),
        (500000, 5400020),
    ])

    output = tmp_path / f"report_polygon{polygon_cloud.suffix}"
    report_path = output.with_suffix(".json")

    report_path.write_text(
        '{"existing": true}',
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError):
        clip_las(
            polygon_cloud,
            output,
            geometry=polygon,
            write_report=True,
        )

    assert not output.exists()
    assert report_path.read_text(
        encoding="utf-8"
    ) == '{"existing": true}'
