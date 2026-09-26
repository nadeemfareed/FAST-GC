"""Integration tests for the FAST-GIS multi-plot runner."""

import csv
import json

import laspy
import numpy as np
import pytest

from fastgc.gis.plot_runner import extract_plots


@pytest.fixture(params=["las", "laz"])
def sample(tmp_path, request):
    source = tmp_path / f"source.{request.param}"

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [500000, 5400000, 0]
    header.add_crs(__import__("pyproj").CRS.from_epsg(25832))
    header.add_extra_dim(
        laspy.ExtraBytesParams(name="test_extra", type=np.float32)
    )

    las = laspy.LasData(header)
    las.x = np.array(
        [500000, 500005, 500010, 500015, 500020],
        dtype=float,
    )
    las.y = np.array(
        [5400000, 5400005, 5400010, 5400015, 5400020],
        dtype=float,
    )
    las.z = np.array([100, 101, 102, 103, 104], dtype=float)
    las.test_extra = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    las.write(source)

    plots = tmp_path / "plots.csv"
    with plots.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["plot_name", "x", "y", "width", "height"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "plot_name": "First Plot",
                "x": 500005,
                "y": 5400005,
                "width": 10,
                "height": 10,
            }
        )
        writer.writerow(
            {
                "plot_name": "Second Plot",
                "x": 500015,
                "y": 5400015,
                "width": 10,
                "height": 10,
            }
        )

    return source, plots


def _load_plots_manifest(workspace):
    workspace_manifest = json.loads(
        (workspace / "workspace_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    plots_manifest = json.loads(
        (workspace / workspace_manifest["plots_manifest"]).read_text(
            encoding="utf-8"
        )
    )

    return workspace_manifest, plots_manifest


def test_multiplot_exact_and_buffer(sample, tmp_path):
    source, plots = sample
    workspace = tmp_path / "workspace"

    manifest = extract_plots(
        source,
        plots,
        workspace,
        source_crs="EPSG:25832",
        buffer=5,
        chunk_size=2,
    )

    assert manifest["plot_count"] == 2
    assert (workspace / "workspace_manifest.json").is_file()

    workspace_manifest, plots_manifest = _load_plots_manifest(workspace)

    assert workspace_manifest["plot_count"] == 2
    assert plots_manifest["plot_count"] == 2
    assert len(plots_manifest["plots"]) == 2

    original = laspy.read(source)

    # A single output point cloud now contains core + buffer.
    expected_buffered = [
        [0, 1, 2],
        [2, 3, 4],
    ]

    for entry, expected in zip(
        plots_manifest["plots"],
        expected_buffered,
    ):
        point_path = workspace / entry["point_file"]

        assert point_path.is_file()

        # Exactly one LAS/LAZ point cloud per plot directory.
        plot_dir = point_path.parent
        point_files = list(plot_dir.glob("*.las")) + list(
            plot_dir.glob("*.laz")
        )
        assert len(point_files) == 1
        assert point_files[0] == point_path

        clipped = laspy.read(point_path)
        np.testing.assert_array_equal(
            clipped.points.array,
            original.points.array[expected],
        )

        metadata_path = workspace / entry["metadata"]
        assert metadata_path.is_file()

        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        assert metadata["point_file"] == entry["point_file"]
        assert metadata["point_extent"] == "core_plus_buffer"
        assert metadata["buffer_m"] == 5.0
        assert metadata["points_written"] == 3

        # The former two-file contract must not return.
        assert "core_file" not in entry
        assert "buffer_file" not in entry
        assert not list(plot_dir.glob("*_buffer*.las"))
        assert not list(plot_dir.glob("*_buffer*.laz"))


def test_existing_workspace_is_protected(sample, tmp_path):
    source, plots = sample
    workspace = tmp_path / "workspace"

    workspace.mkdir()
    sentinel = workspace / "sentinel.txt"
    sentinel.write_text("untouched")

    with pytest.raises(FileExistsError):
        extract_plots(
            source,
            plots,
            workspace,
            source_crs="EPSG:25832",
        )

    assert sentinel.read_text() == "untouched"


def test_requires_source_crs_for_csv(sample, tmp_path):
    source, plots = sample

    with pytest.raises(ValueError):
        extract_plots(
            source,
            plots,
            tmp_path / "workspace",
        )

    assert not (tmp_path / "workspace").exists()


def test_no_buffer_creates_only_core(sample, tmp_path):
    source, plots = sample
    workspace = tmp_path / "workspace"

    manifest = extract_plots(
        source,
        plots,
        workspace,
        source_crs="EPSG:25832",
        buffer=0,
    )

    assert manifest["plot_count"] == 2

    workspace_manifest, plots_manifest = _load_plots_manifest(workspace)

    assert workspace_manifest["plot_count"] == 2
    assert plots_manifest["plot_count"] == 2
    assert len(plots_manifest["plots"]) == 2

    for entry in plots_manifest["plots"]:
        point_path = workspace / entry["point_file"]
        metadata_path = workspace / entry["metadata"]

        assert point_path.is_file()
        assert metadata_path.is_file()

        plot_dir = point_path.parent

        point_files = list(plot_dir.glob("*.las")) + list(
            plot_dir.glob("*.laz")
        )

        # Still exactly one point cloud when buffer == 0.
        assert len(point_files) == 1

        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        assert metadata["buffer_m"] == 0.0
        assert metadata["point_extent"] == "core"
        assert metadata["point_file"] == entry["point_file"]

        assert "core_file" not in entry
        assert "buffer_file" not in entry
        assert not list(plot_dir.glob("*_buffer*.las"))
        assert not list(plot_dir.glob("*_buffer*.laz"))

    # Preserve the input compression type.
    if source.suffix == ".laz":
        assert len(list(workspace.rglob("*.laz"))) == 2
        assert len(list(workspace.rglob("*.las"))) == 0
    else:
        assert len(list(workspace.rglob("*.las"))) == 2
        assert len(list(workspace.rglob("*.laz"))) == 0

def test_crsless_las_with_explicit_source_crs(tmp_path):
    import json
    import laspy
    import numpy as np
    from pyproj import CRS
    from shapely.geometry import Point, mapping
    from fastgc.gis.plot_runner import extract_plots

    source = tmp_path / "local_header.las"
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.01, 0.01, 0.01])
    las = laspy.LasData(header)
    las.x = np.array([500000.0, 500001.0, 500002.0])
    las.y = np.array([7000000.0, 7000001.0, 7000002.0])
    las.z = np.array([100.0, 101.0, 102.0])
    las.write(source)
    assert laspy.read(source).header.parse_crs() is None

    definitions = tmp_path / "plots.geojson"
    geometry = Point(500001.0, 7000001.0).buffer(5.0)
    definitions.write_text(json.dumps({
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:32756"}},
        "features": [{
            "type": "Feature",
            "properties": {"plot_name": "Plot_001"},
            "geometry": mapping(geometry),
        }],
    }), encoding="utf-8")

    output = tmp_path / "extracted"
    result = extract_plots(
        source, definitions, output,
        source_crs="EPSG:32756",
        sensor_mode="TLS", buffer=0.0,
    )
    assert result["plot_count"] == 1
    metadata = json.loads(
        (output / "Plot_001" / "Clip_Plot_001.json").read_text(encoding="utf-8")
    )
    assert CRS.from_user_input(metadata["processing_crs"]) == CRS.from_epsg(32756)
    assert metadata["points_written"] == 3
    assert (output / "Plot_001" / "Plot_001.las").is_file()
