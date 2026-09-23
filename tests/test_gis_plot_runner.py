"""Integration tests for the standalone FAST-GIS multi-plot runner."""
import csv
import json
from pathlib import Path

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
    header.add_extra_dim(laspy.ExtraBytesParams(name="test_extra", type=np.float32))
    las = laspy.LasData(header)
    las.x = np.array([500000, 500005, 500010, 500015, 500020], dtype=float)
    las.y = np.array([5400000, 5400005, 5400010, 5400015, 5400020], dtype=float)
    las.z = np.array([100, 101, 102, 103, 104], dtype=float)
    las.test_extra = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    las.write(source)
    plots = tmp_path / "plots.csv"
    with plots.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["plot_name", "x", "y", "width", "height"])
        writer.writeheader()
        writer.writerow({"plot_name": "First Plot", "x": 500005, "y": 5400005, "width": 10, "height": 10})
        writer.writerow({"plot_name": "Second Plot", "x": 500015, "y": 5400015, "width": 10, "height": 10})
    return source, plots


def test_multiplot_exact_and_buffer(sample, tmp_path):
    source, plots = sample
    workspace = tmp_path / "workspace"
    manifest = extract_plots(source, plots, workspace, source_crs="EPSG:25832", buffer=5, chunk_size=2)
    assert manifest["plot_count"] == 2
    assert (workspace / "workspace_manifest.json").is_file()
    original = laspy.read(source)
    for entry, expected in zip(manifest["plots"], [[0, 1, 2], [2, 3, 4]]):
        core = laspy.read(workspace / entry["core_file"])
        np.testing.assert_array_equal(core.points.array, original.points.array[expected])
        assert entry["core_points"] == 3
        assert entry["buffer_file"] is not None
        assert (workspace / entry["buffer_file"]).is_file()
        assert (workspace / entry["plot_name"] / f"Clip_{entry['plot_name']}.json").is_file()
    saved = json.loads((workspace / "workspace_manifest.json").read_text())
    assert saved["plot_count"] == 2


def test_existing_workspace_is_protected(sample, tmp_path):
    source, plots = sample
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sentinel = workspace / "sentinel.txt"
    sentinel.write_text("untouched")
    with pytest.raises(FileExistsError):
        extract_plots(source, plots, workspace, source_crs="EPSG:25832")
    assert sentinel.read_text() == "untouched"


def test_requires_source_crs_for_csv(sample, tmp_path):
    source, plots = sample
    with pytest.raises(ValueError):
        extract_plots(source, plots, tmp_path / "workspace")
    assert not (tmp_path / "workspace").exists()


def test_no_buffer_creates_only_core(sample, tmp_path):
    source, plots = sample
    workspace = tmp_path / "workspace"
    manifest = extract_plots(source, plots, workspace, source_crs="EPSG:25832", buffer=0)
    assert all(entry["buffer_file"] is None for entry in manifest["plots"])
    assert len(list(workspace.rglob("*.laz"))) == (2 if source.suffix == ".laz" else 0)
    assert len(list(workspace.rglob("*.las"))) == (2 if source.suffix == ".las" else 0)
