import json
import laspy
import numpy as np
from pyproj import CRS
from shapely.geometry import shape as shapely_shape

from fastgc.gis.geometry import plot_geometry
from fastgc.gis.plot_runner import extract_plots


def _write(path, xyz):
    h = laspy.LasHeader(point_format=3, version="1.2")
    h.scales = [0.01, 0.01, 0.01]
    h.add_crs(CRS.from_epsg(32618))
    las = laspy.LasData(h)
    arr = np.asarray(xyz, float)
    las.x, las.y, las.z = arr[:, 0], arr[:, 1], arr[:, 2]
    las.write(path)


def test_hexagon_uses_circumradius():
    g = plot_geometry(100, 200, shape="hexagon", radius=15)
    coords = list(g.exterior.coords)[:-1]
    assert len(coords) == 6
    assert all(abs(((x-100)**2 + (y-200)**2)**0.5 - 15) < 1e-9 for x, y in coords)


def test_one_buffered_file_and_hierarchical_metadata(tmp_path):
    src = tmp_path / "source.las"
    _write(src, [
        (0, 0, 1),
        (14, 0, 2),
        (18, 0, 3),
        (30, 0, 4),
        (0, 1, 1),
    ])
    core = plot_geometry(0, 0, shape="hexagon", radius=15)
    plots = tmp_path / "plots.geojson"
    plots.write_text(json.dumps({
        "type":"FeatureCollection", "name":"plots",
        "crs":{"type":"name","properties":{"name":"EPSG:32618"}},
        "features":[{"type":"Feature","properties":{"plot_name":"Plot_001","shape":"hexagon","radius_m":15},
                     "geometry":core.__geo_interface__}]
    }))
    out = tmp_path / "ALS_plots"
    manifest = extract_plots(src, plots, out, buffer=5, sensor_mode="ALS",
                             sampling_method="random_observed_coverage", shape="hexagon", radius=15)
    point_files = list((out / "Plot_001").glob("*.las"))
    assert [p.name for p in point_files] == ["Plot_001.las"]
    assert not list(out.rglob("*_buffer*.las"))
    meta = json.loads((out / "Plot_001" / "Clip_Plot_001.json").read_text())
    assert meta["sensor_mode"] == "ALS"
    assert meta["point_extent"] == "core_plus_buffer"
    assert meta["buffer_m"] == 5.0
    assert meta["core_radius_m"] == 15.0
    assert meta["radius_definition"] == "circumradius"
    assert meta["source_definition_file"] == "plots.geojson"
    assert "_stage_" not in meta["source_definition_file"]
    assert meta["points_written"] == 4
    assert meta["core_points"] == 3
    assert shapely_shape(meta["buffered_geometry"]).contains(shapely_shape(meta["core_geometry"]))
    queue = json.loads((out / "plots_manifest.json").read_text())
    assert queue["sensor_mode"] == "ALS"
    assert queue["plots"][0]["point_file"] == "Plot_001/Plot_001.las"
    assert manifest["schema_version"] == 3
