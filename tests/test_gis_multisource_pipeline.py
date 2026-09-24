import json
import laspy
import numpy as np
from pyproj import CRS
from fastgc.gis.plot_runner import extract_plots


def _write(path, xyz):
    h=laspy.LasHeader(point_format=3,version="1.2"); h.scales=[.01,.01,.01]; h.add_crs(CRS.from_epsg(32618))
    las=laspy.LasData(h); a=np.asarray(xyz,float); las.x,las.y,las.z=a[:,0],a[:,1],a[:,2]; las.write(path)


def test_plot_crosses_tiles_and_exact_xyz_is_deduplicated(tmp_path):
    src=tmp_path/"survey"; src.mkdir()
    _write(src/"a.las",[(0,0,1),(5,5,2),(10,5,3)]); _write(src/"b.las",[(10,5,3),(15,5,4),(20,10,5)])
    plots=tmp_path/"plots.geojson"; plots.write_text(json.dumps({"type":"FeatureCollection","name":"plots",
      "crs":{"type":"name","properties":{"name":"EPSG:32618"}},"features":[{"type":"Feature","properties":{"plot_name":"P1"},
      "geometry":{"type":"Polygon","coordinates":[[[4,4],[16,4],[16,6],[4,6],[4,4]]]}}]}))
    out=tmp_path/"out"; manifest=extract_plots(src,plots,out,buffer=0,sensor_mode="ALS")
    meta=json.loads((out/"P1"/"Clip_P1.json").read_text())
    assert manifest["schema_version"]==3
    assert meta["source_count"]==2 and meta["duplicates_removed"]==1 and meta["points_written"]==3
    with laspy.open(out/"P1"/"P1.las") as r: assert r.header.point_count==3
