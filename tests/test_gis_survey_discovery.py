import laspy
import numpy as np
from pyproj import CRS

from fastgc.gis.survey_catalog import build_survey_catalog
from fastgc.gis.survey_regions import discover_survey_regions


def _tile(path, x, y, *, crs=True):
    h = laspy.LasHeader(point_format=3, version='1.2')
    h.scales = [0.01, 0.01, 0.01]
    if crs:
        h.add_crs(CRS.from_epsg(25832))
    las = laspy.LasData(h)
    las.x = np.array([x, x+10])
    las.y = np.array([y, y+10])
    las.z = np.array([1, 2])
    las.write(path)


def test_three_disconnected_regions(tmp_path):
    for name, x in [('a', 0), ('b', 10), ('c', 1000), ('d', 1010), ('e', 5000)]:
        _tile(tmp_path / f'{name}.las', x, 0)
    catalog = build_survey_catalog(tmp_path)
    result = discover_survey_regions(catalog)
    assert catalog['ready_count'] == 5
    assert sorted(r['tile_count'] for r in result['regions']) == [1, 2, 2]
    assert result['overall_bounds'] == [0, 0, 5010, 10]


def test_sidecar_crs_recovery(tmp_path):
    path = tmp_path / 'tile.las'
    _tile(path, 0, 0, crs=False)
    path.with_suffix('.prj').write_text(CRS.from_epsg(25832).to_wkt(), encoding='utf-8')
    catalog = build_survey_catalog(path)
    assert catalog['tiles'][0]['status'] == 'ready'
    assert catalog['tiles'][0]['crs_source'] == 'sidecar_prj'


def test_unresolved_crs_is_excluded(tmp_path):
    _tile(tmp_path / 'tile.las', 0, 0, crs=False)
    catalog = build_survey_catalog(tmp_path)
    assert catalog['tiles'][0]['status'] == 'unresolved_crs'
    assert discover_survey_regions(catalog)['regions'] == []
