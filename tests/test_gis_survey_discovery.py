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

def test_explicit_epsg_filename_resolves_crs(tmp_path):
    import laspy
    import numpy as np
    from pyproj import CRS

    path = tmp_path / "Woodford_EPSG32756.las"

    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.x = np.array([476000.0, 476001.0])
    las.y = np.array([7023000.0, 7023001.0])
    las.z = np.array([200.0, 201.0])
    las.write(path)

    catalog = build_survey_catalog(path)
    tile = catalog["tiles"][0]

    assert tile["status"] == "ready"
    assert tile["crs_source"] == "path_epsg"
    assert CRS.from_wkt(tile["crs_wkt"]) == CRS.from_epsg(32756)


def test_ambiguous_utm_filename_remains_unresolved(tmp_path):
    import laspy
    import numpy as np

    path = tmp_path / "Woodford_UTM56S.las"

    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.x = np.array([476000.0, 476001.0])
    las.y = np.array([7023000.0, 7023001.0])
    las.z = np.array([200.0, 201.0])
    las.write(path)

    catalog = build_survey_catalog(path)
    tile = catalog["tiles"][0]

    assert tile["status"] == "unresolved_crs"
    assert tile["crs_source"] == "unresolved"


def test_path_epsg_conflict_with_embedded_crs_is_error(tmp_path):
    import laspy
    import numpy as np

    path = tmp_path / "Plot_EPSG32756.las"

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_crs(CRS.from_epsg(32656))

    las = laspy.LasData(header)
    las.x = np.array([476000.0, 476001.0])
    las.y = np.array([7023000.0, 7023001.0])
    las.z = np.array([200.0, 201.0])
    las.write(path)

    catalog = build_survey_catalog(path)
    tile = catalog["tiles"][0]

    assert tile["status"] == "error"
    assert "Conflicting CRS metadata" in tile["error"]

def test_utm_filename_hint_is_recorded_but_not_promoted_to_crs(tmp_path):
    import laspy
    import numpy as np

    path = tmp_path / "Woodford_UTM56S.las"

    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.x = np.array([476000.0, 476001.0])
    las.y = np.array([7023000.0, 7023001.0])
    las.z = np.array([200.0, 201.0])
    las.write(path)

    catalog = build_survey_catalog(path)
    tile = catalog["tiles"][0]

    assert tile["status"] == "unresolved_crs"
    assert tile["crs_wkt"] is None
    assert tile["crs_source"] == "unresolved"
    assert tile["crs_hint"] == {
        "type": "utm",
        "zone": 56,
        "hemisphere": "S",
        "datum": None,
        "source": "path_hint",
    }


def test_no_crs_hint_is_recorded_as_none(tmp_path):
    import laspy
    import numpy as np

    path = tmp_path / "Woodford.las"

    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.x = np.array([10.0, 11.0])
    las.y = np.array([20.0, 21.0])
    las.z = np.array([2.0, 3.0])
    las.write(path)

    catalog = build_survey_catalog(path)
    tile = catalog["tiles"][0]

    assert tile["status"] == "unresolved_crs"
    assert tile["crs_hint"] is None
