import io
import zipfile

import numpy as np
import pytest

from pyproj import Transformer

from fastgc.gis.plot_io import import_plots


KML = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>

<Placemark>
<name>Plot_1</name>
<Polygon>
<outerBoundaryIs>
<LinearRing>
<coordinates>
9.0,49.0,0
9.001,49.0,0
9.001,49.001,0
9.0,49.001,0
9.0,49.0,0
</coordinates>
</LinearRing>
</outerBoundaryIs>
</Polygon>
</Placemark>

<Placemark>
<name>Plot_1</name>
<Polygon>
<outerBoundaryIs>
<LinearRing>
<coordinates>
9.002,49.0,0
9.003,49.0,0
9.003,49.001,0
9.002,49.001,0
9.002,49.0,0
</coordinates>
</LinearRing>
</outerBoundaryIs>
</Polygon>
</Placemark>

</Document>
</kml>
"""


def test_kml_multiple_plots(tmp_path):

    path = tmp_path / "plots.kml"
    path.write_text(KML, encoding="utf-8")

    plots = import_plots(
        path,
        destination_crs="EPSG:25832",
    )

    assert len(plots) == 2

    assert plots[0].plot_name == "Plot_1"
    assert plots[1].plot_name == "Plot_1_2"

    assert all(
        plot.geometry.is_valid
        for plot in plots
    )


def test_kmz_import(tmp_path):

    path = tmp_path / "plots.kmz"

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "doc.kml",
            KML,
        )

    plots = import_plots(
        path,
        destination_crs="EPSG:25832",
    )

    assert len(plots) == 2


def test_kml_coordinate_transformation(tmp_path):

    path = tmp_path / "plots.kml"
    path.write_text(KML, encoding="utf-8")

    plots = import_plots(
        path,
        destination_crs="EPSG:25832",
    )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:25832",
        always_xy=True,
    )

    x, y = transformer.transform(
        9.0,
        49.0,
    )

    assert plots[0].geometry.bounds[0] == pytest.approx(
        x,
        abs=0.01,
    )

    assert plots[0].geometry.bounds[1] == pytest.approx(
        y,
        abs=0.01,
    )


def test_csv_plot_centers(tmp_path):

    path = tmp_path / "plots.csv"

    path.write_text(
        "plot_name,x,y,width,height\n"
        "Plot_1,500000,5400000,30,30\n"
        "Plot_2,500100,5400100,50,40\n",
        encoding="utf-8",
    )

    plots = import_plots(
        path,
        destination_crs="EPSG:25832",
        source_crs="EPSG:25832",
    )

    assert len(plots) == 2

    assert plots[0].geometry.area == pytest.approx(900)
    assert plots[1].geometry.area == pytest.approx(2000)


def test_unsupported_format(tmp_path):

    path = tmp_path / "plots.txt"
    path.write_text("test", encoding="utf-8")

    with pytest.raises(ValueError):
        import_plots(
            path,
            destination_crs="EPSG:25832",
        )
