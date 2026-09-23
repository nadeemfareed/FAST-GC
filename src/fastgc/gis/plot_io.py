"""Unified vector and Google Earth plot importer for FAST-GIS.

Supported:
    KML, KMZ, SHP, GPKG, GeoJSON and CSV plot centers.

All returned geometries use the requested destination CRS.
"""

from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
import zipfile

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd

from pyproj import CRS, Transformer
from shapely.geometry import (
    Polygon,
    MultiPolygon,
    shape,
)
from shapely.ops import transform


@dataclass
class PlotDefinition:
    plot_id: str
    plot_name: str
    geometry: Any
    source_crs: str
    processing_crs: str
    source_file: str
    source_feature_id: str
    attributes: dict = field(default_factory=dict)


def _safe_name(value: str) -> str:
    """Create a filesystem-safe plot name."""

    name = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        str(value).strip(),
    )

    return name.strip("_")


def _transform_geometry(geometry, source_crs, destination_crs):

    src = CRS.from_user_input(source_crs)
    dst = CRS.from_user_input(destination_crs)

    if src == dst:
        return geometry

    transformer = Transformer.from_crs(
        src,
        dst,
        always_xy=True,
    )

    return transform(transformer.transform, geometry)


def _coordinates(text):

    coordinates = []

    for item in (text or "").strip().split():

        parts = item.split(",")

        if len(parts) < 2:
            raise ValueError("Invalid KML coordinate.")

        coordinates.append(
            (float(parts[0]), float(parts[1]))
        )

    return coordinates


def _parse_kml_polygon(element):

    outer = element.find(
        "./{*}outerBoundaryIs/{*}LinearRing/{*}coordinates"
    )

    if outer is None:
        raise ValueError("KML polygon has no outer boundary.")

    shell = _coordinates(outer.text)

    holes = []

    for inner in element.findall(
        "./{*}innerBoundaryIs/{*}LinearRing/{*}coordinates"
    ):
        holes.append(_coordinates(inner.text))

    return Polygon(shell, holes)


def _parse_kml(data):

    root = ET.fromstring(data)

    features = []

    for index, placemark in enumerate(
        root.findall(".//{*}Placemark"),
        start=1,
    ):

        name_element = placemark.find("./{*}name")

        name = (
            name_element.text
            if name_element is not None
            else None
        )

        polygons = []

        for polygon_element in placemark.findall(
            ".//{*}Polygon"
        ):
            polygons.append(
                _parse_kml_polygon(polygon_element)
            )

        if not polygons:
            continue

        geometry = (
            polygons[0]
            if len(polygons) == 1
            else MultiPolygon(polygons)
        )

        features.append(
            {
                "name": name,
                "geometry": geometry,
                "feature_id": str(index),
                "attributes": {},
            }
        )

    return features


def _read_kmz(path):

    with zipfile.ZipFile(path) as archive:

        names = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".kml")
        ]

        if not names:
            raise ValueError("KMZ contains no KML file.")

        # Prefer the conventional primary KML document.
        primary = next(
            (
                name
                for name in names
                if Path(name).name.lower() == "doc.kml"
            ),
            names[0],
        )

        return archive.read(primary)


def _read_vector(path, layer=None):

    frame = gpd.read_file(
        path,
        layer=layer,
        engine="pyogrio",
    )

    if frame.crs is None:
        raise ValueError(
            f"Vector file has no CRS: {path}"
        )

    features = []

    for index, row in frame.iterrows():

        attributes = row.drop(
            labels=["geometry"]
        ).to_dict()

        name = attributes.get(
            "plot_name",
            attributes.get("name"),
        )

        features.append(
            {
                "name": name,
                "geometry": row.geometry,
                "feature_id": str(index),
                "attributes": attributes,
            }
        )

    return features, frame.crs


def _read_csv(path, source_crs):

    if source_crs is None:
        raise ValueError(
            "CSV plot centers require source_crs."
        )

    features = []

    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        reader = csv.DictReader(handle)

        for index, row in enumerate(reader, start=1):

            x = float(row["x"])
            y = float(row["y"])

            width = float(row["width"])
            height = float(row["height"])

            if width <= 0 or height <= 0:
                raise ValueError(
                    "Plot width and height must be positive."
                )

            geometry = Polygon(
                [
                    (x - width / 2, y - height / 2),
                    (x + width / 2, y - height / 2),
                    (x + width / 2, y + height / 2),
                    (x - width / 2, y + height / 2),
                ]
            )

            features.append(
                {
                    "name": row.get("plot_name"),
                    "geometry": geometry,
                    "feature_id": str(index),
                    "attributes": dict(row),
                }
            )

    return features


def import_plots(
    path,
    destination_crs,
    *,
    source_crs=None,
    layer=None,
):
    """Import named plot polygons into a common destination CRS.

    CSV x/y/width/height values must use a projected CRS
    with linear units appropriate for the plot dimensions.
    """

    path = Path(path).resolve()

    if not path.is_file():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    if suffix == ".kml":

        features = _parse_kml(
            path.read_bytes()
        )

        input_crs = CRS.from_epsg(4326)

    elif suffix == ".kmz":

        features = _parse_kml(
            _read_kmz(path)
        )

        input_crs = CRS.from_epsg(4326)

    elif suffix in {
        ".shp",
        ".gpkg",
        ".geojson",
    }:

        features, input_crs = _read_vector(
            path,
            layer=layer,
        )

    elif suffix == ".csv":

        input_crs = CRS.from_user_input(
            source_crs
        )

        if not input_crs.is_projected:
            raise ValueError(
                "CSV plot dimensions require a projected CRS."
            )

        features = _read_csv(
            path,
            source_crs,
        )

    else:

        raise ValueError(
            f"Unsupported plot format: {suffix}"
        )

    destination_crs = CRS.from_user_input(
        destination_crs
    )

    plots = []

    used_names = set()

    for index, feature in enumerate(
        features,
        start=1,
    ):

        geometry = feature["geometry"]

        if geometry is None or geometry.is_empty:
            continue

        if not geometry.is_valid:
            raise ValueError(
                f"Invalid geometry in feature {index}."
            )

        if geometry.geom_type not in {
            "Polygon",
            "MultiPolygon",
        }:
            raise ValueError(
                f"Unsupported plot geometry: {geometry.geom_type}"
            )

        geometry = _transform_geometry(
            geometry,
            input_crs,
            destination_crs,
        )

        requested_name = _safe_name(
            feature.get("name") or ""
        )

        base_name = (
            requested_name
            or f"Plot_{index}"
        )

        name = base_name
        duplicate = 2

        while name.lower() in used_names:

            name = f"{base_name}_{duplicate}"
            duplicate += 1

        used_names.add(name.lower())

        plots.append(
            PlotDefinition(
                plot_id=f"plot_{index}",
                plot_name=name,
                geometry=geometry,
                source_crs=input_crs.to_string(),
                processing_crs=destination_crs.to_string(),
                source_file=str(path),
                source_feature_id=feature["feature_id"],
                attributes=feature["attributes"],
            )
        )

    return plots
