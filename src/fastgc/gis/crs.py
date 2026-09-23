"""Read-only coordinate reference system inspection for FAST-GIS.

This module does not modify source datasets or change the CRS
handling used by established FAST-GC processing workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import laspy
import pyogrio
import rasterio
from pyproj import CRS


POINT_EXTENSIONS = {".las", ".laz"}
RASTER_EXTENSIONS = {".tif", ".tiff"}
VECTOR_EXTENSIONS = {
    ".shp",
    ".gpkg",
    ".geojson",
    ".json",
    ".kml",
}


@dataclass(frozen=True)
class CRSInfo:
    """Coordinate reference system information for a spatial dataset."""

    path: Path
    data_type: str
    crs: CRS | None
    status: str
    epsg: int | None
    is_geographic: bool | None
    is_projected: bool | None


def inspect_crs(
    path: str | Path,
    *,
    layer: str | None = None,
) -> CRSInfo:
    """Inspect CRS metadata without modifying the input dataset.

    Vector containers with multiple layers require an explicit layer.
    Missing CRS metadata is reported, never inferred.
    """

    path = Path(path).resolve()

    if not path.is_file():
        raise FileNotFoundError(path)

    extension = path.suffix.lower()

    if extension in POINT_EXTENSIONS:
        data_type = "point_cloud"

        raw_crs = inspect_las_crs_records(path)

    elif extension in RASTER_EXTENSIONS:
        data_type = "raster"

        with rasterio.open(path) as dataset:
            raw_crs = dataset.crs

    elif extension in VECTOR_EXTENSIONS:
        data_type = "vector"

        layers = pyogrio.list_layers(path)

        if layer is None:
            if len(layers) != 1:
                raise ValueError(
                    "Vector dataset contains multiple layers. "
                    "Specify the layer explicitly."
                )

            layer = str(layers[0][0])

        available_layers = {
            str(item[0]) for item in layers
        }

        if layer not in available_layers:
            raise ValueError(
                f"Layer {layer!r} not found. "
                f"Available layers: {sorted(available_layers)}"
            )

        metadata = pyogrio.read_info(
            path,
            layer=layer,
        )

        raw_crs = metadata.get("crs")

    else:
        raise ValueError(
            f"Unsupported spatial format: {extension}"
        )

    if raw_crs is None or raw_crs == "":
        return CRSInfo(
            path=path,
            data_type=data_type,
            crs=None,
            status="missing",
            epsg=None,
            is_geographic=None,
            is_projected=None,
        )

    try:
        crs = CRS.from_user_input(raw_crs)
    except Exception as exc:
        raise ValueError(
            f"Invalid CRS metadata in {path}"
        ) from exc

    return CRSInfo(
        path=path,
        data_type=data_type,
        crs=crs,
        status="identified",
        epsg=crs.to_epsg(),
        is_geographic=crs.is_geographic,
        is_projected=crs.is_projected,
    )


# ============================================================
# Coordinate transformations and spatial compatibility
# ============================================================

from pyproj.aoi import AreaOfInterest
from shapely.geometry import box
from shapely.ops import transform as shapely_transform


def extract_horizontal_crs(
    crs: CRS | str | int | None,
) -> CRS:
    """Extract the horizontal 2D component of a CRS.

    Geographic 3D and compound CRS definitions are reduced to
    their horizontal components. Vertical-only and unsupported
    coordinate systems are rejected.

    This function does not transform elevation values.
    """

    if crs is None:
        raise ValueError(
            "CRS is missing. Supply an explicit CRS before "
            "performing spatial operations."
        )

    resolved = CRS.from_user_input(crs)

    if resolved.is_compound:
        horizontal_components = [
            component
            for component in resolved.sub_crs_list
            if component.is_geographic or component.is_projected
        ]

        if len(horizontal_components) != 1:
            raise ValueError(
                "Compound CRS must contain exactly one "
                "recognized horizontal component."
            )

        resolved = horizontal_components[0]

    if resolved.is_vertical:
        raise ValueError(
            "A vertical-only CRS cannot be used for horizontal "
            "spatial operations."
        )

    if not (resolved.is_geographic or resolved.is_projected):
        raise ValueError(
            "CRS does not provide a supported geographic or "
            "projected horizontal coordinate system."
        )

    horizontal = resolved.to_2d()

    if len(horizontal.axis_info) != 2:
        raise ValueError(
            "Horizontal CRS extraction did not produce "
            "a two-dimensional coordinate system."
        )

    return horizontal


def require_horizontal_crs(
    crs: CRS | str | int | None,
) -> CRS:
    """Return a validated two-dimensional horizontal CRS."""

    return extract_horizontal_crs(crs)


def transform_geometry(
    geometry,
    source_crs: CRS | str | int,
    target_crs: CRS | str | int,
    *,
    area_of_interest: AreaOfInterest | None = None,
    max_accuracy_m: float | None = None,
    require_known_accuracy: bool = False,
):
    """Transform horizontal coordinates without PROJ network access.

    Selects an available local transformation and keeps PROJ
    networking disabled throughout its execution. Existing geometry
    Z coordinates are preserved; vertical datums are not transformed.
    """

    source = require_horizontal_crs(source_crs)
    target = require_horizontal_crs(target_crs)

    if source.equals(target):
        return geometry

    with _offline_proj_context():
        transformer = select_horizontal_transformer(
            source,
            target,
            area_of_interest=area_of_interest,
            max_accuracy_m=max_accuracy_m,
            require_known_accuracy=require_known_accuracy,
        )

        return shapely_transform(
            lambda x, y, z=None: transformer.transform(
                x,
                y,
                z,
                errcheck=True,
            ) if z is not None else transformer.transform(
                x,
                y,
                errcheck=True,
            ),
            geometry,
        )


def spatially_overlaps(
    bounds_a,
    crs_a,
    bounds_b,
    crs_b,
) -> bool:
    """Check spatial overlap using a common horizontal CRS.

    Bounds must be ordered as xmin, ymin, xmax, ymax.
    """

    source_a = require_horizontal_crs(crs_a)
    source_b = require_horizontal_crs(crs_b)

    geometry_a = box(*bounds_a)
    geometry_b = box(*bounds_b)

    if not source_b.equals(source_a):
        geometry_b = transform_geometry(
            geometry_b,
            source_b,
            source_a,
        )

    return bool(
        geometry_a.intersects(geometry_b)
    )



# ============================================================
# Independent LAS/LAZ CRS record validation
# ============================================================

from laspy.vlrs.known import (
    GeoKeyDirectoryVlr,
    WktCoordinateSystemVlr,
)


def inspect_las_crs_records(
    path: str | Path,
) -> CRS | None:
    """Validate all recognized LAS CRS records independently.

    Both ordinary VLRs and extended VLRs are inspected.

    Returns None when no recognized CRS records exist.
    Rejects unparseable records and conflicting CRS definitions.

    This function does not modify the source point cloud.
    """

    path = Path(path).resolve()

    if path.suffix.lower() not in POINT_EXTENSIONS:
        raise ValueError(
            "LAS CRS validation requires a .las or .laz file."
        )

    records = []

    with laspy.open(path) as reader:
        header = reader.header

        records.extend(header.vlrs)

        if header.evlrs is not None:
            records.extend(header.evlrs)

    recognized = (
        WktCoordinateSystemVlr,
        GeoKeyDirectoryVlr,
    )

    parsed = []

    for record in records:
        if not isinstance(record, recognized):
            continue

        record_name = type(record).__name__

        try:
            raw_crs = record.parse_crs()

            if raw_crs is None:
                raise ValueError(
                    "CRS record could not be interpreted."
                )

            resolved = CRS.from_user_input(raw_crs)

        except Exception as exc:
            raise ValueError(
                f"Invalid {record_name} in {path}"
            ) from exc

        parsed.append(
            (record_name, resolved)
        )

    if not parsed:
        return None

    reference_name, reference_crs = parsed[0]

    for record_name, candidate in parsed[1:]:
        if not reference_crs.equals(
            candidate,
            ignore_axis_order=True,
        ):
            raise ValueError(
                "Conflicting LAS CRS metadata: "
                f"{reference_name} and {record_name} "
                f"describe different coordinate systems "
                f"in {path}"
            )

    return reference_crs


from contextlib import contextmanager
from math import isfinite

from pyproj import network
from pyproj.transformer import TransformerGroup


@contextmanager
def _offline_proj_context():
    """Temporarily disable PROJ network access."""

    was_enabled = network.is_network_enabled()

    try:
        network.set_network_enabled(False)
        yield
    finally:
        network.set_network_enabled(was_enabled)


def select_horizontal_transformer(
    source_crs: CRS | str | int,
    target_crs: CRS | str | int,
    *,
    area_of_interest: AreaOfInterest | None = None,
    max_accuracy_m: float | None = None,
    require_known_accuracy: bool = False,
):
    """Select an available horizontal coordinate transformation."""

    source = extract_horizontal_crs(source_crs)
    target = extract_horizontal_crs(target_crs)

    if max_accuracy_m is not None:
        if (
            not isfinite(max_accuracy_m)
            or max_accuracy_m < 0
        ):
            raise ValueError(
                "max_accuracy_m must be finite and nonnegative."
            )

    with _offline_proj_context():
        group = TransformerGroup(
            source,
            target,
            always_xy=True,
            area_of_interest=area_of_interest,
            allow_ballpark=False,
        )

    if not group.best_available:
        missing_grids = sorted({
            grid.short_name
            for operation in group.unavailable_operations
            for grid in operation.grids
            if not grid.available
        })

        raise RuntimeError(
            "The preferred horizontal transformation is "
            "unavailable with locally installed PROJ resources. "
            f"Missing grids: {missing_grids}"
        )

    if not group.transformers:
        raise RuntimeError(
            "No suitable horizontal coordinate transformation "
            "is available."
        )

    transformer = group.transformers[0]
    accuracy = transformer.accuracy

    if accuracy < 0:
        if max_accuracy_m is not None or require_known_accuracy:
            raise ValueError(
                "Selected coordinate transformation has "
                "unknown numerical accuracy."
            )

    elif (
        max_accuracy_m is not None
        and accuracy > max_accuracy_m
    ):
        raise ValueError(
            "Selected coordinate transformation exceeds "
            "the permitted accuracy threshold: "
            f"{accuracy} m > {max_accuracy_m} m."
        )

    if area_of_interest is not None:
        validate_transformer_coverage(
            transformer,
            area_of_interest,
        )

    return transformer


def validate_transformer_coverage(
    transformer,
    area_of_interest: AreaOfInterest,
) -> None:
    """Validate geographic coverage of a selected operation."""

    if not isinstance(area_of_interest, AreaOfInterest):
        raise TypeError(
            "area_of_interest must be an AreaOfInterest instance."
        )

    west = area_of_interest.west_lon_degree
    south = area_of_interest.south_lat_degree
    east = area_of_interest.east_lon_degree
    north = area_of_interest.north_lat_degree

    coordinates = (west, south, east, north)

    if not all(isfinite(value) for value in coordinates):
        raise ValueError(
            "Area-of-interest coordinates must be finite."
        )

    if not (
        -180 <= west <= 180
        and -180 <= east <= 180
        and -90 <= south <= 90
        and -90 <= north <= 90
    ):
        raise ValueError(
            "Area-of-interest coordinates exceed "
            "geographic coordinate limits."
        )

    if west >= east:
        raise ValueError(
            "Invalid or antimeridian-crossing longitude bounds."
        )

    if south >= north:
        raise ValueError(
            "Area-of-interest southern boundary must "
            "be below its northern boundary."
        )

    coverage = transformer.area_of_use

    if coverage is None:
        raise ValueError(
            "Selected transformation does not report "
            "geographic coverage."
        )

    if coverage.west > coverage.east:
        raise ValueError(
            "Antimeridian-crossing transformation coverage "
            "requires dedicated handling."
        )

    tolerance = 1e-9

    covered = (
        west >= coverage.west - tolerance
        and east <= coverage.east + tolerance
        and south >= coverage.south - tolerance
        and north <= coverage.north + tolerance
    )

    if not covered:
        raise ValueError(
            "Selected transformation does not cover "
            "the entire requested geographic area. "
            f"Requested: {(west, south, east, north)}. "
            f"Operation coverage: "
            f"{(coverage.west, coverage.south, coverage.east, coverage.north)}."
        )
