"""Vector GIS outputs for FAST_TERRAIN hydrology.

Hydrologic analysis remains raster/topology based. This module converts
validated analytical products to user-facing vector GIS products.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _require_vector_stack():
    try:
        import geopandas as gpd
        from shapely.geometry import LineString, Point, shape
        import rasterio.features
    except ImportError as exc:
        raise RuntimeError(
            "FAST_TERRAIN vector export requires geopandas, shapely, "
            "pyogrio, and rasterio."
        ) from exc

    return gpd, LineString, Point, shape


def _cell_xy(transform, row: int, col: int) -> tuple[float, float]:
    """Return raster-cell centre coordinates."""
    x, y = transform * (float(col) + 0.5, float(row) + 0.5)
    return float(x), float(y)


def stream_lines_from_topology(
    stream_mask: np.ndarray,
    stream_order: np.ndarray,
    stream_link: np.ndarray,
    receiver: np.ndarray,
    transform,
    *,
    contributing_area: np.ndarray | None = None,
    min_order: int = 1,
    crs=None,
):
    """Create connected stream-link LineStrings from D8 topology.

    Filtering affects vector export only. Raster stream topology remains
    unchanged. Geometry is restricted to cells satisfying min_order;
    filtered-out downstream cells are never appended to exported lines.
    """
    gpd, LineString, _, _ = _require_vector_stack()

    stream = np.asarray(stream_mask, dtype=bool)
    order = np.asarray(stream_order)
    links = np.asarray(stream_link)
    recv = np.asarray(receiver, dtype=np.int64).reshape(-1)

    if stream.ndim != 2:
        raise ValueError("stream_mask must be a 2-D array.")

    if stream.shape != order.shape or stream.shape != links.shape:
        raise ValueError(
            "stream_mask, stream_order, and stream_link must have "
            "identical shapes."
        )

    if recv.size != stream.size:
        raise ValueError(
            "Receiver array size must match stream raster."
        )

    min_order = int(min_order)
    if min_order < 1:
        raise ValueError("min_order must be >= 1.")

    area_flat = None
    if contributing_area is not None:
        area = np.asarray(contributing_area, dtype=np.float64)
        if area.shape != stream.shape:
            raise ValueError(
                "contributing_area must match stream raster shape."
            )
        area_flat = area.reshape(-1)

    rows, cols = stream.shape
    stream_flat = stream.reshape(-1)
    order_flat = order.reshape(-1)
    links_flat = links.reshape(-1)

    selected = (
        stream_flat
        & (links_flat > 0)
        & (order_flat >= min_order)
    )

    link_ids = np.unique(links_flat[selected])
    records = []

    for link_id in link_ids:
        cells = np.flatnonzero(
            selected & (links_flat == link_id)
        )

        if cells.size == 0:
            continue

        cell_set = set(int(i) for i in cells)
        upstream_count = {int(i): 0 for i in cells}

        for i in cells:
            j = int(recv[int(i)])
            if j in cell_set:
                upstream_count[j] += 1

        starts = sorted(
            i for i, n in upstream_count.items()
            if n == 0
        )

        # A valid D8 link is a single directed chain. Multiple starts
        # indicate inconsistent link topology.
        if len(starts) > 1:
            raise RuntimeError(
                "Stream-link topology is not a single directed chain."
            )

        start = starts[0] if starts else int(np.min(cells))

        path = []
        visited = []
        seen = set()
        i = start

        while i in cell_set:
            if i in seen:
                raise RuntimeError(
                    "Stream-link topology contains a cycle."
                )

            seen.add(i)
            visited.append(i)

            r, c = divmod(i, cols)
            path.append(_cell_xy(transform, r, c))

            j = int(recv[i])

            if j not in cell_set:
                # Only append the receiver when it remains part of the
                # selected stream network. This preserves connectivity
                # between adjacent exported links while preventing a
                # min_order filter from extending geometry into an
                # excluded stream cell.
                if (
                    0 <= j < stream.size
                    and selected[j]
                ):
                    rr, cc = divmod(j, cols)
                    path.append(_cell_xy(transform, rr, cc))
                break

            i = j

        if len(visited) != cells.size:
            raise RuntimeError(
                "Stream-link topology contains disconnected cells."
            )

        if len(path) < 2:
            continue

        link_orders = order_flat[cells]
        max_order = int(np.nanmax(link_orders))

        geometry = LineString(path)

        rec = {
            "stream_link": int(link_id),
            "stream_order": max_order,
            "n_cells": int(cells.size),
            "length_m": float(geometry.length),
            "geometry": geometry,
        }

        if area_flat is not None:
            ca = area_flat[cells]
            finite = ca[np.isfinite(ca)]

            rec["area_max_m2"] = (
                float(np.max(finite))
                if finite.size
                else np.nan
            )

            # Outlet/downstream contributing area is more informative
            # than maximum alone and should normally equal the maximum
            # for valid D8 stream topology.
            downstream_cell = int(visited[-1])
            value = float(area_flat[downstream_cell])

            rec["area_outlet_m2"] = (
                value if np.isfinite(value) else np.nan
            )

        records.append(rec)

    columns = [
        "stream_link",
        "stream_order",
        "n_cells",
        "length_m",
    ]

    if contributing_area is not None:
        columns.extend([
            "area_max_m2",
            "area_outlet_m2",
        ])

    columns.append("geometry")

    return gpd.GeoDataFrame(
        records,
        columns=columns,
        geometry="geometry",
        crs=crs,
    )






def stream_nodes_to_points(
    mask: np.ndarray,
    transform,
    *,
    node_type: str,
    stream_order: np.ndarray | None = None,
    contributing_area: np.ndarray | None = None,
    crs=None,
):
    """Convert a stream-node mask to GIS point features."""
    gpd, _, Point, _ = _require_vector_stack()

    mask = np.asarray(mask, dtype=bool)
    rows, cols = np.nonzero(mask)

    records = []

    for r, c in zip(rows, cols):
        rec = {
            "node_type": str(node_type),
            "row": int(r),
            "col": int(c),
            "geometry": Point(
                *_cell_xy(transform, int(r), int(c))
            ),
        }

        if stream_order is not None:
            rec["stream_order"] = int(stream_order[r, c])

        if contributing_area is not None:
            value = contributing_area[r, c]
            rec["area_m2"] = (
                float(value)
                if np.isfinite(value)
                else np.nan
            )

        records.append(rec)

    return gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=crs,
    )


def labels_to_polygons(
    labels: np.ndarray,
    transform,
    *,
    crs=None,
    label_field: str = "basin_id",
):
    """Polygonize positive integer catchment/basin labels."""
    gpd, _, _, shape = _require_vector_stack()
    import rasterio.features

    arr = np.asarray(labels)

    mask = np.isfinite(arr) & (arr > 0)

    records = []

    for geom, value in rasterio.features.shapes(
        arr.astype(np.int32),
        mask=mask,
        transform=transform,
        connectivity=8,
    ):
        label = int(value)
        if label <= 0:
            continue

        records.append(
            {
                label_field: label,
                "geometry": shape(geom),
            }
        )

    gdf = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=crs,
    )

    if not gdf.empty:
        gdf = gdf.dissolve(
            by=label_field,
            as_index=False,
        )

    return gdf


def polygon_boundaries(polygons, *, id_field: str):
    """Convert catchment/watershed polygons to boundary geometry."""
    gdf = polygons[[id_field, "geometry"]].copy()
    gdf["geometry"] = gdf.geometry.boundary
    return gdf


def write_geopackage_layer(
    gdf,
    path: str | Path,
    *,
    layer: str,
):
    """Write one GeoPackage layer using the installed pyogrio engine."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if gdf.empty:
        return None

    gdf.to_file(
        path,
        layer=layer,
        driver="GPKG",
        engine="pyogrio",
    )

    return str(path)
