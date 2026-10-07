import numpy as np
from affine import Affine

from fastgc.terrain_vector import (
    labels_to_polygons,
    polygon_boundaries,
    stream_lines_from_topology,
    stream_nodes_to_points,
)


def test_stream_order_threshold_only_exports_selected_links():
    stream = np.array(
        [[1, 1, 1, 1, 1]],
        dtype=bool,
    )

    order = np.array(
        [[2, 2, 4, 4, 4]],
        dtype=np.int16,
    )

    links = np.array(
        [[1, 1, 2, 2, 2]],
        dtype=np.int32,
    )

    receiver = np.array(
        [1, 2, 3, 4, -1],
        dtype=np.int64,
    )

    transform = Affine.translation(0, 1) * Affine.scale(1, -1)

    gdf = stream_lines_from_topology(
        stream,
        order,
        links,
        receiver,
        transform,
        min_order=4,
    )

    assert len(gdf) == 1
    assert int(gdf.iloc[0]["stream_link"]) == 2
    assert int(gdf.iloc[0]["stream_order"]) == 4


def test_stream_vector_threshold_does_not_modify_input_raster():
    stream = np.ones((1, 4), dtype=bool)
    order = np.array([[1, 2, 3, 4]], dtype=np.int16)
    original = order.copy()
    links = np.ones((1, 4), dtype=np.int32)
    receiver = np.array([1, 2, 3, -1], dtype=np.int64)

    transform = Affine.identity()

    stream_lines_from_topology(
        stream,
        order,
        links,
        receiver,
        transform,
        min_order=3,
    )

    assert np.array_equal(order, original)


def test_node_points():
    mask = np.array(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=bool,
    )

    gdf = stream_nodes_to_points(
        mask,
        Affine.identity(),
        node_type="headwater",
    )

    assert len(gdf) == 2
    assert set(gdf["node_type"]) == {"headwater"}


def test_label_polygonization_and_boundaries():
    labels = np.array(
        [
            [1, 1, 2],
            [1, 1, 2],
            [3, 3, 2],
        ],
        dtype=np.int32,
    )

    gdf = labels_to_polygons(
        labels,
        Affine.identity(),
        label_field="basin_id",
    )

    assert set(gdf["basin_id"]) == {1, 2, 3}

    boundaries = polygon_boundaries(
        gdf,
        id_field="basin_id",
    )

    assert len(boundaries) == 3
    assert all(
        geom.geom_type in {
            "LineString",
            "MultiLineString",
        }
        for geom in boundaries.geometry
    )
