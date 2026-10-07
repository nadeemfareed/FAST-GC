import numpy as np
import pytest
from affine import Affine

from fastgc.terrain_vector import (
    stream_lines_from_topology,
)


def _transform():
    return (
        Affine.translation(0.0, 1.0)
        * Affine.scale(1.0, -1.0)
    )


def test_filtered_vector_does_not_enter_excluded_cell():
    stream = np.ones((1, 4), dtype=bool)

    order = np.array(
        [[4, 4, 2, 2]],
        dtype=np.int16,
    )

    links = np.array(
        [[1, 1, 2, 2]],
        dtype=np.int32,
    )

    receiver = np.array(
        [1, 2, 3, -1],
        dtype=np.int64,
    )

    gdf = stream_lines_from_topology(
        stream,
        order,
        links,
        receiver,
        _transform(),
        min_order=4,
    )

    assert len(gdf) == 1

    coords = list(gdf.iloc[0].geometry.coords)

    # Only centres of cells 0 and 1 may appear.
    assert len(coords) == 2
    assert coords[-1][0] == pytest.approx(1.5)


def test_adjacent_selected_links_remain_connected():
    stream = np.ones((1, 4), dtype=bool)

    order = np.array(
        [[4, 4, 4, 4]],
        dtype=np.int16,
    )

    links = np.array(
        [[1, 1, 2, 2]],
        dtype=np.int32,
    )

    receiver = np.array(
        [1, 2, 3, -1],
        dtype=np.int64,
    )

    gdf = stream_lines_from_topology(
        stream,
        order,
        links,
        receiver,
        _transform(),
        min_order=4,
    )

    assert len(gdf) == 2

    first = gdf.loc[
        gdf["stream_link"] == 1
    ].iloc[0]

    second = gdf.loc[
        gdf["stream_link"] == 2
    ].iloc[0]

    assert (
        list(first.geometry.coords)[-1]
        == list(second.geometry.coords)[0]
    )


def test_stream_length_is_physical_geometry_length():
    stream = np.ones((1, 3), dtype=bool)
    order = np.ones((1, 3), dtype=np.int16)
    links = np.ones((1, 3), dtype=np.int32)

    receiver = np.array(
        [1, 2, -1],
        dtype=np.int64,
    )

    transform = (
        Affine.translation(0.0, 1.0)
        * Affine.scale(2.0, -2.0)
    )

    gdf = stream_lines_from_topology(
        stream,
        order,
        links,
        receiver,
        transform,
    )

    assert len(gdf) == 1
    assert gdf.iloc[0]["length_m"] == pytest.approx(4.0)


def test_area_attributes_are_reported():
    stream = np.ones((1, 3), dtype=bool)
    order = np.ones((1, 3), dtype=np.int16)
    links = np.ones((1, 3), dtype=np.int32)

    receiver = np.array(
        [1, 2, -1],
        dtype=np.int64,
    )

    area = np.array(
        [[10.0, 20.0, 30.0]]
    )

    gdf = stream_lines_from_topology(
        stream,
        order,
        links,
        receiver,
        _transform(),
        contributing_area=area,
    )

    row = gdf.iloc[0]

    assert row["area_max_m2"] == pytest.approx(30.0)
    assert row["area_outlet_m2"] == pytest.approx(30.0)


def test_contributing_area_shape_must_match():
    stream = np.ones((1, 3), dtype=bool)
    order = np.ones((1, 3), dtype=np.int16)
    links = np.ones((1, 3), dtype=np.int32)
    receiver = np.array([1, 2, -1], dtype=np.int64)

    with pytest.raises(
        ValueError,
        match="contributing_area",
    ):
        stream_lines_from_topology(
            stream,
            order,
            links,
            receiver,
            _transform(),
            contributing_area=np.ones((2, 2)),
        )


def test_receiver_size_must_match():
    stream = np.ones((1, 3), dtype=bool)
    order = np.ones((1, 3), dtype=np.int16)
    links = np.ones((1, 3), dtype=np.int32)

    with pytest.raises(
        ValueError,
        match="Receiver",
    ):
        stream_lines_from_topology(
            stream,
            order,
            links,
            np.array([1, -1]),
            _transform(),
        )


def test_disconnected_cells_with_same_link_id_rejected():
    stream = np.ones((1, 3), dtype=bool)
    order = np.ones((1, 3), dtype=np.int16)

    # Same link ID, but topology does not form one chain.
    links = np.ones((1, 3), dtype=np.int32)

    receiver = np.array(
        [-1, -1, -1],
        dtype=np.int64,
    )

    with pytest.raises(
        RuntimeError,
        match="single directed chain",
    ):
        stream_lines_from_topology(
            stream,
            order,
            links,
            receiver,
            _transform(),
        )
