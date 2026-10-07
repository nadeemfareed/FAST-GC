import numpy as np

from fastgc.hydrology import (
    extract_d8_stream_network,
    strahler_stream_order,
    stream_mask_from_contributing_area,
    stream_node_masks,
)


def _synthetic_y_network():
    # Flattened topology:
    #
    # 0 ----\
    #        2 -> 3 -> 4
    # 1 ----/
    #
    # Five cells represented as a 1x5 logical stream network.
    stream = np.ones((1, 5), dtype=bool)

    receiver = np.array(
        [2, 2, 3, 4, -1],
        dtype=np.int64,
    )

    return stream, receiver


def test_stream_threshold_uses_physical_area():
    area = np.array(
        [[10, 20, 30, 40]],
        dtype=float,
    )

    stream = stream_mask_from_contributing_area(
        area,
        25.0,
    )

    np.testing.assert_array_equal(
        stream,
        [[False, False, True, True]],
    )


def test_y_network_nodes():
    stream, receiver = _synthetic_y_network()

    nodes = stream_node_masks(
        stream,
        receiver,
    )

    assert nodes["headwaters"][0, 0]
    assert nodes["headwaters"][0, 1]

    assert nodes["junctions"][0, 2]

    assert nodes["outlets"][0, 4]


def test_strahler_y_network():
    stream, receiver = _synthetic_y_network()

    order = strahler_stream_order(
        stream,
        receiver,
    )

    # Two order-1 tributaries produce order 2.
    assert order[0, 0] == 1
    assert order[0, 1] == 1
    assert order[0, 2] == 2
    assert order[0, 3] == 2
    assert order[0, 4] == 2


def test_single_channel_remains_order_one():
    stream = np.ones((1, 5), dtype=bool)

    receiver = np.array(
        [1, 2, 3, 4, -1],
        dtype=np.int64,
    )

    order = strahler_stream_order(
        stream,
        receiver,
    )

    np.testing.assert_array_equal(
        order,
        np.ones((1, 5), dtype=np.int32),
    )


def test_complete_network_products():
    stream, receiver = _synthetic_y_network()

    area = np.array(
        [[100, 100, 200, 300, 400]],
        dtype=float,
    )

    out = extract_d8_stream_network(
        area,
        receiver,
        threshold_area=50.0,
    )

    assert set(out) == {
        "stream_mask",
        "stream_order",
        "stream_link",
        "headwaters",
        "junctions",
        "stream_outlets",
    }

    assert np.all(out["stream_mask"])
    assert out["junctions"][0, 2]
    assert out["stream_outlets"][0, 4]


def test_nonstream_cells_have_zero_order():
    stream = np.array(
        [[False, True, True]],
        dtype=bool,
    )

    receiver = np.array(
        [-1, 2, -1],
        dtype=np.int64,
    )

    order = strahler_stream_order(
        stream,
        receiver,
    )

    assert order[0, 0] == 0
    assert order[0, 1] == 1
    assert order[0, 2] == 1
