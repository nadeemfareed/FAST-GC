import numpy as np

from fastgc.hydrology import (
    stream_link_ids,
    stream_node_masks,
    strahler_stream_order,
)


def _y_network():
    stream = np.ones((1, 5), dtype=bool)
    receiver = np.array(
        [2, 2, 3, 4, -1],
        dtype=np.int64,
    )
    return stream, receiver


def test_y_network_has_three_links():
    stream, receiver = _y_network()

    links = stream_link_ids(
        stream,
        receiver,
    )

    ids = np.unique(links[stream])

    assert ids.size == 3
    assert np.all(ids > 0)


def test_every_stream_cell_gets_exactly_one_link():
    stream, receiver = _y_network()

    links = stream_link_ids(
        stream,
        receiver,
    )

    assert np.all(links[stream] > 0)
    assert np.all(links[~stream] == 0)


def test_downstream_of_junction_starts_new_link():
    stream, receiver = _y_network()

    links = stream_link_ids(
        stream,
        receiver,
    )

    # Junction is cell 2. Cell 3 is its downstream continuation.
    assert links[0, 3] != links[0, 2]

    # Downstream reach remains continuous to outlet.
    assert links[0, 3] == links[0, 4]


def test_two_headwater_tributaries_are_distinct_links():
    stream, receiver = _y_network()

    links = stream_link_ids(
        stream,
        receiver,
    )

    assert links[0, 0] != links[0, 1]


def test_single_channel_is_single_link():
    stream = np.ones((1, 6), dtype=bool)
    receiver = np.array(
        [1, 2, 3, 4, 5, -1],
        dtype=np.int64,
    )

    links = stream_link_ids(
        stream,
        receiver,
    )

    assert np.unique(links[stream]).size == 1


def test_link_ids_are_deterministic():
    stream, receiver = _y_network()

    a = stream_link_ids(stream, receiver)
    b = stream_link_ids(stream, receiver)

    np.testing.assert_array_equal(a, b)


def test_nodes_and_strahler_remain_consistent():
    stream, receiver = _y_network()

    nodes = stream_node_masks(
        stream,
        receiver,
    )

    order = strahler_stream_order(
        stream,
        receiver,
    )

    assert np.count_nonzero(nodes["headwaters"]) == 2
    assert np.count_nonzero(nodes["junctions"]) == 1
    assert np.count_nonzero(nodes["outlets"]) == 1

    assert order[0, 0] == 1
    assert order[0, 1] == 1
    assert order[0, 2] == 2
    assert order[0, 4] == 2


def test_nonstream_cells_keep_zero_link_id():
    stream = np.array(
        [[False, True, True, False]],
        dtype=bool,
    )

    receiver = np.array(
        [-1, 2, -1, -1],
        dtype=np.int64,
    )

    links = stream_link_ids(
        stream,
        receiver,
    )

    assert links[0, 0] == 0
    assert links[0, 3] == 0
    assert links[0, 1] > 0
    assert links[0, 2] > 0
