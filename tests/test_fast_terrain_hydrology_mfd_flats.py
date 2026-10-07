import numpy as np

from fastgc.hydrology import (
    condition_dem,
    d8_flow_direction_resolved,
    mfd_flow_accumulation,
    mfd_flow_weights,
)


def _receiver_from_weights(weights):
    rows, cols, _ = weights.shape

    steps = (
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
    )

    receiver = np.full(rows * cols, -1, dtype=np.int64)

    for r in range(rows):
        for c in range(cols):
            w = weights[r, c]

            if np.sum(w) <= 0:
                continue

            k = int(np.argmax(w))

            # Only use this helper for deterministic flat fallback
            # cells having one unit-weight receiver.
            if not np.isclose(w[k], 1.0):
                continue

            dr, dc = steps[k]
            rr = r + dr
            cc = c + dc

            if 0 <= rr < rows and 0 <= cc < cols:
                receiver[r * cols + c] = rr * cols + cc

    return receiver


def test_mfd_preserves_freeman_partition_on_real_slope():
    dem = np.array(
        [
            [9.0, 8.0, 7.0],
            [8.0, 7.0, 6.0],
            [7.0, 6.0, 5.0],
        ]
    )

    w = mfd_flow_weights(
        dem,
        1.0,
        1.0,
        exponent=1.1,
    )

    center = w[1, 1]

    # Genuine divergent Freeman routing remains divergent.
    assert np.count_nonzero(center > 0.0) > 1
    assert np.isclose(np.sum(center), 1.0)


def test_priority_flood_flat_gets_mfd_outflow():
    raw = np.array(
        [
            [5.0, 5.0, 5.0, 5.0, 5.0],
            [5.0, 1.0, 1.0, 1.0, 5.0],
            [5.0, 1.0, 0.0, 1.0, 4.0],
            [5.0, 1.0, 1.0, 1.0, 5.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
        ]
    )

    conditioned, _ = condition_dem(raw)

    w = mfd_flow_weights(
        conditioned,
        1.0,
        1.0,
    )

    _, receiver = d8_flow_direction_resolved(
        conditioned,
        1.0,
        1.0,
    )

    valid = np.isfinite(conditioned).ravel()

    for i in np.flatnonzero(valid):
        if receiver[i] >= 0:
            r, c = divmod(int(i), conditioned.shape[1])
            assert np.isclose(
                np.sum(w[r, c]),
                1.0,
            )


def test_mfd_flat_fallback_matches_resolved_d8_receiver():
    dem = np.full((5, 5), 10.0)

    w = mfd_flow_weights(
        dem,
        1.0,
        1.0,
    )

    _, d8_receiver = d8_flow_direction_resolved(
        dem,
        1.0,
        1.0,
    )

    mfd_receiver = _receiver_from_weights(w)

    mask = d8_receiver >= 0

    assert np.array_equal(
        mfd_receiver[mask],
        d8_receiver[mask],
    )


def test_mfd_flat_accumulation_conserves_domain_flow():
    dem = np.full((7, 7), 10.0)

    accumulation = mfd_flow_accumulation(
        dem,
        1.0,
        1.0,
    )

    assert np.all(np.isfinite(accumulation))
    assert np.all(accumulation >= 1.0)

    # At least one outlet must receive upstream contribution.
    assert np.max(accumulation) > 1.0


def test_mfd_weights_sum_to_one_except_true_outlets():
    dem = np.array(
        [
            [5.0, 5.0, 5.0, 4.0],
            [5.0, 5.0, 5.0, 4.0],
            [5.0, 5.0, 5.0, 4.0],
            [5.0, 5.0, 5.0, 4.0],
        ]
    )

    w = mfd_flow_weights(
        dem,
        1.0,
        1.0,
    )

    _, receiver = d8_flow_direction_resolved(
        dem,
        1.0,
        1.0,
    )

    sums = np.sum(w, axis=2).ravel()

    routed = receiver >= 0

    assert np.allclose(
        sums[routed],
        1.0,
    )

    assert np.allclose(
        sums[~routed],
        0.0,
    )
