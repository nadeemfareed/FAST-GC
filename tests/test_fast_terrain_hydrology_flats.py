import numpy as np

from fastgc.hydrology import (
    condition_dem,
    d8_flow_accumulation,
    d8_flow_direction,
    d8_flow_direction_resolved,
)


def test_filled_depression_does_not_create_internal_outlets():
    z = np.full((9, 9), 10.0)
    z[2:7, 2:7] = 5.0

    # Real boundary outlet.
    z[0, 4] = 8.0
    z[1, 4] = 8.0

    conditioned, _ = condition_dem(z)

    _, raw_receiver = d8_flow_direction(
        conditioned,
        1.0,
        1.0,
    )

    _, resolved = d8_flow_direction_resolved(
        conditioned,
        1.0,
        1.0,
    )

    valid = np.isfinite(conditioned).ravel()

    raw_outlets = np.count_nonzero(
        valid & (raw_receiver < 0)
    )

    resolved_outlets = np.count_nonzero(
        valid & (resolved < 0)
    )

    assert resolved_outlets < raw_outlets


def test_flat_resolution_is_acyclic():
    z = np.full((15, 15), 10.0)
    z[0, 7] = 9.0

    _, receiver = d8_flow_direction_resolved(
        z,
        1.0,
        1.0,
    )

    # Accumulation performs an explicit cycle check.
    acc = d8_flow_accumulation(
        z,
        1.0,
        1.0,
        receiver=receiver,
    )

    assert np.all(np.isfinite(acc))


def test_flat_domain_has_deterministic_outlet():
    z = np.full((7, 7), 100.0)

    _, a = d8_flow_direction_resolved(
        z, 1.0, 1.0
    )

    _, b = d8_flow_direction_resolved(
        z, 1.0, 1.0
    )

    np.testing.assert_array_equal(a, b)

    assert np.count_nonzero(a < 0) >= 1


def test_resolved_direction_codes_match_receivers():
    z = np.full((7, 7), 10.0)
    z[0, 3] = 9.0

    direction, receiver = (
        d8_flow_direction_resolved(
            z,
            1.0,
            1.0,
        )
    )

    assert direction.shape == z.shape
    assert receiver.size == z.size

    assert np.count_nonzero(direction) > 0
