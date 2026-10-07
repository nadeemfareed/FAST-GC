import numpy as np

from fastgc.hydrology import (
    condition_dem,
    d8_flow_accumulation,
    d8_flow_direction_resolved,
    d8_basin_labels,
)


def _outlet_mass(dem, receiver, accumulation):
    valid = np.isfinite(dem).ravel()
    rec = np.asarray(receiver).reshape(-1)
    acc = np.asarray(accumulation).reshape(-1)

    outlets = valid & (rec < 0)

    return (
        float(np.sum(acc[outlets])),
        int(np.count_nonzero(valid)),
    )


def test_priority_flood_flat_conserves_all_source_cells():
    raw = np.full((21, 21), 100.0)

    raw[4:17, 4:17] = 90.0
    raw[10, 10] = 70.0

    # Real boundary outlet corridor.
    raw[10, 17:] = np.array(
        [90.0, 89.0, 88.0, 87.0]
    )

    dem, _ = condition_dem(raw)

    _, receiver = d8_flow_direction_resolved(
        dem,
        1.0,
        1.0,
    )

    acc = d8_flow_accumulation(
        dem,
        1.0,
        1.0,
        receiver=receiver,
    )

    mass, cells = _outlet_mass(
        dem,
        receiver,
        acc,
    )

    assert np.isclose(mass, cells)


def test_large_flat_domain_conserves_mass_across_open_boundaries():
    dem = np.full((31, 41), 100.0)

    _, receiver = d8_flow_direction_resolved(
        dem,
        1.0,
        1.0,
    )

    acc = d8_flow_accumulation(
        dem,
        1.0,
        1.0,
        receiver=receiver,
    )

    mass, cells = _outlet_mass(
        dem,
        receiver,
        acc,
    )

    assert np.isclose(mass, cells)


def test_nodata_domain_conserves_mass():
    dem = np.full((25, 25), 50.0)

    dem[8:17, 8:17] = 40.0
    dem[12, 12] = np.nan

    dem, _ = condition_dem(dem)

    _, receiver = d8_flow_direction_resolved(
        dem,
        2.0,
        3.0,
    )

    acc = d8_flow_accumulation(
        dem,
        2.0,
        3.0,
        receiver=receiver,
    )

    mass, cells = _outlet_mass(
        dem,
        receiver,
        acc,
    )

    assert np.isclose(mass, cells)


def test_basin_labels_are_defined_for_routed_domain():
    dem = np.full((17, 17), 20.0)

    dem[8, -1] = 19.0

    _, receiver = d8_flow_direction_resolved(
        dem,
        1.0,
        1.0,
    )

    labels = d8_basin_labels(
        dem,
        receiver,
    )

    valid = np.isfinite(dem)

    assert labels.shape == dem.shape
    assert np.all(labels[valid] >= 0)


def test_basin_labels_are_deterministic_after_flat_resolution():
    dem = np.full((19, 23), 100.0)

    dem[5, 0] = 99.0
    dem[14, -1] = 99.0

    outputs = []

    for _ in range(4):
        _, receiver = d8_flow_direction_resolved(
            dem,
            1.0,
            1.0,
        )

        outputs.append(
            d8_basin_labels(
                dem,
                receiver,
            )
        )

    for labels in outputs[1:]:
        np.testing.assert_array_equal(
            labels,
            outputs[0],
        )


def test_rectangular_cells_preserve_accumulation_mass():
    dem = np.full((15, 27), 80.0)
    dem[7, -1] = 79.0

    _, receiver = d8_flow_direction_resolved(
        dem,
        dx=2.0,
        dy=7.0,
    )

    acc = d8_flow_accumulation(
        dem,
        2.0,
        7.0,
        receiver=receiver,
    )

    mass, cells = _outlet_mass(
        dem,
        receiver,
        acc,
    )

    assert np.isclose(mass, cells)
