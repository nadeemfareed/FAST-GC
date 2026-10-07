import numpy as np

from fastgc.hydrology import (
    mfd_flow_accumulation,
    mfd_contributing_area,
    mfd_specific_catchment_area,
)


def test_contributing_area_equals_cells_times_cell_area():
    dem = np.array(
        [
            [5., 4., 3.],
            [5., 4., 3.],
            [5., 4., 3.],
        ]
    )

    dx = 2.0
    dy = 3.0

    accumulation = mfd_flow_accumulation(
        dem,
        dx,
        dy,
    )

    area = mfd_contributing_area(
        dem,
        dx,
        dy,
        accumulation=accumulation,
    )

    assert np.allclose(
        area,
        accumulation * dx * dy,
    )


def test_contributing_area_has_square_length_scaling():
    dem = np.array(
        [
            [5., 4., 3.],
            [5., 4., 3.],
            [5., 4., 3.],
        ]
    )

    a1 = mfd_contributing_area(
        dem,
        1.0,
        1.0,
    )

    a2 = mfd_contributing_area(
        dem,
        2.0,
        2.0,
    )

    # Same cell topology, four times cell area.
    assert np.allclose(
        a2,
        4.0 * a1,
    )


def test_square_cell_sca_equals_area_divided_by_cell_width():
    dem = np.array(
        [
            [5., 4., 3.],
            [5., 4., 3.],
            [5., 4., 3.],
        ]
    )

    resolution = 5.0

    accumulation = mfd_flow_accumulation(
        dem,
        resolution,
        resolution,
    )

    area = mfd_contributing_area(
        dem,
        resolution,
        resolution,
        accumulation=accumulation,
    )

    sca = mfd_specific_catchment_area(
        dem,
        resolution,
        resolution,
        accumulation=accumulation,
    )

    assert np.allclose(
        sca,
        area / resolution,
    )


def test_sca_has_length_units_under_uniform_scaling():
    dem = np.array(
        [
            [5., 4., 3.],
            [5., 4., 3.],
            [5., 4., 3.],
        ]
    )

    sca1 = mfd_specific_catchment_area(
        dem,
        1.0,
        1.0,
    )

    sca2 = mfd_specific_catchment_area(
        dem,
        2.0,
        2.0,
    )

    # Specific catchment area has dimensions of length.
    assert np.allclose(
        sca2,
        2.0 * sca1,
    )


def test_rectangular_cell_sca_uses_documented_effective_width():
    dem = np.array(
        [
            [5., 4., 3.],
            [5., 4., 3.],
            [5., 4., 3.],
        ]
    )

    dx = 2.0
    dy = 8.0

    accumulation = mfd_flow_accumulation(
        dem,
        dx,
        dy,
    )

    area = mfd_contributing_area(
        dem,
        dx,
        dy,
        accumulation=accumulation,
    )

    sca = mfd_specific_catchment_area(
        dem,
        dx,
        dy,
        accumulation=accumulation,
    )

    effective_width = np.sqrt(dx * dy)

    assert np.allclose(
        sca,
        area / effective_width,
    )


def test_area_and_sca_preserve_nodata():
    dem = np.array(
        [
            [np.nan, 5., 4.],
            [5., 4., 3.],
            [4., 3., np.nan],
        ]
    )

    area = mfd_contributing_area(
        dem,
        2.0,
        2.0,
    )

    sca = mfd_specific_catchment_area(
        dem,
        2.0,
        2.0,
    )

    invalid = ~np.isfinite(dem)

    assert np.all(np.isnan(area[invalid]))
    assert np.all(np.isnan(sca[invalid]))

    assert np.all(np.isfinite(area[~invalid]))
    assert np.all(np.isfinite(sca[~invalid]))
