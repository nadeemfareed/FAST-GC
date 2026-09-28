import numpy as np
import pytest

from fastgc.io_las import (
    _apply_support_mask,
    _build_surface_from_points,
    _grid_support,
    _idw_grid,
    _rasterize_stat,
    _support_mask_from_points,
)


def _two_layer_grid():
    """
    Four 1 x 1 cells, each containing a lower and upper return.
    Expected per-cell lower=1, upper=9, mean=5.
    """
    x = np.array([
        0.25, 0.25,
        1.25, 1.25,
        0.25, 0.25,
        1.25, 1.25,
    ], dtype=np.float64)

    y = np.array([
        0.25, 0.25,
        0.25, 0.25,
        1.25, 1.25,
        1.25, 1.25,
    ], dtype=np.float64)

    z = np.array([
        1.0, 9.0,
        1.0, 9.0,
        1.0, 9.0,
        1.0, 9.0,
    ], dtype=np.float64)

    return x, y, z


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("min", 1.0),
        ("max", 9.0),
        ("mean", 5.0),
    ],
)
def test_grid_support_preserves_requested_cell_statistic(mode, expected):
    x, y, z = _two_layer_grid()

    sx, sy, sz = _grid_support(x, y, z, cell=1.0, mode=mode)

    assert sx.size == 4
    assert sy.size == 4
    assert sz.size == 4
    assert np.allclose(sz, expected)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("min", 1.0),
        ("max", 9.0),
        ("mean", 5.0),
    ],
)
def test_surface_stat_methods_recover_controlled_horizontal_layers(method, expected):
    x, y, z = _two_layer_grid()

    grid, xs, ys = _build_surface_from_points(
        x,
        y,
        z,
        bounds=(0.0, 0.0, 2.0, 2.0),
        res=1.0,
        method=method,
        mask_kind="none",
    )

    assert grid.shape == (2, 2)
    assert xs.size == 2
    assert ys.size == 2
    assert np.all(np.isfinite(grid))
    assert np.allclose(grid, expected)


def test_idw_returns_finite_surface_inside_controlled_domain():
    x = np.array([0.0, 2.0, 0.0, 2.0], dtype=np.float64)
    y = np.array([0.0, 0.0, 2.0, 2.0], dtype=np.float64)
    z = np.array([0.0, 2.0, 2.0, 4.0], dtype=np.float64)

    grid, xs, ys = _idw_grid(
        x,
        y,
        z,
        bounds=(0.0, 0.0, 2.0, 2.0),
        res=1.0,
    )

    assert grid.shape == (2, 2)
    assert np.all(np.isfinite(grid))
    assert np.nanmin(grid) >= np.min(z)
    assert np.nanmax(grid) <= np.max(z)


def test_support_mask_prevents_full_extent_extrapolation():
    x = np.array([0.25], dtype=np.float64)
    y = np.array([0.25], dtype=np.float64)

    mask = _support_mask_from_points(
        x,
        y,
        bounds=(0.0, 0.0, 5.0, 5.0),
        res=1.0,
        grow_cells=0,
    )

    assert mask.shape == (5, 5)
    assert np.count_nonzero(mask) == 1

    grid = np.ones((5, 5), dtype=np.float32)
    masked = _apply_support_mask(grid, mask)

    assert np.count_nonzero(np.isfinite(masked)) == 1
    assert np.count_nonzero(~np.isfinite(masked)) == 24


def test_default_surface_support_mask_is_local():
    x = np.array([0.25], dtype=np.float64)
    y = np.array([0.25], dtype=np.float64)
    z = np.array([10.0], dtype=np.float64)

    grid, _, _ = _build_surface_from_points(
        x,
        y,
        z,
        bounds=(0.0, 0.0, 5.0, 5.0),
        res=1.0,
        method="nearest",
    )

    finite = np.isfinite(grid)

    assert np.count_nonzero(finite) > 0
    assert np.count_nonzero(finite) < grid.size


def test_rasterize_max_uses_highest_return_per_cell():
    x, y, z = _two_layer_grid()

    grid, xmin, ymax = _rasterize_stat(
        x,
        y,
        z,
        bounds=(0.0, 0.0, 2.0, 2.0),
        res=1.0,
        mode="max",
    )

    assert grid.shape == (2, 2)
    assert xmin == pytest.approx(0.0)
    assert ymax == pytest.approx(2.0)
    assert np.allclose(grid, 9.0)


def test_spikefree_does_not_exceed_observed_cell_max_plus_buffer():
    # Dense regular support with one isolated high return.
    xx, yy = np.meshgrid(
        np.arange(0.25, 4.0, 0.5),
        np.arange(0.25, 4.0, 0.5),
    )

    x = xx.ravel().astype(np.float64)
    y = yy.ravel().astype(np.float64)
    z = np.full(x.shape, 10.0, dtype=np.float64)

    # Artificial isolated spike.
    spike_idx = np.argmin((x - 2.25) ** 2 + (y - 2.25) ** 2)
    z[spike_idx] = 50.0

    grid, _, _ = _build_surface_from_points(
        x,
        y,
        z,
        bounds=(0.0, 0.0, 4.0, 4.0),
        res=0.5,
        method="spikefree",
        spikefree_freeze_distance=2.0,
        spikefree_insertion_buffer=0.25,
        mask_kind="none",
    )

    assert np.any(np.isfinite(grid))
    assert np.nanmin(grid) >= 10.0 - 1e-6
    assert np.nanmax(grid) <= 50.25 + 1e-6


def test_invalid_surface_method_fails_explicitly():
    x = np.array([0.25], dtype=np.float64)
    y = np.array([0.25], dtype=np.float64)
    z = np.array([1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="Unsupported raster method"):
        _build_surface_from_points(
            x,
            y,
            z,
            bounds=(0.0, 0.0, 1.0, 1.0),
            res=1.0,
            method="not-a-method",
        )


def test_nonpositive_resolution_fails_explicitly():
    x = np.array([0.25], dtype=np.float64)
    y = np.array([0.25], dtype=np.float64)
    z = np.array([1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="res must be > 0"):
        _build_surface_from_points(
            x,
            y,
            z,
            bounds=(0.0, 0.0, 1.0, 1.0),
            res=0.0,
            method="max",
        )
