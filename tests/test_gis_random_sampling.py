import numpy as np
import pytest

from fastgc.gis.random_sampling import _allocate_equal


def test_equal_region_allocation_is_deterministic():
    assert _allocate_equal(30, 3) == [10, 10, 10]
    assert _allocate_equal(10, 3) == [4, 3, 3]
    assert _allocate_equal(2, 3) == [1, 1, 0]


def test_equal_region_allocation_rejects_invalid_counts():
    with pytest.raises(ValueError):
        _allocate_equal(0, 3)
    with pytest.raises(ValueError):
        _allocate_equal(3, 0)


def test_random_sampling_requested_count_is_upper_limit(monkeypatch):
    import numpy as np
    from fastgc.gis.random_sampling import choose_random_plots

    catalog = {
        "tiles": [
            {"tile_id": "tile_1", "status": "ready"}
        ]
    }
    regions = {
        "regions": [
            {
                "region_id": "region_001",
                "tile_ids": ["tile_1"],
                "bounds": [0.0, 0.0, 200.0, 200.0],
            }
        ]
    }

    occupied = np.ones((9, 9), dtype=bool)

    def fake_occupancy(region, tile_by_id, **kwargs):
        return occupied, (0.0, 0.0, 45.0, 45.0)

    monkeypatch.setattr(
        "fastgc.gis.random_sampling._region_occupancy",
        fake_occupancy,
    )

    chosen, quotas = choose_random_plots(
        catalog,
        regions,
        count=30,
        radius=5.0,
        buffer=0.0,
        seed=42,
        cell_size=5.0,
    )

    assert quotas == [30]
    assert 0 < len(chosen) < 30

