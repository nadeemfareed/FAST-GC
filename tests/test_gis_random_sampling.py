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
