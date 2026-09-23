import numpy as np
import pytest
from rasterio.transform import from_origin
from fastgc.gis.rough_sampling import choose_circular_plots


def test_equal_strata_reproducible():
    h = np.empty((160, 480), dtype=float)
    h[:, :160], h[:, 160:320], h[:, 320:] = 2., 10., 22.
    valid = np.ones_like(h, dtype=bool)
    kw = dict(count=6, radius=3., buffer=2., seed=42)
    a = choose_circular_plots(h, valid, from_origin(0, 160, 1, 1), **kw)
    b = choose_circular_plots(h, valid, from_origin(0, 160, 1, 1), **kw)
    assert a == b
    assert [x['stratum'] for x in a].count('low') == 2
    assert [x['stratum'] for x in a].count('medium') == 2
    assert [x['stratum'] for x in a].count('high') == 2


def test_insufficient_stratum_fails_without_partial_result():
    h = np.ones((100, 100))
    with pytest.raises(ValueError, match='Insufficient eligible area'):
        choose_circular_plots(h, np.ones_like(h, bool), from_origin(0, 100, 1, 1),
                              count=3, radius=3., buffer=2.)


def test_invalid_plot_count():
    h = np.ones((10, 10))
    with pytest.raises(ValueError, match='multiple of 3'):
        choose_circular_plots(h, np.ones_like(h, bool), from_origin(0, 10, 1, 1), count=4)
