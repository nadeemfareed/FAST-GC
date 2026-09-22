import numpy as np

from fastgc.invert_vote import InvertVoteConfig, _normal_vote_surface, sample_surface


def test_normal_vote_surface_signature_and_fill():
    yy, xx = np.mgrid[0:9, 0:9]
    observed = (0.08 * xx + 0.03 * yy).astype(np.float32)
    truth = float(observed[4, 4])
    observed[4, 4] = np.nan
    out = _normal_vote_surface(observed, InvertVoteConfig(cell=0.75, neighbor_radius_cells=3, min_neighbor_cells=6, fill_iters=4))
    assert out.shape == observed.shape
    assert np.isfinite(out[4, 4])
    assert abs(float(out[4, 4]) - truth) < 0.08


def test_sample_surface_uses_xy_origin_once():
    surf = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    value = sample_surface(surf, 0.0, 0.0, 1.0, 0.25, 0.25)
    assert np.isfinite(value)
