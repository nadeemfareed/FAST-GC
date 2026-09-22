import numpy as np

from fastgc.surface_consensus_guard import _terrain_impurity_consensus_als


def _sheet(slope: float, contaminated: bool):
    xs=np.arange(0.0,10.0,0.5)
    ys=np.arange(0.0,10.0,0.5)
    X,Y=np.meshgrid(xs,ys)
    x=X.ravel(); y=Y.ravel()
    z=slope*x + 0.02*np.sin(y)
    g=np.ones(x.size,dtype=bool)
    target=None
    if contaminated:
        target=int(np.argmin((x-5.0)**2+(y-5.0)**2))
        z[target]+=1.0
        # Nearby non-ground column provides 3-D vegetation evidence.
        x=np.r_[x,[5.05,5.10,4.95,5.00]]
        y=np.r_[y,[5.00,5.05,5.00,4.95]]
        z=np.r_[z,[slope*5+0.25,slope*5+0.35,slope*5+0.45,slope*5+0.55]]
        g=np.r_[g,[False,False,False,False]]
    return x,y,z,g,target


def test_v3_removes_isolated_pyramid_on_flat_surface():
    x,y,z,g,target=_sheet(0.0,True)
    out,rep=_terrain_impurity_consensus_als(x=x,y=y,z=z,ground_mask=g,cfg={},workspace=None)
    assert target is not None
    assert not out[target]
    assert rep['demoted_points'] >= 1


def test_v3_removes_slope_normal_pyramid_on_extreme_slope():
    x,y,z,g,target=_sheet(2.0,True)
    out,rep=_terrain_impurity_consensus_als(x=x,y=y,z=z,ground_mask=g,cfg={},workspace=None)
    assert target is not None
    assert not out[target]
    assert rep['demoted_points'] >= 1


def test_v3_does_not_erode_clean_extreme_slope():
    x,y,z,g,_=_sheet(2.0,False)
    out,rep=_terrain_impurity_consensus_als(x=x,y=y,z=z,ground_mask=g,cfg={},workspace=None)
    assert np.array_equal(out,g)
    assert rep['demoted_points'] == 0
