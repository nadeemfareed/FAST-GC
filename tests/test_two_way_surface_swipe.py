import numpy as np
from fastgc.surface_consensus_guard import build_surface_consensus_workspace, two_way_surface_classification_swipe


def _curved_slope(seed=5):
    rng=np.random.default_rng(seed)
    xs=np.linspace(0,40,161); ys=np.linspace(0,20,81)
    xx,yy=np.meshgrid(xs,ys)
    x=xx.ravel(); y=yy.ravel()
    z=0.18*x+0.04*y+0.15*np.sin(x/8.0)+rng.normal(0,0.012,x.size)
    g=np.ones(x.size,dtype=bool)
    return rng,x,y,z,g


def test_ground_sheet_separates_low_understory_and_canopy_leaks_without_eating_terrain():
    rng,x,y,z,g=_curved_slope()
    nterrain=len(x)
    n=4000
    xv=rng.uniform(8,32,n); yv=rng.uniform(2,18,n)
    base=0.18*xv+0.04*yv+0.15*np.sin(xv/8.0)
    heights=rng.choice([0.22,0.35,1.2,4.0],size=n,p=[0.35,0.35,0.20,0.10])
    zv=base+heights+rng.normal(0,0.035,n)
    gv=np.zeros(n,dtype=bool)
    leak=rng.choice(n,size=300,replace=False); gv[leak]=True
    x=np.r_[x,xv]; y=np.r_[y,yv]; z=np.r_[z,zv]; g=np.r_[g,gv]
    ws=build_surface_consensus_workspace(x,y,z,1.0)
    out,rep=two_way_surface_classification_swipe(x=x,y=y,z=z,ground_mask=g,sensor_mode='ALS',workspace=ws)
    leak_global=nterrain+leak
    assert np.count_nonzero(~out[leak_global]) >= 285
    assert np.count_nonzero(~out[:nterrain]) == 0
    assert rep['demoted_points'] >= 285


def test_final_manifold_recovers_isolated_stolen_terrain_points():
    rng,x,y,z,g=_curved_slope(seed=11)
    stolen=np.arange(100,160,5)
    g[stolen]=False
    ws=build_surface_consensus_workspace(x,y,z,1.0)
    out,rep=two_way_surface_classification_swipe(x=x,y=y,z=z,ground_mask=g,sensor_mode='ALS',workspace=ws)
    assert np.count_nonzero(out[stolen]) >= int(0.75*len(stolen))
    assert rep['promoted_points'] > 0
