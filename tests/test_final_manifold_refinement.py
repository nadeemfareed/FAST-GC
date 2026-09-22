import numpy as np
from fastgc.surface_consensus_guard import (
    build_surface_consensus_workspace,
    two_way_surface_classification_swipe,
)


def _terrain(seed=1):
    rng=np.random.default_rng(seed)
    gx=np.arange(0.0,30.0,0.45)
    gy=np.arange(0.0,14.0,0.55)
    xx,yy=np.meshgrid(gx,gy)
    x=xx.ravel(); y=yy.ravel()
    # slope + convex/concave bends in both axes
    z=(0.075*x + 0.018*y
       + 0.006*(x-15.0)**2
       - 0.035*np.exp(-((x-16.0)**2/18.0 + (y-7.0)**2/8.0)))
    z=z+rng.normal(0,0.012,size=z.size)
    return x,y,z


def test_recovers_curved_depression_omissions_without_promoting_shrubs():
    x,y,z=_terrain()
    n=len(x)
    ground=np.ones(n,dtype=bool)
    # stolen ground: a broad curved strip through depression/slope transition
    lost=((x>12.0)&(x<18.5)&(y>4.0)&(y<9.5)&(((np.arange(n)%3)!=0)))
    ground[lost]=False
    true_ground=np.ones(n,dtype=bool)
    # add shrubs/understory near surface and canopy -- all non-ground
    rng=np.random.default_rng(4)
    take=rng.choice(n,700,replace=False)
    vx=x[take]+rng.normal(0,0.08,len(take)); vy=y[take]+rng.normal(0,0.08,len(take))
    vz=z[take]+rng.choice([0.35,0.65,1.5,4.0],len(take),p=[.35,.30,.25,.10])
    X=np.r_[x,vx]; Y=np.r_[y,vy]; Z=np.r_[z,vz]
    G=np.r_[ground,np.zeros(len(vx),dtype=bool)]
    ws=build_surface_consensus_workspace(X,Y,Z,1.0)
    out,rep=two_way_surface_classification_swipe(x=X,y=Y,z=Z,ground_mask=G,sensor_mode='ALS',workspace=ws)
    recovered=np.count_nonzero(out[:n] & lost)
    assert recovered >= int(0.70*np.count_nonzero(lost))
    # Do not flood vegetation: at most a tiny fraction may enter ground.
    assert np.count_nonzero(out[n:]) <= 5
    # Original true ground outside the deliberately lost strip should remain.
    assert np.count_nonzero(~out[:n] & ~lost) <= 5
    assert rep['promoted_points'] > 0


def test_removes_small_detached_ground_islands_but_preserves_curved_sheet():
    x,y,z=_terrain(seed=7)
    n=len(x)
    rng=np.random.default_rng(8)
    # Add two small false-ground islands above the established sheet.
    centers=[(8.0,4.5,0.55),(22.0,9.0,1.4)]
    fx=[]; fy=[]; fz=[]
    for cx,cy,h in centers:
        ids=np.argsort((x-cx)**2+(y-cy)**2)[:18]
        fx.append(x[ids]+rng.normal(0,0.04,len(ids)))
        fy.append(y[ids]+rng.normal(0,0.04,len(ids)))
        fz.append(z[ids]+h+rng.normal(0,0.015,len(ids)))
    fx=np.concatenate(fx); fy=np.concatenate(fy); fz=np.concatenate(fz)
    X=np.r_[x,fx]; Y=np.r_[y,fy]; Z=np.r_[z,fz]
    G=np.ones(len(X),dtype=bool)
    ws=build_surface_consensus_workspace(X,Y,Z,1.0)
    out,rep=two_way_surface_classification_swipe(x=X,y=Y,z=Z,ground_mask=G,sensor_mode='ALS',workspace=ws)
    # Most detached points should be rejected.
    assert np.count_nonzero(~out[n:]) >= int(0.75*len(fx))
    # Main curved/sloping terrain should be essentially untouched.
    assert np.count_nonzero(~out[:n]) <= 8
    assert rep['demoted_points'] > 0


def test_no_changes_on_clean_curved_terrain():
    x,y,z=_terrain(seed=12)
    g=np.ones(len(x),dtype=bool)
    ws=build_surface_consensus_workspace(x,y,z,1.0)
    out,rep=two_way_surface_classification_swipe(x=x,y=y,z=z,ground_mask=g,sensor_mode='ALS',workspace=ws)
    assert np.count_nonzero(out != g) <= 3
