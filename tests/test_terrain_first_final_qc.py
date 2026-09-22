import numpy as np
from fastgc.surface_consensus_guard import (
    recover_surface_false_negatives,
    demote_flying_ground_outliers,
)


def _terrain(kind="convex", n=25, per=6, seed=4):
    rng=np.random.default_rng(seed)
    x=[]; y=[]; z=[]
    for iy in range(n):
        for ix in range(n):
            xc=ix-(n-1)/2; yc=iy-(n-1)/2
            for _ in range(per):
                xx=xc+rng.uniform(-0.35,0.35)
                yy=yc+rng.uniform(-0.35,0.35)
                if kind=="convex": zz=0.04*xx+0.012*xx*xx+0.006*yy*yy
                elif kind=="concave": zz=0.04*xx-0.012*xx*xx-0.006*yy*yy
                else: zz=0.03*yy+0.20*abs(xx)
                x.append(xx); y.append(yy); z.append(zz+rng.normal(0,0.008))
    return np.asarray(x),np.asarray(y),np.asarray(z)


def test_large_curved_voids_are_recovered():
    for kind in ("convex","concave","ridge"):
        x,y,z=_terrain(kind)
        g=np.ones(x.size,dtype=bool)
        hole=(np.abs(x)<3.0)&(np.abs(y)<3.0)
        g[hole]=False
        out,rep=recover_surface_false_negatives(
            x=x,y=y,z=z,ground_mask=g,sensor_mode="ALS",
            cfg={"terrain_final_recovery_passes":6},
        )
        assert np.all(out[hole]), (kind,rep)


def test_elevated_vegetation_is_not_promoted_by_terrain_swipe():
    rng=np.random.default_rng(3)
    x=[];y=[];z=[];g=[];isveg=[]
    for iy in range(12):
        for ix in range(12):
            base=0.03*ix+0.02*iy
            for _ in range(4):
                x.append(ix+rng.uniform(-.3,.3)); y.append(iy+rng.uniform(-.3,.3))
                z.append(base+rng.normal(0,.008)); g.append(True); isveg.append(False)
            for _ in range(2):
                x.append(ix+rng.uniform(-.3,.3)); y.append(iy+rng.uniform(-.3,.3))
                z.append(base+0.80+rng.normal(0,.02)); g.append(False); isveg.append(True)
    x=np.asarray(x);y=np.asarray(y);z=np.asarray(z);g=np.asarray(g,bool);isveg=np.asarray(isveg,bool)
    out,_=recover_surface_false_negatives(x=x,y=y,z=z,ground_mask=g,sensor_mode="ALS",cfg={})
    assert not np.any(out[isveg])


def test_only_isolated_flying_ground_is_demoted():
    x=[];y=[];z=[];g=[]
    for iy in range(15):
        for ix in range(15):
            xx=ix+0.1; yy=iy+0.1; zz=0.03*xx+0.02*yy
            if ix==7 and iy==7: zz+=3.0
            x.append(xx);y.append(yy);z.append(zz);g.append(True)
    x=np.asarray(x);y=np.asarray(y);z=np.asarray(z);g=np.asarray(g,bool)
    out,rep=demote_flying_ground_outliers(x=x,y=y,z=z,ground_mask=g,sensor_mode="ALS",cfg={})
    assert not out.reshape(15,15)[7,7]
    assert np.count_nonzero(~out)==1, rep
