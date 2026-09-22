import importlib.util
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'src'/'fastgc'

def load(name):
    p=ROOT/f'{name}.py'; spec=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m


def test_global_grid_phase_and_planar_surface():
    m=load('tls_vote'); cfg=m.TlsInvertDsmVoteConfig(cell=0.35,propagation_iters=8,fill_iters=4)
    xx,yy=np.meshgrid(np.linspace(1000.02,1008,100),np.linspace(2000.11,2008,100))
    z=10+0.05*xx+0.02*yy
    surf,x0,y0=m.build_tls_surface_invert_dsm_vote(xx.ravel(),yy.ravel(),z.ravel(),cfg)
    assert abs((x0/cfg.cell)-round(x0/cfg.cell))<1e-9
    assert abs((y0/cfg.cell)-round(y0/cfg.cell))<1e-9
    pred=m._bilinear_sample(surf,xx.ravel(),yy.ravel(),x0,y0,cfg.cell)
    ok=np.isfinite(pred)
    assert np.median(np.abs(pred[ok]-z.ravel()[ok]))<0.08


def test_convex_concave_curved_surface_is_retained():
    m=load('tls_vote'); cfg=m.TlsInvertDsmVoteConfig(cell=0.35,propagation_iters=12,fill_iters=6)
    rng=np.random.default_rng(2)
    x=rng.uniform(0,20,30000); y=rng.uniform(0,20,30000)
    z=0.03*x + 0.002*(x-10)**2 - 0.0015*(y-10)**2 + rng.normal(0,0.01,x.size)
    # add vegetation well above ground
    xv=rng.uniform(0,20,8000); yv=rng.uniform(0,20,8000); zv=0.03*xv+0.002*(xv-10)**2-0.0015*(yv-10)**2+rng.uniform(0.5,6,8000)
    xa=np.r_[x,xv]; ya=np.r_[y,yv]; za=np.r_[z,zv]
    surf,x0,y0=m.build_tls_surface_invert_dsm_vote(xa,ya,za,cfg)
    g=m.classify_tls_by_surface(xa,ya,za,surf,x0,y0,cfg)
    recall=g[:x.size].mean(); fp=g[x.size:].mean()
    assert recall>0.90
    assert fp<0.03


def test_membrane_and_recovery_are_compatibility_noops():
    mm=load('tls_membrane_refine'); rr=load('tls_terrain_recover')
    x=np.arange(10.0); y=x.copy(); z=x.copy(); g=np.array([1,0,1,0,1,0,1,0,1,0],bool)
    a=mm.refine_tls_ground_membrane(x=x,y=y,z=z,candidate_ground=g,config=mm.TlsMembraneConfig())
    b=rr.recover_tls_microtopography(x=x,y=y,z=z,trusted_ground=g,candidate_pool=~g,config=rr.TlsTerrainRecoveryConfig())
    assert np.array_equal(a.final_ground,g)
    assert np.array_equal(b.final_ground,g)
