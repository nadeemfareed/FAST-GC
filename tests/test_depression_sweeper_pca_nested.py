import numpy as np

from fastgc.depression_sweeper import sweep_ground_depressions as sweep


def _cloud_from_grid(xs, ys, zfun, clsfun):
    X, Y = np.meshgrid(xs, ys)
    x = X.ravel(); y = Y.ravel(); z = zfun(x,y)
    g = clsfun(x,y,z)
    return x,y,z,g


def test_tls_unchanged():
    x=np.arange(30,dtype=float); y=np.zeros_like(x); z=np.zeros_like(x); g=np.ones_like(x,dtype=bool)
    g[10:15]=False
    out, rep = sweep(x=x,y=y,z=z,ground_mask=g,sensor_mode='TLS',cfg={},return_report=True)
    assert np.array_equal(out,g)
    assert rep['recovered_points']==0


def test_elevated_object_rejected():
    xs=np.arange(0,8,0.4); ys=np.arange(0,8,0.4)
    X,Y=np.meshgrid(xs,ys); x=X.ravel(); y=Y.ravel(); z=np.zeros_like(x); g=np.ones_like(x,dtype=bool)
    hole=(x>3)&(x<5)&(y>3)&(y<5)
    z[hole]=1.0; g[hole]=False
    out,rep=sweep(x=x,y=y,z=z,ground_mask=g,sensor_mode='ULS',cfg={},return_report=True)
    assert np.count_nonzero(out & ~g)==0


def test_linear_ditch_arbitrary_orientation_recovers():
    rng=np.random.default_rng(3)
    # Dense points on sloping ground plane.
    xs=np.arange(-6,6.001,0.2); ys=np.arange(-5,5.001,0.2)
    X,Y=np.meshgrid(xs,ys); x=X.ravel(); y=Y.ravel()
    z=0.02*x + 0.01*y
    # Rotated linear ditch centred on line y=0.35*x, width ~0.8 m.
    theta=np.deg2rad(23.0)
    u=x*np.cos(theta)+y*np.sin(theta)
    v=-x*np.sin(theta)+y*np.cos(theta)
    ditch=(np.abs(v)<0.45)&(np.abs(u)<4.4)
    z[ditch]-=0.35*(1-(np.abs(v[ditch])/0.45)**2)
    g=np.ones_like(x,dtype=bool)
    # Misclassify compact ditch bottom only, leave banks as ground.
    bad=ditch & (np.abs(v)<0.26)
    g[bad]=False
    out,rep=sweep(x=x,y=y,z=z,ground_mask=g,sensor_mode='ULS',cfg={
        'ditch_sweeper_linear_min_section_fraction':0.30,
        'ditch_sweeper_linear_min_sections':3,
    },return_report=True)
    recovered=np.count_nonzero(out & ~g)
    assert recovered>20, rep
    assert rep['linear_components']>=1, rep


def test_nested_trench_second_pass_recovers():
    # Broad V-shaped depression plus narrower trench in its bottom.
    xs=np.arange(-6,6.001,0.2); ys=np.arange(-4,4.001,0.2)
    X,Y=np.meshgrid(xs,ys); x=X.ravel(); y=Y.ravel()
    base=0.01*x
    broad=np.maximum(0.0, 1.0-np.abs(y)/2.0)
    z=base - 0.28*broad
    trench=np.maximum(0.0,1.0-np.abs(y)/0.35)
    z-=0.24*trench
    g=np.ones_like(x,dtype=bool)
    # First-level broad-bottom stripe plus nested deeper stripe are non-ground.
    broad_bad=(np.abs(y)<0.8)&(np.abs(x)<4.5)
    nested=(np.abs(y)<0.22)&(np.abs(x)<4.2)
    g[broad_bad]=False
    # Keep some broad depression edge ground to provide lateral support.
    keep_edge=(np.abs(y)>0.55)&(np.abs(y)<0.9)&(np.abs(x)<4.5)
    g[keep_edge]=True
    out,rep=sweep(x=x,y=y,z=z,ground_mask=g,sensor_mode='ULS',cfg={
        'ditch_sweeper_linear_min_section_fraction':0.25,
        'ditch_sweeper_linear_min_sections':3,
        'ditch_sweeper_nested_passes':2,
        'ditch_sweeper_linear_max_longitudinal_resid_m':0.25,
    },return_report=True)
    recovered=np.count_nonzero(out & ~g)
    assert recovered>25, rep
    # At least some deepest trench points should be recovered.
    assert np.count_nonzero(out[nested] & ~g[nested])>10, rep


def test_isolated_low_noise_rejected():
    xs=np.arange(0,8,0.4); ys=np.arange(0,8,0.4)
    X,Y=np.meshgrid(xs,ys); x=X.ravel(); y=Y.ravel(); z=np.zeros_like(x); g=np.ones_like(x,dtype=bool)
    idx=np.argmin((x-4)**2+(y-4)**2); z[idx]=-0.8; g[idx]=False
    out,rep=sweep(x=x,y=y,z=z,ground_mask=g,sensor_mode='ULS',cfg={},return_report=True)
    assert not out[idx]
