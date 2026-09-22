import numpy as np
import fastgc.als_statistical_sheet_cleaner as m

def test_steep_bare_sheet_is_preserved():
    xs = np.linspace(0,20,201); ys = np.linspace(0,4,41); xx,yy = np.meshgrid(xs,ys)
    x = xx.ravel(); y = yy.ravel(); z = 3*x + 0.02*np.sin(y); g = np.ones_like(z,dtype=bool)
    out,rep = m.clean_statistical_ground_sheet(x=x,y=y,z=z,ground_mask=g,sensor_mode='ALS',cfg={},return_report=True)
    assert np.array_equal(out,g); assert rep['demoted_points'] == 0

def test_forest_canopy_ground_contamination_is_demoted():
    rng = np.random.default_rng(42)
    gx,gy = np.meshgrid(np.linspace(0,10,101),np.linspace(0,5,51))
    xg = gx.ravel(); yg = gy.ravel(); zg = .15*xg + .03*np.sin(yg); gg = np.ones(xg.size,dtype=bool)
    n = 5000; xc = rng.uniform(0,10,n); yc = rng.uniform(0,5,n); base = .15*xc + .03*np.sin(yc); zc = base + rng.uniform(2,12,n)
    contam = np.zeros(n,dtype=bool); contam[:1200] = True
    x = np.r_[xg,xc]; y = np.r_[yg,yc]; z = np.r_[zg,zc]; g = np.r_[gg,contam]
    out,rep = m.clean_statistical_ground_sheet(x=x,y=y,z=z,ground_mask=g,sensor_mode='ALS',cfg={},return_report=True)
    assert np.count_nonzero(contam & ~out[-n:]) > 800
    assert np.count_nonzero(~out[:xg.size]) == 0

def test_tls_noop():
    x=np.arange(20,dtype=float); y=np.zeros(20); z=np.zeros(20); g=np.ones(20,dtype=bool)
    out,rep = m.clean_statistical_ground_sheet(x=x,y=y,z=z,ground_mask=g,sensor_mode='TLS',cfg={},return_report=True)
    assert np.array_equal(out,g)

