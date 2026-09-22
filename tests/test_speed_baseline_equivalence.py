import numpy as np

from fastgc.invert_vote import _fill_nan_median_step
from fastgc.surface_consensus_guard import build_surface_consensus_workspace


def _legacy_fill_step(grid):
    grid=np.asarray(grid,dtype=np.float32)
    ny,nx=grid.shape
    out=grid.copy()
    for j in range(ny):
        for i in range(nx):
            if np.isfinite(grid[j,i]):
                continue
            neigh=grid[max(0,j-1):min(ny,j+2),max(0,i-1):min(nx,i+2)]
            neigh=neigh[np.isfinite(neigh)]
            if neigh.size:
                out[j,i]=float(np.nanmedian(neigh))
    return out


def _legacy_cell_stats(x,y,z,g,cell):
    x0=float(np.min(x)); y0=float(np.min(y))
    ix=np.floor((x-x0)/cell).astype(np.int32); iy=np.floor((y-y0)/cell).astype(np.int32)
    nx=int(ix.max())+1; ny=int(iy.max())+1; key=iy.astype(np.int64)*nx+ix.astype(np.int64); n=nx*ny
    count=np.bincount(key,minlength=n).reshape(ny,nx)
    gcount=np.bincount(key[g],minlength=n).reshape(ny,nx)
    gz=np.full(n,np.nan); low=np.full(n,np.nan)
    order=np.argsort(key,kind='mergesort'); ks=key[order]; zs=z[order]; gs=g[order]
    starts=np.r_[0,1+np.flatnonzero(ks[1:]!=ks[:-1])]; ends=np.r_[starts[1:],len(ks)]
    for a,b in zip(starts,ends):
        k=int(ks[a]); block=zs[a:b]; gb=block[gs[a:b]]
        low[k]=float(np.quantile(block,0.12))
        if gb.size: gz[k]=float(np.median(gb))
    return count,gcount,gz.reshape(ny,nx),low.reshape(ny,nx),key


def test_vectorized_fill_is_exact_legacy_step():
    rng=np.random.default_rng(41)
    for shape in ((3,4),(17,23),(31,29)):
        for _ in range(8):
            a=rng.normal(size=shape).astype(np.float32)
            a[rng.random(shape)<0.62]=np.nan
            old=_legacy_fill_step(a)
            new=_fill_nan_median_step(a)
            assert np.array_equal(np.isnan(old),np.isnan(new))
            assert np.allclose(old,new,rtol=0.0,atol=0.0,equal_nan=True)


def test_workspace_reuses_exact_legacy_cell_statistics():
    rng=np.random.default_rng(7)
    x=rng.uniform(0,50,7000); y=rng.uniform(0,40,7000)
    z=.03*x-.015*y+rng.normal(0,.05,7000)
    g=rng.random(7000)>.35
    old_count,old_gcount,old_gz,old_low,old_key=_legacy_cell_stats(x,y,z,g,1.0)
    ws=build_surface_consensus_workspace(x,y,z,1.0)
    new_gcount,new_gz=ws.ground_stats(z,g)
    assert np.array_equal(ws.key,old_key)
    assert np.array_equal(ws.count,old_count)
    assert np.array_equal(new_gcount,old_gcount)
    assert np.allclose(new_gz,old_gz,rtol=0.0,atol=0.0,equal_nan=True)
    assert np.allclose(ws.low,old_low,rtol=0.0,atol=0.0,equal_nan=True)
