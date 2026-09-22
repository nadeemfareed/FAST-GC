
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import median_filter


@dataclass
class InvertVoteConfig:
    cell: float = 0.25
    top_m: int = 1
    neighbor_radius_cells: int = 6
    min_neighbor_cells: int = 10
    max_robust_z: float = 2.8
    mad_floor: float = 0.05
    fill_iters: int = 120
    smooth_sigma_cells: float = 1.25
    ground_threshold: float = 0.2
    slope_adapt_k: float = 0.35

    # Global support-layer controls
    use_offset_swipe: bool = True
    # Keep snapping local by default; larger radii can reintroduce banding.
    support_snap_radius_cells: int = 0


def _grid_index(
    x: np.ndarray,
    y: np.ndarray,
    cell: float,
    x0: float | None = None,
    y0: float | None = None,
):
    if x0 is None:
        x0 = float(np.min(x))
    if y0 is None:
        y0 = float(np.min(y))
    ix = np.floor((x - x0) / cell).astype(np.int32)
    iy = np.floor((y - y0) / cell).astype(np.int32)
    return float(x0), float(y0), ix, iy


def _conv_same_len(v: np.ndarray, k1: np.ndarray) -> np.ndarray:
    """
    np.convolve(..., mode="same") returns length max(len(v), len(k1)).
    That breaks when len(k1) > len(v). We always trim back to len(v).
    """
    out = np.convolve(v, k1, mode="same")
    n = int(v.size)
    if out.size == n:
        return out.astype(np.float32, copy=False)
    start = max(0, (out.size - n) // 2)
    end = start + n
    return out[start:end].astype(np.float32, copy=False)


def _conv1d_nan(a: np.ndarray, k1: np.ndarray, axis: int):
    out = np.full_like(a, np.nan, dtype=np.float32)
    if axis == 0:
        for i in range(a.shape[1]):
            col = a[:, i].astype(np.float32, copy=False)
            num = _conv_same_len(np.nan_to_num(col, nan=0.0), k1)
            den = _conv_same_len(np.isfinite(col).astype(np.float32), k1)
            tmp = np.full(col.shape, np.nan, dtype=np.float32)
            np.divide(num, den, out=tmp, where=den > 1e-6)
            out[:, i] = tmp
    else:
        for j in range(a.shape[0]):
            row = a[j, :].astype(np.float32, copy=False)
            num = _conv_same_len(np.nan_to_num(row, nan=0.0), k1)
            den = _conv_same_len(np.isfinite(row).astype(np.float32), k1)
            tmp = np.full(row.shape, np.nan, dtype=np.float32)
            np.divide(num, den, out=tmp, where=den > 1e-6)
            out[j, :] = tmp
    return out


def _fill_nan_median_step(grid: np.ndarray) -> np.ndarray:
    """One exact 3x3 NaN-median propagation step, vectorized.

    This is mathematically identical to the former nested Python j/i loops:
    finite cells are frozen; each NaN cell receives the median of finite values
    in its truncated 3x3 neighborhood; cells with no finite neighbor stay NaN.
    NaN padding reproduces the old edge truncation exactly.
    """
    a=np.asarray(grid,dtype=np.float32)
    ny,nx=a.shape
    p=np.pad(a,((1,1),(1,1)),mode='constant',constant_values=np.nan)
    stack=np.stack([p[dy:dy+ny,dx:dx+nx] for dy in range(3) for dx in range(3)],axis=0)
    finite=np.isfinite(stack)
    # Avoid all-NaN warnings while preserving all-NaN cells as NaN.
    work=np.where(finite,stack,np.inf)
    count=np.sum(finite,axis=0)
    # Sort only 9 values per cell; median of finite values is exact, including
    # the average of the two central values when the support count is even.
    work.sort(axis=0)
    lo=np.maximum((count-1)//2,0)
    hi=count//2
    yy,xx=np.indices((ny,nx))
    med=(work[lo,yy,xx]+work[hi,yy,xx])*0.5
    med[count==0]=np.nan
    out=a.copy()
    missing=~np.isfinite(a)
    out[missing]=med[missing].astype(np.float32,copy=False)
    return out


def _build_point_bins(ix: np.ndarray, iy: np.ndarray, nx: int, ny: int):
    bins = [[] for _ in range(nx * ny)]
    for p in range(ix.size):
        i = int(ix[p])
        j = int(iy[p])
        if 0 <= i < nx and 0 <= j < ny:
            bins[j * nx + i].append(p)
    return bins


def _initial_surface_from_swipe(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cell: float,
    x0_base: float,
    y0_base: float,
    nx: int,
    ny: int,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
):
    """Vectorized published lower-support swipe.

    Semantics are unchanged: each offset cell contributes its lowest observed
    point, which is then mapped back to the base grid and lower-envelope merged.
    """
    x0 = x0_base + x_offset
    y0 = y0_base + y_offset
    ix = np.floor((x - x0) / cell).astype(np.int32)
    iy = np.floor((y - y0) / cell).astype(np.int32)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    surf = np.full((ny, nx), np.nan, dtype=np.float32)
    if not np.any(valid):
        return surf

    ixv = ix[valid]; iyv = iy[valid]
    xv = x[valid]; yv = y[valid]; zv = z[valid]
    lin = iyv.astype(np.int64) * int(nx) + ixv.astype(np.int64)

    # Sort by swipe-cell then z so the first member of each cell is the
    # published max-inverted/min-original support point.
    order = np.lexsort((zv, lin))
    lin_s = lin[order]
    first = np.r_[True, lin_s[1:] != lin_s[:-1]]
    pick = order[first]

    xs = xv[pick]; ys = yv[pick]; zs = zv[pick]
    ib = np.floor((xs - x0_base) / cell).astype(np.int32)
    jb = np.floor((ys - y0_base) / cell).astype(np.int32)
    good = (ib >= 0) & (ib < nx) & (jb >= 0) & (jb < ny)
    if not np.any(good):
        return surf

    base_lin = jb[good].astype(np.int64) * int(nx) + ib[good].astype(np.int64)
    vals = zs[good].astype(np.float32, copy=False)
    flat = np.full(nx * ny, np.inf, dtype=np.float32)
    np.minimum.at(flat, base_lin, vals)
    flat[~np.isfinite(flat) | (flat == np.inf)] = np.nan
    return flat.reshape(ny, nx)


def _merge_surfaces_lower(s1: np.ndarray, s2: np.ndarray):
    out = s1.copy()
    mask2 = np.isfinite(s2)
    mask1 = np.isfinite(out)
    take2 = mask2 & (~mask1 | (s2 < out))
    out[take2] = s2[take2]
    return out


def _bilinear_sample(
    grid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    y0: float,
    cell: float,
):
    ny, nx = grid.shape
    gx = (x - x0) / cell
    gy = (y - y0) / cell

    ix = np.floor(gx).astype(np.int32)
    iy = np.floor(gy).astype(np.int32)
    fx = gx - ix
    fy = gy - iy

    ix0 = np.clip(ix, 0, max(0, nx - 2))
    iy0 = np.clip(iy, 0, max(0, ny - 2))
    ix1 = np.clip(ix0 + 1, 0, nx - 1)
    iy1 = np.clip(iy0 + 1, 0, ny - 1)

    g00 = grid[iy0, ix0]
    g10 = grid[iy0, ix1]
    g01 = grid[iy1, ix0]
    g11 = grid[iy1, ix1]

    nn = grid[np.clip(iy, 0, ny - 1), np.clip(ix, 0, nx - 1)]

    ok = np.isfinite(g00) & np.isfinite(g10) & np.isfinite(g01) & np.isfinite(g11)
    out = np.where(
        ok,
        (1.0 - fx) * (1.0 - fy) * g00
        + fx * (1.0 - fy) * g10
        + (1.0 - fx) * fy * g01
        + fx * fy * g11,
        nn,
    )
    return out


def _select_surface_snap_points(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    provisional: np.ndarray,
    x0: float,
    y0: float,
    cfg: InvertVoteConfig,
):
    """
    For each base-grid support location, choose the closest competitor to the
    provisional surface from the nearest point above and the nearest point below.
    No tolerance band.
    """
    cell = float(cfg.cell)
    x0b, y0b, ix, iy = _grid_index(x, y, cell, x0=x0, y0=y0)
    assert x0b == x0 and y0b == y0

    ny, nx = provisional.shape
    r = int(max(0, cfg.support_snap_radius_cells))
    chosen = np.full((ny, nx), np.nan, dtype=np.float32)

    # Published default is radius=0.  Handle that common case without Python
    # point-bin lists: choose the observed point closest to provisional support
    # in each occupied base cell.
    if r == 0:
        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        if np.any(valid):
            ixv = ix[valid]; iyv = iy[valid]; zv = z[valid]
            lin = iyv.astype(np.int64) * int(nx) + ixv.astype(np.int64)
            prov = provisional[iyv, ixv]
            ok = np.isfinite(prov)
            if np.any(ok):
                lin2 = lin[ok]; z2 = zv[ok]; d2 = np.abs(z2 - prov[ok])
                order = np.lexsort((d2, lin2))
                ls = lin2[order]
                first = np.r_[True, ls[1:] != ls[:-1]]
                chosen.flat[ls[first]] = z2[order[first]].astype(np.float32, copy=False)
        return chosen

    bins = _build_point_bins(ix, iy, nx, ny)

    for j in range(ny):
        for i in range(nx):
            zs = provisional[j, i]
            if not np.isfinite(zs):
                continue

            j0 = max(0, j - r)
            j1 = min(ny, j + r + 1)
            i0 = max(0, i - r)
            i1 = min(nx, i + r + 1)

            best_above = np.nan
            best_above_abs = np.inf
            best_below = np.nan
            best_below_abs = np.inf

            for jj in range(j0, j1):
                base = jj * nx
                for ii in range(i0, i1):
                    pts = bins[base + ii]
                    if not pts:
                        continue
                    vals = z[np.asarray(pts, dtype=np.int32)]

                    above = vals[vals >= zs]
                    if above.size:
                        idx = int(np.argmin(above - zs))
                        cand = float(above[idx])
                        d = abs(cand - zs)
                        if d < best_above_abs:
                            best_above_abs = d
                            best_above = cand

                    below = vals[vals < zs]
                    if below.size:
                        idx = int(np.argmin(zs - below))
                        cand = float(below[idx])
                        d = abs(cand - zs)
                        if d < best_below_abs:
                            best_below_abs = d
                            best_below = cand

            if np.isfinite(best_above) and np.isfinite(best_below):
                chosen[j, i] = np.float32(best_above if best_above_abs <= best_below_abs else best_below)
            elif np.isfinite(best_below):
                chosen[j, i] = np.float32(best_below)
            elif np.isfinite(best_above):
                chosen[j, i] = np.float32(best_above)

    return chosen


def _robust_quadratic_fit(dx, dy, zz, good0, *, mad_floor: float, max_rz: float):
    """Compatibility helper retained for the 0.2.1 API; fast backbone does not call it."""
    if np.count_nonzero(good0) < 10:
        return None
    A = np.column_stack((dx, dy, dx*dx, dx*dy, dy*dy, np.ones_like(dx)))
    try:
        beta, *_ = np.linalg.lstsq(A[good0], zz[good0], rcond=None)
    except np.linalg.LinAlgError:
        return None
    pred = A @ beta
    rr = zz - pred
    med = float(np.median(rr[good0]))
    mad = max(float(np.median(np.abs(rr[good0] - med))), float(mad_floor))
    scale = max(1.4826 * mad, float(mad_floor))
    good = np.abs(rr - med) <= float(max_rz) * scale
    return beta, pred, rr, med, scale, good


def _local_plane_vote(snapped, j, i, *, cell, radius_max, min_nei, max_rz, mad_floor):
    """Compatibility local-plane diagnostic; not used by the fast production path."""
    ny, nx = snapped.shape
    for r in range(1, int(max(1, radius_max)) + 1):
        j0=max(0,j-r); j1=min(ny,j+r+1); i0=max(0,i-r); i1=min(nx,i+r+1)
        sub=snapped[j0:j1,i0:i1]
        yy,xx=np.nonzero(np.isfinite(sub))
        if yy.size < min_nei:
            continue
        zz=sub[yy,xx].astype(np.float64,copy=False)
        dx=(xx+i0-i).astype(np.float64)*float(cell); dy=(yy+j0-j).astype(np.float64)*float(cell)
        A=np.column_stack((dx,dy,np.ones_like(dx)))
        try: beta,*_=np.linalg.lstsq(A,zz,rcond=None)
        except np.linalg.LinAlgError: continue
        pred=A@beta; rr=zz-pred
        med=float(np.median(rr)); mad=max(float(np.median(np.abs(rr-med))),float(mad_floor))
        good=np.abs(rr-med) <= float(max_rz)*1.4826*mad
        if np.count_nonzero(good) >= min_nei:
            return float(beta[2]), float(mad), int(np.count_nonzero(good))
    return None


def _normal_vote_surface(snapped: np.ndarray, cfg: InvertVoteConfig) -> np.ndarray:
    """Published robust scalar vote, exposed as the stable 0.2.1 helper API."""
    snapped=np.asarray(snapped,dtype=np.float32)
    ny,nx=snapped.shape
    grid=np.full((ny,nx),np.nan,dtype=np.float32)
    r=int(max(1,cfg.neighbor_radius_cells)); min_nei=int(max(1,cfg.min_neighbor_cells))
    max_rz=float(cfg.max_robust_z); mad_floor=float(cfg.mad_floor)
    for j in range(ny):
        j0=max(0,j-r); j1=min(ny,j+r+1)
        for i in range(nx):
            i0=max(0,i-r); i1=min(nx,i+r+1)
            vals=snapped[j0:j1,i0:i1]; vals=vals[np.isfinite(vals)]
            if vals.size < min_nei: continue
            med=float(np.median(vals)); mad=max(float(np.median(np.abs(vals-med))),mad_floor)
            good=vals[np.abs(vals-med) <= max_rz*1.4826*mad]
            if good.size >= min_nei: grid[j,i]=np.float32(np.median(good))
    return grid


def build_surface_invert_vote(x: np.ndarray, y: np.ndarray, z: np.ndarray, cfg: InvertVoteConfig):
    """
    Global support-layer version:
      1) explicit inversion + scalar provisional support surface
      2) optional half-cell offset swipe
      3) single above/below competition snap per support location
      4) robust vote on the scalar snapped surface
      5) hole-fill + nan-safe smoothing

    Returns: (surf_z_grid, x0, y0)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    if x.size == 0:
        return np.empty((0, 0), dtype=np.float32), 0.0, 0.0

    cell = float(cfg.cell)
    x0 = float(np.min(x))
    y0 = float(np.min(y))
    ix = np.floor((x - x0) / cell).astype(np.int32)
    iy = np.floor((y - y0) / cell).astype(np.int32)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1

    surf0 = _initial_surface_from_swipe(x, y, z, cell, x0, y0, nx, ny, 0.0, 0.0)
    provisional = surf0

    if bool(getattr(cfg, "use_offset_swipe", True)):
        surf1 = _initial_surface_from_swipe(
            x, y, z, cell, x0, y0, nx, ny, 0.5 * cell, 0.5 * cell
        )
        provisional = _merge_surfaces_lower(surf0, surf1)

    snapped = _select_surface_snap_points(x, y, z, provisional, x0, y0, cfg)

    # Published robust scalar vote; the newer multi-radius plane/quadratic
    # Alternate implementations are not used by the production path.
    grid = _normal_vote_surface(snapped, cfg)

    # hole fill -- exact former semantics, but vectorized over the grid.
    for _ in range(int(max(0, cfg.fill_iters))):
        if np.isfinite(grid).all():
            break
        g2=_fill_nan_median_step(grid)
        # If no frontier cell could be filled, stop rather than repeating the
        # remaining configured iterations over an unchanged grid.
        if np.array_equal(np.isfinite(g2),np.isfinite(grid)):
            grid=g2
            break
        grid=g2

    # nan-safe smoothing (separable)
    sigma = float(max(0.01, cfg.smooth_sigma_cells))
    rad = int(np.ceil(3.0 * sigma))
    xs = np.arange(-rad, rad + 1, dtype=np.float32)
    k = np.exp(-(xs * xs) / (2.0 * sigma * sigma)).astype(np.float32)
    k /= float(k.sum())

    grid = _conv1d_nan(grid, k, axis=0)
    grid = _conv1d_nan(grid, k, axis=1)

    return grid, x0, y0


def sample_surface(surf_z: np.ndarray, x0: float, y0: float, cell: float, x: float, y: float) -> float:
    out = _bilinear_sample(
        surf_z,
        np.array([x], dtype=np.float64),
        np.array([y], dtype=np.float64),
        x0,
        y0,
        float(cell),
    )
    return float(out[0])


def sample_slope(surf_z: np.ndarray, x0: float, y0: float, cell: float, x: float, y: float) -> float:
    if surf_z.size == 0:
        return float("nan")
    gy, gx = np.gradient(surf_z, float(cell), float(cell))
    slope_mag = np.sqrt(gx * gx + gy * gy)
    out = _bilinear_sample(
        slope_mag,
        np.array([x], dtype=np.float64),
        np.array([y], dtype=np.float64),
        x0,
        y0,
        float(cell),
    )
    return float(out[0])


def classify_by_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    surf_z: np.ndarray,
    x0: float,
    y0: float,
    cfg: InvertVoteConfig,
) -> np.ndarray:
    """Fast ALS/ULS membership with breakline-aware tolerance.

    The published slope-adaptive gate remains the backbone.  A small additional
    allowance is activated only where the already-derived terrain surface shows
    local breakline relief (crest/concavity/slope transition).  This avoids a
    global threshold increase and therefore leaves smooth terrain behavior
    unchanged.
    """
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64); z=np.asarray(z,dtype=np.float64)
    cell=float(cfg.cell); thr0=float(cfg.ground_threshold); slope_k=float(cfg.slope_adapt_k)
    zhat=_bilinear_sample(surf_z,x,y,x0,y0,cell)

    # Fill only for derivative diagnostics; classification still requires a
    # finite original interpolated surface.
    sf=np.asarray(surf_z,dtype=np.float32)
    med0=float(np.nanmedian(sf)) if np.any(np.isfinite(sf)) else 0.0
    dense=np.where(np.isfinite(sf),sf,med0).astype(np.float32,copy=False)
    gy,gx=np.gradient(dense,cell,cell)
    slope_mag=np.sqrt(gx*gx+gy*gy)
    slope_here=_bilinear_sample(slope_mag,x,y,x0,y0,cell)

    # Local breakline relief is measured in metres, not a unitless curvature.
    # On planar/constant slopes this is nearly zero. At convex/concave bends it
    # rises, providing a bounded local recovery allowance.
    local_med=median_filter(dense,size=3,mode='nearest')
    relief=np.abs(dense-local_med)
    relief_here=_bilinear_sample(relief,x,y,x0,y0,cell)
    transition_extra=np.clip(1.5*relief_here,0.0,0.25)

    thr=thr0 + slope_k*slope_here + transition_extra
    thr=np.clip(thr,0.05,0.65)
    resid=z-zhat
    return np.isfinite(zhat) & (resid <= thr) & (resid >= -2.0*thr)

