from __future__ import annotations



from dataclasses import dataclass

import math

import numpy as np

from scipy.ndimage import label, binary_dilation, median_filter

from scipy.spatial import cKDTree





@dataclass(frozen=True)

class SurfaceConsensusConfig:

    cell_m: float

    recover_min_points: int

    recover_weak_ground_ratio: float

    recover_density_fraction: float

    recover_radius_m: float

    recover_min_support: int

    recover_min_sectors: int

    recover_normal_tol_m: float

    recover_low_band_m: float

    demote_support_inner_m: float

    demote_support_outer_m: float

    demote_min_support: int

    demote_min_sectors: int

    demote_seed_residual_m: float

    demote_grow_residual_m: float

    demote_same_surface_band_m: float

    demote_min_nonground_fraction: float

    demote_strong_nonground_fraction: float

    demote_max_area_m2: float

    demote_max_component_rmse_m: float





@dataclass

class SurfaceConsensusWorkspace:

    """Reusable XY cell index for final terrain QC.



    This object is deliberately *classification neutral*: it caches only static

    geometry (cell membership, point order, counts, low-return quantiles). Ground

    counts and ground medians are recomputed from the current mask each pass.

    Reusing it avoids repeated O(N log N) sorts and repeated O(N) whole-cloud

    scans without changing any acceptance/demotion rule.

    """

    x0: float

    y0: float

    cell_m: float

    nx: int

    ny: int

    ix: np.ndarray

    iy: np.ndarray

    key: np.ndarray

    count: np.ndarray

    order: np.ndarray

    cell_start: np.ndarray

    cell_end: np.ndarray

    occupied_keys: np.ndarray

    low: np.ndarray



    @property

    def n_cells(self) -> int:

        return int(self.nx * self.ny)



    def point_indices(self, linear_key: int) -> np.ndarray:

        k=int(linear_key)

        if k < 0 or k >= self.n_cells:

            return np.empty(0, dtype=np.int64)

        a=int(self.cell_start[k])

        if a < 0:

            return np.empty(0, dtype=np.int64)

        b=int(self.cell_end[k])

        return self.order[a:b]



    def ground_stats(self, z: np.ndarray, g: np.ndarray):

        g=np.asarray(g, dtype=bool)

        z=np.asarray(z, dtype=float)

        gcount=np.bincount(self.key[g], minlength=self.n_cells).reshape(self.ny,self.nx)

        gz=np.full(self.n_cells, np.nan, dtype=float)

        # Exact same per-cell median as the pre-optimization implementation,

        # but reuse the pre-sorted point membership instead of sorting again.

        for k in self.occupied_keys:

            idx=self.point_indices(int(k))

            gb=z[idx[g[idx]]]

            if gb.size:

                gz[int(k)]=float(np.median(gb))

        return gcount, gz.reshape(self.ny,self.nx)





def build_surface_consensus_workspace(x, y, z, cell_m: float) -> SurfaceConsensusWorkspace:

    x=np.asarray(x, dtype=float); y=np.asarray(y, dtype=float); z=np.asarray(z, dtype=float)

    x0=float(np.min(x)); y0=float(np.min(y)); cell=float(cell_m)

    ix=np.floor((x-x0)/cell).astype(np.int32)

    iy=np.floor((y-y0)/cell).astype(np.int32)

    nx=int(ix.max())+1; ny=int(iy.max())+1; n=int(nx*ny)

    key=iy.astype(np.int64)*nx+ix.astype(np.int64)

    count=np.bincount(key,minlength=n).reshape(ny,nx)

    order=np.argsort(key,kind='mergesort')

    ks=key[order]

    if len(ks):

        starts=np.r_[0,1+np.flatnonzero(ks[1:]!=ks[:-1])]

        ends=np.r_[starts[1:],len(ks)]

        occupied=ks[starts].astype(np.int64,copy=False)

    else:

        starts=np.empty(0,dtype=np.int64); ends=np.empty(0,dtype=np.int64); occupied=np.empty(0,dtype=np.int64)

    cell_start=np.full(n,-1,dtype=np.int64); cell_end=np.full(n,-1,dtype=np.int64)

    if occupied.size:

        cell_start[occupied]=starts; cell_end[occupied]=ends

    low=np.full(n,np.nan,dtype=float)

    for k,a,b in zip(occupied,starts,ends):

        low[int(k)]=float(np.quantile(z[order[int(a):int(b)]],0.12))

    return SurfaceConsensusWorkspace(

        x0=x0,y0=y0,cell_m=cell,nx=nx,ny=ny,ix=ix,iy=iy,key=key,

        count=count,order=order,cell_start=cell_start,cell_end=cell_end,

        occupied_keys=occupied,low=low.reshape(ny,nx),

    )







def _config(sensor_mode: str, cfg: dict | None = None) -> SurfaceConsensusConfig:

    cfg = cfg or {}

    sm = str(sensor_mode).upper().strip()

    if sm == "ULS":

        d = dict(cell_m=0.60, recover_min_points=6, recover_weak_ground_ratio=0.22,

                 recover_density_fraction=0.30, recover_radius_m=5.0, recover_min_support=7,

                 recover_min_sectors=3, recover_normal_tol_m=0.14, recover_low_band_m=0.10,

                 demote_support_inner_m=1.5, demote_support_outer_m=7.0,

                 demote_min_support=12, demote_min_sectors=5,

                 demote_seed_residual_m=0.38, demote_grow_residual_m=0.24,

                 demote_same_surface_band_m=0.45, demote_min_nonground_fraction=0.24,

                 demote_strong_nonground_fraction=0.52, demote_max_area_m2=700.0,

                 demote_max_component_rmse_m=0.28)

    else:

        d = dict(cell_m=1.00, recover_min_points=4, recover_weak_ground_ratio=0.18,

                 recover_density_fraction=0.25, recover_radius_m=7.0, recover_min_support=6,

                 recover_min_sectors=3, recover_normal_tol_m=0.22, recover_low_band_m=0.16,

                 demote_support_inner_m=2.0, demote_support_outer_m=9.0,

                 demote_min_support=10, demote_min_sectors=4,

                 demote_seed_residual_m=0.52, demote_grow_residual_m=0.30,

                 demote_same_surface_band_m=0.60, demote_min_nonground_fraction=0.20,

                 demote_strong_nonground_fraction=0.48, demote_max_area_m2=900.0,

                 demote_max_component_rmse_m=0.35)

    for k in list(d):

        key = f"surface_consensus_{k}"

        if key in cfg:

            d[k] = type(d[k])(cfg[key])

    return SurfaceConsensusConfig(**d)





def _sector_count(dx: np.ndarray, dy: np.ndarray) -> int:

    if dx.size == 0:

        return 0

    ang = (np.arctan2(dy, dx) + 2*np.pi) % (2*np.pi)

    return int(np.unique(np.floor(ang/(np.pi/4)).astype(np.int8)).size)





def _robust_plane_python_reference(x: np.ndarray, y: np.ndarray, z: np.ndarray, *, asymmetric_high: bool=False):

    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    if keep.sum() < 3:

        return None

    xc = float(np.median(x[keep])); yc = float(np.median(y[keep]))

    xx=x-xc; yy=y-yc

    coef=None

    for _ in range(5):

        if keep.sum() < 3: return None

        A=np.column_stack((xx[keep],yy[keep],np.ones(keep.sum())))

        try: coef,*_=np.linalg.lstsq(A,z[keep],rcond=None)

        except np.linalg.LinAlgError: return None

        r=z-(coef[0]*xx+coef[1]*yy+coef[2])

        med=float(np.median(r[keep])); mad=1.4826*float(np.median(np.abs(r[keep]-med)))

        sig=max(mad,0.025)

        if asymmetric_high:

            # terrain baseline: reject elevated roofs/canopy more strongly than low terrain

            nk=keep & (r >= med-4.0*sig) & (r <= med+2.25*sig)

        else:

            nk=keep & (np.abs(r-med) <= 3.25*sig)

        if nk.sum()<3 or np.array_equal(nk,keep): break

        keep=nk

    if coef is None: return None

    pred=coef[0]*xx[keep]+coef[1]*yy[keep]+coef[2]

    rmse=float(np.sqrt(np.mean((z[keep]-pred)**2)))

    return float(coef[0]),float(coef[1]),float(coef[2]),xc,yc,rmse


def _robust_plane_native_hybrid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    asymmetric_high: bool=False,
):
    from .backend.native import surface_plane_iteration_native

    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if keep.sum() < 3:
        return None

    xc = float(np.median(x[keep]))
    yc = float(np.median(y[keep]))
    xx = x - xc
    yy = y - yc
    coef = None

    for _ in range(5):
        if keep.sum() < 3:
            return None

        A = np.column_stack(
            (xx[keep], yy[keep], np.ones(keep.sum()))
        )

        try:
            coef, *_ = np.linalg.lstsq(
                A,
                z[keep],
                rcond=None,
            )
        except np.linalg.LinAlgError:
            return None

        nk = surface_plane_iteration_native(
            xx,
            yy,
            z,
            keep,
            coef,
            asymmetric_high,
        )

        if nk.sum() < 3 or np.array_equal(nk, keep):
            break

        keep = nk

    if coef is None:
        return None

    pred = (
        coef[0] * xx[keep]
        + coef[1] * yy[keep]
        + coef[2]
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (z[keep] - pred) ** 2
            )
        )
    )

    return (
        float(coef[0]),
        float(coef[1]),
        float(coef[2]),
        xc,
        yc,
        rmse,
    )


def _robust_plane(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    asymmetric_high: bool=False,
):
    import os

    backend = os.environ.get(
        "FASTGC_SURFACE_PLANE_BACKEND",
        "reference",
    ).strip().lower()

    if backend == "native":
        try:
            return _robust_plane_native_hybrid(
                x,
                y,
                z,
                asymmetric_high=asymmetric_high,
            )
        except Exception:
            return _robust_plane_python_reference(
                x,
                y,
                z,
                asymmetric_high=asymmetric_high,
            )

    return _robust_plane_python_reference(
        x,
        y,
        z,
        asymmetric_high=asymmetric_high,
    )





def _robust_quadratic(x: np.ndarray, y: np.ndarray, z: np.ndarray, *, asymmetric_high: bool=False):

    keep=np.isfinite(x)&np.isfinite(y)&np.isfinite(z)

    if keep.sum()<10: return None

    xc=float(np.median(x[keep])); yc=float(np.median(y[keep])); xx=x-xc; yy=y-yc; coef=None

    for _ in range(5):

        if keep.sum()<10: return None

        A=np.column_stack((np.ones(keep.sum()),xx[keep],yy[keep],xx[keep]**2,xx[keep]*yy[keep],yy[keep]**2))

        try: coef,*_=np.linalg.lstsq(A,z[keep],rcond=None)

        except np.linalg.LinAlgError: return None

        pred=coef[0]+coef[1]*xx+coef[2]*yy+coef[3]*xx**2+coef[4]*xx*yy+coef[5]*yy**2

        r=z-pred; med=float(np.median(r[keep])); mad=1.4826*float(np.median(np.abs(r[keep]-med))); sig=max(mad,0.025)

        if asymmetric_high: nk=keep&(r>=med-4.0*sig)&(r<=med+2.25*sig)

        else: nk=keep&(np.abs(r-med)<=3.25*sig)

        if nk.sum()<10 or np.array_equal(nk,keep): break

        keep=nk

    if coef is None:return None

    pred=coef[0]+coef[1]*xx[keep]+coef[2]*yy[keep]+coef[3]*xx[keep]**2+coef[4]*xx[keep]*yy[keep]+coef[5]*yy[keep]**2

    rmse=float(np.sqrt(np.mean((z[keep]-pred)**2)))

    return np.asarray(coef,float),xc,yc,rmse



def _quad_predict(model, x, y):

    coef,xc,yc,_=model; xx=np.asarray(x,float)-xc; yy=np.asarray(y,float)-yc

    pred=coef[0]+coef[1]*xx+coef[2]*yy+coef[3]*xx**2+coef[4]*xx*yy+coef[5]*yy**2

    gx=coef[1]+2*coef[3]*xx+coef[4]*yy; gy=coef[2]+coef[4]*xx+2*coef[5]*yy

    return pred,np.sqrt(1.0+gx*gx+gy*gy)



def _cell_stats(x,y,z,g,cell,workspace: SurfaceConsensusWorkspace | None = None):

    ws=workspace if workspace is not None else build_surface_consensus_workspace(x,y,z,cell)

    # Defensive rebuild only if a caller supplied a workspace at another scale.

    if abs(float(ws.cell_m)-float(cell)) > 1e-12 or len(ws.key) != len(x):

        ws=build_surface_consensus_workspace(x,y,z,cell)

    gcount,gz=ws.ground_stats(z,g)

    return ws.x0,ws.y0,ws.ix,ws.iy,ws.key,ws.nx,ws.ny,ws.count,gcount,gz,ws.low





def recover_surface_false_negatives(*,x,y,z,ground_mask,sensor_mode,cfg=None,workspace: SurfaceConsensusWorkspace | None = None):

    """Multi-pass terrain-first recovery of observed low-layer false negatives.



    The routine never interpolates synthetic points.  It uses current class-2

    cells as trusted terrain anchors, fits a robust local plane/quadratic

    manifold, and promotes only observed low-layer returns whose orthogonal

    residual is small.  Repeating the conservative pass allows support to move

    inward through a large systematic void while retaining the same local tests

    at every step.

    """

    sm=str(sensor_mode).upper().strip(); g=np.asarray(ground_mask,bool).copy()

    rep={'enabled':sm in {'ALS','ULS'},'passes_run':0,'candidate_cells':0,

         'validated_cells':0,'recovered_points':0}

    if sm not in {'ALS','ULS'} or len(x)<32 or g.sum()<12: return g,rep

    cfg=cfg or {}; c=_config(sm,cfg)

    x=np.asarray(x,float); y=np.asarray(y,float); z=np.asarray(z,float)

    default_passes=4 if sm=='ALS' else 3

    max_passes=max(1,min(6,int(cfg.get('terrain_final_recovery_passes',default_passes))))



    ws=workspace if workspace is not None else build_surface_consensus_workspace(x,y,z,c.cell_m)

    if abs(float(ws.cell_m)-float(c.cell_m)) > 1e-12 or len(ws.key) != len(x):

        ws=build_surface_consensus_workspace(x,y,z,c.cell_m)

    x0,y0,ix,iy,key,nx,ny,count=ws.x0,ws.y0,ws.ix,ws.iy,ws.key,ws.nx,ws.ny,ws.count

    local_density=median_filter(count.astype(float),size=5,mode='nearest')



    recovered_all=[]

    for _pass in range(max_passes):

        rep['passes_run'] += 1

        gcount,gz=ws.ground_stats(z,g)

        low=ws.low

        ratio=gcount/np.maximum(count,1)



        # Density-relative occupancy keeps this viable for sparse historical ALS,

        # dense modern ALS/ULS, and non-uniform scan patterns.  A zero-ground cell

        # can still be examined when actual returns are present.

        # Adaptive minimum occupancy: old sparse line-scanner ALS may have only

        # one or two returns in a 1 m cell, while modern ALS/ULS can have many.

        # Require density-relative support instead of a fixed point count.

        adaptive_min=np.clip(

            np.ceil(0.35*np.maximum(local_density,1.0)),

            1,

            c.recover_min_points,

        )

        rich=(count>=adaptive_min) & (

            count>=c.recover_density_fraction*np.maximum(local_density,1.0)

        )

        # Terrain-first recovery also repairs partial within-cell omissions;

        # a cell need not be completely empty of class-2 points.  The geometric

        # manifold test below, not the ground ratio alone, decides promotion.

        weak_ratio=max(float(c.recover_weak_ground_ratio),

                       float(cfg.get('terrain_recover_weak_ground_ratio',0.90)))

        weak=rich & (ratio<weak_ratio)

        cells=np.argwhere(weak)

        rep['candidate_cells'] += int(len(cells))

        if len(cells)==0: break



        oy,ox=np.nonzero(np.isfinite(gz))

        ifx=x0+(ox+.5)*c.cell_m; ify=y0+(oy+.5)*c.cell_m; ifz=gz[oy,ox]

        if len(ifx)<c.recover_min_support: break

        tree=cKDTree(np.c_[ifx,ify])

        recovered_this=[]

        # Batch the radius lookup in compiled SciPy code.  The returned neighbor

        # lists are identical to one-query-per-cell, but avoid thousands of

        # Python/C boundary crossings on dense tiles.

        cell_centers=np.column_stack((

            x0+(cells[:,1].astype(float)+.5)*c.cell_m,

            y0+(cells[:,0].astype(float)+.5)*c.cell_m,

        ))

        neighbor_lists=tree.query_ball_point(cell_centers,c.recover_radius_m)



        for (gy,gx), ids0 in zip(cells,neighbor_lists):

            qx=x0+(gx+.5)*c.cell_m; qy=y0+(gy+.5)*c.cell_m

            ids=np.asarray(ids0,dtype=int)

            if ids.size<c.recover_min_support: continue

            dx=ifx[ids]-qx; dy=ify[ids]-qy

            if _sector_count(dx,dy)<c.recover_min_sectors: continue



            # Prefer nearby support to avoid crossing from one valley wall/ridge

            # flank to another.  Angular-sector support prevents a dense scan line

            # from masquerading as full 2-D terrain support.

            if ids.size>32:

                rr=np.hypot(dx,dy); ids=ids[np.argsort(rr)[:32]]



            # Reject target cells with nothing eligible to recover before paying

            # for an iterative terrain-model fit.  These checks depend only on the

            # current pass state and therefore do not alter classification logic.

            pidx=ws.point_indices(int(gy*nx+gx))

            if pidx.size:

                pidx=pidx[~g[pidx]]

            if pidx.size==0:

                continue



            # Only an observed bottom cluster is eligible.  This is deliberately

            # not raw neighborhood Z-range: steep terrain can have large Z range.

            zlo=float(np.quantile(z[pidx],0.12))

            cand=pidx[z[pidx] <= zlo+c.recover_low_band_m]

            if cand.size==0:

                continue



            # Quadratic is the preferred model.  Fit the plane only as fallback.

            quad=_robust_quadratic(ifx[ids],ify[ids],ifz[ids],asymmetric_high=True)

            pl=None

            if quad is None:

                pl=_robust_plane(ifx[ids],ify[ids],ifz[ids],asymmetric_high=True)

                if pl is None:

                    continue

                a,b,k,xc,yc,_=pl



            if quad is not None:

                pred,scale=_quad_predict(quad,x[cand],y[cand])

            else:

                pred=a*(x[cand]-xc)+b*(y[cand]-yc)+k

                scale=math.sqrt(1+a*a+b*b)



            # Orthogonal/surface-normal residual: steep slopes are not punished

            # merely because vertical Z changes rapidly with XY.

            rn=(z[cand]-pred)/scale

            seed_ok=cand[np.abs(rn)<=c.recover_normal_tol_m]

            if seed_ok.size==0: continue



            # Once the observed low layer validates the cell, evaluate ALL

            # non-ground returns in that cell against the same local manifold.

            # This recovers the uphill side of a steep/ridged cell instead of

            # keeping only its lowest Z band. Vegetation above the terrain still

            # fails the orthogonal residual test.

            if quad is not None:

                pred_all,scale_all=_quad_predict(quad,x[pidx],y[pidx])

            else:

                pred_all=a*(x[pidx]-xc)+b*(y[pidx]-yc)+k

                scale_all=math.sqrt(1+a*a+b*b)

            rn_all=(z[pidx]-pred_all)/scale_all

            cand_all=pidx[np.abs(rn_all)<=c.recover_normal_tol_m]

            if cand_all.size==0: continue



            # No long-range fabrication: every promoted point must remain close

            # to at least one trusted occupied terrain cell.

            d,_=tree.query(np.c_[x[cand_all],y[cand_all]],k=1)

            cand_all=cand_all[d<=c.recover_radius_m]

            if cand_all.size==0: continue



            recovered_this.append(cand_all)

            rep['validated_cells'] += 1



        if not recovered_this: break

        idx=np.unique(np.concatenate(recovered_this))

        idx=idx[~g[idx]]

        if idx.size==0: break

        g[idx]=True

        recovered_all.append(idx)



    if recovered_all:

        rep['recovered_points']=int(np.unique(np.concatenate(recovered_all)).size)

    return g,rep



def demote_detached_ground_consensus(*,x,y,z,ground_mask,context_x,context_y,context_z,sensor_mode,cfg=None,workspace: SurfaceConsensusWorkspace | None = None):

    """Demote localized elevated ground only when terrain residual AND non-ground consensus agree.



    Intended for residual roofs/buildings and canopy butterflies. A steep natural

    slope remains attached to the surrounding terrain plane and therefore has a

    small normal residual. The function never promotes or semantically labels

    objects; it only changes false class-2 candidates back to non-ground.

    """

    sm=str(sensor_mode).upper().strip(); g=np.asarray(ground_mask,bool).copy()

    rep={'enabled':sm in {'ALS','ULS'},'candidate_components':0,'validated_components':0,'demoted_points':0}

    if sm not in {'ALS','ULS'} or len(x)<32 or g.sum()<12: return g,rep

    c=_config(sm,cfg); x=np.asarray(x,float); y=np.asarray(y,float); z=np.asarray(z,float)

    cx=np.asarray(context_x,float); cy=np.asarray(context_y,float); cz=np.asarray(context_z,float)

    ws=workspace if workspace is not None else build_surface_consensus_workspace(x,y,z,c.cell_m)

    if abs(float(ws.cell_m)-float(c.cell_m)) > 1e-12 or len(ws.key) != len(x):

        ws=build_surface_consensus_workspace(x,y,z,c.cell_m)

    x0,y0,ix,iy,key,nx,ny,count,gcount,gz,low=_cell_stats(x,y,z,g,c.cell_m,workspace=ws)

    oy,ox=np.nonzero(np.isfinite(gz)); qx=x0+(ox+.5)*c.cell_m; qy=y0+(oy+.5)*c.cell_m; qz=gz[oy,ox]

    if len(qx)<c.demote_min_support: return g,rep

    tree=cKDTree(np.c_[qx,qy]); residual=np.full((ny,nx),np.nan)

    for gy,gx,px,py in zip(oy,ox,qx,qy):

        ids=np.asarray(tree.query_ball_point([px,py],c.demote_support_outer_m),int)

        if ids.size<c.demote_min_support: continue

        rr=np.hypot(qx[ids]-px,qy[ids]-py); ids=ids[(rr>=c.demote_support_inner_m)]

        if ids.size<c.demote_min_support: continue

        if _sector_count(qx[ids]-px,qy[ids]-py)<c.demote_min_sectors: continue

        if ids.size>36:

            rr=np.hypot(qx[ids]-px,qy[ids]-py); ids=ids[np.argsort(rr)[:36]]

        pl=_robust_plane(qx[ids],qy[ids],qz[ids],asymmetric_high=True)

        if pl is None: continue

        quad=_robust_quadratic(qx[ids],qy[ids],qz[ids],asymmetric_high=True)

        if quad is not None:

            pred,scale=_quad_predict(quad,px,py)

            residual[gy,gx]=(gz[gy,gx]-float(pred))/float(scale)

        else:

            a,b,k,xc,yc,_=pl; scale=math.sqrt(1+a*a+b*b)

            residual[gy,gx]=(gz[gy,gx]-(a*(px-xc)+b*(py-yc)+k))/scale

    seed=np.isfinite(residual)&(residual>=c.demote_seed_residual_m)

    # Grow only across cells that remain positively detached from terrain.

    grow=np.isfinite(residual)&(residual>=c.demote_grow_residual_m)

    labs,n=label(grow,np.ones((3,3),dtype=np.uint8)); rep['candidate_components']=int(n)

    if n==0:return g,rep



    # Context non-ground corroboration indexed in the same XY grid.

    cix=np.floor((cx-x0)/c.cell_m).astype(int); ciy=np.floor((cy-y0)/c.cell_m).astype(int)

    inside=(cix>=0)&(cix<nx)&(ciy>=0)&(ciy<ny)

    cix=cix[inside]; ciy=ciy[inside]; cz=cz[inside]

    # Ground work points are part of context too; estimate non-ground by excluding

    # points that exactly belong to current ground positions is expensive. Instead

    # corroborate with context returns near the candidate surface and require that

    # their local vertical distribution contains substantial support beyond the

    # candidate ground sample. This works because most building/canopy returns are

    # already non-ground in the input classification stage.

    # Build per-cell context z lists once.

    ckey=ciy.astype(np.int64)*nx+cix.astype(np.int64); order=np.argsort(ckey,kind='mergesort'); cks=ckey[order]; czs=cz[order]

    starts=np.r_[0,1+np.flatnonzero(cks[1:]!=cks[:-1])] if len(cks) else np.array([],int); ends=np.r_[starts[1:],len(cks)] if len(starts) else np.array([],int)

    slices={int(cks[a]):czs[a:b] for a,b in zip(starts,ends)}



    demote_points=np.zeros(len(x),dtype=bool)

    for lid in range(1,n+1):

        comp=(labs==lid)

        if not np.any(comp & seed): continue

        yy,xx=np.nonzero(comp); area=len(xx)*c.cell_m*c.cell_m

        if area>c.demote_max_area_m2: continue

        # Ignore components touching the working-grid boundary; overlap/seam logic

        # owns those areas and context accounting is incomplete there.

        if np.any(xx <= 1) or np.any(xx >= nx-2) or np.any(yy <= 1) or np.any(yy >= ny-2): continue

        rvals=residual[yy,xx]; medr=float(np.nanmedian(rvals))

        if medr<c.demote_grow_residual_m: continue

        # Candidate surface itself should be coherent enough to be an object sheet.

        if len(xx)>=3:

            sx=x0+(xx+.5)*c.cell_m; sy=y0+(yy+.5)*c.cell_m; sz=gz[yy,xx]

            cp=_robust_plane(sx,sy,sz,asymmetric_high=False)

            crmse=cp[-1] if cp is not None else 999.0

        else: crmse=0.0



        same_ctx=0; ground_samples=0; total_ctx=0

        for gy,gx in zip(yy,xx):

            vals=slices.get(int(gy*nx+gx))

            if vals is None: continue

            total_ctx+=len(vals); surf=gz[gy,gx]

            same_ctx+=int(np.count_nonzero(np.abs(vals-surf)<=c.demote_same_surface_band_m))

            ground_samples+=int(gcount[gy,gx])

        if total_ctx==0: continue

        # Approximate non-ground same-surface returns by subtracting current ground

        # samples from all same-surface context returns.

        ng_same=max(0,same_ctx-ground_samples)

        frac=ng_same/max(1,ng_same+ground_samples)

        strong=(frac>=c.demote_strong_nonground_fraction)

        coherent=(crmse<=c.demote_max_component_rmse_m)

        if frac<c.demote_min_nonground_fraction: continue

        very_high = medr >= max(0.90, 1.75*c.demote_seed_residual_m)

        if not (coherent or strong or very_high): continue

        # Final point-level decision against a terrain plane fitted outside the

        # candidate component. This preserves genuine ground returns that may

        # coexist in the same XY cells below a roof/canopy sheet.

        ccx=float(np.mean(x0+(xx+.5)*c.cell_m)); ccy=float(np.mean(y0+(yy+.5)*c.cell_m))

        rad=max(c.demote_support_inner_m, 0.5*math.hypot((xx.max()-xx.min()+1)*c.cell_m,(yy.max()-yy.min()+1)*c.cell_m))

        ids=np.asarray(tree.query_ball_point([ccx,ccy], max(c.demote_support_outer_m,rad+3.0)),int)

        if ids.size < c.demote_min_support: continue

        cellx=np.floor((qx[ids]-x0)/c.cell_m).astype(int); celly=np.floor((qy[ids]-y0)/c.cell_m).astype(int)

        outside=~comp[np.clip(celly,0,ny-1),np.clip(cellx,0,nx-1)]

        ids=ids[outside]

        if ids.size < c.demote_min_support: continue

        pl=_robust_plane(qx[ids],qy[ids],qz[ids],asymmetric_high=True)

        if pl is None: continue

        comp_keys=(yy.astype(np.int64)*nx+xx.astype(np.int64))

        chunks=[ws.point_indices(int(kk)) for kk in comp_keys]

        pidx=np.concatenate([v for v in chunks if v.size]) if any(v.size for v in chunks) else np.empty(0,dtype=np.int64)

        if pidx.size:

            pidx=pidx[g[pidx]]

        if pidx.size==0: continue

        quad=_robust_quadratic(qx[ids],qy[ids],qz[ids],asymmetric_high=True)

        if quad is not None:

            pred,scale=_quad_predict(quad,x[pidx],y[pidx])

            rn=(z[pidx]-pred)/scale

        else:

            a,b,k,xc,yc,_=pl; scale=math.sqrt(1+a*a+b*b)

            rn=(z[pidx]-(a*(x[pidx]-xc)+b*(y[pidx]-yc)+k))/scale

        # Stronger point threshold than component grow threshold.

        bad=pidx[rn >= c.demote_grow_residual_m]

        if bad.size==0: continue

        demote_points[bad]=True; rep['validated_components']+=1



    dem=g & demote_points

    if np.any(dem): g[dem]=False; rep['demoted_points']=int(dem.sum())

    return g,rep



def demote_flying_ground_outliers(*, x, y, z, ground_mask, sensor_mode, cfg=None, workspace: SurfaceConsensusWorkspace | None = None):

    """Extremely conservative cleanup of isolated flying class-2 fragments.



    Terrain preservation has priority over object removal.  Buildings, roofs,

    ridge lines and other coherent surfaces are intentionally protected.  A cell

    can be demoted only if all of the following agree:

      * it is a very large positive orthogonal residual from a robust curved

        terrain model;

      * the surrounding detrended residual distribution is thin (robust range

        and MAD are small);

      * the supporting neighborhood is a thin PCA surface;

      * same-surface support is essentially absent; and

      * the suspicious cells form only a tiny isolated component.



    Raw Z range is not used because it scales directly with slope.

    """

    sm=str(sensor_mode).upper().strip(); g=np.asarray(ground_mask,bool).copy()

    rep={'enabled':sm in {'ALS','ULS'},'candidate_cells':0,

         'candidate_components':0,'demoted_points':0}

    if sm not in {'ALS','ULS'} or len(x)<32 or g.sum()<12: return g,rep

    cfg=cfg or {}

    if not bool(cfg.get('terrain_flying_guard_enabled', True)):

        rep['enabled']=False; return g,rep



    c=_config(sm,cfg); x=np.asarray(x,float); y=np.asarray(y,float); z=np.asarray(z,float)

    ws=workspace if workspace is not None else build_surface_consensus_workspace(x,y,z,c.cell_m)

    if abs(float(ws.cell_m)-float(c.cell_m)) > 1e-12 or len(ws.key) != len(x):

        ws=build_surface_consensus_workspace(x,y,z,c.cell_m)

    x0,y0,ix,iy,key,nx,ny,count,gcount,gz,low=_cell_stats(x,y,z,g,c.cell_m,workspace=ws)

    oy,ox=np.nonzero(np.isfinite(gz))

    if len(ox)<12:return g,rep

    qx=x0+(ox+.5)*c.cell_m; qy=y0+(oy+.5)*c.cell_m; qz=gz[oy,ox]

    tree=cKDTree(np.c_[qx,qy])

    radius=max(4.0*c.cell_m, 0.65*c.demote_support_outer_m)



    candidate=np.zeros((ny,nx),dtype=bool)

    models={}

    max_surface_variation=float(cfg.get('terrain_flying_pca_variation_max',0.030))

    min_abs_residual=float(cfg.get('terrain_flying_min_normal_residual_m',0.75))

    max_component_cells=max(1,int(cfg.get('terrain_flying_max_component_cells',2)))



    for gy,gx,px,py,pzz in zip(oy,ox,qx,qy,qz):

        ids=np.asarray(tree.query_ball_point([px,py],radius),int)

        if ids.size<10: continue

        rr=np.hypot(qx[ids]-px,qy[ids]-py)

        ids=ids[rr>0.25*c.cell_m]

        if ids.size<9:continue

        if ids.size>28:

            rr=np.hypot(qx[ids]-px,qy[ids]-py); ids=ids[np.argsort(rr)[:28]]

        if _sector_count(qx[ids]-px,qy[ids]-py)<4:continue



        quad=_robust_quadratic(qx[ids],qy[ids],qz[ids],asymmetric_high=True)

        if quad is not None:

            pred,scale=_quad_predict(quad,px,py); pred=float(pred); scale=float(scale)

            npred,nscale=_quad_predict(quad,qx[ids],qy[ids]); nr=(qz[ids]-npred)/nscale

        else:

            pl=_robust_plane(qx[ids],qy[ids],qz[ids],asymmetric_high=True)

            if pl is None:continue

            a,b,k,xc,yc,_=pl

            pred=float(a*(px-xc)+b*(py-yc)+k); scale=math.sqrt(1+a*a+b*b)

            nr=(qz[ids]-(a*(qx[ids]-xc)+b*(qy[ids]-yc)+k))/scale



        rn=(float(pzz)-pred)/scale

        med=float(np.median(nr))

        mad=1.4826*float(np.median(np.abs(nr-med))); sigma=max(mad,0.04)

        q05,q95=np.quantile(nr,[0.05,0.95])

        robust_range=float(q95-q05)



        # PCA surface variation of trusted support.  Curved/rough/ambiguous

        # neighborhoods are protected by simply refusing to demote anything.

        xyz=np.column_stack((qx[ids],qy[ids],qz[ids]))

        xyz=xyz-np.mean(xyz,axis=0,keepdims=True)

        cov=(xyz.T@xyz)/max(1,xyz.shape[0]-1)

        try:

            eig=np.linalg.eigvalsh(cov)

        except np.linalg.LinAlgError:

            continue

        eig=np.maximum(eig,0.0)

        variation=float(eig[0]/max(float(np.sum(eig)),1e-12))

        if variation>max_surface_variation: continue



        # Stronger than a standard outlier gate.  q95+0.40 uses the robust local

        # residual range explicitly while the 6*MAD condition handles density/noise.

        hard=max(min_abs_residual, med+6.0*sigma, float(q95)+0.40,

                 2.5*max(robust_range,0.10))



        # Surface-support check is slope-normal, not equal-Z.  Neighbor residuals

        # near rn would indicate a coherent ridge/roof/surface and therefore veto

        # demotion.

        same=int(np.count_nonzero(np.abs(nr-rn)<=max(0.16,2.5*sigma)))

        if rn>=hard and same<=1:

            candidate[int(gy),int(gx)]=True

            models[(int(gy),int(gx))]=(pred,scale,hard)



    rep['candidate_cells']=int(np.count_nonzero(candidate))

    if not np.any(candidate):return g,rep



    labs,nlab=label(candidate,np.ones((3,3),dtype=np.uint8))

    rep['candidate_components']=int(nlab)

    dem=np.zeros(len(g),bool)

    for lid in range(1,int(nlab)+1):

        comp=(labs==lid)

        ncell=int(np.count_nonzero(comp))

        if ncell==0 or ncell>max_component_cells:

            # Coherent/extended surfaces are protected, including building roofs

            # and narrow terrain ridges.

            continue

        yy,xx=np.nonzero(comp)

        for gy,gx in zip(yy,xx):

            model=models.get((int(gy),int(gx)))

            if model is None:continue

            pred,scale,hard=model

            ids=ws.point_indices(int(gy*nx+gx))

            if ids.size:

                ids=ids[g[ids]]

            if ids.size==0:continue

            rn=(z[ids]-pred)/scale

            dem[ids[rn>=hard]]=True



    if np.any(dem):

        g[dem]=False; rep['demoted_points']=int(dem.sum())

    return g,rep







def _recover_manifold_frontier(*, x, y, z, ground_mask, original_ground_mask, sensor_mode, cfg, workspace):

    """Short, candidate-only recovery of stolen terrain along the final ground sheet.



    This is intentionally not a new classifier.  It starts from final trusted

    ground and only examines non-ground points in mixed cells or cells touching

    the trusted sheet.  A local plane handles ordinary slopes; a quadratic is

    used only when the plane leaves coherent curvature.  Every accepted point

    must fit the local surface in surface-normal distance and remain close to

    original trusted-ground anchors.  At most a few passes are allowed.

    """

    sm=str(sensor_mode).upper().strip(); cfg=cfg or {}

    g=np.asarray(ground_mask,bool).copy(); orig=np.asarray(original_ground_mask,bool)

    rep={'passes':0,'frontier_cells':0,'candidate_points':0,'promoted_points':0,'models_built':0,'quadratic_models':0}

    if sm not in {'ALS','ULS'} or g.sum()<12: return g,rep

    x=np.asarray(x,float); y=np.asarray(y,float); z=np.asarray(z,float)

    c=_config(sm,cfg)

    ws=workspace

    fit_radius=float(cfg.get('manifold_fit_radius_m',4.0 if sm=='ALS' else 3.25))

    max_radius=float(cfg.get('manifold_max_fit_radius_m',6.0 if sm=='ALS' else 5.0))

    min_support=max(6,int(cfg.get('manifold_min_support_cells',8 if sm=='ALS' else 10)))

    min_orig=max(4,int(cfg.get('manifold_min_original_support_cells',5 if sm=='ALS' else 6)))

    max_support=max(16,int(cfg.get('manifold_max_support_cells',28)))

    min_sectors=max(2,int(cfg.get('manifold_min_support_sectors',3)))

    curve_rmse=float(cfg.get('manifold_plane_curve_rmse_m',0.075 if sm=='ALS' else 0.055))

    tol0=float(cfg.get('manifold_recover_normal_tol_m',0.16 if sm=='ALS' else 0.12))

    tolmax=float(cfg.get('manifold_recover_normal_tol_max_m',0.28 if sm=='ALS' else 0.22))

    mad_k=float(cfg.get('manifold_recover_mad_k',3.0))

    zcap=float(cfg.get('manifold_recover_vertical_cap_m',0.55 if sm=='ALS' else 0.40))

    max_passes=max(1,min(3,int(cfg.get('manifold_recovery_passes',2))))

    stop_min=max(1,int(cfg.get('manifold_stop_min_new_points',4)))

    frontier_iter=max(1,min(2,int(cfg.get('manifold_frontier_dilate_cells',1))))

    orig_idx=np.flatnonzero(orig)

    if orig_idx.size<min_orig:return g,rep

    orig_tree=cKDTree(np.column_stack((x[orig_idx],y[orig_idx])))



    def support(mask):

        vals=np.full(ws.n_cells,np.nan,float); ocount=np.zeros(ws.n_cells,dtype=np.int16)

        for kk in ws.occupied_keys:

            ii=ws.point_indices(int(kk)); gi=ii[mask[ii]]

            if gi.size==0:continue

            zz=z[gi]; vals[int(kk)]=float(np.min(zz) if zz.size<=2 else np.quantile(zz,0.18))

            ocount[int(kk)]=int(np.count_nonzero(orig[ii]))

        keys=np.flatnonzero(np.isfinite(vals)); yy=(keys//ws.nx).astype(int); xx=(keys%ws.nx).astype(int)

        sx=ws.x0+(xx+.5)*c.cell_m; sy=ws.y0+(yy+.5)*c.cell_m; sz=vals[keys]

        return vals,ocount,keys,xx,yy,sx,sy,sz



    for ipass in range(max_passes):

        vals,ocount,keys,kx,ky,sx,sy,sz=support(g)

        if keys.size<min_support:break

        stree=cKDTree(np.column_stack((sx,sy))); row={int(k):i for i,k in enumerate(keys)}

        occ=np.zeros((ws.ny,ws.nx),bool); occ[ky,kx]=True

        frontier=binary_dilation(occ,iterations=frontier_iter)&(~occ)

        mixed=np.zeros_like(frontier)

        nonground_occ=np.zeros_like(frontier)

        # Only occupied cells can contribute recoverable points.  The previous

        # implementation iterated every geometrical frontier cell, including

        # large numbers of empty cells.  Mark occupied non-ground cells once

        # and intersect them with the frontier before fitting any models.

        for kk in ws.occupied_keys:

            ii=ws.point_indices(int(kk))

            if not ii.size:

                continue

            ng_here=np.any(~g[ii])

            if ng_here:

                nonground_occ[int(kk)//ws.nx,int(kk)%ws.nx]=True

            if ng_here and np.any(g[ii]):

                mixed[int(kk)//ws.nx,int(kk)%ws.nx]=True

        cells=np.argwhere((frontier & nonground_occ)|mixed); rep['frontier_cells']+=int(len(cells))

        if len(cells)==0:break

        cache={}; promoted=[]

        for gy,gx in cells:

            kk=int(gy*ws.nx+gx); ii=ws.point_indices(kk); ng=ii[~g[ii]]

            if ng.size==0:continue

            rep['candidate_points']+=int(ng.size)

            px=ws.x0+(gx+.5)*c.cell_m; py=ws.y0+(gy+.5)*c.cell_m

            ids=np.asarray(stree.query_ball_point([px,py],fit_radius),int)

            if ids.size<min_support:ids=np.asarray(stree.query_ball_point([px,py],max_radius),int)

            tr=row.get(kk)

            if tr is not None:ids=ids[ids!=tr]

            if ids.size<min_support:continue

            rr=np.hypot(sx[ids]-px,sy[ids]-py)

            if ids.size>max_support:ids=ids[np.argsort(rr)[:max_support]]

            if _sector_count(sx[ids]-px,sy[ids]-py)<min_sectors:continue

            if np.count_nonzero(ocount[keys[ids]]>0)<min_orig:continue

            pl=_robust_plane(sx[ids],sy[ids],sz[ids],asymmetric_high=True)

            if pl is None:continue

            a,b,k0,xc,yc,prmse=pl; pscale=math.sqrt(1+a*a+b*b)

            pr=(sz[ids]-(a*(sx[ids]-xc)+b*(sy[ids]-yc)+k0))/pscale

            pmad=max(1.4826*float(np.median(np.abs(pr-np.median(pr)))),0.015)

            model=('plane',pl,pmad)

            if prmse>=curve_rmse and ids.size>=10:

                q=_robust_quadratic(sx[ids],sy[ids],sz[ids],asymmetric_high=True)

                if q is not None and float(q[-1])<0.92*float(prmse):

                    qp,qs=_quad_predict(q,sx[ids],sy[ids]); qr=(sz[ids]-qp)/qs

                    qmad=max(1.4826*float(np.median(np.abs(qr-np.median(qr)))),0.015)

                    model=('quad',q,qmad); rep['quadratic_models']+=1

            rep['models_built']+=1

            if model[0]=='quad':pred,scale=_quad_predict(model[1],x[ng],y[ng])

            else:

                a,b,k0,xc,yc,_=model[1]; scale=math.sqrt(1+a*a+b*b); pred=a*(x[ng]-xc)+b*(y[ng]-yc)+k0

            rn=(z[ng]-pred)/scale; dz=z[ng]-pred

            tol=min(tolmax,max(tol0,mad_k*float(model[2])))

            good=ng[(np.abs(rn)<=tol)&(np.abs(dz)<=zcap)]

            if good.size==0:continue

            dd,_=orig_tree.query(np.column_stack((x[good],y[good])),k=1)

            good=good[dd<=max_radius]

            if good.size:promoted.append(good)

        rep['passes']=ipass+1

        if not promoted:break

        new=np.unique(np.concatenate(promoted)); new=new[~g[new]]

        if new.size<stop_min:break

        g[new]=True; rep['promoted_points']+=int(new.size)

    return g,rep



def _legacy_two_way_surface_classification_swipe(*, x, y, z, ground_mask, sensor_mode, cfg=None, workspace: SurfaceConsensusWorkspace | None = None):

    """Final slope-aware ground-sheet separation swipe.



    The current FAST-GC ground is assumed to contain a strong, coherent terrain

    sheet plus occasional detached class-2 contamination.  The detector first

    works at the ~1 m support-cell scale to find *only* anomalous ground cells:



      1. mixed cells whose class-2 vertical span is too large; or

      2. cells whose lower class-2 representative sits above the median of nearby

         lower-ground cells.



    Those fine anomalies are the point-cloud analogue of the bright dots/patches

    in a Z-range raster.  They are grouped inside a 10 m diagnostic layer, but

    raw Z range is never itself a classification rule.



    Each suspicious 1 m cell is then judged against a 30 m-diameter terrain

    context.  The target cell and other upward-anomalous support cells are

    excluded from the fit, preventing flying points from defining the surface

    used to judge themselves.  A quadratic model is used first, with plane

    fallback, and residuals are measured normal to that surface.



    High detached class-2 points can be removed from geometry alone when the

    lower terrain sheet has strong support.  Near-surface shrub/understory leaks

    require a majority non-ground vote in *surface-residual space*, which stays

    slope aware.  Non-ground -> ground promotion is disabled by default; broad

    void recovery remains the job of the earlier FAST-GC terrain-recovery stage.

    """

    sm=str(sensor_mode).upper().strip()

    g0=np.asarray(ground_mask,dtype=bool)

    g=g0.copy()

    original_ground=g0.copy()

    rep={

        'enabled':sm in {'ALS','ULS'}, 'fine_cells_flagged':0,

        'coarse_cells_flagged':0,'candidate_ground_points':0,

        'candidate_nonground_points':0,'demoted_points':0,

        'promoted_points':0,'models_built':0,

        'near_surface_demoted':0,'detached_demoted':0,

    }

    if sm not in {'ALS','ULS'} or len(x)<32 or int(g.sum())<12:

        return g,rep

    cfg=cfg or {}

    if not bool(cfg.get('two_way_surface_swipe_enabled',True)):

        rep['enabled']=False; return g,rep



    x=np.asarray(x,float); y=np.asarray(y,float); z=np.asarray(z,float)

    c=_config(sm,cfg)

    ws=workspace if workspace is not None else build_surface_consensus_workspace(x,y,z,c.cell_m)

    if abs(float(ws.cell_m)-float(c.cell_m))>1e-12 or len(ws.key)!=len(x):

        ws=build_surface_consensus_workspace(x,y,z,c.cell_m)



    coarse_cell=float(cfg.get('two_way_flag_cell_m',10.0))

    context_radius=float(cfg.get('two_way_context_radius_m',15.0))  # 30 m diameter

    fine_neighbor_k=max(6,int(cfg.get('two_way_fine_neighbor_k',16)))

    fine_low_offset=float(cfg.get('two_way_fine_low_offset_trigger_m',0.10 if sm=='ALS' else 0.08))

    fine_span_trigger=float(cfg.get('two_way_fine_ground_span_trigger_m',0.14 if sm=='ALS' else 0.11))

    local_vote_radius=float(cfg.get('two_way_vote_radius_m',1.50 if sm=='ALS' else 1.00))

    residual_band=float(cfg.get('two_way_same_residual_band_m',0.14 if sm=='ALS' else 0.11))

    min_vote_neighbors=max(2,int(cfg.get('two_way_min_same_level_neighbors',3)))

    near_ng_frac=float(cfg.get('two_way_demote_nonground_fraction',0.52))

    near_abs=float(cfg.get('two_way_demote_min_normal_residual_m',0.10 if sm=='ALS' else 0.08))

    near_sigma=float(cfg.get('two_way_demote_mad_k',3.0))

    detached_abs=float(cfg.get('two_way_detached_normal_residual_m',0.42 if sm=='ALS' else 0.32))

    detached_sigma=float(cfg.get('two_way_detached_mad_k',5.0))

    min_support_cells=max(8,int(cfg.get('two_way_min_support_cells',12 if sm=='ALS' else 14)))

    min_support_sectors=max(3,int(cfg.get('two_way_min_support_sectors',4)))

    max_model_cells=max(min_support_cells,int(cfg.get('two_way_max_model_support_cells',96)))

    enable_promotion=False  # manifold frontier recovery below owns non-ground -> ground

    promote_g_frac=float(cfg.get('two_way_promote_ground_fraction',0.78))

    promote_normal=float(cfg.get('two_way_promote_max_normal_residual_m',0.09 if sm=='ALS' else 0.07))

    promote_vertical=float(cfg.get('two_way_promote_max_vertical_residual_m',0.18))



    # ------------------------------------------------------------------

    # 1) Robust lower/upper class-2 representatives for each ~1 m support cell.

    # ------------------------------------------------------------------

    n_cells=ws.n_cells

    glow=np.full(n_cells,np.nan,float)

    ghigh=np.full(n_cells,np.nan,float)

    for kk in ws.occupied_keys:

        idx=ws.point_indices(int(kk)); gi=idx[g[idx]]

        if gi.size==0: continue

        vals=z[gi]

        if vals.size<=2:

            glow[int(kk)]=float(np.min(vals)); ghigh[int(kk)]=float(np.max(vals))

        else:

            qlo,qhi=np.quantile(vals,[0.12,0.98])

            glow[int(kk)]=float(qlo); ghigh[int(kk)]=float(qhi)

    valid_keys=np.flatnonzero(np.isfinite(glow))

    if valid_keys.size<min_support_cells:

        return g,rep

    vy=(valid_keys//ws.nx).astype(np.int32); vx=(valid_keys%ws.nx).astype(np.int32)

    sx=ws.x0+(vx.astype(float)+0.5)*c.cell_m

    sy=ws.y0+(vy.astype(float)+0.5)*c.cell_m

    sz=glow[valid_keys]

    shi=ghigh[valid_keys]

    support_xy=np.column_stack((sx,sy))

    support_tree=cKDTree(support_xy)

    all_tree=cKDTree(np.column_stack((x,y)))



    # ------------------------------------------------------------------

    # 2) Cheap fine-cell anomaly screen.  A cell is suspicious when its lower

    #    class-2 layer sits above surrounding lower terrain OR when class-2 inside

    #    the cell has a large vertical span.  This catches sparse flying-only

    #    cells and mixed terrain+vegetation cells without a Q95 gate.

    # ------------------------------------------------------------------

    kk=min(fine_neighbor_k,int(valid_keys.size))

    dd,nn=support_tree.query(support_xy,k=kk)

    if kk==1:

        nn=nn[:,None]

    neigh_z=sz[nn]

    # Exclude self (first nearest) where possible.

    local_med=np.median(neigh_z[:,1:] if kk>1 else neigh_z,axis=1)

    low_offset=sz-local_med

    cell_span=shi-sz

    suspicious_support=(low_offset>=fine_low_offset)|(cell_span>=fine_span_trigger)

    suspicious_keys=set(map(int,valid_keys[suspicious_support]))

    rep['fine_cells_flagged']=len(suspicious_keys)

    if not suspicious_keys:

        return g,rep



    # Coarse 10-m grouping is retained for diagnostics/reporting only.

    cx0=float(np.min(x)); cy0=float(np.min(y))

    sfix=np.floor((sx-cx0)/coarse_cell).astype(np.int32)

    sfiy=np.floor((sy-cy0)/coarse_cell).astype(np.int32)

    cfnx=max(1,int(np.floor((float(np.max(x))-cx0)/coarse_cell))+1)

    sfkey=sfiy.astype(np.int64)*cfnx+sfix.astype(np.int64)

    rep['coarse_cells_flagged']=int(np.unique(sfkey[suspicious_support]).size)



    # Fast map from global 1-m cell key -> support-array row.

    support_row={int(k):i for i,k in enumerate(valid_keys)}

    suspicious_rows=np.flatnonzero(suspicious_support)

    suspicious_row_set=set(map(int,suspicious_rows))



    model_cache={}

    def get_model(cell_key:int):

        cell_key=int(cell_key)

        if cell_key in model_cache: return model_cache[cell_key]

        row=support_row.get(cell_key)

        if row is None:

            model_cache[cell_key]=None; return None

        px=float(sx[row]); py=float(sy[row])

        ids=np.asarray(support_tree.query_ball_point([px,py],context_radius),dtype=np.int64)

        if ids.size:

            # Leave target 1-m cell out, and suppress other upward-anomalous cells

            # from the terrain fit so a shrub patch cannot pull the sheet upward.

            ids=ids[ids!=row]

            if ids.size:

                ids=ids[np.array([int(j) not in suspicious_row_set for j in ids],dtype=bool)]

        if ids.size<min_support_cells:

            # If the anomaly mask is locally dense, fall back to excluding only

            # the target cell; robust asymmetric fitting still downweights highs.

            ids=np.asarray(support_tree.query_ball_point([px,py],context_radius),dtype=np.int64)

            ids=ids[ids!=row]

        if ids.size<min_support_cells:

            model_cache[cell_key]=None; return None

        rr=np.hypot(sx[ids]-px,sy[ids]-py)

        if ids.size>max_model_cells:

            ids=ids[np.argsort(rr)[:max_model_cells]]

        if _sector_count(sx[ids]-px,sy[ids]-py)<min_support_sectors:

            model_cache[cell_key]=None; return None

        # Plane-first fast path.  Most ALS terrain is locally well described by

        # a plane even on substantial slopes.  Only pay for the six-parameter

        # quadratic fit when the robust plane residual indicates real curvature.

        pl=_robust_plane(sx[ids],sy[ids],sz[ids],asymmetric_high=True)

        if pl is None:

            model_cache[cell_key]=None; return None

        a,b,k0,xc,yc,prmse=pl; scale=math.sqrt(1+a*a+b*b)

        rn=(sz[ids]-(a*(sx[ids]-xc)+b*(sy[ids]-yc)+k0))/scale

        med=float(np.median(rn)); mad=1.4826*float(np.median(np.abs(rn-med)))

        model=('plane',pl,max(mad,0.015))

        curve_gate=float(cfg.get('two_way_plane_curve_rmse_m',0.080 if sm=='ALS' else 0.060))

        if prmse>=curve_gate and ids.size>=12:

            quad=_robust_quadratic(sx[ids],sy[ids],sz[ids],asymmetric_high=True)

            if quad is not None and float(quad[-1]) < 0.88*float(prmse):

                pred,qscale=_quad_predict(quad,sx[ids],sy[ids]); qrn=(sz[ids]-pred)/qscale

                qmed=float(np.median(qrn)); qmad=1.4826*float(np.median(np.abs(qrn-qmed)))

                model=('quad',quad,max(qmad,0.015))

        model_cache[cell_key]=model; rep['models_built']+=1; return model



    def residual(model,pidx):

        pidx=np.asarray(pidx,dtype=np.int64)

        if model[0]=='quad':

            pred,scale=_quad_predict(model[1],x[pidx],y[pidx])

        else:

            a,b,k0,xc,yc,_=model[1]; scale=math.sqrt(1+a*a+b*b)

            pred=a*(x[pidx]-xc)+b*(y[pidx]-yc)+k0

        return (z[pidx]-pred)/scale, z[pidx]-pred



    def residual_layer_vote(pi:int,model,current_g:np.ndarray):

        ids=np.asarray(all_tree.query_ball_point([x[pi],y[pi]],local_vote_radius),dtype=np.int64)

        ids=ids[ids!=pi]

        if ids.size==0: return 0,0,0.0,0.0

        r0,_=residual(model,[pi]); rr,_=residual(model,ids)

        ids=ids[np.abs(rr-float(r0[0]))<=residual_band]

        if ids.size==0: return 0,0,0.0,0.0

        gg=int(np.count_nonzero(current_g[ids])); ng=int(ids.size-gg)

        return gg,ng,gg/max(1,ids.size),ng/max(1,ids.size)



    # ------------------------------------------------------------------

    # 3) Ground -> non-ground only inside suspicious fine cells.

    # ------------------------------------------------------------------

    demote=[]; near_count=0; detached_count=0; cand_count=0

    for cell_key in suspicious_keys:

        idx=ws.point_indices(int(cell_key)); pts=idx[g[idx]]

        if pts.size==0: continue

        cand_count+=int(pts.size)

        model=get_model(int(cell_key))

        if model is None: continue

        rn,dz=residual(model,pts); mad=float(model[2])

        # MAD is allowed to adapt, but cannot raise the near-surface threshold so

        # high that 15-40 cm shrub leakage becomes invisible.  The local residual

        # majority vote is the protection against over-cleaning real terrain.

        near_thr=max(near_abs,min(0.18,near_sigma*mad))

        detached_thr=max(detached_abs,min(0.55,detached_sigma*mad))

        high=pts[(rn>=detached_thr)&(dz>=detached_abs)]

        if high.size:

            demote.extend(map(int,high)); detached_count+=int(high.size)

        near=pts[(rn>=near_thr)&(dz>=near_abs)&(rn<detached_thr)]

        for pi in near:

            gg,ng,gfrac,ngfrac=residual_layer_vote(int(pi),model,g)

            if gg+ng<min_vote_neighbors: continue

            if ngfrac>=near_ng_frac:

                demote.append(int(pi)); near_count+=1

    rep['candidate_ground_points']=int(cand_count)

    if demote:

        demote=np.unique(np.asarray(demote,dtype=np.int64)); g[demote]=False

        rep['demoted_points']=int(demote.size)

    rep['near_surface_demoted']=int(near_count)

    rep['detached_demoted']=int(detached_count)



    # Optional reverse swipe, disabled by default.  It is intentionally limited

    # to the same suspicious fine cells and does not region-grow.

    if enable_promotion:

        promote=[]; cand_ng_count=0

        for cell_key in suspicious_keys:

            idx=ws.point_indices(int(cell_key)); pts=idx[~g[idx]]

            if pts.size==0: continue

            cand_ng_count+=int(pts.size)

            model=get_model(int(cell_key))

            if model is None: continue

            rn,dz=residual(model,pts)

            pts=pts[(np.abs(rn)<=promote_normal)&(np.abs(dz)<=promote_vertical)]

            for pi in pts:

                gg,ng,gfrac,ngfrac=residual_layer_vote(int(pi),model,g)

                if gg+ng<min_vote_neighbors: continue

                if gfrac>=promote_g_frac: promote.append(int(pi))

        rep['candidate_nonground_points']=int(cand_ng_count)

        if promote:

            promote=np.unique(np.asarray(promote,dtype=np.int64)); g[promote]=True

            rep['promoted_points']=int(promote.size)

    # Final short manifold recovery.  This is intentionally last: detached/flying

    # class-2 points are removed first so they cannot seed terrain growth.

    if bool(cfg.get('manifold_recovery_enabled', True)):

        g,mrep=_recover_manifold_frontier(x=x,y=y,z=z,ground_mask=g,original_ground_mask=original_ground,

                                         sensor_mode=sm,cfg=cfg,workspace=ws)

        rep['manifold_recovery']=mrep

        rep['promoted_points']=int(rep.get('promoted_points',0))+int(mrep.get('promoted_points',0))

    return g,rep







# =============================================================================

# ALS FEATURE-PRESERVING TERRAIN-SURFACE IMPURITY CONSENSUS (V3)

# =============================================================================

#

# Integrate the 0.5 m DTM impurity assessment into production processing.

# It deliberately does NOT replace FAST-GC's stable upstream terrain classifier.

# Instead it treats the mature class-2 sheet as a trusted baseline and searches

# only for small, DTM-influential local protrusions ("pyramids/tents/domes") that

# are inconsistent with the surrounding terrain in surface-normal coordinates.

#

# The implementation is candidate-first:

#   1) a cheap 0.5 m raster screen identifies locally anomalous ground cells;

#   2) only those cells receive robust leave-one-out plane/quadratic fitting;

#   3) demotion requires multiple independent signals (normal residual, weak

#      same-sheet continuation, fine/coarse normal disagreement and/or local

#      non-ground / perpendicular-column evidence);

#   4) steep/rough but coherent terrain is explicitly protected;

#   5) the already validated manifold-frontier recovery is retained after

#      demotion to recover omitted terrain conservatively.

#

# Set ``surface_impurity_consensus_enabled=False`` to recover the exact legacy

# two-way swipe while preserving the configured processing route.





def _fill_nearest_2d(a: np.ndarray) -> np.ndarray:

    a=np.asarray(a,dtype=float)

    valid=np.isfinite(a)

    if np.all(valid): return a.copy()

    if not np.any(valid): return np.zeros_like(a,dtype=float)

    from scipy.ndimage import distance_transform_edt

    _,inds=distance_transform_edt(~valid,return_indices=True)

    return a[tuple(inds)]





def _fine_ground_surface_stats(x, y, z, g, *, cell_m: float):

    """Return robust lower/upper ground representatives on a fine XY grid.



    The lower representative is intentionally close to the lower envelope used

    by a high-resolution DTM, while still being robust when a cell contains many

    returns.  The second-lowest gap is retained as a direct surface-influence

    diagnostic: a lone bad ground return has a large influence on the 0.5 m DTM.

    """

    x=np.asarray(x,float); y=np.asarray(y,float); z=np.asarray(z,float); g=np.asarray(g,bool)

    x0=float(np.floor(np.min(x)/cell_m)*cell_m); y0=float(np.floor(np.min(y)/cell_m)*cell_m)

    ix=np.floor((x-x0)/cell_m).astype(np.int32); iy=np.floor((y-y0)/cell_m).astype(np.int32)

    nx=int(ix.max())+1; ny=int(iy.max())+1; ncell=nx*ny

    key=iy.astype(np.int64)*nx+ix.astype(np.int64)

    order=np.argsort(key,kind='mergesort'); ks=key[order]

    if ks.size:

        starts=np.r_[0,1+np.flatnonzero(ks[1:]!=ks[:-1])]; ends=np.r_[starts[1:],ks.size]

        occupied=ks[starts].astype(np.int64,copy=False)

    else:

        starts=np.empty(0,dtype=np.int64); ends=np.empty(0,dtype=np.int64); occupied=np.empty(0,dtype=np.int64)

    cell_start=np.full(ncell,-1,dtype=np.int64); cell_end=np.full(ncell,-1,dtype=np.int64)

    if occupied.size:

        cell_start[occupied]=starts; cell_end[occupied]=ends



    low=np.full(ncell,np.nan,float); high=np.full(ncell,np.nan,float)

    xrep=np.full(ncell,np.nan,float); yrep=np.full(ncell,np.nan,float)

    gcount=np.zeros(ncell,dtype=np.int32); low_gap=np.full(ncell,np.nan,float)

    for kk in occupied:

        a=int(cell_start[int(kk)]); b=int(cell_end[int(kk)]); ids=order[a:b]

        gi=ids[g[ids]]

        if gi.size==0: continue

        zz=np.sort(z[gi])

        gcount[int(kk)]=int(zz.size)

        xrep[int(kk)]=float(np.median(x[gi])); yrep[int(kk)]=float(np.median(y[gi]))

        # DTM influence must follow the actual lower envelope.  For dense cells

        # q10 suppresses one pathological low spike; for sparse cells use min.

        low[int(kk)]=float(np.min(zz) if zz.size<5 else np.quantile(zz,0.10))

        high[int(kk)]=float(np.max(zz) if zz.size<5 else np.quantile(zz,0.98))

        if zz.size>=2: low_gap[int(kk)]=float(zz[1]-zz[0])

    return dict(x0=x0,y0=y0,ix=ix,iy=iy,key=key,nx=nx,ny=ny,

                order=order,cell_start=cell_start,cell_end=cell_end,

                occupied=occupied,low=low.reshape(ny,nx),high=high.reshape(ny,nx),

                xrep=xrep.reshape(ny,nx),yrep=yrep.reshape(ny,nx),

                gcount=gcount.reshape(ny,nx),low_gap=low_gap.reshape(ny,nx))





def _grid_candidate_screen(low2, high2, *, cell_m: float, cfg: dict):

    """Cheap raster screen for local tent/dome/pyramid-like terrain defects.



    This stage is intentionally permissive: it NEVER changes labels.  Its only

    job is to reduce millions of points to a small set of cells that deserve a

    robust leave-one-out surface test.

    """

    from scipy.ndimage import gaussian_filter, median_filter

    filled=_fill_nearest_2d(low2)

    # Fine geometry and a ~5 m regional trend.  The regional surface is evidence

    # only; it never replaces the 0.5 m DTM or directly reclassifies points.

    fine_sigma=max(0.0,float(cfg.get('surface_impurity_fine_sigma_cells',0.75)))

    coarse_scale=max(3.0,float(cfg.get('surface_impurity_coarse_scale_m',5.0)))

    coarse_sigma=max(1.0,coarse_scale/max(cell_m,1e-6)/2.355)

    fine=gaussian_filter(filled,sigma=fine_sigma,mode='nearest') if fine_sigma>0 else filled

    coarse=gaussian_filter(filled,sigma=coarse_sigma,mode='nearest')

    fy,fx=np.gradient(fine,cell_m,cell_m); cy,cx=np.gradient(coarse,cell_m,cell_m)

    fmag=np.hypot(fx,fy); cmag=np.hypot(cx,cy)

    fn=np.stack((-fx,-fy,np.ones_like(fx)),axis=-1); cn=np.stack((-cx,-cy,np.ones_like(cx)),axis=-1)

    fn/=np.maximum(np.linalg.norm(fn,axis=-1,keepdims=True),1e-12)

    cn/=np.maximum(np.linalg.norm(cn,axis=-1,keepdims=True),1e-12)

    dot=np.clip(np.sum(fn*cn,axis=-1),-1.0,1.0); angle=np.degrees(np.arccos(dot))



    # Local median residual is a simple representation of the ArcGIS/TIN

    # "pimple" signature.  Because this is merely a screen, steep terrain is

    # allowed through and later protected by leave-one-out normal geometry.

    w=max(3,int(cfg.get('surface_impurity_screen_window_cells',5)))

    if w%2==0: w+=1

    med=median_filter(fine,size=w,mode='nearest')

    local_bump=fine-med

    span=high2-low2

    finite=np.isfinite(low2)

    bump_thr=float(cfg.get('surface_impurity_screen_bump_m',0.10))

    span_thr=float(cfg.get('surface_impurity_screen_span_m',0.18))

    angle_thr=float(cfg.get('surface_impurity_screen_angle_deg',8.0))

    slope_disagree=np.degrees(np.arctan(fmag))-np.degrees(np.arctan(cmag))

    slope_thr=float(cfg.get('surface_impurity_screen_slope_disagreement_deg',8.0))

    cand=finite & ((local_bump>=bump_thr)|(span>=span_thr)|(angle>=angle_thr)|(slope_disagree>=slope_thr))

    return cand,dict(fine=fine,coarse=coarse,fx=fx,fy=fy,cx=cx,cy=cy,

                     fine_slope=fmag,coarse_slope=cmag,normal_angle=angle,

                     local_bump=local_bump,span=span)





def _surface_normal_angle(a1,b1,a2,b2) -> float:

    n1=np.array([-float(a1),-float(b1),1.0],dtype=float); n2=np.array([-float(a2),-float(b2),1.0],dtype=float)

    n1/=max(float(np.linalg.norm(n1)),1e-12); n2/=max(float(np.linalg.norm(n2)),1e-12)

    return float(np.degrees(np.arccos(np.clip(float(np.dot(n1,n2)),-1.0,1.0))))





def _fit_support_model(sx,sy,sz,ids,*,px,py,curve_gate: float):

    ids=np.asarray(ids,dtype=np.int64)

    if ids.size<6: return None

    pl=_robust_plane(sx[ids],sy[ids],sz[ids],asymmetric_high=True)

    if pl is None: return None

    a,b,k,xc,yc,prmse=pl; scale=math.sqrt(1+a*a+b*b)

    rr=(sz[ids]-(a*(sx[ids]-xc)+b*(sy[ids]-yc)+k))/scale

    med=float(np.median(rr)); mad=max(1.4826*float(np.median(np.abs(rr-med))),0.015)

    model=('plane',pl,mad)

    if prmse>=curve_gate and ids.size>=12:

        q=_robust_quadratic(sx[ids],sy[ids],sz[ids],asymmetric_high=True)

        if q is not None and float(q[-1])<0.88*float(prmse):

            qp,qs=_quad_predict(q,sx[ids],sy[ids]); qr=(sz[ids]-qp)/qs

            qmed=float(np.median(qr)); qmad=max(1.4826*float(np.median(np.abs(qr-qmed))),0.015)

            model=('quad',q,qmad)

    return model





def _model_predict(model, xx, yy):

    xx=np.asarray(xx,float); yy=np.asarray(yy,float)

    if model[0]=='quad':

        return _quad_predict(model[1],xx,yy)

    a,b,k,xc,yc,_=model[1]; scale=math.sqrt(1+a*a+b*b)

    pred=a*(xx-xc)+b*(yy-yc)+k

    return pred,np.full(np.shape(pred),scale,dtype=float)





def _model_plane_slopes(model, px: float, py: float):

    if model[0]=='plane':

        a,b,*_=model[1]; return float(a),float(b)

    # z=a*x^2+b*y^2+c*x*y+d*x+e*y+f in centered coordinates used by helper.

    q=model[1]

    # _robust_quadratic returns (coef, xc, yc, rmse) in this module.

    coef,xc,yc,_=q

    dx=float(px)-float(xc); dy=float(py)-float(yc)

    coef=np.asarray(coef,float)

    gx=float(coef[1]+2.0*coef[3]*dx+coef[4]*dy)

    gy=float(coef[2]+coef[4]*dx+2.0*coef[5]*dy)

    return gx,gy





def _terrain_impurity_consensus_als(*, x, y, z, ground_mask, cfg, workspace):

    """Conservative ALS-only G->NG impurity cleanup for the 0.5 m terrain surface."""

    x=np.asarray(x,float); y=np.asarray(y,float); z=np.asarray(z,float)

    g=np.asarray(ground_mask,bool).copy(); cfg=cfg or {}

    rep={

        'enabled':True,'candidate_cells':0,'candidate_ground_points':0,'models_built':0,

        'confirmed_cells':0,'demoted_points':0,'near_surface_demoted':0,'detached_demoted':0,

        'protected_coherent_slope_cells':0,'protected_same_sheet_cells':0,

        'rejected_weak_evidence_cells':0,'surface_changed_cells':0,

    }

    if x.size<32 or int(g.sum())<16: return g,rep



    cell=float(cfg.get('surface_impurity_cell_m',0.50))

    fine=_fine_ground_surface_stats(x,y,z,g,cell_m=cell)

    low2=fine['low']; high2=fine['high']

    screen,geom=_grid_candidate_screen(low2,high2,cell_m=cell,cfg=cfg)

    cy,cx=np.nonzero(screen); rep['candidate_cells']=int(len(cx))

    if len(cx)==0: return g,rep



    # Fine lower-support cloud.  This is the terrain evidence used by local fits.

    valid=np.isfinite(low2); syi,sxi=np.nonzero(valid)

    skey=syi.astype(np.int64)*fine['nx']+sxi.astype(np.int64)

    sx=fine['xrep'][syi,sxi]

    sy=fine['yrep'][syi,sxi]

    sz=low2[syi,sxi]

    stree=cKDTree(np.column_stack((sx,sy)))

    row={int(k):i for i,k in enumerate(skey)}

    all_tree=cKDTree(np.column_stack((x,y)))



    fit_r=float(cfg.get('surface_impurity_fit_radius_m',3.0))

    coarse_r=float(cfg.get('surface_impurity_coarse_radius_m',7.5))

    min_support=max(8,int(cfg.get('surface_impurity_min_support_cells',10)))

    max_support=max(min_support,int(cfg.get('surface_impurity_max_support_cells',48)))

    min_sectors=max(3,int(cfg.get('surface_impurity_min_support_sectors',4)))

    curve_gate=float(cfg.get('surface_impurity_curve_rmse_m',0.075))

    base_resid=float(cfg.get('surface_impurity_min_normal_residual_m',0.12))

    strong_resid=float(cfg.get('surface_impurity_strong_normal_residual_m',0.35))

    mad_k=float(cfg.get('surface_impurity_mad_k',3.5))

    angle_soft=float(cfg.get('surface_impurity_normal_angle_soft_deg',10.0))

    angle_strong=float(cfg.get('surface_impurity_normal_angle_strong_deg',20.0))

    same_sheet_band=float(cfg.get('surface_impurity_same_sheet_band_m',0.12))

    max_same_sheet=max(1,int(cfg.get('surface_impurity_max_same_sheet_neighbors',2)))

    vote_r=float(cfg.get('surface_impurity_vote_radius_m',1.25))

    column_min=max(2,int(cfg.get('surface_impurity_column_min_points',3)))

    ng_frac_thr=float(cfg.get('surface_impurity_nonground_fraction',0.55))

    column_upper=float(cfg.get('surface_impurity_column_upper_m',0.60))

    high_slope_deg=float(cfg.get('surface_impurity_steep_slope_deg',35.0))

    steep_protect_angle=float(cfg.get('surface_impurity_steep_protect_angle_deg',8.0))

    lowgap_influence=float(cfg.get('surface_impurity_lowgap_influence_m',0.08))

    max_demote_fraction=float(cfg.get('surface_impurity_max_demote_fraction',0.010))



    demote=[]; dem=np.empty(0,dtype=np.int64); confirmed_cells=set()

    model_cache={}

    for gy,gx in zip(cy,cx):

        kk=int(gy*fine['nx']+gx); sr=row.get(kk)

        if sr is None: continue

        px=float(sx[sr]); py=float(sy[sr])



        ids=np.asarray(stree.query_ball_point([px,py],fit_r),dtype=np.int64)

        if ids.size:

            ids=ids[ids!=sr]  # leave target cell out: critical pyramid/tent test

        if ids.size<min_support: continue

        rr=np.hypot(sx[ids]-px,sy[ids]-py)

        if ids.size>max_support: ids=ids[np.argsort(rr)[:max_support]]

        if _sector_count(sx[ids]-px,sy[ids]-py)<min_sectors: continue

        model=_fit_support_model(sx,sy,sz,ids,px=px,py=py,curve_gate=curve_gate)

        if model is None: continue

        rep['models_built']+=1

        pred,scale=_model_predict(model,[px],[py]); pred=float(pred[0]); scale=float(scale[0])

        cell_rn=(float(low2[gy,gx])-pred)/scale

        tol=max(base_resid,mad_k*float(model[2]))

        if cell_rn<tol: continue



        # Larger-scale terrain orientation.  This is only a protection/evidence

        # layer; it never directly moves or labels points.

        cids=np.asarray(stree.query_ball_point([px,py],coarse_r),dtype=np.int64)

        if cids.size: cids=cids[cids!=sr]

        if cids.size>max_support*2:

            cr=np.hypot(sx[cids]-px,sy[cids]-py); cids=cids[np.argsort(cr)[:max_support*2]]

        coarse_model=_fit_support_model(sx,sy,sz,cids,px=px,py=py,curve_gate=max(curve_gate,0.10)) if cids.size>=min_support else None

        fa,fb=_model_plane_slopes(model,px,py)

        fine_slope_deg=float(np.degrees(np.arctan(np.hypot(fa,fb))))

        normal_angle=float(geom['normal_angle'][gy,gx])

        if coarse_model is not None:

            ca,cb=_model_plane_slopes(coarse_model,px,py)

            normal_angle=_surface_normal_angle(fa,fb,ca,cb)



        # Same-sheet continuation veto: evaluate neighboring lower-support cells

        # in the target model.  A real ridge/slope tends to have peers at the same

        # normal level; an isolated canopy vertex does not.

        npred,nscale=_model_predict(model,sx[ids],sy[ids]); nr=(sz[ids]-npred)/nscale

        same=int(np.count_nonzero(np.abs(nr-cell_rn)<=same_sheet_band))

        if same>max_same_sheet:

            rep['protected_same_sheet_cells']+=1; continue



        # Coherent extreme slopes are sacred.  If fine and regional terrain normals

        # agree, do not demote merely because global-Z geometry is dramatic.

        if fine_slope_deg>=high_slope_deg and normal_angle<=steep_protect_angle and cell_rn<strong_resid:

            rep['protected_coherent_slope_cells']+=1; continue



        # 3-D / perpendicular-column evidence from ALL current labels.

        pids=np.asarray(all_tree.query_ball_point([px,py],vote_r),dtype=np.int64)

        if pids.size:

            pp,ps=_model_predict(model,x[pids],y[pids]); pr=(z[pids]-pp)/ps

            nearcol=(pr>=max(0.03,0.25*tol))&(pr<=column_upper)

            col_ids=pids[nearcol]

        else:

            col_ids=np.empty(0,dtype=np.int64)

        if col_ids.size:

            ng=int(np.count_nonzero(~g[col_ids])); gg=int(col_ids.size-ng); ngfrac=ng/float(col_ids.size)

        else:

            ng=gg=0; ngfrac=0.0

        vegetation_column=(col_ids.size>=column_min and ngfrac>=ng_frac_thr)



        # Surface-influence evidence.  A sparse cell or one whose lower support is

        # separated from the next observed ground return can disproportionately

        # control a 0.5 m TIN/DTM.  Cell residual itself is the strongest signal.

        gc=int(fine['gcount'][gy,gx]); lg=float(fine['low_gap'][gy,gx]) if np.isfinite(fine['low_gap'][gy,gx]) else 0.0

        influential=(gc<=2) or (lg>=lowgap_influence) or (cell_rn>=strong_resid)

        if not influential:

            rep['rejected_weak_evidence_cells']+=1; continue



        evidence=int(normal_angle>=angle_soft)+int(vegetation_column)+int(float(geom['local_bump'][gy,gx])>=base_resid)

        strong_geom=(cell_rn>=strong_resid and (normal_angle>=angle_soft or vegetation_column))

        if not (strong_geom or evidence>=2):

            rep['rejected_weak_evidence_cells']+=1; continue

        if fine_slope_deg>=high_slope_deg and normal_angle<angle_strong and not vegetation_column and cell_rn<strong_resid:

            rep['protected_coherent_slope_cells']+=1; continue



        # Demote only the supporting ground points that are themselves above the

        # leave-one-out terrain.  Genuine low points in a mixed cell are retained.

        a=int(fine['cell_start'][kk]); b=int(fine['cell_end'][kk])

        if a<0: continue

        pts=fine['order'][a:b]; pts=pts[g[pts]]

        if pts.size==0: continue

        pp,ps=_model_predict(model,x[pts],y[pts]); rn=(z[pts]-pp)/ps

        bad=pts[rn>=tol]

        if bad.size==0: continue

        demote.extend(map(int,bad)); confirmed_cells.add(kk)

        if cell_rn>=strong_resid: rep['detached_demoted']+=int(bad.size)

        else: rep['near_surface_demoted']+=int(bad.size)



    rep['candidate_ground_points']=int(sum(int(fine['gcount'][gy,gx]) for gy,gx in zip(cy,cx)))

    if demote:

        dem=np.unique(np.asarray(demote,dtype=np.int64))

        # Global safety cap is intentionally very small.  This stage targets a

        # sparse, high-impact impurity population; broad erosion is a hard failure.

        cap=max(1,int(np.ceil(max_demote_fraction*max(1,int(g.sum())))))

        if dem.size>cap:

            # Rank by point normal residual to each cell model would require

            # retaining all models.  Conservative fallback: abort broad cleanup

            # rather than silently erode terrain.

            dem=np.empty(0,dtype=np.int64); confirmed_cells.clear()

            rep['safety_cap_triggered']=True

        else:

            g[dem]=False

    rep['confirmed_cells']=int(len(confirmed_cells)); rep['demoted_points']=int(len(dem))

    rep['surface_changed_cells']=int(len(confirmed_cells))

    return g,rep





def two_way_surface_classification_swipe(*, x, y, z, ground_mask, sensor_mode, cfg=None, workspace: SurfaceConsensusWorkspace | None = None):

    """Mature two-way swipe plus conservative ALS surface-impurity consensus.



    This wrapper deliberately preserves the stable FAST-GC baseline.  The mature

    legacy two-way swipe runs first, unchanged.  ALS then receives one additional

    candidate-first, G->NG-only terrain-surface impurity pass that targets the

    sparse 0.5 m pyramid/tent/dome defects found in the DTM ablation study.



    ``surface_impurity_consensus_enabled=False`` returns the exact legacy result,

    without altering any upstream classifier stage.

    """

    sm=str(sensor_mode).upper().strip(); cfg=cfg or {}



    # Stable baseline first -- never replace or bypass it.

    base,base_rep=_legacy_two_way_surface_classification_swipe(

        x=x,y=y,z=z,ground_mask=ground_mask,sensor_mode=sensor_mode,cfg=cfg,workspace=workspace

    )

    if sm!='ALS' or not bool(cfg.get('surface_impurity_consensus_enabled',True)):

        return base,base_rep



    # New evidence is additive and demotion-only.  Existing manifold recovery and

    # all historical two-way behavior have already occurred in ``base``.

    out,v3=_terrain_impurity_consensus_als(

        x=x,y=y,z=z,ground_mask=base,cfg=cfg,workspace=workspace

    )



    # Preserve historical report fields expected by CLI/tests and append explicit

    # V3 diagnostics so every extra change can be audited independently.

    rep=dict(base_rep)

    rep['surface_impurity_consensus']=v3

    rep['surface_impurity_candidate_cells']=int(v3.get('candidate_cells',0))

    rep['surface_impurity_confirmed_cells']=int(v3.get('confirmed_cells',0))

    rep['surface_impurity_demoted_points']=int(v3.get('demoted_points',0))

    rep['surface_impurity_surface_changed_cells']=int(v3.get('surface_changed_cells',0))

    rep['demoted_points']=int(base_rep.get('demoted_points',0))+int(v3.get('demoted_points',0))

    rep['ground_after_surface_impurity']=int(np.count_nonzero(out))

    return out,rep





__all__ = ['SurfaceConsensusWorkspace','build_surface_consensus_workspace','recover_surface_false_negatives',

           'demote_detached_ground_consensus','demote_flying_ground_outliers',

           'two_way_surface_classification_swipe']



