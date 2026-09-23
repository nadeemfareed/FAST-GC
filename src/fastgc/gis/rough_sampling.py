"""FAST-GIS: reproducible rough-height circular sampling from an existing DSM.

No ground classification is required. Local low-return elevations are only a
proxy for terrain; results must not be described as validated canopy heights.
Uses the existing FAST-GIS plot importer and transactional clipping runner.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import laspy
import numpy as np
import rasterio
from pyproj import CRS
from rasterio.transform import rowcol
from scipy.ndimage import distance_transform_edt, median_filter
from shapely.geometry import Point, mapping

from .plot_runner import extract_plots, _metre_crs
from .progress import gis_progress


def local_low_surface(source, reference, *, chunk_size=1_000_000):
    """Minimum observed Z per DSM cell, locally median-filtered; not a DEM."""
    with rasterio.open(reference) as ds:
        if ds.crs is None or not _metre_crs(CRS.from_user_input(ds.crs)):
            raise ValueError('DSM requires a projected CRS with metre units')
        if ds.transform.b != 0 or ds.transform.d != 0:
            raise ValueError('Rotated DSM grids are not supported in this first sampler')
        transform, shape = ds.transform, (ds.height, ds.width)
        raster_crs = CRS.from_user_input(ds.crs)
        dsm = ds.read(1, masked=True)
        valid_dsm = ~np.ma.getmaskarray(dsm) & np.isfinite(dsm.data)
        valid_dsm &= dsm.data > -1e6
    with laspy.open(source) as reader:
        crs = reader.header.parse_crs()
        if crs is None or not CRS.from_user_input(crs).equals(raster_crs):
            raise ValueError('LAS and DSM must have matching embedded projected CRS')
        low = np.full(shape, np.inf, dtype=np.float64)
        total = int(reader.header.point_count)
        bar = gis_progress("ROUGH HEIGHT", total, unit="point")
        try:
            for chunk in reader.chunk_iterator(chunk_size):
                x, y, z = np.asarray(chunk.x), np.asarray(chunk.y), np.asarray(chunk.z)
                cols = np.floor((x - transform.c) / transform.a).astype(np.int64)
                rows = np.floor((y - transform.f) / transform.e).astype(np.int64)
                good = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1]) & np.isfinite(z)
                np.minimum.at(low, (rows[good], cols[good]), z[good])
                bar.update(len(chunk))
        finally:
            bar.close()
    observed = np.isfinite(low)
    # No extrapolation into empty cells: all sampling cells must have raw returns.
    if not observed.any():
        raise ValueError('No source points intersect the DSM raster')
    # Median filter is applied only to a fully observed local neighborhood.
    # Keep raw local minimum elsewhere; this is a proxy, not ground truth.
    local = np.where(observed, low, 0.0)
    full_window = median_filter(observed.astype(np.uint8), size=5, mode='constant') == 1
    # median_filter of binary mask is insufficient to assert complete coverage;
    # use a 5x5 minimum filter for that condition.
    from scipy.ndimage import minimum_filter
    full_window = minimum_filter(observed.astype(np.uint8), size=5, mode='constant') == 1
    filtered = median_filter(local, size=5, mode='nearest')
    local = np.where(full_window, filtered, local)
    valid = observed & valid_dsm
    height = np.maximum(0.0, dsm.data.astype(np.float64) - local)
    height[~valid] = np.nan
    return height, valid, transform, raster_crs


def choose_circular_plots(height, valid, transform, *, count=30, radius=10.0,
                          buffer=10.0, thresholds=(5.0, 15.0), seed=42,
                          min_valid_fraction=0.95, max_attempts=250_000):
    """Select nonoverlapping plot+buffer footprints, equally per height stratum."""
    if count < 1 or count % 3:
        raise ValueError('count must be a positive multiple of 3 for equal three-stratum allocation')
    if radius <= 0 or buffer < 0 or not (0 < min_valid_fraction <= 1):
        raise ValueError('Invalid radius, buffer or minimum valid fraction')
    if transform.b != 0 or transform.d != 0:
        raise ValueError('Rotated rasters not supported')
    dx, dy = abs(transform.a), abs(transform.e)
    clearance = distance_transform_edt(valid, sampling=(dy, dx))
    # Require full observed coverage over processing footprint, not only the core.
    eligible = np.argwhere(clearance >= radius + buffer + max(dx, dy))
    if not len(eligible):
        raise ValueError('No eligible fully covered plot+buffer footprints')
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(eligible))
    quota = count // 3
    chosen = []
    counts = [0, 0, 0]
    core_rows = math.ceil(radius / dy)
    core_cols = math.ceil(radius / dx)
    for ix in order[:max_attempts]:
        r, c = eligible[ix]
        rs = slice(max(0, r-core_rows), min(height.shape[0], r+core_rows+1))
        cs = slice(max(0, c-core_cols), min(height.shape[1], c+core_cols+1))
        yy = (np.arange(rs.start, rs.stop)-r)*dy
        xx = (np.arange(cs.start, cs.stop)-c)*dx
        circle = yy[:, None]**2 + xx[None, :]**2 <= radius**2
        values = height[rs, cs][circle]
        finite = values[np.isfinite(values)]
        if not len(values) or len(finite)/len(values) < min_valid_fraction:
            continue
        med = float(np.median(finite))
        stratum = 0 if med < thresholds[0] else (1 if med < thresholds[1] else 2)
        if counts[stratum] >= quota:
            continue
        x, y = transform * (float(c)+0.5, float(r)+0.5)
        if any((x - item['x'])**2 + (y-item['y'])**2 < (2*(radius+buffer))**2 for item in chosen):
            continue
        chosen.append({'x': x, 'y': y, 'median_rough_height_m': med,
                       'stratum': ('low', 'medium', 'high')[stratum],
                       'valid_fraction': len(finite)/len(values)})
        counts[stratum] += 1
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError(f'Insufficient eligible area: selected {counts} of {[quota]*3}; no partial plot set published')
    return chosen


def sample_and_clip(source, dsm_path, output_dir, *, count=30, radius=10., buffer=10.,
                    thresholds=(5., 15.), seed=42, chunk_size=500_000, sensor_mode="ALS"):
    """Create sampling GeoJSON and publish an existing-runner clip workspace.

    Output root must be new. Original LAS and existing FAST-GC outputs are untouched.
    """
    import shutil
    import tempfile
    source, dsm_path, target = Path(source).resolve(), Path(dsm_path).resolve(), Path(output_dir).resolve()
    sensor_mode = str(sensor_mode).upper()
    if sensor_mode not in {'ALS','ULS','TLS'}:
        raise ValueError('sensor_mode must be ALS, ULS, or TLS')
    if not source.is_file() or not dsm_path.is_file():
        raise FileNotFoundError('Input LAS/LAZ and existing SpikeFree DSM are both required')
    if target.exists():
        raise FileExistsError(target)
    if target in source.parents or target == source.parent:
        raise ValueError('Output cannot replace input')
    height, valid, transform, crs = local_low_surface(source, dsm_path, chunk_size=chunk_size)
    chosen = choose_circular_plots(height, valid, transform, count=count, radius=radius,
                                   buffer=buffer, thresholds=thresholds, seed=seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{target.name}_stage_', dir=target.parent))
    try:
        features = []
        for i, item in enumerate(chosen, 1):
            name = f'Plot_{i:03d}'
            geom = Point(item['x'], item['y']).buffer(radius, quad_segs=32)
            features.append({'type':'Feature', 'id':name, 'properties':{'plot_name':name, **item},
                             'geometry':mapping(geom)})
        plots = stage/'sample_plots.geojson'
        plots.write_text(json.dumps({'type':'FeatureCollection', 'name':'FAST_GIS_rough_height',
                                      'crs':{'type':'name','properties':{'name':crs.to_string()}},
                                      'features':features}, indent=2), encoding='utf-8')
        (stage/'sampling_manifest.json').write_text(json.dumps({
            'schema_version':1, 'method':'rough_height_local_min_spikefree',
            'warning':'Rough heights are DSM minus local low-return proxy, not validated canopy heights',
            'source_las':str(source), 'dsm':str(dsm_path), 'crs':crs.to_string(),
            'sensor_mode':sensor_mode, 'seed':seed, 'plot_count':count, 'radius_m':radius, 'buffer_m':buffer,
            'strata_m':[0, thresholds[0], thresholds[1], None], 'plots':chosen
        }, indent=2), encoding='utf-8')
        # Existing transactional clip runner: outputs include workspace_manifest.json.
        clip_root = stage/f'{sensor_mode}_plots'
        extract_plots(source, plots, clip_root, buffer=buffer, chunk_size=chunk_size)
        # Publish root only after clipping completes.
        stage.rename(target)
        return target
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--in_path', required=True)
    p.add_argument('--dsm_path', required=True, help='Existing FAST_DSM SpikeFree GeoTIFF')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--plots', type=int, default=30)
    p.add_argument('--radius', type=float, default=10.)
    p.add_argument('--buffer', type=float, default=10.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--low_max', type=float, default=5.)
    p.add_argument('--medium_max', type=float, default=15.)
    p.add_argument('--chunk_size', type=int, default=500_000)
    a = p.parse_args(argv)
    if not 0 < a.low_max < a.medium_max:
        p.error('Require 0 < low_max < medium_max')
    print(sample_and_clip(a.in_path, a.dsm_path, a.out_dir, count=a.plots,
                          radius=a.radius, buffer=a.buffer, seed=a.seed,
                          thresholds=(a.low_max,a.medium_max), chunk_size=a.chunk_size))

if __name__ == '__main__':
    main()
