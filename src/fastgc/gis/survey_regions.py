"""Candidate disconnected regions based on tile bounding boxes.

Bounding-box connectivity is preliminary, not observed point coverage.
"""
from __future__ import annotations

from pyproj import CRS
from shapely.geometry import box
from shapely.strtree import STRtree


def discover_survey_regions(catalog, *, adjacency_tolerance=0.0):
    if adjacency_tolerance < 0:
        raise ValueError('adjacency_tolerance must be nonnegative')
    tiles = [t for t in catalog['tiles'] if t['status'] == 'ready']
    if not tiles:
        return {'regions': [], 'overall_bounds': None, 'excluded_tile_ids':
                [t['tile_id'] for t in catalog['tiles']]}
    # Different CRSs cannot share a meaningful bounding-box coordinate space.
    crs = CRS.from_wkt(tiles[0]['crs_wkt'])
    if any(not crs.equals(CRS.from_wkt(t['crs_wkt'])) for t in tiles[1:]):
        raise ValueError('Ready tiles have different CRSs; transform to a common CRS first')
    shapes = [box(*t['bounds']).buffer(adjacency_tolerance) for t in tiles]
    index = STRtree(shapes)
    parent = list(range(len(tiles)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, shape in enumerate(shapes):
        for j in index.query(shape, predicate='intersects'):
            j = int(j)
            a, b = find(i), find(j)
            if a != b:
                parent[max(a,b)] = min(a,b)
    groups = {}
    for i, tile in enumerate(tiles):
        groups.setdefault(find(i), []).append(tile)
    regions = []
    for k, members in enumerate(sorted(groups.values(), key=lambda g: min(t['tile_id'] for t in g)), 1):
        bounds = [min(t['bounds'][0] for t in members), min(t['bounds'][1] for t in members),
                  max(t['bounds'][2] for t in members), max(t['bounds'][3] for t in members)]
        regions.append({'region_id': f'region_{k:03d}', 'tile_ids': [t['tile_id'] for t in members],
                        'tile_count': len(members), 'bounds': bounds,
                        'connectivity_method': 'tile_bbox_preliminary'})
    return {'regions': regions,
            'overall_bounds': [min(r['bounds'][0] for r in regions), min(r['bounds'][1] for r in regions),
                               max(r['bounds'][2] for r in regions), max(r['bounds'][3] for r in regions)],
            'excluded_tile_ids': [t['tile_id'] for t in catalog['tiles'] if t['status'] != 'ready']}
