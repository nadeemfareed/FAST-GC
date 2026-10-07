"""Hydrologic terrain conditioning for FAST_TERRAIN.

Scientific basis
----------------
Barnes, R., Lehman, C., & Mulla, D. J. (2014).
Priority-Flood: An optimal depression-filling and watershed-labeling
algorithm for digital elevation models. Computers & Geosciences, 62,
117-127.

The routines here operate on a complete analytical DEM domain.
They are not intended for independent hydrologic processing of
ordinary LiDAR tiles.
"""

from __future__ import annotations

from collections import deque
import heapq

import numpy as np


_NEIGHBORS_8 = (
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
)


def priority_flood_fill(
    dem: np.ndarray,
) -> np.ndarray:
    """Return an 8-connected depression-filled DEM.

    NaN cells are treated as outside the hydrologic domain and are
    preserved as NaN. Valid cells adjacent to NaData therefore act as
    domain boundaries/outlets, consistent with an irregular DEM domain.

    The input DEM is never modified.
    """
    z = np.asarray(dem, dtype=np.float64)

    if z.ndim != 2:
        raise ValueError("DEM must be a 2-D array.")

    rows, cols = z.shape

    if rows == 0 or cols == 0:
        return z.copy()

    filled = z.copy()
    valid = np.isfinite(z)

    if not np.any(valid):
        return filled

    closed = ~valid.copy()

    open_heap: list[tuple[float, int, int]] = []
    pit: deque[tuple[float, int, int]] = deque()

    def is_domain_boundary(r: int, c: int) -> bool:
        if (
            r == 0
            or c == 0
            or r == rows - 1
            or c == cols - 1
        ):
            return True

        for dr, dc in _NEIGHBORS_8:
            rr = r + dr
            cc = c + dc

            if not valid[rr, cc]:
                return True

        return False

    # Seed every valid hydrologic-domain boundary cell.
    for r in range(rows):
        for c in range(cols):
            if (
                valid[r, c]
                and is_domain_boundary(r, c)
            ):
                closed[r, c] = True
                heapq.heappush(
                    open_heap,
                    (float(filled[r, c]), r, c),
                )

    while open_heap or pit:

        if pit:
            elevation, r, c = pit.popleft()
        else:
            elevation, r, c = heapq.heappop(
                open_heap
            )

        for dr, dc in _NEIGHBORS_8:
            rr = r + dr
            cc = c + dc

            if (
                rr < 0
                or rr >= rows
                or cc < 0
                or cc >= cols
                or closed[rr, cc]
            ):
                continue

            closed[rr, cc] = True

            neighbor_z = float(filled[rr, cc])

            if neighbor_z <= elevation:
                filled[rr, cc] = elevation

                pit.append(
                    (
                        elevation,
                        rr,
                        cc,
                    )
                )
            else:
                heapq.heappush(
                    open_heap,
                    (
                        neighbor_z,
                        rr,
                        cc,
                    ),
                )

    filled[~valid] = np.nan

    return filled


def depression_depth(
    dem: np.ndarray,
    conditioned_dem: np.ndarray | None = None,
) -> np.ndarray:
    """Return elevation increase caused by depression filling."""
    original = np.asarray(
        dem,
        dtype=np.float64,
    )

    if conditioned_dem is None:
        conditioned = priority_flood_fill(
            original
        )
    else:
        conditioned = np.asarray(
            conditioned_dem,
            dtype=np.float64,
        )

    if conditioned.shape != original.shape:
        raise ValueError(
            "Original and conditioned DEM shapes must match."
        )

    depth = conditioned - original

    valid = (
        np.isfinite(original)
        & np.isfinite(conditioned)
    )

    depth[~valid] = np.nan

    # Numerical protection only.
    depth[
        valid
        & (np.abs(depth) < 1.0e-12)
    ] = 0.0

    if np.any(depth[valid] < -1.0e-10):
        raise RuntimeError(
            "Conditioning lowered terrain unexpectedly."
        )

    return depth


def condition_dem(
    dem: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return conditioned DEM and depression-depth raster."""
    conditioned = priority_flood_fill(dem)

    depth = depression_depth(
        dem,
        conditioned,
    )

    return conditioned, depth


# ---------------------------------------------------------------------
# D8 FLOW TOPOLOGY
#
# Direction codes:
#   0 = outlet / unresolved flat
#   1 = E
#   2 = SE
#   4 = S
#   8 = SW
#  16 = W
#  32 = NW
#  64 = N
# 128 = NE
#
# This follows the common ESRI-style D8 coding convention.
# ---------------------------------------------------------------------

_D8 = (
    ( 0,  1,   1),
    ( 1,  1,   2),
    ( 1,  0,   4),
    ( 1, -1,   8),
    ( 0, -1,  16),
    (-1, -1,  32),
    (-1,  0,  64),
    (-1,  1, 128),
)


def d8_flow_direction(
    dem: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return D8 direction codes and receiver indices.

    Flow is assigned to the neighboring cell with maximum positive
    elevation drop per unit horizontal distance.

    Parameters
    ----------
    dem
        Hydrologically conditioned DEM.
    dx, dy
        Positive raster cell dimensions in map units.

    Returns
    -------
    direction
        uint8 D8 direction raster. Zero marks outlets or cells without
        a strictly lower neighbor.
    receiver
        int64 flattened receiver index. -1 marks no receiver / outlet.
    """
    z = np.asarray(dem, dtype=np.float64)

    if z.ndim != 2:
        raise ValueError("DEM must be a 2-D array.")

    dx = abs(float(dx))
    dy = abs(float(dy))

    if (
        not np.isfinite(dx)
        or not np.isfinite(dy)
        or dx <= 0.0
        or dy <= 0.0
    ):
        raise ValueError(
            "D8 pixel dimensions must be finite and > 0."
        )

    rows, cols = z.shape
    valid = np.isfinite(z)

    direction = np.zeros(
        z.shape,
        dtype=np.uint8,
    )

    receiver = np.full(
        z.size,
        -1,
        dtype=np.int64,
    )

    for r in range(rows):
        for c in range(cols):

            if not valid[r, c]:
                continue

            center = float(z[r, c])

            best_slope = 0.0
            best_code = 0
            best_receiver = -1

            for dr, dc, code in _D8:
                rr = r + dr
                cc = c + dc

                if (
                    rr < 0
                    or rr >= rows
                    or cc < 0
                    or cc >= cols
                    or not valid[rr, cc]
                ):
                    continue

                drop = center - float(z[rr, cc])

                if drop <= 0.0:
                    continue

                distance = float(
                    np.hypot(
                        float(dc) * dx,
                        float(dr) * dy,
                    )
                )

                slope = drop / distance

                # Deterministic tie handling:
                # _D8 order above wins exact equal-slope ties.
                if slope > best_slope:
                    best_slope = slope
                    best_code = code
                    best_receiver = (
                        rr * cols + cc
                    )

            direction[r, c] = best_code
            receiver[
                r * cols + c
            ] = best_receiver

    return direction, receiver


def d8_flow_accumulation(
    dem: np.ndarray,
    dx: float,
    dy: float,
    *,
    receiver: np.ndarray | None = None,
) -> np.ndarray:
    """Return D8 upslope accumulation in number of cells.

    Each valid cell contributes one cell, including itself.
    """
    z = np.asarray(dem, dtype=np.float64)

    if z.ndim != 2:
        raise ValueError("DEM must be a 2-D array.")

    rows, cols = z.shape
    valid = np.isfinite(z)
    valid_flat = valid.ravel()

    if receiver is None:
        _, receiver = d8_flow_direction(
            z,
            dx,
            dy,
        )

    receiver = np.asarray(
        receiver,
        dtype=np.int64,
    ).reshape(-1)

    if receiver.size != z.size:
        raise ValueError(
            "Receiver array size must match DEM size."
        )

    indegree = np.zeros(
        z.size,
        dtype=np.int64,
    )

    for i in range(z.size):
        if not valid_flat[i]:
            continue

        j = int(receiver[i])

        if j >= 0:
            if (
                j >= z.size
                or not valid_flat[j]
            ):
                raise ValueError(
                    "Receiver references an invalid DEM cell."
                )

            indegree[j] += 1

    accumulation = np.zeros(
        z.size,
        dtype=np.float64,
    )

    accumulation[valid_flat] = 1.0

    queue = deque(
        int(i)
        for i in np.flatnonzero(
            valid_flat & (indegree == 0)
        )
    )

    processed = 0

    while queue:
        i = queue.popleft()
        processed += 1

        j = int(receiver[i])

        if j < 0:
            continue

        accumulation[j] += accumulation[i]

        indegree[j] -= 1

        if indegree[j] == 0:
            queue.append(j)

    n_valid = int(np.count_nonzero(valid_flat))

    if processed != n_valid:
        raise RuntimeError(
            "D8 receiver topology contains a cycle."
        )

    out = accumulation.reshape(z.shape)
    out[~valid] = np.nan

    return out


def d8_contributing_area(
    dem: np.ndarray,
    dx: float,
    dy: float,
    *,
    accumulation: np.ndarray | None = None,
) -> np.ndarray:
    """Return D8 contributing area in squared map units."""
    dx = abs(float(dx))
    dy = abs(float(dy))

    if accumulation is None:
        accumulation = d8_flow_accumulation(
            dem,
            dx,
            dy,
        )

    return (
        np.asarray(
            accumulation,
            dtype=np.float64,
        )
        * dx
        * dy
    )


def d8_specific_catchment_area(
    dem: np.ndarray,
    dx: float,
    dy: float,
    *,
    accumulation: np.ndarray | None = None,
) -> np.ndarray:
    """Return D8 specific catchment area in map-length units.

    Contributing area is divided by representative contour width.
    For rectangular cells, sqrt(dx*dy) is used as the representative
    cell width.
    """
    area = d8_contributing_area(
        dem,
        dx,
        dy,
        accumulation=accumulation,
    )

    width = float(
        np.sqrt(
            abs(float(dx) * float(dy))
        )
    )

    return area / width


def d8_hydrology(
    dem: np.ndarray,
    dx: float,
    dy: float,
) -> dict[str, np.ndarray]:
    """Build the core D8 hydrologic topology for a conditioned DEM."""
    direction, receiver = d8_flow_direction(
        dem,
        dx,
        dy,
    )

    accumulation = d8_flow_accumulation(
        dem,
        dx,
        dy,
        receiver=receiver,
    )

    area = d8_contributing_area(
        dem,
        dx,
        dy,
        accumulation=accumulation,
    )

    sca = d8_specific_catchment_area(
        dem,
        dx,
        dy,
        accumulation=accumulation,
    )

    return {
        "flow_direction": direction,
        "receiver": receiver,
        "flow_accumulation": accumulation,
        "contributing_area": area,
        "specific_catchment_area": sca,
    }


# ---------------------------------------------------------------------
# DRAINAGE NETWORK
# ---------------------------------------------------------------------

def _validate_stream_threshold(
    threshold_area: float,
) -> float:
    threshold_area = float(threshold_area)

    if (
        not np.isfinite(threshold_area)
        or threshold_area <= 0.0
    ):
        raise ValueError(
            "Stream initiation area threshold must be finite "
            "and > 0 square map units."
        )

    return threshold_area


def stream_mask_from_contributing_area(
    contributing_area: np.ndarray,
    threshold_area: float,
) -> np.ndarray:
    """Return stream cells using a physical contributing-area threshold."""
    area = np.asarray(
        contributing_area,
        dtype=np.float64,
    )

    threshold_area = _validate_stream_threshold(
        threshold_area
    )

    return (
        np.isfinite(area)
        & (area >= threshold_area)
    )


def _stream_indegree(
    stream_mask: np.ndarray,
    receiver: np.ndarray,
) -> np.ndarray:
    stream = np.asarray(
        stream_mask,
        dtype=bool,
    )

    rec = np.asarray(
        receiver,
        dtype=np.int64,
    ).reshape(-1)

    if rec.size != stream.size:
        raise ValueError(
            "Receiver array size must match stream raster."
        )

    stream_flat = stream.ravel()

    indegree = np.zeros(
        stream.size,
        dtype=np.int32,
    )

    for i in np.flatnonzero(stream_flat):
        j = int(rec[i])

        if (
            j >= 0
            and j < stream.size
            and stream_flat[j]
        ):
            indegree[j] += 1

    return indegree.reshape(stream.shape)


def stream_node_masks(
    stream_mask: np.ndarray,
    receiver: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return headwater, junction, and outlet masks."""
    stream = np.asarray(
        stream_mask,
        dtype=bool,
    )

    rec = np.asarray(
        receiver,
        dtype=np.int64,
    ).reshape(-1)

    indegree = _stream_indegree(
        stream,
        rec,
    )

    stream_flat = stream.ravel()

    headwaters = (
        stream
        & (indegree == 0)
    )

    junctions = (
        stream
        & (indegree >= 2)
    )

    outlets_flat = np.zeros(
        stream.size,
        dtype=bool,
    )

    for i in np.flatnonzero(stream_flat):
        j = int(rec[i])

        if (
            j < 0
            or j >= stream.size
            or not stream_flat[j]
        ):
            outlets_flat[i] = True

    return {
        "headwaters": headwaters,
        "junctions": junctions,
        "outlets": outlets_flat.reshape(
            stream.shape
        ),
        "stream_indegree": indegree,
    }


def strahler_stream_order(
    stream_mask: np.ndarray,
    receiver: np.ndarray,
) -> np.ndarray:
    """Return Strahler order for a D8 stream network."""
    stream = np.asarray(
        stream_mask,
        dtype=bool,
    )

    rec = np.asarray(
        receiver,
        dtype=np.int64,
    ).reshape(-1)

    if rec.size != stream.size:
        raise ValueError(
            "Receiver array size must match stream raster."
        )

    stream_flat = stream.ravel()
    indegree = _stream_indegree(
        stream,
        rec,
    ).ravel().astype(np.int64)

    order = np.zeros(
        stream.size,
        dtype=np.int32,
    )

    max_upstream_order = np.zeros(
        stream.size,
        dtype=np.int32,
    )

    max_order_count = np.zeros(
        stream.size,
        dtype=np.int32,
    )

    queue = deque(
        int(i)
        for i in np.flatnonzero(
            stream_flat & (indegree == 0)
        )
    )

    for i in queue:
        order[i] = 1

    processed = 0

    while queue:
        i = queue.popleft()
        processed += 1

        j = int(rec[i])

        if (
            j < 0
            or j >= stream.size
            or not stream_flat[j]
        ):
            continue

        oi = int(order[i])

        if oi > max_upstream_order[j]:
            max_upstream_order[j] = oi
            max_order_count[j] = 1

        elif oi == max_upstream_order[j]:
            max_order_count[j] += 1

        indegree[j] -= 1

        if indegree[j] == 0:
            upstream_max = int(
                max_upstream_order[j]
            )

            if upstream_max <= 0:
                order[j] = 1
            elif max_order_count[j] >= 2:
                order[j] = upstream_max + 1
            else:
                order[j] = upstream_max

            queue.append(j)

    n_stream = int(
        np.count_nonzero(stream_flat)
    )

    if processed != n_stream:
        raise RuntimeError(
            "Stream topology contains a cycle."
        )

    return order.reshape(stream.shape)


def stream_link_ids(
    stream_mask: np.ndarray,
    receiver: np.ndarray,
) -> np.ndarray:
    """Assign deterministic IDs to stream links in O(N).

    A link begins at each headwater and immediately downstream of
    each junction. It terminates at the next junction or stream outlet.

    Junction cells belong deterministically to one incoming link;
    the downstream continuation begins a new link.
    """
    stream = np.asarray(
        stream_mask,
        dtype=bool,
    )

    rec = np.asarray(
        receiver,
        dtype=np.int64,
    ).reshape(-1)

    if rec.size != stream.size:
        raise ValueError(
            "Receiver array size must match stream raster."
        )

    stream_flat = stream.ravel()

    indegree = _stream_indegree(
        stream,
        rec,
    ).ravel()

    links = np.zeros(
        stream.size,
        dtype=np.int32,
    )

    # O(N) start discovery:
    #   1. every headwater;
    #   2. receiver immediately downstream of every junction.
    starts: set[int] = set(
        int(i)
        for i in np.flatnonzero(
            stream_flat & (indegree == 0)
        )
    )

    for junction in np.flatnonzero(
        stream_flat & (indegree >= 2)
    ):
        downstream = int(rec[junction])

        if (
            0 <= downstream < stream.size
            and stream_flat[downstream]
        ):
            starts.add(downstream)

    link_id = 0

    for start in sorted(starts):
        if links[start] != 0:
            continue

        link_id += 1
        current = start

        while (
            0 <= current < stream.size
            and stream_flat[current]
            and links[current] == 0
        ):
            links[current] = link_id

            # A junction closes the current incoming reach.
            if (
                current != start
                and indegree[current] >= 2
            ):
                break

            nxt = int(rec[current])

            if (
                nxt < 0
                or nxt >= stream.size
                or not stream_flat[nxt]
            ):
                break

            # If this start itself is a junction, its downstream
            # continuation is intentionally represented separately.
            if (
                current == start
                and indegree[current] >= 2
            ):
                break

            current = nxt

    # Any remaining stream cell indicates an unusual but valid
    # disconnected fragment. Assign deterministically while preserving
    # complete stream coverage.
    for i in np.flatnonzero(
        stream_flat & (links == 0)
    ):
        link_id += 1
        current = int(i)

        while (
            0 <= current < stream.size
            and stream_flat[current]
            and links[current] == 0
        ):
            links[current] = link_id

            if (
                current != i
                and indegree[current] >= 2
            ):
                break

            nxt = int(rec[current])

            if (
                nxt < 0
                or nxt >= stream.size
                or not stream_flat[nxt]
            ):
                break

            current = nxt

    return links.reshape(stream.shape)




def extract_d8_stream_network(
    contributing_area: np.ndarray,
    receiver: np.ndarray,
    *,
    threshold_area: float,
) -> dict[str, np.ndarray]:
    """Extract core drainage-network products from D8 topology."""
    stream = stream_mask_from_contributing_area(
        contributing_area,
        threshold_area,
    )

    nodes = stream_node_masks(
        stream,
        receiver,
    )

    order = strahler_stream_order(
        stream,
        receiver,
    )

    links = stream_link_ids(
        stream,
        receiver,
    )

    return {
        "stream_mask": stream,
        "stream_order": order,
        "stream_link": links,
        "headwaters": nodes["headwaters"],
        "junctions": nodes["junctions"],
        "stream_outlets": nodes["outlets"],
    }


# ---------------------------------------------------------------------
# BASINS / CATCHMENTS / WATERSHEDS
# ---------------------------------------------------------------------

def d8_basin_labels(
    dem: np.ndarray,
    receiver: np.ndarray,
) -> np.ndarray:
    """Label every valid cell by its terminal D8 outlet.

    Basin IDs are deterministic positive integers. NaData is 0.
    """
    z = np.asarray(dem, dtype=np.float64)
    valid = np.isfinite(z)
    rec = np.asarray(receiver, dtype=np.int64).reshape(-1)

    if rec.size != z.size:
        raise ValueError(
            "Receiver array size must match DEM size."
        )

    valid_flat = valid.ravel()
    terminal = np.full(z.size, -2, dtype=np.int64)
    terminal[~valid_flat] = -1

    for start in np.flatnonzero(valid_flat):
        start = int(start)

        if terminal[start] >= 0:
            continue

        path = []
        seen = set()
        current = start

        while True:
            if current in seen:
                raise RuntimeError(
                    "D8 receiver topology contains a cycle."
                )

            seen.add(current)

            if terminal[current] >= 0:
                outlet = int(terminal[current])
                break

            path.append(current)
            nxt = int(rec[current])

            if nxt < 0:
                outlet = current
                break

            if (
                nxt >= z.size
                or not valid_flat[nxt]
            ):
                outlet = current
                break

            current = nxt

        for cell in path:
            terminal[cell] = outlet

    outlets = sorted(
        set(
            int(v)
            for v in terminal[valid_flat]
            if v >= 0
        )
    )

    outlet_to_id = {
        outlet: i + 1
        for i, outlet in enumerate(outlets)
    }

    labels = np.zeros(z.size, dtype=np.int32)

    for i in np.flatnonzero(valid_flat):
        labels[i] = outlet_to_id[
            int(terminal[i])
        ]

    return labels.reshape(z.shape)


def watershed_boundary_mask(
    basin_labels: np.ndarray,
) -> np.ndarray:
    """Return cells touching a different positive basin label."""
    labels = np.asarray(
        basin_labels,
        dtype=np.int32,
    )

    if labels.ndim != 2:
        raise ValueError(
            "Basin labels must be a 2-D array."
        )

    rows, cols = labels.shape
    boundary = np.zeros(
        labels.shape,
        dtype=bool,
    )

    # Use 4-neighbor boundaries so diagonal contact alone
    # does not create an artificially thick divide.
    neighbors = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    )

    for r in range(rows):
        for c in range(cols):
            center = int(labels[r, c])

            if center <= 0:
                continue

            for dr, dc in neighbors:
                rr = r + dr
                cc = c + dc

                if (
                    rr < 0
                    or rr >= rows
                    or cc < 0
                    or cc >= cols
                ):
                    continue

                other = int(labels[rr, cc])

                if (
                    other > 0
                    and other != center
                ):
                    boundary[r, c] = True
                    break

    return boundary


def upstream_mask(
    receiver: np.ndarray,
    pour_index: int,
    shape: tuple[int, int],
) -> np.ndarray:
    """Return all cells draining to a specified D8 pour cell."""
    rec = np.asarray(
        receiver,
        dtype=np.int64,
    ).reshape(-1)

    size = int(shape[0] * shape[1])

    if rec.size != size:
        raise ValueError(
            "Receiver array size does not match requested shape."
        )

    pour_index = int(pour_index)

    if pour_index < 0 or pour_index >= size:
        raise ValueError(
            "Pour-point index is outside the raster."
        )

    donors: list[list[int]] = [
        [] for _ in range(size)
    ]

    for i, j in enumerate(rec):
        j = int(j)

        if 0 <= j < size:
            donors[j].append(i)

    mask = np.zeros(size, dtype=bool)
    queue = deque([pour_index])
    mask[pour_index] = True

    while queue:
        current = queue.popleft()

        for donor in donors[current]:
            if not mask[donor]:
                mask[donor] = True
                queue.append(donor)

    return mask.reshape(shape)



def snap_pour_point(
    contributing_area: np.ndarray,
    *,
    row: int,
    col: int,
    dx: float,
    dy: float,
    radius_m: float,
    stream_mask: np.ndarray | None = None,
) -> tuple[int, int]:
    """Snap a pour point to the strongest nearby drainage cell.

    Candidate cells must lie within the requested physical Euclidean
    radius. The candidate with maximum finite contributing area wins.
    Ties are resolved by shortest physical distance and then raster
    row/column order for deterministic behavior.

    If stream_mask is supplied, candidates are restricted to stream
    cells.
    """
    area = np.asarray(
        contributing_area,
        dtype=np.float64,
    )

    if area.ndim != 2:
        raise ValueError(
            "Contributing area must be a 2-D array."
        )

    rows, cols = area.shape
    row = int(row)
    col = int(col)

    if (
        row < 0
        or row >= rows
        or col < 0
        or col >= cols
    ):
        raise ValueError(
            "Pour point lies outside the raster."
        )

    dx = abs(float(dx))
    dy = abs(float(dy))
    radius_m = float(radius_m)

    if (
        not np.isfinite(dx)
        or not np.isfinite(dy)
        or dx <= 0.0
        or dy <= 0.0
    ):
        raise ValueError(
            "Pixel dimensions must be finite and > 0."
        )

    if (
        not np.isfinite(radius_m)
        or radius_m < 0.0
    ):
        raise ValueError(
            "Pour-point snap radius must be finite and >= 0."
        )

    if stream_mask is not None:
        stream = np.asarray(
            stream_mask,
            dtype=bool,
        )

        if stream.shape != area.shape:
            raise ValueError(
                "Stream mask must match contributing-area shape."
            )
    else:
        stream = None

    rr = int(np.floor(radius_m / dy))
    cc = int(np.floor(radius_m / dx))

    r0 = max(0, row - rr)
    r1 = min(rows - 1, row + rr)
    c0 = max(0, col - cc)
    c1 = min(cols - 1, col + cc)

    candidates = []

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            distance = float(
                np.hypot(
                    (c - col) * dx,
                    (r - row) * dy,
                )
            )

            if distance > radius_m + 1.0e-12:
                continue

            value = float(area[r, c])

            if not np.isfinite(value):
                continue

            if stream is not None and not stream[r, c]:
                continue

            candidates.append(
                (
                    -value,
                    distance,
                    r,
                    c,
                )
            )

    if not candidates:
        raise ValueError(
            "No valid pour-point snap candidate exists "
            "within the requested radius."
        )

    candidates.sort()

    _, _, snapped_row, snapped_col = candidates[0]

    return int(snapped_row), int(snapped_col)


def delineate_watershed_snapped(
    dem: np.ndarray,
    receiver: np.ndarray,
    contributing_area: np.ndarray,
    *,
    pour_row: int,
    pour_col: int,
    dx: float,
    dy: float,
    snap_radius_m: float,
    stream_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Snap a pour point and delineate its D8 upstream watershed."""
    snapped = snap_pour_point(
        contributing_area,
        row=pour_row,
        col=pour_col,
        dx=dx,
        dy=dy,
        radius_m=snap_radius_m,
        stream_mask=stream_mask,
    )

    mask = delineate_watershed(
        dem,
        receiver,
        pour_row=snapped[0],
        pour_col=snapped[1],
    )

    return mask, snapped


def delineate_watershed(
    dem: np.ndarray,
    receiver: np.ndarray,
    *,
    pour_row: int,
    pour_col: int,
) -> np.ndarray:
    """Delineate the upstream watershed of one raster pour point."""
    z = np.asarray(dem, dtype=np.float64)

    if z.ndim != 2:
        raise ValueError("DEM must be a 2-D array.")

    rows, cols = z.shape

    pour_row = int(pour_row)
    pour_col = int(pour_col)

    if (
        pour_row < 0
        or pour_row >= rows
        or pour_col < 0
        or pour_col >= cols
    ):
        raise ValueError(
            "Pour point lies outside the DEM."
        )

    if not np.isfinite(
        z[pour_row, pour_col]
    ):
        raise ValueError(
            "Pour point lies on NoData."
        )

    pour_index = (
        pour_row * cols + pour_col
    )

    mask = upstream_mask(
        receiver,
        pour_index,
        z.shape,
    )

    mask[~np.isfinite(z)] = False

    return mask


def subcatchment_labels(
    stream_mask: np.ndarray,
    receiver: np.ndarray,
) -> np.ndarray:
    """Assign each cell to its first downstream stream cell.

    Non-contributing/unresolved cells receive zero.
    Stream cells identify their own local subcatchment.
    """
    stream = np.asarray(
        stream_mask,
        dtype=bool,
    )

    rec = np.asarray(
        receiver,
        dtype=np.int64,
    ).reshape(-1)

    if rec.size != stream.size:
        raise ValueError(
            "Receiver array size must match stream raster."
        )

    stream_flat = stream.ravel()

    stream_cells = np.flatnonzero(
        stream_flat
    )

    stream_id = {
        int(cell): i + 1
        for i, cell in enumerate(stream_cells)
    }

    labels = np.zeros(
        stream.size,
        dtype=np.int32,
    )

    for start in range(stream.size):
        current = start
        path = []
        seen = set()
        label = 0

        while current >= 0:
            if current in seen:
                raise RuntimeError(
                    "D8 receiver topology contains a cycle."
                )

            seen.add(current)

            if labels[current] > 0:
                label = int(labels[current])
                break

            if stream_flat[current]:
                label = stream_id[current]
                break

            path.append(current)

            nxt = int(rec[current])

            if nxt < 0 or nxt >= stream.size:
                break

            current = nxt

        if label > 0:
            if stream_flat[current]:
                labels[current] = label

            for cell in path:
                labels[cell] = label

    return labels.reshape(stream.shape)


def d8_watershed_products(
    dem: np.ndarray,
    receiver: np.ndarray,
    *,
    stream_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build basin/divide and optional stream-subcatchment products."""
    basins = d8_basin_labels(
        dem,
        receiver,
    )

    products = {
        "basin_labels": basins,
        "watershed_boundaries":
            watershed_boundary_mask(basins),
    }

    if stream_mask is not None:
        products["subcatchment_labels"] = (
            subcatchment_labels(
                stream_mask,
                receiver,
            )
        )

    return products


# ---------------------------------------------------------------------
# D8 FLAT RESOLUTION
# ---------------------------------------------------------------------

def resolve_d8_flats(
    dem: np.ndarray,
    receiver: np.ndarray,
) -> np.ndarray:
    """Resolve drainable D8 flats using a Barnes-style flat mask.

    The construction follows Barnes, Lehman & Mulla (2014):

    1. identify unresolved equal-elevation flat components;
    2. identify high edges adjacent to higher terrain;
    3. identify drainage edges adjoining an already resolved
       equal-elevation cell or an open hydrologic-domain boundary;
    4. construct a gradient away from higher terrain;
    5. superimpose a stronger gradient toward drainage edges;
    6. route unresolved cells down the resulting integer mask.

    Existing strict-downslope receivers are preserved.

    A flat without a legitimate drainage edge is left unresolved.
    No artificial outlet is manufactured for a closed interior flat.

    FAST-GC treats the outer DEM boundary and valid cells adjacent to
    NoData as open hydrologic-domain boundaries, consistent with
    priority_flood_fill().
    """
    z = np.asarray(dem, dtype=np.float64)

    if z.ndim != 2:
        raise ValueError("DEM must be a 2-D array.")

    rows, cols = z.shape
    size = z.size

    valid = np.isfinite(z)
    valid_flat = valid.ravel()
    zflat = z.ravel()

    rec = np.asarray(
        receiver,
        dtype=np.int64,
    ).reshape(-1).copy()

    if rec.size != size:
        raise ValueError(
            "Receiver array size must match DEM size."
        )

    unresolved = valid_flat & (rec < 0)
    visited = np.zeros(size, dtype=bool)

    def neighbors(i: int):
        r, c = divmod(i, cols)

        for dr, dc in _NEIGHBORS_8:
            rr = r + dr
            cc = c + dc

            if (
                rr < 0
                or rr >= rows
                or cc < 0
                or cc >= cols
            ):
                continue

            yield rr * cols + cc

    def is_open_boundary(i: int) -> bool:
        r, c = divmod(i, cols)

        if (
            r == 0
            or c == 0
            or r == rows - 1
            or c == cols - 1
        ):
            return True

        for j in neighbors(i):
            if not valid_flat[j]:
                return True

        return False

    for seed in np.flatnonzero(unresolved):
        seed = int(seed)

        if visited[seed]:
            continue

        elevation = float(zflat[seed])

        # ----------------------------------------------------------
        # Identify this equal-elevation unresolved flat component.
        # ----------------------------------------------------------
        component = []
        q = deque([seed])
        visited[seed] = True

        while q:
            i = q.popleft()
            component.append(i)

            for j in neighbors(i):
                if (
                    visited[j]
                    or not unresolved[j]
                    or not np.isclose(
                        float(zflat[j]),
                        elevation,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                ):
                    continue

                visited[j] = True
                q.append(j)

        component_set = set(component)

        high_edges = []
        low_edges = []
        low_targets: dict[int, int] = {}

        # ----------------------------------------------------------
        # Barnes-style high and low edge identification.
        # ----------------------------------------------------------
        for i in component:
            has_higher = False

            for j in neighbors(i):
                if not valid_flat[j]:
                    continue

                zj = float(zflat[j])

                if zj > elevation + 1.0e-12:
                    has_higher = True

                # An equal-elevation neighbour outside this unresolved
                # component which already has a receiver provides a
                # legitimate drainage edge.
                if (
                    j not in component_set
                    and rec[j] >= 0
                    and np.isclose(
                        zj,
                        elevation,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                ):
                    low_edges.append(i)
                    low_targets.setdefault(i, j)

            if has_higher:
                high_edges.append(i)

            # FAST-GC open-domain convention.
            if is_open_boundary(i):
                low_edges.append(i)
                low_targets.setdefault(i, -1)

        high_edges = sorted(set(high_edges))
        low_edges = sorted(set(low_edges))

        # Barnes: an undrainable flat must remain unresolved.
        if not low_edges:
            continue

        # ----------------------------------------------------------
        # Gradient away from higher terrain.
        #
        # Distance starts at 1 at the high edge. Cells farther from
        # higher terrain receive larger values. The Barnes mask then
        # inverts this contribution.
        # ----------------------------------------------------------
        away_distance = {
            i: 0
            for i in component
        }

        if high_edges:
            q = deque(high_edges)

            for i in high_edges:
                away_distance[i] = 1

            while q:
                i = q.popleft()
                d = away_distance[i]

                for j in neighbors(i):
                    if (
                        j not in component_set
                        or away_distance[j] != 0
                    ):
                        continue

                    away_distance[j] = d + 1
                    q.append(j)

            flat_height = max(
                away_distance.values()
            )
        else:
            flat_height = 0

        # Barnes first contribution:
        # FlatHeight - away_distance.
        mask = {}

        for i in component:
            if flat_height > 0:
                mask[i] = (
                    flat_height
                    - away_distance[i]
                )
            else:
                mask[i] = 0

        # ----------------------------------------------------------
        # Stronger gradient toward lower/drainage terrain.
        #
        # Barnes uses 2 * distance so drainage-to-low dominates
        # the away-from-high contribution.
        # ----------------------------------------------------------
        toward_distance = {
            i: -1
            for i in component
        }

        q = deque()

        for i in low_edges:
            toward_distance[i] = 1
            q.append(i)

        while q:
            i = q.popleft()
            d = toward_distance[i]

            for j in neighbors(i):
                if (
                    j not in component_set
                    or toward_distance[j] >= 0
                ):
                    continue

                toward_distance[j] = d + 1
                q.append(j)

        for i in component:
            d = toward_distance[i]

            if d < 0:
                # Defensive: drainable component should have been
                # reached from its low edges.
                continue

            mask[i] += 2 * d

        # ----------------------------------------------------------
        # Drainage-edge cells retain their real target/outlet.
        # ----------------------------------------------------------
        for i in low_edges:
            rec[i] = low_targets[i]

        # ----------------------------------------------------------
        # Route unresolved interior flat cells to a D8 neighbour
        # having a strictly lower Barnes mask.
        #
        # Deterministic tie-breaking follows _NEIGHBORS_8 order.
        # ----------------------------------------------------------
        low_edge_set = set(low_edges)

        for i in component:
            if i in low_edge_set:
                continue

            best = -1
            best_mask = mask[i]

            for j in neighbors(i):
                if j not in component_set:
                    continue

                candidate = mask[j]

                if candidate < best_mask:
                    best_mask = candidate
                    best = j

            if best >= 0:
                rec[i] = best

    return rec




def d8_flow_direction_resolved(
    dem: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return D8 directions with conditioned-flat routing resolved."""
    _, receiver = d8_flow_direction(
        dem,
        dx,
        dy,
    )

    receiver = resolve_d8_flats(
        dem,
        receiver,
    )

    rows, cols = np.asarray(dem).shape

    code_lookup = {
        (dr, dc): code
        for dr, dc, code in _D8
    }

    direction = np.zeros(
        (rows, cols),
        dtype=np.uint8,
    )

    for i, j in enumerate(receiver):
        if j < 0:
            continue

        r, c = divmod(i, cols)
        rr, cc = divmod(int(j), cols)

        direction[r, c] = code_lookup[
            (rr - r, cc - c)
        ]

    return direction, receiver


# ---------------------------------------------------------------------
# MULTIPLE FLOW DIRECTION (MFD)
# ---------------------------------------------------------------------

def mfd_flow_weights(
    dem: np.ndarray,
    dx: float,
    dy: float,
    *,
    exponent: float = 1.1,
) -> np.ndarray:
    """Return Freeman-style MFD weights to 8 neighboring cells.

    Downslope flow is partitioned in proportion to slope**exponent.
    Neighbor ordering follows _D8.
    """
    z = np.asarray(dem, dtype=np.float64)

    if z.ndim != 2:
        raise ValueError("DEM must be a 2-D array.")

    dx = abs(float(dx))
    dy = abs(float(dy))
    exponent = float(exponent)

    if dx <= 0.0 or dy <= 0.0:
        raise ValueError(
            "Pixel dimensions must be > 0."
        )

    if (
        not np.isfinite(exponent)
        or exponent <= 0.0
    ):
        raise ValueError(
            "MFD exponent must be finite and > 0."
        )

    rows, cols = z.shape
    valid = np.isfinite(z)

    weights = np.zeros(
        (rows, cols, 8),
        dtype=np.float64,
    )

    for r in range(rows):
        for c in range(cols):

            if not valid[r, c]:
                continue

            center = float(z[r, c])
            raw = np.zeros(8, dtype=np.float64)

            for k, (dr, dc, _) in enumerate(_D8):
                rr = r + dr
                cc = c + dc

                if (
                    rr < 0
                    or rr >= rows
                    or cc < 0
                    or cc >= cols
                    or not valid[rr, cc]
                ):
                    continue

                drop = center - float(z[rr, cc])

                if drop <= 0.0:
                    continue

                distance = float(
                    np.hypot(
                        dc * dx,
                        dr * dy,
                    )
                )

                slope = drop / distance
                raw[k] = slope ** exponent

            total = float(np.sum(raw))

            if total > 0.0:
                weights[r, c, :] = raw / total

    # Freeman MFD is defined from positive downslope gradients.
    # Depression filling can, however, create mathematically flat
    # surfaces. Cells on such flats would otherwise retain zero
    # outflow and become artificial sinks.
    #
    # Preserve Freeman partitioning wherever a genuine downslope
    # gradient exists. Only zero-outflow cells receive a deterministic
    # unit-weight fallback along the already resolved, acyclic D8
    # flat-drainage graph.
    _, flat_receiver = d8_flow_direction_resolved(
        z,
        dx,
        dy,
    )

    for i in np.flatnonzero(valid.ravel()):
        r, c = divmod(int(i), cols)

        if float(np.sum(weights[r, c, :])) > 0.0:
            continue

        j = int(flat_receiver[int(i)])

        if j < 0:
            continue

        rr, cc = divmod(j, cols)
        step = (rr - r, cc - c)

        for k, (dr, dc, _) in enumerate(_D8):
            if (dr, dc) == step:
                weights[r, c, k] = 1.0
                break

    return weights


def mfd_flow_accumulation(
    dem: np.ndarray,
    dx: float,
    dy: float,
    *,
    exponent: float = 1.1,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return MFD accumulation in equivalent contributing cells."""
    z = np.asarray(dem, dtype=np.float64)

    if z.ndim != 2:
        raise ValueError("DEM must be a 2-D array.")

    rows, cols = z.shape
    valid = np.isfinite(z)

    if weights is None:
        weights = mfd_flow_weights(
            z,
            dx,
            dy,
            exponent=exponent,
        )

    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    if weights.shape != (rows, cols, 8):
        raise ValueError(
            "MFD weights must have shape (rows, cols, 8)."
        )

    accumulation = np.zeros(
        z.shape,
        dtype=np.float64,
    )

    accumulation[valid] = 1.0

    # Build the actual directed MFD graph and process it in
    # topological order.
    #
    # Elevation sorting alone is insufficient because conditioned
    # flats can contain equal-elevation routing edges supplied by the
    # deterministic D8 flat resolver. Kahn ordering guarantees that
    # every upstream contribution reaches a cell before that cell
    # distributes its accumulated flow.
    indegree = np.zeros(
        z.size,
        dtype=np.int64,
    )

    valid_flat = valid.ravel()

    for flat_i in np.flatnonzero(valid_flat):
        r, c = divmod(int(flat_i), cols)

        for k, (dr, dc, _) in enumerate(_D8):
            if float(weights[r, c, k]) <= 0.0:
                continue

            rr = r + dr
            cc = c + dc

            if (
                rr < 0
                or rr >= rows
                or cc < 0
                or cc >= cols
                or not valid[rr, cc]
            ):
                continue

            indegree[rr * cols + cc] += 1

    ready = deque(
        int(i)
        for i in np.flatnonzero(
            valid_flat & (indegree == 0)
        )
    )

    processed = 0

    while ready:
        flat_i = ready.popleft()
        processed += 1

        r, c = divmod(flat_i, cols)
        source = float(accumulation[r, c])

        for k, (dr, dc, _) in enumerate(_D8):
            weight = float(weights[r, c, k])

            if weight <= 0.0:
                continue

            rr = r + dr
            cc = c + dc

            if (
                rr < 0
                or rr >= rows
                or cc < 0
                or cc >= cols
                or not valid[rr, cc]
            ):
                continue

            accumulation[rr, cc] += (
                source * weight
            )

            j = rr * cols + cc
            indegree[j] -= 1

            if indegree[j] == 0:
                ready.append(j)

    expected = int(np.count_nonzero(valid))

    if processed != expected:
        raise RuntimeError(
            "MFD routing graph contains a cycle or "
            "inconsistent routing weights: "
            f"processed {processed} of {expected} valid cells."
        )

    accumulation[~valid] = np.nan

    return accumulation


def mfd_contributing_area(
    dem: np.ndarray,
    dx: float,
    dy: float,
    *,
    exponent: float = 1.1,
    accumulation: np.ndarray | None = None,
) -> np.ndarray:
    """Return MFD contributing area in squared map units."""
    if accumulation is None:
        accumulation = mfd_flow_accumulation(
            dem,
            dx,
            dy,
            exponent=exponent,
        )

    return (
        np.asarray(accumulation, dtype=np.float64)
        * abs(float(dx))
        * abs(float(dy))
    )


def mfd_specific_catchment_area(
    dem: np.ndarray,
    dx: float,
    dy: float,
    *,
    exponent: float = 1.1,
    accumulation: np.ndarray | None = None,
) -> np.ndarray:
    """Return MFD specific catchment area in map-length units."""
    area = mfd_contributing_area(
        dem,
        dx,
        dy,
        exponent=exponent,
        accumulation=accumulation,
    )

    width = float(
        np.sqrt(
            abs(float(dx) * float(dy))
        )
    )

    return area / width


def mfd_hydrology(
    dem: np.ndarray,
    dx: float,
    dy: float,
    *,
    exponent: float = 1.1,
) -> dict[str, np.ndarray]:
    """Build core Freeman-style MFD hydrology products."""
    weights = mfd_flow_weights(
        dem,
        dx,
        dy,
        exponent=exponent,
    )

    accumulation = mfd_flow_accumulation(
        dem,
        dx,
        dy,
        exponent=exponent,
        weights=weights,
    )

    area = mfd_contributing_area(
        dem,
        dx,
        dy,
        exponent=exponent,
        accumulation=accumulation,
    )

    sca = mfd_specific_catchment_area(
        dem,
        dx,
        dy,
        exponent=exponent,
        accumulation=accumulation,
    )

    return {
        "flow_weights": weights,
        "flow_accumulation": accumulation,
        "contributing_area": area,
        "specific_catchment_area": sca,
    }


# ---------------------------------------------------------------------
# HYDROLOGIC TERRAIN INDICES AND FLOW-PATH LENGTH
# ---------------------------------------------------------------------

def topographic_wetness_index(
    specific_catchment_area: np.ndarray,
    slope_radians: np.ndarray,
    *,
    min_slope_radians: float = 1.0e-6,
) -> np.ndarray:
    """Return TWI = ln(a / tan(beta)).

    a is specific catchment area in map-length units.
    beta is local slope angle in radians.
    """
    a = np.asarray(
        specific_catchment_area,
        dtype=np.float64,
    )

    beta = np.asarray(
        slope_radians,
        dtype=np.float64,
    )

    if a.shape != beta.shape:
        raise ValueError(
            "Specific catchment area and slope must have "
            "the same shape."
        )

    min_slope_radians = float(min_slope_radians)

    if (
        not np.isfinite(min_slope_radians)
        or min_slope_radians <= 0.0
    ):
        raise ValueError(
            "Minimum slope angle must be finite and > 0."
        )

    out = np.full(
        a.shape,
        np.nan,
        dtype=np.float64,
    )

    finite_negative_slope = (
        np.isfinite(beta)
        & (beta < 0.0)
    )

    if np.any(finite_negative_slope):
        raise ValueError(
            "Slope angle must be >= 0 radians."
        )

    valid = (
        np.isfinite(a)
        & np.isfinite(beta)
        & (a > 0.0)
    )

    beta_safe = np.maximum(
        beta[valid],
        min_slope_radians,
    )

    tan_beta = np.tan(beta_safe)

    out[valid] = np.log(
        a[valid] / tan_beta
    )

    return out


def stream_power_index(
    specific_catchment_area: np.ndarray,
    slope_radians: np.ndarray,
) -> np.ndarray:
    """Return SPI = a * tan(beta)."""
    a = np.asarray(
        specific_catchment_area,
        dtype=np.float64,
    )

    beta = np.asarray(
        slope_radians,
        dtype=np.float64,
    )

    if a.shape != beta.shape:
        raise ValueError(
            "Specific catchment area and slope must have "
            "the same shape."
        )

    out = np.full(
        a.shape,
        np.nan,
        dtype=np.float64,
    )

    finite_negative_slope = (
        np.isfinite(beta)
        & (beta < 0.0)
    )

    if np.any(finite_negative_slope):
        raise ValueError(
            "Slope angle must be >= 0 radians."
        )

    valid = (
        np.isfinite(a)
        & np.isfinite(beta)
        & (a >= 0.0)
    )

    out[valid] = (
        a[valid]
        * np.tan(beta[valid])
    )

    return out


def _receiver_step_length(
    i: int,
    j: int,
    cols: int,
    dx: float,
    dy: float,
) -> float:
    r, c = divmod(int(i), cols)
    rr, cc = divmod(int(j), cols)

    dr = abs(rr - r)
    dc = abs(cc - c)

    if dr > 1 or dc > 1 or (dr == 0 and dc == 0):
        raise ValueError(
            "D8 receiver must be one of the 8 adjacent cells."
        )

    return float(
        np.hypot(
            dc * abs(float(dx)),
            dr * abs(float(dy)),
        )
    )


def d8_downslope_flow_length(
    dem: np.ndarray,
    receiver: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Return distance from each valid cell to its terminal outlet."""
    z = np.asarray(dem, dtype=np.float64)

    if z.ndim != 2:
        raise ValueError("DEM must be a 2-D array.")

    rows, cols = z.shape
    valid = np.isfinite(z).ravel()

    rec = np.asarray(
        receiver,
        dtype=np.int64,
    ).reshape(-1)

    if rec.size != z.size:
        raise ValueError(
            "Receiver array size must match DEM size."
        )

    length = np.full(
        z.size,
        np.nan,
        dtype=np.float64,
    )

    state = np.zeros(
        z.size,
        dtype=np.uint8,
    )

    def solve(start: int) -> float:
        if not valid[start]:
            return np.nan

        if state[start] == 2:
            return float(length[start])

        path = []
        current = start

        while True:
            if state[current] == 1:
                raise RuntimeError(
                    "D8 receiver topology contains a cycle."
                )

            if state[current] == 2:
                base = float(length[current])
                break

            state[current] = 1
            path.append(current)

            nxt = int(rec[current])

            if nxt < 0:
                base = 0.0
                break

            if (
                nxt >= z.size
                or not valid[nxt]
            ):
                base = 0.0
                break

            current = nxt

        while path:
            cell = path.pop()
            nxt = int(rec[cell])

            if (
                nxt >= 0
                and nxt < z.size
                and valid[nxt]
            ):
                base += _receiver_step_length(
                    cell,
                    nxt,
                    cols,
                    dx,
                    dy,
                )

            length[cell] = base
            state[cell] = 2

        return float(length[start])

    for i in np.flatnonzero(valid):
        solve(int(i))

    return length.reshape(z.shape)


def d8_longest_upslope_flow_length(
    dem: np.ndarray,
    receiver: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Return longest D8 flow-path distance reaching each cell."""
    z = np.asarray(dem, dtype=np.float64)

    if z.ndim != 2:
        raise ValueError("DEM must be a 2-D array.")

    rows, cols = z.shape
    valid = np.isfinite(z).ravel()

    rec = np.asarray(
        receiver,
        dtype=np.int64,
    ).reshape(-1)

    if rec.size != z.size:
        raise ValueError(
            "Receiver array size must match DEM size."
        )

    indegree = np.zeros(
        z.size,
        dtype=np.int64,
    )

    for i in np.flatnonzero(valid):
        j = int(rec[i])

        if (
            j >= 0
            and j < z.size
            and valid[j]
        ):
            indegree[j] += 1

    longest = np.zeros(
        z.size,
        dtype=np.float64,
    )

    queue = deque(
        int(i)
        for i in np.flatnonzero(
            valid & (indegree == 0)
        )
    )

    processed = 0

    while queue:
        i = queue.popleft()
        processed += 1

        j = int(rec[i])

        if (
            j < 0
            or j >= z.size
            or not valid[j]
        ):
            continue

        candidate = (
            longest[i]
            + _receiver_step_length(
                i,
                j,
                cols,
                dx,
                dy,
            )
        )

        if candidate > longest[j]:
            longest[j] = candidate

        indegree[j] -= 1

        if indegree[j] == 0:
            queue.append(j)

    if processed != int(np.count_nonzero(valid)):
        raise RuntimeError(
            "D8 receiver topology contains a cycle."
        )

    longest[~valid] = np.nan

    return longest.reshape(z.shape)


# ---------------------------------------------------------------------
# RUSLE TOPOGRAPHIC FACTORS
#
# These are topographic factors only. They are NOT soil-loss estimates.
# ---------------------------------------------------------------------

def rusle_s_factor(
    slope_radians: np.ndarray,
) -> np.ndarray:
    """Return RUSLE slope-steepness factor S.

    Uses the McCool/RUSLE piecewise relationship:
      S = 10.8 sin(beta) + 0.03   for slope < 9%
      S = 16.8 sin(beta) - 0.50   for slope >= 9%

    beta is slope angle in radians.
    """
    beta = np.asarray(
        slope_radians,
        dtype=np.float64,
    )

    out = np.full(
        beta.shape,
        np.nan,
        dtype=np.float64,
    )

    finite_negative_slope = (
        np.isfinite(beta)
        & (beta < 0.0)
    )

    if np.any(finite_negative_slope):
        raise ValueError(
            "Slope angle must be >= 0 radians."
        )

    valid = np.isfinite(beta)

    if not np.any(valid):
        return out

    sin_beta = np.sin(beta[valid])
    slope_fraction = np.tan(beta[valid])

    values = np.where(
        slope_fraction < 0.09,
        10.8 * sin_beta + 0.03,
        16.8 * sin_beta - 0.50,
    )

    # Numerical protection for effectively horizontal terrain.
    values = np.maximum(values, 0.0)

    out[valid] = values
    return out


def contributing_area_ls_factor(
    specific_catchment_area: np.ndarray,
    slope_radians: np.ndarray,
    *,
    m: float = 0.4,
    n: float = 1.3,
    reference_length: float = 22.13,
    reference_sine: float = 0.0896,
) -> np.ndarray:
    """Return contributing-area LS topographic factor.

    Form:
        LS = (a / 22.13)^m
             * (sin(beta) / 0.0896)^n

    where:
      a    = specific catchment area [length]
      beta = slope angle [radians]

    This is the Moore/Burch-style distributed contributing-area
    topographic formulation. It is not the complete RUSLE soil-loss
    equation.

    Note that this implementation intentionally does not include the
    additional (m + 1) multiplier used in the related Mitasova /
    RUSLE3D formulation. The two formulations must therefore not be
    treated as numerically identical.
    """
    a = np.asarray(
        specific_catchment_area,
        dtype=np.float64,
    )

    beta = np.asarray(
        slope_radians,
        dtype=np.float64,
    )

    if a.shape != beta.shape:
        raise ValueError(
            "Specific catchment area and slope must have "
            "the same shape."
        )

    m = float(m)
    n = float(n)
    reference_length = float(reference_length)
    reference_sine = float(reference_sine)

    for name, value in (
        ("m", m),
        ("n", n),
        ("reference_length", reference_length),
        ("reference_sine", reference_sine),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{name} must be finite and > 0."
            )

    out = np.full(
        a.shape,
        np.nan,
        dtype=np.float64,
    )

    finite_negative_slope = (
        np.isfinite(beta)
        & (beta < 0.0)
    )

    if np.any(finite_negative_slope):
        raise ValueError(
            "Slope angle must be >= 0 radians."
        )

    valid = (
        np.isfinite(a)
        & np.isfinite(beta)
        & (a >= 0.0)
    )

    if not np.any(valid):
        return out

    area_term = (
        a[valid] / reference_length
    ) ** m

    slope_term = (
        np.maximum(
            np.sin(beta[valid]),
            0.0,
        )
        / reference_sine
    ) ** n

    out[valid] = area_term * slope_term

    return out


def topographic_erosion_factors(
    specific_catchment_area: np.ndarray,
    slope_radians: np.ndarray,
    *,
    ls_m: float = 0.4,
    ls_n: float = 1.3,
) -> dict[str, np.ndarray]:
    """Return explicitly named topographic erosion factors."""
    return {
        "rusle_s_factor": rusle_s_factor(
            slope_radians
        ),
        "contributing_area_ls_factor":
            contributing_area_ls_factor(
                specific_catchment_area,
                slope_radians,
                m=ls_m,
                n=ls_n,
            ),
    }
