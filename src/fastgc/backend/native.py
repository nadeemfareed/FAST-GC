"""Optional exact native kernels for FAST-GC.

Scientific behavior remains defined by the Python reference
implementation. Native kernels are execution accelerators only.
"""

from __future__ import annotations

import numpy as np

try:
    from fastgc import _fastgc_native
except Exception:
    try:
        import _fastgc_native
    except Exception:
        _fastgc_native = None


def native_available() -> bool:
    return _fastgc_native is not None


def lower_scaffold_native(
    x,
    y,
    z,
    ids,
    cell_m,
):
    """Validated exact native implementation of FAST-GC lower scaffold."""

    if _fastgc_native is None:
        raise RuntimeError(
            "FAST-GC native extension is unavailable"
        )

    ids = np.asarray(ids, dtype=np.int64)

    if ids.size == 0:
        return ids.copy()

    return np.asarray(
        _fastgc_native.lower_scaffold(
            np.ascontiguousarray(
                ids,
                dtype=np.int64,
            ),
            np.ascontiguousarray(
                np.asarray(x)[ids],
                dtype=np.float64,
            ),
            np.ascontiguousarray(
                np.asarray(y)[ids],
                dtype=np.float64,
            ),
            np.ascontiguousarray(
                np.asarray(z)[ids],
                dtype=np.float64,
            ),
            float(cell_m),
        ),
        dtype=np.int64,
    )


def surface_plane_iteration_native(
    xx,
    yy,
    z,
    keep,
    coef,
    asymmetric_high=False,
):
    """Validated exact native robust-plane iteration kernel."""

    if _fastgc_native is None:
        raise RuntimeError(
            "FAST-GC native extension is unavailable"
        )

    if not hasattr(_fastgc_native, "surface_plane_iteration"):
        raise RuntimeError(
            "FAST-GC native surface-plane kernel is unavailable"
        )

    finite = (
        np.isfinite(xx)
        & np.isfinite(yy)
        & np.isfinite(z)
    )

    return np.asarray(
        _fastgc_native.surface_plane_iteration(
            np.ascontiguousarray(xx, dtype=np.float64),
            np.ascontiguousarray(yy, dtype=np.float64),
            np.ascontiguousarray(z, dtype=np.float64),
            np.ascontiguousarray(finite, dtype=np.bool_),
            np.ascontiguousarray(keep, dtype=np.bool_),
            np.ascontiguousarray(coef, dtype=np.float64),
            bool(asymmetric_high),
        ),
        dtype=np.bool_,
    )


def tls_final_plane_iteration_native(
    px,
    py,
    pz,
    keep,
    beta,
    cx,
    cy,
    robust_z,
    mad_floor,
):
    """
    Exact native iteration primitive for the TLS final
    ground-vote robust plane.

    np.linalg.lstsq remains in the Python reference path.
    """

    if _fastgc_native is None:
        raise RuntimeError(
            "FAST-GC native extension is unavailable"
        )

    if not hasattr(
        _fastgc_native,
        "tls_final_plane_iteration",
    ):
        raise RuntimeError(
            "FAST-GC native TLS final-plane kernel "
            "is unavailable"
        )

    return np.asarray(
        _fastgc_native.tls_final_plane_iteration(
            np.ascontiguousarray(
                px,
                dtype=np.float64,
            ),
            np.ascontiguousarray(
                py,
                dtype=np.float64,
            ),
            np.ascontiguousarray(
                pz,
                dtype=np.float64,
            ),
            np.ascontiguousarray(
                keep,
                dtype=np.bool_,
            ),
            np.ascontiguousarray(
                beta,
                dtype=np.float64,
            ),
            float(cx),
            float(cy),
            float(robust_z),
            float(mad_floor),
        ),
        dtype=np.bool_,
    )


def uls_terrain_plane_iteration_native(
    xx,
    yy,
    z,
    finite,
    keep,
    coef,
):
    """Exact native ULS terrain robust-plane iteration kernel."""

    if _fastgc_native is None:
        raise RuntimeError(
            "FAST-GC native extension is unavailable"
        )

    if not hasattr(
        _fastgc_native,
        "uls_terrain_plane_iteration",
    ):
        raise RuntimeError(
            "FAST-GC native ULS terrain-plane kernel is unavailable"
        )

    return np.asarray(
        _fastgc_native.uls_terrain_plane_iteration(
            np.ascontiguousarray(xx, dtype=np.float64),
            np.ascontiguousarray(yy, dtype=np.float64),
            np.ascontiguousarray(z, dtype=np.float64),
            np.ascontiguousarray(finite, dtype=np.bool_),
            np.ascontiguousarray(keep, dtype=np.bool_),
            np.ascontiguousarray(coef, dtype=np.float64),
        ),
        dtype=np.bool_,
    )



def airborne_prepare_support_native(
    x,
    y,
    z,
    ground,
    nbr,
    point_id,
    exclusion_radius_m,
    scaffold_cell_m,
    min_support_points,
    min_support_sectors,
):
    """Prepare exact ULS hard-guard support."""

    if _fastgc_native is None:
        raise RuntimeError(
            "FAST-GC native extension is unavailable"
        )

    if not hasattr(
        _fastgc_native,
        "airborne_prepare_support",
    ):
        raise RuntimeError(
            "FAST-GC native ULS hard-guard support preparation is unavailable"
        )

    ids, sectors = _fastgc_native.airborne_prepare_support(
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(z, dtype=np.float64),
        np.ascontiguousarray(ground, dtype=np.bool_),
        np.ascontiguousarray(nbr, dtype=np.int64),
        int(point_id),
        float(exclusion_radius_m),
        float(scaffold_cell_m),
        int(min_support_points),
        int(min_support_sectors),
    )

    return (
        np.asarray(ids, dtype=np.int64),
        int(sectors),
    )


def airborne_prepare_support_multi_native(
    x,
    y,
    z,
    ground,
    nbr_max,
    point_id,
    radii,
    exclusion_radius_m,
    scaffold_cell_m,
    min_support_points,
    min_support_sectors,
):
    """Prepare exact multi-radius ULS support."""

    if _fastgc_native is None:
        raise RuntimeError(
            "FAST-GC native extension is unavailable"
        )

    if not hasattr(
        _fastgc_native,
        "airborne_prepare_support_multi",
    ):
        raise RuntimeError(
            "FAST-GC native multi-radius ULS support preparation is unavailable"
        )

    raw = _fastgc_native.airborne_prepare_support_multi(
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(z, dtype=np.float64),
        np.ascontiguousarray(ground, dtype=np.bool_),
        np.ascontiguousarray(nbr_max, dtype=np.int64),
        int(point_id),
        np.ascontiguousarray(radii, dtype=np.float64),
        float(exclusion_radius_m),
        float(scaffold_cell_m),
        int(min_support_points),
        int(min_support_sectors),
    )

    return [
        (np.asarray(ids, dtype=np.int64), int(sectors))
        for ids, sectors in raw
    ]

def airborne_prepare_support_multi_batch_native(
    x,
    y,
    z,
    ground,
    candidate_ids,
    neighbor_ids,
    offsets,
    radii,
    exclusion_radius_m,
    scaffold_cell_m,
    min_support_points,
    min_support_sectors,
):
    """Prepare exact batched ULS support from CSR neighborhoods."""

    if _fastgc_native is None:
        raise RuntimeError(
            "FAST-GC native extension is unavailable"
        )

    if not hasattr(
        _fastgc_native,
        "airborne_prepare_support_multi_batch",
    ):
        raise RuntimeError(
            "FAST-GC native batched ULS support preparation is unavailable"
        )

    raw = _fastgc_native.airborne_prepare_support_multi_batch(
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(z, dtype=np.float64),
        np.ascontiguousarray(ground, dtype=np.bool_),
        np.ascontiguousarray(candidate_ids, dtype=np.int64),
        np.ascontiguousarray(neighbor_ids, dtype=np.int64),
        np.ascontiguousarray(offsets, dtype=np.int64),
        np.ascontiguousarray(radii, dtype=np.float64),
        float(exclusion_radius_m),
        float(scaffold_cell_m),
        int(min_support_points),
        int(min_support_sectors),
    )

    return [
        [
            (np.asarray(ids, dtype=np.int64), int(sectors))
            for ids, sectors in candidate_output
        ]
        for candidate_output in raw
    ]



def airborne_prepare_support_multi_batch_rayon_native(
    x, y, z, ground,
    candidate_ids, neighbor_ids, offsets, radii,
    exclusion_radius_m, scaffold_cell_m,
    min_support_points, min_support_sectors,
):
    """Prepare exact bounded parallel ULS support."""

    if _fastgc_native is None:
        raise RuntimeError("FAST-GC native extension is unavailable")

    name = "airborne_prepare_support_multi_batch_rayon"

    if not hasattr(_fastgc_native, name):
        raise RuntimeError("FAST-GC native bounded ULS support preparation is unavailable")

    raw = getattr(_fastgc_native, name)(
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(z, dtype=np.float64),
        np.ascontiguousarray(ground, dtype=np.bool_),
        np.ascontiguousarray(candidate_ids, dtype=np.int64),
        np.ascontiguousarray(neighbor_ids, dtype=np.int64),
        np.ascontiguousarray(offsets, dtype=np.int64),
        np.ascontiguousarray(radii, dtype=np.float64),
        float(exclusion_radius_m),
        float(scaffold_cell_m),
        int(min_support_points),
        int(min_support_sectors),
    )

    return [
        [
            (np.asarray(ids, dtype=np.int64), int(sectors))
            for ids, sectors in candidate_output
        ]
        for candidate_output in raw
    ]
