
import numpy as np
from scipy.spatial import cKDTree

from fastgc.als_terrain_blob_guard import (
    _v5_canopy_ridge_authorization,
)


def test_long_bare_ridge_is_restored():

    # Long narrow candidate component.
    x = np.arange(
        0.0,
        21.0,
        0.5,
    )

    y = np.zeros_like(
        x
    )

    z = (
        100.0
        + 0.4 * x
    )

    n = x.size

    before = np.ones(
        n,
        dtype=bool,
    )

    after = np.zeros(
        n,
        dtype=bool,
    )

    tree = cKDTree(
        np.column_stack(
            (
                x,
                y,
            )
        )
    )

    result, report = _v5_canopy_ridge_authorization(
        x=x,
        y=y,
        z=z,
        ground_before=before,
        ground_after=after,
        tree=tree,
    )

    assert np.all(result)

    assert (
        report[
            "ridge_protected_components"
        ]
        >= 1
    )

    assert (
        report[
            "authorized_demotions"
        ]
        == 0
    )


def test_compact_bare_rock_is_restored():

    gx, gy = np.meshgrid(
        np.linspace(
            0.0,
            1.0,
            3,
        ),
        np.linspace(
            0.0,
            1.0,
            3,
        ),
    )

    x = gx.ravel()
    y = gy.ravel()

    z = (
        50.0
        + 0.2 * x
        + 0.1 * y
    )

    before = np.ones(
        x.size,
        dtype=bool,
    )

    after = np.zeros(
        x.size,
        dtype=bool,
    )

    tree = cKDTree(
        np.column_stack(
            (
                x,
                y,
            )
        )
    )

    result, report = _v5_canopy_ridge_authorization(
        x=x,
        y=y,
        z=z,
        ground_before=before,
        ground_after=after,
        tree=tree,
    )

    assert np.all(result)

    assert (
        report[
            "authorized_demotions"
        ]
        == 0
    )


def test_compact_candidate_inside_canopy_is_authorized():

    # --------------------------------------------------------
    # Candidate false-ground component.
    # --------------------------------------------------------

    cx = np.asarray(
        [
            0.0,
            0.4,
            0.0,
            0.4,
        ]
    )

    cy = np.asarray(
        [
            0.0,
            0.0,
            0.4,
            0.4,
        ]
    )

    cz = np.asarray(
        [
            5.0,
            5.1,
            5.0,
            5.1,
        ]
    )


    # --------------------------------------------------------
    # Existing non-ground canopy around, above and below.
    # --------------------------------------------------------

    ngx = np.asarray(
        [
            -0.8,
             0.9,
            -0.7,
             0.8,
            -0.6,
             0.7,
             0.2,
             0.2,
        ]
    )

    ngy = np.asarray(
        [
             0.1,
             0.1,
             0.8,
             0.8,
            -0.6,
            -0.5,
             0.9,
            -0.8,
        ]
    )

    ngz = np.asarray(
        [
            3.0,
            3.8,
            4.6,
            5.0,
            5.5,
            6.0,
            6.8,
            7.5,
        ]
    )


    x = np.concatenate(
        (
            cx,
            ngx,
        )
    )

    y = np.concatenate(
        (
            cy,
            ngy,
        )
    )

    z = np.concatenate(
        (
            cz,
            ngz,
        )
    )


    before = np.zeros(
        x.size,
        dtype=bool,
    )

    # Candidate was ground before cleanup.
    before[:4] = True

    after = before.copy()

    # V4 attempted to demote candidate.
    after[:4] = False


    tree = cKDTree(
        np.column_stack(
            (
                x,
                y,
            )
        )
    )


    result, report = _v5_canopy_ridge_authorization(
        x=x,
        y=y,
        z=z,
        ground_before=before,
        ground_after=after,
        tree=tree,
    )


    # Candidate remains demoted.
    assert not np.any(
        result[:4]
    )

    assert (
        report[
            "canopy_confirmed_components"
        ]
        >= 1
    )

    assert (
        report[
            "authorized_demotions"
        ]
        == 4
    )


def test_newly_demoted_points_cannot_self_confirm():

    # Four attempted false-ground candidates, but absolutely
    # no pre-existing non-ground observations.
    x = np.asarray(
        [
            0.0,
            0.3,
            0.0,
            0.3,
        ]
    )

    y = np.asarray(
        [
            0.0,
            0.0,
            0.3,
            0.3,
        ]
    )

    z = np.asarray(
        [
            5.0,
            5.1,
            5.0,
            5.1,
        ]
    )

    before = np.ones(
        4,
        dtype=bool,
    )

    after = np.zeros(
        4,
        dtype=bool,
    )

    tree = cKDTree(
        np.column_stack(
            (
                x,
                y,
            )
        )
    )

    result, report = _v5_canopy_ridge_authorization(
        x=x,
        y=y,
        z=z,
        ground_before=before,
        ground_after=after,
        tree=tree,
    )

    # Must restore because newly demoted points cannot count as
    # their own non-ground evidence.
    assert np.all(result)

    assert (
        report[
            "authorized_demotions"
        ]
        == 0
    )
