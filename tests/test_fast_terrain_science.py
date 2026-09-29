import numpy as np

from fastgc.terrain import (
    _resolve_terrain_products,
    compute_curvature,
    compute_gaussian_curvature,
    compute_local_relief,
    compute_mean_curvature,
    compute_planform_curvature,
    compute_profile_curvature,
    compute_tangential_curvature,
    compute_roughness,
    compute_slope_degrees,
    compute_slope_percent,
    compute_tpi,
    compute_tri,
)


def _grid(n=41, dx=1.0, dy=1.0):
    x = (np.arange(n, dtype=np.float64) - n // 2) * dx
    y = (np.arange(n, dtype=np.float64) - n // 2) * dy
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def _interior(arr, pad=4):
    return np.asarray(arr)[pad:-pad, pad:-pad]


def test_all_includes_tier1_morphometry():
    products = _resolve_terrain_products(["all"])

    required = {
        "profile_curvature",
        "tangential_curvature",
        "planform_curvature",
        "mean_curvature",
        "gaussian_curvature",
        "tri",
        "roughness",
        "local_relief",
    }

    assert required.issubset(set(products))


def test_flat_plane_has_zero_slope_and_intrinsic_curvature():
    dem = np.full((31, 31), 100.0, dtype=np.float64)

    slope_deg = compute_slope_degrees(dem, 1.0, 1.0)
    slope_pct = compute_slope_percent(dem, 1.0, 1.0)
    mean = compute_mean_curvature(dem, 1.0, 1.0)
    gaussian = compute_gaussian_curvature(dem, 1.0, 1.0)

    assert np.allclose(_interior(slope_deg), 0.0, atol=1e-7)
    assert np.allclose(_interior(slope_pct), 0.0, atol=1e-7)
    assert np.allclose(_interior(mean), 0.0, atol=1e-7)
    assert np.allclose(_interior(gaussian), 0.0, atol=1e-7)


def test_45_degree_plane_is_100_percent_slope():
    xx, yy = _grid()
    dem = xx.copy()

    slope_deg = compute_slope_degrees(dem, 1.0, 1.0)
    slope_pct = compute_slope_percent(dem, 1.0, 1.0)

    assert np.allclose(_interior(slope_deg), 45.0, atol=1e-5)
    assert np.allclose(_interior(slope_pct), 100.0, atol=1e-5)


def test_rectangular_pixels_preserve_known_plane_slope():
    dx = 2.0
    dy = 3.0
    xx, yy = _grid(dx=dx, dy=dy)

    # dz/dx = 0.3, dz/dy = 0.4 -> gradient magnitude = 0.5
    dem = 0.3 * xx + 0.4 * yy

    slope_pct = compute_slope_percent(dem, dx, dy)
    slope_deg = compute_slope_degrees(dem, dx, dy)

    expected_deg = np.degrees(np.arctan(0.5))

    assert np.allclose(_interior(slope_pct), 50.0, atol=1e-4)
    assert np.allclose(_interior(slope_deg), expected_deg, atol=1e-4)


def test_plane_has_zero_all_curvatures():
    xx, yy = _grid()
    dem = 0.2 * xx - 0.35 * yy + 12.0

    legacy = compute_curvature(dem, 1.0, 1.0)
    profile = compute_profile_curvature(dem, 1.0, 1.0)
    tangential = compute_tangential_curvature(dem, 1.0, 1.0)
    planform = compute_planform_curvature(dem, 1.0, 1.0)
    mean = compute_mean_curvature(dem, 1.0, 1.0)
    gaussian = compute_gaussian_curvature(dem, 1.0, 1.0)

    assert np.allclose(_interior(legacy), 0.0, atol=1e-6)
    assert np.allclose(_interior(profile), 0.0, atol=1e-6)
    assert np.allclose(_interior(tangential), 0.0, atol=1e-6)
    assert np.allclose(_interior(planform), 0.0, atol=1e-6)
    assert np.allclose(_interior(mean), 0.0, atol=1e-6)
    assert np.allclose(_interior(gaussian), 0.0, atol=1e-6)


def test_bowl_center_has_negative_mean_and_positive_gaussian_curvature():
    """Concave bowl is negative under the FAST geomorphometric convention."""
    xx, yy = _grid()
    dem = 0.01 * xx**2 + 0.02 * yy**2

    mean = compute_mean_curvature(dem, 1.0, 1.0)
    gaussian = compute_gaussian_curvature(dem, 1.0, 1.0)

    c = dem.shape[0] // 2

    # At the bowl center:
    # r = 0.02, t = 0.04, p=q=s=0.
    # The unsigned/orientation-dependent geometric expression has
    # magnitude 0.03. FAST selects the geomorphometric orientation:
    # concave negative, convex positive.
    # Gaussian curvature is orientation-independent.
    assert np.isclose(mean[c, c], -0.03, atol=1e-5)
    assert np.isclose(gaussian[c, c], 0.0008, atol=1e-6)


def test_dome_center_has_positive_mean_and_gaussian_curvature():
    """Convex dome is positive under the FAST geomorphometric convention."""
    xx, yy = _grid()
    dem = -(0.01 * xx**2 + 0.02 * yy**2)

    mean = compute_mean_curvature(dem, 1.0, 1.0)
    gaussian = compute_gaussian_curvature(dem, 1.0, 1.0)

    c = dem.shape[0] // 2

    assert np.isclose(mean[c, c], 0.03, atol=1e-5)
    assert np.isclose(gaussian[c, c], 0.0008, atol=1e-6)


def test_saddle_center_has_negative_gaussian_curvature():
    xx, yy = _grid()
    dem = 0.01 * xx**2 - 0.02 * yy**2

    gaussian = compute_gaussian_curvature(dem, 1.0, 1.0)

    c = dem.shape[0] // 2

    # r=0.02, t=-0.04, s=0 -> K=-0.0008
    assert np.isclose(gaussian[c, c], -0.0008, atol=1e-6)


def test_directional_curvatures_are_finite_away_from_flat_bowl_center():
    xx, yy = _grid()
    dem = 0.01 * xx**2 + 0.02 * yy**2

    profile = compute_profile_curvature(dem, 1.0, 1.0)
    tangential = compute_tangential_curvature(dem, 1.0, 1.0)
    planform = compute_planform_curvature(dem, 1.0, 1.0)

    c = dem.shape[0] // 2

    assert np.isnan(profile[c, c])
    assert np.isnan(tangential[c, c])
    assert np.isnan(planform[c, c])

    assert np.isfinite(profile[c, c + 5])
    assert np.isfinite(tangential[c, c + 5])
    assert np.isfinite(planform[c, c + 5])


def test_tangential_and_planform_curvature_relationship():
    xx, yy = _grid()

    # Nonzero linear tilt avoids an undefined contour direction at
    # the center while quadratic terms provide curvature.
    dem = (
        0.25 * xx
        + 0.15 * yy
        + 0.01 * xx**2
        + 0.02 * yy**2
    )

    tangential = compute_tangential_curvature(dem, 1.0, 1.0)
    planform = compute_planform_curvature(dem, 1.0, 1.0)

    c = dem.shape[0] // 2

    g = np.sqrt(0.25**2 + 0.15**2)

    expected = (
        planform[c, c]
        * g
        / np.sqrt(1.0 + g * g)
    )

    assert np.isclose(
        tangential[c, c],
        expected,
        atol=1e-6,
    )


def test_tpi_excludes_center_from_neighbor_mean():
    dem = np.zeros((3, 3), dtype=np.float64)
    dem[1, 1] = 9.0

    tpi = compute_tpi(dem, radius=1)

    # Surrounding eight cells all equal zero.
    assert np.isclose(tpi[1, 1], 9.0)


def test_tpi_flat_surface_is_zero():
    dem = np.full((9, 9), 27.0, dtype=np.float64)

    tpi = compute_tpi(dem, radius=2)

    assert np.allclose(tpi, 0.0, atol=1e-7)


def test_riley_tri_exact_controlled_neighborhood():
    dem = np.zeros((3, 3), dtype=np.float64)
    dem[1, 1] = 1.0

    tri = compute_tri(dem)

    # Eight neighbors differ from center by -1:
    # sqrt(8 * 1^2) = sqrt(8)
    assert np.isclose(tri[1, 1], np.sqrt(8.0), atol=1e-6)


def test_roughness_is_standard_full_neighborhood_range():
    """Roughness is max minus min across the complete 3x3 neighborhood."""
    dem = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 10.0],
        ],
        dtype=np.float64,
    )

    roughness = compute_roughness(dem)

    # Standard Wilson/GDAL roughness:
    #
    #     max(z_3x3) - min(z_3x3)
    #     = 10 - 1
    #     = 9
    #
    # This intentionally distinguishes standard terrain roughness
    # from the former focal-to-neighbor maximum difference (5).
    assert np.isclose(
        roughness[1, 1],
        9.0,
        atol=1e-7,
    )


def test_local_relief_respects_requested_radius():
    dem = np.zeros((7, 7), dtype=np.float64)
    dem[3, 3] = 10.0

    relief_r1 = compute_local_relief(dem, radius=1)
    relief_r2 = compute_local_relief(dem, radius=2)

    # One cell two positions away cannot see the peak at radius 1,
    # but can see it at radius 2.
    assert np.isclose(relief_r1[3, 1], 0.0)
    assert np.isclose(relief_r2[3, 1], 10.0)


def test_neighborhood_metrics_ignore_nan_neighbors_but_preserve_nan_center():
    dem = np.zeros((5, 5), dtype=np.float64)
    dem[1, 1] = np.nan
    dem[2, 2] = 5.0

    tpi = compute_tpi(dem, radius=1)
    tri = compute_tri(dem)
    roughness = compute_roughness(dem)
    relief = compute_local_relief(dem, radius=1)

    assert np.isnan(tpi[1, 1])
    assert np.isnan(tri[1, 1])
    assert np.isnan(roughness[1, 1])
    assert np.isnan(relief[1, 1])

    assert np.isfinite(tpi[2, 2])
    assert np.isfinite(tri[2, 2])
    assert np.isfinite(roughness[2, 2])
    assert np.isfinite(relief[2, 2])


def test_nodata_hole_influence_is_spatially_local():
    """
    A nodata hole must not perturb terrain derivatives arbitrarily far
    into otherwise valid terrain.

    The implementation may require local support around missing cells,
    but a compact missing region must not contaminate distant cells.
    """
    from fastgc.terrain import (
        compute_slope_percent,
        compute_curvature,
    )

    # Exact plane: z = 2x + 3y.
    #
    # Away from nodata:
    # slope_percent = sqrt(2^2 + 3^2) * 100
    # curvature = 0
    ny = 41
    nx = 41

    y, x = np.mgrid[0:ny, 0:nx]

    clean = (
        2.0 * x.astype(np.float64)
        + 3.0 * y.astype(np.float64)
    )

    damaged = clean.copy()

    # Compact 3 x 3 internal nodata hole.
    damaged[19:22, 19:22] = np.nan

    clean_slope = compute_slope_percent(
        clean,
        1.0,
        1.0,
    )

    hole_slope = compute_slope_percent(
        damaged,
        1.0,
        1.0,
    )

    clean_curv = compute_curvature(
        clean,
        1.0,
        1.0,
    )

    hole_curv = compute_curvature(
        damaged,
        1.0,
        1.0,
    )

    # Compare only cells comfortably separated from the hole.
    yy, xx = np.mgrid[0:ny, 0:nx]

    far = (
        (np.maximum(
            np.abs(xx - 20),
            np.abs(yy - 20),
        ) >= 5)
        & (xx >= 2)
        & (xx < nx - 2)
        & (yy >= 2)
        & (yy < ny - 2)
    )

    assert np.all(np.isfinite(hole_slope[far]))
    assert np.all(np.isfinite(hole_curv[far]))

    np.testing.assert_allclose(
        hole_slope[far],
        clean_slope[far],
        rtol=0.0,
        atol=1.0e-10,
    )

    np.testing.assert_allclose(
        hole_curv[far],
        clean_curv[far],
        rtol=0.0,
        atol=1.0e-10,
    )


def test_original_nodata_cells_remain_nodata_after_product_processing(
    tmp_path,
):
    """
    End-to-end terrain processing must preserve the original DEM nodata
    footprint in the written product.
    """
    import rasterio
    from rasterio.transform import from_origin

    from fastgc.terrain import run_terrain_from_dem

    dem_fp = tmp_path / "synthetic_FAST_DEM.tif"
    out_root = tmp_path / "FAST_TERRAIN"

    y, x = np.mgrid[0:21, 0:21]

    dem = (
        x.astype(np.float32)
        + 0.5 * y.astype(np.float32)
    )

    dem[9:12, 9:12] = np.nan

    profile = {
        "driver": "GTiff",
        "height": dem.shape[0],
        "width": dem.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32617",
        "transform": from_origin(
            0.0,
            21.0,
            1.0,
            1.0,
        ),
        "nodata": np.nan,
    }

    with rasterio.open(
        dem_fp,
        "w",
        **profile,
    ) as dst:
        dst.write(dem, 1)

    run_terrain_from_dem(
        dem_fp=dem_fp,
        output_root=out_root,
        terrain_products=[
            "slope_percent",
            "curvature",
        ],
        overwrite=True,
        n_jobs=1,
    )

    for product in [
        "slope_percent",
        "curvature",
    ]:
        files = list(
            (out_root / product).glob("*.tif")
        )

        assert len(files) == 1

        with rasterio.open(files[0]) as ds:
            arr = ds.read(1)

        # Original hole remains nodata.
        assert np.all(
            np.isnan(arr[9:12, 9:12])
        )

        # Ordinary valid terrain remains finite.
        assert np.isfinite(arr[5, 5])
        assert np.isfinite(arr[15, 15])



# FAST_TERRAIN_CURVATURE_SIGN_CONVENTION_TESTS

def test_curvature_family_geomorphometric_sign_convention():
    """Convex terrain is positive; concave terrain is negative."""
    from fastgc.terrain import (
        compute_profile_curvature,
        compute_tangential_curvature,
        compute_planform_curvature,
        compute_mean_curvature,
        compute_gaussian_curvature,
    )

    n = 101
    x = np.arange(n, dtype=np.float64) - n // 2
    y = np.arange(n, dtype=np.float64) - n // 2
    xx, yy = np.meshgrid(x, y)

    convex = 0.20 * xx - 0.01 * xx**2 - 0.005 * yy**2
    concave = 0.20 * xx + 0.01 * xx**2 + 0.005 * yy**2

    cy = cx = n // 2

    directional = (
        compute_profile_curvature,
        compute_tangential_curvature,
        compute_planform_curvature,
    )

    for fn in directional:
        kc = float(fn(convex, 1.0, 1.0)[cy, cx])
        kv = float(fn(concave, 1.0, 1.0)[cy, cx])

        assert np.isfinite(kc)
        assert np.isfinite(kv)
        assert kc > 0.0
        assert kv < 0.0

    mean_convex = float(
        compute_mean_curvature(convex, 1.0, 1.0)[cy, cx]
    )
    mean_concave = float(
        compute_mean_curvature(concave, 1.0, 1.0)[cy, cx]
    )

    assert mean_convex > 0.0
    assert mean_concave < 0.0

    # Gaussian curvature is orientation-independent.
    gaussian_convex = float(
        compute_gaussian_curvature(convex, 1.0, 1.0)[cy, cx]
    )
    gaussian_concave = float(
        compute_gaussian_curvature(concave, 1.0, 1.0)[cy, cx]
    )

    assert gaussian_convex > 0.0
    assert gaussian_concave > 0.0


def test_gaussian_curvature_is_negative_on_saddle():
    from fastgc.terrain import compute_gaussian_curvature

    n = 101
    x = np.arange(n, dtype=np.float64) - n // 2
    y = np.arange(n, dtype=np.float64) - n // 2
    xx, yy = np.meshgrid(x, y)

    saddle = 0.01 * (xx**2 - yy**2)

    c = n // 2
    value = float(
        compute_gaussian_curvature(
            saddle,
            1.0,
            1.0,
        )[c, c]
    )

    assert value < 0.0


def test_directional_curvature_undefined_at_zero_gradient():
    """Profile/contour directions are undefined where horizontal gradient is zero."""
    from fastgc.terrain import (
        compute_profile_curvature,
        compute_tangential_curvature,
        compute_planform_curvature,
    )

    n = 101
    x = np.arange(n, dtype=np.float64) - n // 2
    y = np.arange(n, dtype=np.float64) - n // 2
    xx, yy = np.meshgrid(x, y)

    dome = -0.01 * (xx**2 + yy**2)
    c = n // 2

    for fn in (
        compute_profile_curvature,
        compute_tangential_curvature,
        compute_planform_curvature,
    ):
        assert np.isnan(
            fn(dome, 1.0, 1.0)[c, c]
        )


def test_legacy_curvature_remains_laplacian_sign():
    """Backward-compatible legacy curvature remains z_xx + z_yy."""
    from fastgc.terrain import compute_curvature

    n = 101
    x = np.arange(n, dtype=np.float64) - n // 2
    y = np.arange(n, dtype=np.float64) - n // 2
    xx, yy = np.meshgrid(x, y)

    dome = -0.01 * (xx**2 + yy**2)
    bowl = +0.01 * (xx**2 + yy**2)

    c = n // 2

    assert float(
        compute_curvature(dome, 1.0, 1.0)[c, c]
    ) < 0.0

    assert float(
        compute_curvature(bowl, 1.0, 1.0)[c, c]
    ) > 0.0


# FAST_TERRAIN_STANDARD_ROUGHNESS_TESTS

def test_roughness_is_full_3x3_elevation_range():
    """Standard roughness is neighborhood max minus neighborhood min."""
    dem = np.array(
        [
            [0.0, 10.0, 10.0],
            [4.0,  5.0,  6.0],
            [4.0,  4.0,  4.0],
        ],
        dtype=np.float64,
    )

    rough = compute_roughness(dem)

    # Center-to-neighbor maximum difference would be 5.
    # Standard 3x3 range roughness is 10 - 0 = 10.
    assert np.isclose(
        rough[1, 1],
        10.0,
        atol=1e-7,
    )


def test_roughness_equals_range_on_simple_surface():
    dem = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float64,
    )

    rough = compute_roughness(dem)

    assert np.isclose(
        rough[1, 1],
        8.0,
        atol=1e-7,
    )


def test_roughness_is_zero_on_flat_surface():
    dem = np.full(
        (7, 7),
        42.0,
        dtype=np.float64,
    )

    rough = compute_roughness(dem)

    assert np.allclose(
        rough,
        0.0,
        atol=1e-7,
    )


def test_roughness_preserves_nodata_center():
    dem = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, np.nan, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float64,
    )

    rough = compute_roughness(dem)

    assert np.isnan(
        rough[1, 1]
    )


# FAST_TERRAIN_TPI_LOCAL_RELIEF_SCIENCE_TESTS

def test_tpi_excludes_focal_cell_from_surrounding_mean():
    """TPI must use surrounding cells, not include the focal elevation."""
    dem = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 9.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    tpi = compute_tpi(
        dem,
        radius=1,
    )

    # Surrounding mean = 0, therefore TPI = 9.
    #
    # If the focal cell were incorrectly included, the mean would
    # be 1 and TPI would instead be 8.
    assert np.isclose(
        tpi[1, 1],
        9.0,
        atol=1e-7,
    )


def test_tpi_is_zero_on_planar_surface_at_interior():
    """A symmetric neighborhood on a plane has zero TPI."""
    y, x = np.mgrid[
        -5:6,
        -5:6,
    ]

    dem = (
        100.0
        + 2.0 * x
        + 3.0 * y
    ).astype(np.float64)

    for radius in (1, 2, 3):
        tpi = compute_tpi(
            dem,
            radius=radius,
        )

        assert np.isclose(
            tpi[5, 5],
            0.0,
            atol=1e-7,
        )


def test_tpi_scale_changes_with_neighborhood_radius():
    """TPI is explicitly scale dependent."""
    y, x = np.mgrid[
        -5:6,
        -5:6,
    ]

    dem = (
        x.astype(np.float64) ** 2
        + y.astype(np.float64) ** 2
    )

    values = []

    for radius in (1, 2, 3):
        tpi = compute_tpi(
            dem,
            radius=radius,
        )

        values.append(
            float(tpi[5, 5])
        )

    # At the bowl center z=0 while surrounding elevations increase
    # with distance. TPI is therefore negative, and its magnitude
    # must increase as the analysis neighborhood grows.
    assert values[0] < 0.0
    assert values[1] < values[0]
    assert values[2] < values[1]


def test_tpi_ignores_nodata_neighbors_but_preserves_nodata_center():
    dem = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 5.0, np.nan],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    tpi = compute_tpi(
        dem,
        radius=1,
    )

    # Seven finite surrounding cells, all equal to 1.
    assert np.isclose(
        tpi[1, 1],
        4.0,
        atol=1e-7,
    )

    assert np.isnan(
        tpi[1, 2]
    )


def test_local_relief_radius1_is_exact_3x3_range():
    dem = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 10.0],
        ],
        dtype=np.float64,
    )

    relief = compute_local_relief(
        dem,
        radius=1,
    )

    assert np.isclose(
        relief[1, 1],
        9.0,
        atol=1e-7,
    )


def test_local_relief_matches_exact_range_at_multiple_radii():
    y, x = np.mgrid[
        -5:6,
        -5:6,
    ]

    dem = (
        100.0
        + x
        + 2.0 * y
    ).astype(np.float64)

    cy = cx = 5

    for radius in (1, 2, 3):

        relief = compute_local_relief(
            dem,
            radius=radius,
        )

        window = dem[
            cy-radius:cy+radius+1,
            cx-radius:cx+radius+1,
        ]

        expected = (
            np.max(window)
            - np.min(window)
        )

        assert np.isclose(
            relief[cy, cx],
            expected,
            atol=1e-7,
        )


def test_local_relief_increases_with_radius_on_nonflat_surface():
    y, x = np.mgrid[
        -5:6,
        -5:6,
    ]

    dem = (
        x.astype(np.float64) ** 2
        + y.astype(np.float64) ** 2
    )

    values = []

    for radius in (1, 2, 3):
        relief = compute_local_relief(
            dem,
            radius=radius,
        )

        values.append(
            float(relief[5, 5])
        )

    assert values[0] < values[1] < values[2]


def test_local_relief_ignores_nodata_neighbors_and_preserves_center_nodata():
    dem = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, np.nan],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float64,
    )

    relief = compute_local_relief(
        dem,
        radius=1,
    )

    assert np.isclose(
        relief[1, 1],
        8.0,
        atol=1e-7,
    )

    assert np.isnan(
        relief[1, 2]
    )


# FAST_TERRAIN_PROVENANCE_TESTS

def _terrain_test_item(
    dem_fp,
    out_fp,
    *,
    analytical_domain="continuous_merged_dem",
    product="slope_degrees",
    skip_existing=True,
    overwrite=False,
):
    return {
        "dem_fp": str(dem_fp),
        "out_fp": str(out_fp),
        "product": product,
        "analytical_domain": analytical_domain,
        "skip_existing": skip_existing,
        "overwrite": overwrite,
        "hillshade_azimuth": 315.0,
        "hillshade_altitude": 45.0,
        "hillshade_z_factor": 1.0,
        "tpi_radius": 3,
        "twi_eps": 1e-6,
        "dtw_max_distance": None,
    }


def test_terrain_provenance_roundtrip(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    from fastgc.terrain import (
        _process_dem_for_product,
        _terrain_provenance,
        _terrain_provenance_matches,
    )

    dem_fp = tmp_path / "example_FAST_DEM.tif"
    out_fp = tmp_path / "example_FAST_TERRAIN_slope_degrees.tif"

    profile = {
        "driver": "GTiff",
        "height": 5,
        "width": 5,
        "count": 1,
        "dtype": "float32",
        "transform": from_origin(
            0.0,
            5.0,
            1.0,
            1.0,
        ),
        "nodata": -9999.0,
    }

    dem = np.arange(
        25,
        dtype=np.float32,
    ).reshape(5, 5)

    with rasterio.open(
        dem_fp,
        "w",
        **profile,
    ) as dst:
        dst.write(dem, 1)

    item = _terrain_test_item(
        dem_fp,
        out_fp,
        skip_existing=False,
    )

    result = _process_dem_for_product(item)

    assert result["status"] == "ok"

    expected = _terrain_provenance(item)

    assert _terrain_provenance_matches(
        out_fp,
        expected,
    )

    with rasterio.open(out_fp) as src:
        tags = src.tags()

    assert tags["FASTGC_PRODUCT"] == "FAST_TERRAIN"
    assert (
        tags["FASTGC_ANALYTICAL_DOMAIN"]
        == "continuous_merged_dem"
    )
    assert (
        tags["FASTGC_SOURCE_DEM"]
        == dem_fp.name
    )
    assert (
        tags["FASTGC_TERRAIN_PRODUCT"]
        == "slope_degrees"
    )


def test_matching_provenance_allows_skip(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    from fastgc.terrain import _process_dem_for_product

    dem_fp = tmp_path / "example_FAST_DEM.tif"
    out_fp = tmp_path / "example_FAST_TERRAIN_slope_degrees.tif"

    profile = {
        "driver": "GTiff",
        "height": 5,
        "width": 5,
        "count": 1,
        "dtype": "float32",
        "transform": from_origin(
            0.0,
            5.0,
            1.0,
            1.0,
        ),
        "nodata": -9999.0,
    }

    dem = np.arange(
        25,
        dtype=np.float32,
    ).reshape(5, 5)

    with rasterio.open(
        dem_fp,
        "w",
        **profile,
    ) as dst:
        dst.write(dem, 1)

    first = _terrain_test_item(
        dem_fp,
        out_fp,
        skip_existing=False,
    )

    assert (
        _process_dem_for_product(first)["status"]
        == "ok"
    )

    second = _terrain_test_item(
        dem_fp,
        out_fp,
        skip_existing=True,
    )

    assert (
        _process_dem_for_product(second)["status"]
        == "skipped"
    )


def test_missing_provenance_forces_recompute(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    from fastgc.terrain import _process_dem_for_product

    dem_fp = tmp_path / "example_FAST_DEM.tif"
    out_fp = tmp_path / "example_FAST_TERRAIN_slope_degrees.tif"

    profile = {
        "driver": "GTiff",
        "height": 5,
        "width": 5,
        "count": 1,
        "dtype": "float32",
        "transform": from_origin(
            0.0,
            5.0,
            1.0,
            1.0,
        ),
        "nodata": -9999.0,
    }

    dem = np.arange(
        25,
        dtype=np.float32,
    ).reshape(5, 5)

    with rasterio.open(
        dem_fp,
        "w",
        **profile,
    ) as dst:
        dst.write(dem, 1)

    # Deliberately create an old/stale output with no FASTGC tags.
    with rasterio.open(
        out_fp,
        "w",
        **profile,
    ) as dst:
        dst.write(
            np.zeros((5, 5), dtype=np.float32),
            1,
        )

    item = _terrain_test_item(
        dem_fp,
        out_fp,
        skip_existing=True,
    )

    result = _process_dem_for_product(item)

    assert result["status"] == "ok"

    with rasterio.open(out_fp) as src:
        tags = src.tags()

    assert (
        tags["FASTGC_ANALYTICAL_DOMAIN"]
        == "continuous_merged_dem"
    )


def test_wrong_analytical_domain_forces_recompute(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    from fastgc.terrain import (
        _process_dem_for_product,
    )

    dem_fp = tmp_path / "example_FAST_DEM.tif"
    out_fp = tmp_path / "example_FAST_TERRAIN_slope_degrees.tif"

    profile = {
        "driver": "GTiff",
        "height": 5,
        "width": 5,
        "count": 1,
        "dtype": "float32",
        "transform": from_origin(
            0.0,
            5.0,
            1.0,
            1.0,
        ),
        "nodata": -9999.0,
    }

    dem = np.arange(
        25,
        dtype=np.float32,
    ).reshape(5, 5)

    with rasterio.open(
        dem_fp,
        "w",
        **profile,
    ) as dst:
        dst.write(dem, 1)

    tile_item = _terrain_test_item(
        dem_fp,
        out_fp,
        analytical_domain="buffered_dem_tile",
        skip_existing=False,
    )

    assert (
        _process_dem_for_product(tile_item)["status"]
        == "ok"
    )

    continuous_item = _terrain_test_item(
        dem_fp,
        out_fp,
        analytical_domain="continuous_merged_dem",
        skip_existing=True,
    )

    result = _process_dem_for_product(
        continuous_item
    )

    assert result["status"] == "ok"

    with rasterio.open(out_fp) as src:
        tags = src.tags()

    assert (
        tags["FASTGC_ANALYTICAL_DOMAIN"]
        == "continuous_merged_dem"
    )
