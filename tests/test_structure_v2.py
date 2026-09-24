import numpy as np
import rasterio
from fastgc.structure import compute_structure_metrics, write_structure_rasters

def _cloud():
    x=np.array([1,2,3,4,5,6,7,8.],float); y=np.array([1,2,3,4,5,6,7,8.],float); z=np.array([0.2,1,2,3,4,5,6,7.],float)
    return x,y,z

def test_core_metric_families():
    x,y,z=_cloud(); r=compute_structure_metrics(x,y,z,sensor_mode='ALS',res=20,min_h=.5,bin_size=1,canopy_threshold=2,bounds=(0,0,20,20)); m=r['metrics']
    expected={'z_min','z_median','z_p95','z_iqr','point_density','vertical_entropy','vertical_richness','vertical_gap_fraction','sigma_z','eigenvalue_1','linearity','planarity','normal_z','slope','robust_scale','n_points_all'}
    assert expected.issubset(m); assert m['n_points_all'][0,0]==8; assert m['n_points'][0,0]==7
    assert np.isclose(m['canopy_cover'][0,0],6/8)

def test_optional_attributes():
    x,y,z=_cloud(); n=z.size
    r=compute_structure_metrics(x,y,z,sensor_mode='ALS',res=20,min_h=.5,bin_size=1,canopy_threshold=2,bounds=(0,0,20,20),intensity=np.arange(n)*10+10,return_number=np.array([1,1,2,1,2,1,1,2]),number_of_returns=np.array([1,2,2,1,2,1,1,2]))
    m=r['metrics']; assert np.isfinite(m['intensity_mean'][0,0]); assert np.isfinite(m['first_return_fraction'][0,0]); assert np.isfinite(m['mean_number_of_returns'][0,0])

def test_writer_integer_nodata(tmp_path):
    x,y,z=_cloud(); r=compute_structure_metrics(x,y,z,sensor_mode='ALS',res=20,min_h=.5,bin_size=1,canopy_threshold=2,bounds=(0,0,20,20)); r['metrics']={'n_points':r['metrics']['n_points'],'z_mean':r['metrics']['z_mean']}
    paths=write_structure_rasters(r,tmp_path)
    with rasterio.open(paths['n_points']) as ds: assert ds.nodata is None and ds.dtypes[0]=='int32'
    with rasterio.open(paths['z_mean']) as ds: assert np.isnan(ds.nodata)


def test_v21_density_semantics_and_orientation_alias():
    import numpy as np

    from fastgc.structure import compute_structure_metrics

    # All points occupy one 1 x 1 m cell.
    # Two points are below min_h=0.5 and four are vegetation points.
    x = np.array([0.10, 0.20, 0.30, 0.40, 0.60, 0.80])
    y = np.array([0.10, 0.25, 0.70, 0.40, 0.80, 0.55])
    z = np.array([0.10, 0.30, 1.00, 2.00, 3.00, 4.00])

    result = compute_structure_metrics(
        x,
        y,
        z,
        sensor_mode="ALS",
        res=1.0,
        min_h=0.5,
        bin_size=0.5,
        canopy_threshold=2.0,
        bounds=(0.0, 0.0, 1.0, 1.0),
    )

    m = result["metrics"]

    assert int(m["n_points_all"][0, 0]) == 6
    assert int(m["n_points"][0, 0]) == 4

    # Explicit semantics.
    assert np.isclose(m["point_density_all"][0, 0], 6.0)
    assert np.isclose(m["vegetation_point_density"][0, 0], 4.0)

    # Backward-compatible historical name remains the vegetation density.
    assert np.isclose(
        m["point_density"][0, 0],
        m["vegetation_point_density"][0, 0],
    )

    # Historical slope is retained as a compatibility alias only.
    assert np.isfinite(m["normal_inclination_deg"][0, 0])
    assert np.isclose(
        m["slope"][0, 0],
        m["normal_inclination_deg"][0, 0],
    )
    assert 0.0 <= m["normal_inclination_deg"][0, 0] <= 90.0


def test_v21_optional_metrics_absent_without_source_dimensions():
    import numpy as np

    from fastgc.structure import compute_structure_metrics

    x = np.array([0.10, 0.20, 0.30, 0.40])
    y = np.array([0.10, 0.20, 0.40, 0.70])
    z = np.array([1.00, 2.00, 3.00, 4.00])

    result = compute_structure_metrics(
        x,
        y,
        z,
        sensor_mode="ALS",
        res=1.0,
        min_h=0.5,
        bin_size=0.5,
        canopy_threshold=2.0,
        bounds=(0.0, 0.0, 1.0, 1.0),
    )

    m = result["metrics"]

    for name in (
        "intensity_mean",
        "intensity_sd",
        "intensity_p50",
        "intensity_p95",
        "first_return_fraction",
        "multi_return_fraction",
        "mean_number_of_returns",
    ):
        assert name not in m


def test_v21_optional_metrics_present_with_source_dimensions():
    import numpy as np

    from fastgc.structure import compute_structure_metrics

    x = np.array([0.10, 0.20, 0.30, 0.40])
    y = np.array([0.10, 0.20, 0.40, 0.70])
    z = np.array([1.00, 2.00, 3.00, 4.00])

    intensity = np.array([100, 200, 300, 400])
    return_number = np.array([1, 2, 1, 2])
    number_of_returns = np.array([1, 2, 2, 3])

    result = compute_structure_metrics(
        x,
        y,
        z,
        sensor_mode="ALS",
        res=1.0,
        min_h=0.5,
        bin_size=0.5,
        canopy_threshold=2.0,
        bounds=(0.0, 0.0, 1.0, 1.0),
        intensity=intensity,
        return_number=return_number,
        number_of_returns=number_of_returns,
    )

    m = result["metrics"]

    for name in (
        "intensity_mean",
        "intensity_sd",
        "intensity_p50",
        "intensity_p95",
        "first_return_fraction",
        "multi_return_fraction",
        "mean_number_of_returns",
    ):
        assert name in m
        assert np.isfinite(m[name][0, 0])

    assert np.isclose(m["intensity_mean"][0, 0], 250.0)
    assert np.isclose(m["first_return_fraction"][0, 0], 0.5)
    assert np.isclose(m["multi_return_fraction"][0, 0], 0.5)
    assert np.isclose(m["mean_number_of_returns"][0, 0], 2.0)


def test_structure_default_preserves_unobserved_cells_as_nan():
    import numpy as np

    from fastgc.structure import compute_structure_metrics

    # Force a 2 x 2 raster while placing observations only in the
    # upper-left cell.
    x = np.array([0.10, 0.20, 0.30, 0.40])
    y = np.array([1.90, 1.80, 1.70, 1.60])
    z = np.array([1.0, 2.0, 3.0, 4.0])

    result = compute_structure_metrics(
        x,
        y,
        z,
        sensor_mode="ALS",
        res=1.0,
        min_h=0.5,
        bin_size=0.5,
        canopy_threshold=2.0,
        na_fill="none",
        bounds=(0.0, 0.0, 2.0, 2.0),
    )

    m = result["metrics"]

    for name in (
        "z_mean",
        "z_max",
        "z_sd",
        "canopy_cover",
        "FHD",
        "VCI",
    ):
        assert np.isfinite(m[name][0, 0])
        assert np.isnan(m[name][0, 1])
        assert np.isnan(m[name][1, 0])
        assert np.isnan(m[name][1, 1])

    # Count rasters intentionally use zero for cells containing no points.
    assert m["n_points_all"][0, 0] == 4
    assert m["n_points_all"][0, 1] == 0
    assert m["n_points_all"][1, 0] == 0
    assert m["n_points_all"][1, 1] == 0


def test_point_density_all_includes_cells_without_vegetation():
    import numpy as np

    from fastgc.structure import compute_structure_metrics

    # Left cell contains vegetation.
    # Right cell contains observations, but all are below min_h.
    x = np.array([
        0.10, 0.20, 0.30,
        1.10, 1.20, 1.30,
    ])
    y = np.array([
        0.10, 0.20, 0.30,
        0.10, 0.20, 0.30,
    ])
    z = np.array([
        1.0, 2.0, 3.0,
        0.10, 0.20, 0.30,
    ])

    result = compute_structure_metrics(
        x,
        y,
        z,
        sensor_mode="ALS",
        res=1.0,
        min_h=0.5,
        bin_size=0.5,
        canopy_threshold=2.0,
        na_fill="none",
        bounds=(0.0, 0.0, 2.0, 1.0),
    )

    m = result["metrics"]

    assert m["n_points_all"][0, 0] == 3
    assert m["n_points_all"][0, 1] == 3

    assert m["n_points"][0, 0] == 3
    assert m["n_points"][0, 1] == 0

    # All-point density must exist in BOTH observed cells.
    assert np.isclose(m["point_density_all"][0, 0], 3.0)
    assert np.isclose(m["point_density_all"][0, 1], 3.0)

    # Vegetation density exists only where vegetation exists.
    assert np.isclose(m["vegetation_point_density"][0, 0], 3.0)
    assert np.isnan(m["vegetation_point_density"][0, 1])

    # Historical alias retains vegetation-density semantics.
    assert np.isnan(m["point_density"][0, 1])

    # Canopy cover is still defined because the right cell was observed.
    assert np.isclose(m["canopy_cover"][0, 1], 0.0)


def test_sigma_z_is_translation_invariant_for_projected_coordinates():
    import numpy as np

    from fastgc.structure import _sigma_z

    # Two-dimensional local XY support with non-planar Z residuals.
    x = np.array([
        0.05, 0.20, 0.35, 0.55,
        0.70, 0.85, 0.15, 0.45,
    ], dtype=np.float64)

    y = np.array([
        0.10, 0.75, 0.30, 0.90,
        0.20, 0.65, 0.50, 0.40,
    ], dtype=np.float64)

    z = np.array([
        1.0, 2.2, 1.7, 3.4,
        2.1, 4.0, 2.8, 3.0,
    ], dtype=np.float64)

    local = _sigma_z(x, y, z)

    # Representative projected-coordinate translation.
    projected = _sigma_z(
        x + 478600.0,
        y + 5427100.0,
        z,
    )

    assert np.isfinite(local)
    assert np.isfinite(projected)

    # Plane residual dispersion must not depend on coordinate origin.
    assert np.isclose(
        local,
        projected,
        rtol=1e-8,
        atol=1e-9,
    )


def test_geometry_invariants_and_projected_coordinate_stability():
    import numpy as np
    from fastgc.structure import _geometry_metrics

    rng = np.random.default_rng(42)
    n = 500

    x = rng.uniform(0.0, 1.0, n)
    y = rng.uniform(0.0, 1.0, n)
    z = 12.0 * x + 4.0 * y + rng.normal(0.0, 1.5, n)

    local = _geometry_metrics(x, y, z)
    projected = _geometry_metrics(
        x + 478600.0,
        y + 5427100.0,
        z,
    )

    assert np.isclose(
        local["linearity"]
        + local["planarity"]
        + local["sphericity"],
        1.0,
        atol=1e-10,
    )

    assert np.isclose(
        local["anisotropy"] + local["sphericity"],
        1.0,
        atol=1e-10,
    )

    assert 0.0 <= local["surface_variation"] <= (1.0 / 3.0 + 1e-12)
    assert 0.0 <= local["eigenentropy"] <= np.log(3.0) + 1e-12
    assert 0.0 <= local["normal_inclination_deg"] <= 90.0

    names = (
        "eigenvalue_1",
        "eigenvalue_2",
        "eigenvalue_3",
        "linearity",
        "planarity",
        "sphericity",
        "anisotropy",
        "surface_variation",
        "eigenentropy",
        "omnivariance",
        "robust_scale",
        "axis_x",
        "axis_y",
        "axis_z",
        "normal_x",
        "normal_y",
        "normal_z",
        "slope",
        "normal_inclination_deg",
    )

    for name in names:
        assert np.isclose(
            local[name],
            projected[name],
            rtol=1e-8,
            atol=1e-9,
        ), name


def test_sigma_z_projected_coordinates_have_full_rank_support():
    import numpy as np
    from fastgc.structure import _sigma_z

    # Dense, genuinely 2-D XY support representative of a raster cell.
    xx, yy = np.meshgrid(
        np.linspace(0.05, 0.95, 12),
        np.linspace(0.05, 0.95, 12),
    )

    x = xx.ravel()
    y = yy.ravel()

    z = (
        8.0
        + 1.7 * x
        - 0.8 * y
        + 0.25 * np.sin(7.0 * x)
        + 0.15 * np.cos(5.0 * y)
    )

    local = _sigma_z(x, y, z)

    projected = _sigma_z(
        x + 478600.0,
        y + 5427100.0,
        z,
    )

    assert np.isfinite(local)
    assert np.isfinite(projected)

    assert np.isclose(
        local,
        projected,
        rtol=1e-8,
        atol=1e-9,
    )


def test_vertical_profile_mathematical_contract():
    import numpy as np
    from fastgc.structure import _vertical_profile_metrics

    # Three occupied 1-m layers with equal probability.
    z = np.array([
        0.2, 0.3,
        1.2, 1.3,
        2.2, 2.3,
    ])

    m = _vertical_profile_metrics(z, 1.0)

    assert np.isclose(m["vertical_entropy"], np.log(3.0))
    assert m["vertical_richness"] == 3.0
    assert np.isclose(m["vertical_evenness"], 1.0)
    assert np.isclose(m["vertical_effective_layers"], 3.0)
    assert np.isclose(m["vertical_dominance"], 1.0 / 3.0)
    assert np.isclose(m["vertical_simpson"], 2.0 / 3.0)
    assert np.isclose(m["vertical_occupancy_fraction"], 1.0)

    assert m["vertical_gap_count"] == 0.0
    assert m["vertical_gap_fraction"] == 0.0
    assert m["max_vertical_gap_m"] == 0.0


def test_vertical_profile_internal_gap_contract():
    import numpy as np
    from fastgc.structure import _vertical_profile_metrics

    # Occupied layers 0 and 2, with one internal empty 1-m layer.
    z = np.array([0.2, 0.3, 2.2, 2.3])

    m = _vertical_profile_metrics(z, 1.0)

    assert m["vertical_richness"] == 2.0
    assert np.isclose(m["vertical_entropy"], np.log(2.0))
    assert np.isclose(m["vertical_evenness"], 1.0)
    assert np.isclose(m["vertical_effective_layers"], 2.0)

    assert m["vertical_gap_count"] == 1.0
    assert np.isclose(m["vertical_gap_fraction"], 1.0 / 3.0)
    assert np.isclose(m["vertical_occupancy_fraction"], 2.0 / 3.0)
    assert np.isclose(m["max_vertical_gap_m"], 1.0)


def test_fhd_vci_are_vertical_profile_aliases():
    import numpy as np
    from fastgc.structure import compute_structure_metrics

    x = np.array([0.1, 0.2, 0.3, 0.4])
    y = np.array([0.1, 0.2, 0.3, 0.4])
    z = np.array([0.6, 0.8, 1.6, 2.6])

    result = compute_structure_metrics(
        x,
        y,
        z,
        sensor_mode="ALS",
        res=1.0,
        min_h=0.5,
        bin_size=1.0,
        canopy_threshold=2.0,
    )

    m = result["metrics"]

    valid = np.isfinite(m["FHD"])

    assert np.allclose(
        m["FHD"][valid],
        m["vertical_entropy"][valid],
    )

    valid = np.isfinite(m["VCI"])

    assert np.allclose(
        m["VCI"][valid],
        m["vertical_evenness"][valid],
    )


def test_height_statistics_mathematical_contract():
    import numpy as np
    from fastgc.structure import compute_structure_metrics

    z = np.array([
        0.6, 0.8, 1.1, 1.5, 2.0,
        2.8, 3.7, 5.0, 7.2, 9.5,
    ], dtype=float)

    x = np.linspace(0.05, 0.95, z.size)
    y = np.linspace(0.95, 0.05, z.size)

    result = compute_structure_metrics(
        x, y, z,
        sensor_mode="ALS",
        res=1.0,
        min_h=0.5,
        bin_size=0.5,
        canopy_threshold=2.0,
    )

    m = result["metrics"]

    def value(name):
        a = m[name]
        v = a[np.isfinite(a)]
        assert v.size == 1
        return float(v[0])

    percentile_names = [
        "z_p01", "z_p05", "z_p10", "z_p20",
        "z_p25", "z_p30", "z_p40", "z_p50",
        "z_p60", "z_p70", "z_p75", "z_p80",
        "z_p90", "z_p95", "z_p99",
    ]

    sequence = (
        [value("z_min")]
        + [value(n) for n in percentile_names]
        + [value("z_max")]
    )

    assert all(
        left <= right
        for left, right in zip(sequence[:-1], sequence[1:])
    )

    assert np.isclose(value("z_median"), value("z_p50"))

    assert np.isclose(
        value("z_iqr"),
        value("z_p75") - value("z_p25"),
    )

    assert np.isclose(
        value("z_range"),
        value("z_max") - value("z_min"),
    )

    assert np.isclose(
        value("z_variance"),
        value("z_sd") ** 2,
        rtol=1e-5,
        atol=1e-6,
    )

    assert np.isclose(
        value("z_cv"),
        value("z_sd") / value("z_mean"),
        rtol=1e-5,
        atol=1e-6,
    )


def test_population_skewness_and_excess_kurtosis_contract():
    import numpy as np
    from fastgc.structure import compute_structure_metrics

    z = np.array([
        0.5, 1.0, 1.5, 2.0,
        3.0, 5.0, 8.0, 13.0,
    ], dtype=float)

    x = np.linspace(0.05, 0.95, z.size)
    y = np.linspace(0.95, 0.05, z.size)

    result = compute_structure_metrics(
        x, y, z,
        sensor_mode="ALS",
        res=1.0,
        min_h=0.5,
        bin_size=0.5,
        canopy_threshold=2.0,
    )

    m = result["metrics"]

    def value(name):
        a = m[name]
        v = a[np.isfinite(a)]
        assert v.size == 1
        return float(v[0])

    mean = np.mean(z)
    centered = z - mean
    sd = np.std(z)

    expected_skew = np.mean(centered ** 3) / (sd ** 3)
    expected_excess = np.mean(centered ** 4) / (sd ** 4) - 3.0

    assert np.isclose(
        value("z_skewness"),
        expected_skew,
        rtol=1e-6,
        atol=1e-7,
    )

    assert np.isclose(
        value("z_kurtosis"),
        expected_excess,
        rtol=1e-6,
        atol=1e-7,
    )

    assert np.isclose(
        value("z_excess_kurtosis"),
        expected_excess,
        rtol=1e-6,
        atol=1e-7,
    )

    assert np.isclose(
        value("z_kurtosis"),
        value("z_excess_kurtosis"),
        rtol=0.0,
        atol=0.0,
    )


def test_intensity_and_return_metrics_follow_vegetation_mask():
    import numpy as np
    from fastgc.structure import compute_structure_metrics

    # First two points are deliberately below min_h and must not
    # contribute to intensity or return metrics.
    z = np.array([0.1, 0.2, 1.0, 2.0, 3.0, 4.0], dtype=float)
    x = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60])
    y = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60])

    intensity = np.array([1000, 2000, 10, 20, 30, 40], dtype=float)
    return_number = np.array([1, 1, 1, 2, 1, 3], dtype=int)
    number_of_returns = np.array([1, 1, 2, 2, 3, 3], dtype=int)

    result = compute_structure_metrics(
        x, y, z,
        sensor_mode="ALS",
        res=1.0,
        min_h=0.5,
        bin_size=0.5,
        canopy_threshold=2.0,
        intensity=intensity,
        return_number=return_number,
        number_of_returns=number_of_returns,
    )

    m = result["metrics"]

    def value(name):
        a = m[name]
        v = a[np.isfinite(a)]
        assert v.size == 1
        return float(v[0])

    retained_intensity = np.array([10, 20, 30, 40], dtype=float)

    assert np.isclose(value("intensity_mean"), retained_intensity.mean())
    assert np.isclose(value("intensity_sd"), retained_intensity.std(ddof=0))
    assert np.isclose(value("intensity_p50"), np.percentile(retained_intensity, 50))
    assert np.isclose(value("intensity_p95"), np.percentile(retained_intensity, 95))

    # Retained return numbers are [1, 2, 1, 3].
    assert np.isclose(value("first_return_fraction"), 0.5)
    assert np.isclose(value("multi_return_fraction"), 0.5)
    assert np.isclose(
        value("first_return_fraction") + value("multi_return_fraction"),
        1.0,
    )

    # This is the point-weighted mean LAS number_of_returns attribute
    # for retained points: [2, 2, 3, 3].
    assert np.isclose(value("mean_number_of_returns"), 2.5)


def test_all_writer_omits_unavailable_optional_metrics(tmp_path):
    import numpy as np
    from fastgc.structure import (
        compute_structure_metrics,
        write_structure_rasters,
        _filter_metrics,
    )

    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)
    z = np.array([0.6, 1.0, 2.0, 3.0, 5.0], dtype=float)

    # Deliberately provide no intensity or LAS return attributes.
    result = compute_structure_metrics(
        x, y, z,
        sensor_mode="ALS",
        res=1.0,
        min_h=0.5,
        bin_size=0.5,
        canopy_threshold=2.0,
    )

    optional = {
        "intensity_mean",
        "intensity_sd",
        "intensity_p50",
        "intensity_p95",
        "first_return_fraction",
        "multi_return_fraction",
        "mean_number_of_returns",
    }

    assert optional.isdisjoint(result["metrics"])

    result["metrics"] = _filter_metrics(result["metrics"], ["all"])
    assert optional.isdisjoint(result["metrics"])

    written = write_structure_rasters(
        result,
        tmp_path,
        crs=None,
        prefix="no_optional",
    )

    assert optional.isdisjoint(written)
    assert set(written) == set(result["metrics"])
    assert written
    assert all(path.exists() for path in written.values())

