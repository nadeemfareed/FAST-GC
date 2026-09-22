from pathlib import Path

import fastgc.chm as chm
from fastgc.chm import registry


def test_chm_public_api_is_preserved():
    expected = {
        "p2r", "p99", "tin", "pitfree", "adaptive_pitfree", "csf_chm",
        "spikefree", "percentile", "percentile_top", "percentile_band",
    }
    assert expected.issubset(set(chm.CHM_METHOD_CHOICES))
    assert callable(chm.build_chm_from_normalized_root)
    assert callable(chm.build_chm_from_dem_and_dsm)
    assert callable(chm.resolve_normalized_root)
    assert callable(chm.chm_output_label)
    assert callable(chm.chm_method_output_dir)


def test_legacy_methods_are_not_silently_replaced():
    for name in registry.LEGACY_METHODS:
        assert registry.get_root_builder(name) is None


def test_chm_scaffold_is_present():
    root = Path(__file__).resolve().parents[1] / "src" / "fastgc" / "chm"
    for rel in (
        "api.py", "registry.py", "config.py", "schemas.py",
        "io", "preprocess", "surfaces", "fusion", "postprocess",
        "products", "quality",
    ):
        assert (root / rel).exists(), rel
