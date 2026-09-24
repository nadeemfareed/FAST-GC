from pathlib import Path

from fastgc import core


def test_derive_only_structure_accepts_direct_normalized_las(
    tmp_path,
    monkeypatch,
):
    normalized = tmp_path / "Plot_001_FAST_NORMALIZED.las"
    normalized.write_bytes(b"dummy")

    calls = []

    def fake_structure(*args, **kwargs):
        calls.append((args, kwargs))
        return tmp_path / "FAST_STRUCTURE"

    monkeypatch.setattr(core, "run_structure_from_root", fake_structure)

    # This must never be reached for an already-normalized LAS.
    def fail_normalization(*args, **kwargs):
        raise AssertionError(
            "Direct FAST_NORMALIZED input was incorrectly sent "
            "through normalization derivation."
        )

    monkeypatch.setattr(
        core,
        "derive_products_from_classified_root",
        fail_normalization,
    )

    core.run_fastgc(
        in_path=str(normalized),
        out_dir=None,
        sensor_mode="ALS",
        products=["FAST_STRUCTURE"],
        workflow="derive-only",
        structure_products=["all"],
        structure_res=1.0,
        structure_min_h=0.5,
        structure_bin_size=0.5,
        canopy_thr=2.0,
        overwrite=True,
    )

    assert len(calls) == 1

    args, kwargs = calls[0]
    source = kwargs.get("source_root")
    if source is None and args:
        source = args[0]

    assert Path(source) == normalized


def test_derive_only_structure_direct_normalized_laz_is_case_insensitive(
    tmp_path,
    monkeypatch,
):
    normalized = tmp_path / "plot_002_fast_normalized.laz"
    normalized.write_bytes(b"dummy")

    calls = []

    def fake_structure(*args, **kwargs):
        calls.append((args, kwargs))
        return tmp_path / "FAST_STRUCTURE"

    monkeypatch.setattr(core, "run_structure_from_root", fake_structure)

    def fail_normalization(*args, **kwargs):
        raise AssertionError(
            "Direct FAST_NORMALIZED input was incorrectly sent "
            "through normalization derivation."
        )

    monkeypatch.setattr(
        core,
        "derive_products_from_classified_root",
        fail_normalization,
    )

    core.run_fastgc(
        in_path=str(normalized),
        out_dir=None,
        sensor_mode="ALS",
        products=["FAST_STRUCTURE"],
        workflow="derive-only",
        structure_products=["all"],
        overwrite=True,
    )

    assert len(calls) == 1
    _, kwargs = calls[0]
    assert Path(kwargs["source_root"]) == normalized
