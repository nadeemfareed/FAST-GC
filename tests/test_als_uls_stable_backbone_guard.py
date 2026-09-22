from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _src(name: str) -> str:
    return (ROOT / "src" / "fastgc" / name).read_text(encoding="utf-8")


def test_als_uls_core_stays_in_invert_vote():
    text = _src("invert_vote.py")
    assert "def build_surface_invert_vote" in text
    assert "def classify_by_surface" in text
    assert "_normal_vote_surface" in text
    assert "_local_plane_vote" in text
    assert "_robust_quadratic_fit" in text
    assert "terrain_manifold" not in text


def test_tls_file_is_not_replaced_by_als_uls_patch():
    io = _src("io_las.py")
    assert 'if sm == "TLS"' in io
    assert 'elif sm in {"ULS", "ALS"}' in io
    assert "build_surface_invert_vote" in io
    assert "build_tls_surface_invert_dsm_vote" in io
