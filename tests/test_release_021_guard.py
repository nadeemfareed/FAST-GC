from __future__ import annotations

import ast
from pathlib import Path

import fastgc


ROOT = Path(__file__).resolve().parents[1]


def _cli_option_strings() -> set[str]:
    tree = ast.parse((ROOT / "src" / "fastgc" / "cli.py").read_text(encoding="utf-8"))
    opts: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                opts.add(arg.value)
    return opts


def test_release_identity_is_021():
    assert fastgc.__version__ == "0.2.1"
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.2.1"' in pyproject


def test_published_command_surface_is_preserved():
    options = _cli_option_strings()
    required = {
        "--in_path",
        "--out_dir",
        "--sensor_mode",
        "--workflow",
        "--tile_size_m",
        "--buffer_m",
        "--products",
        "--grid_res",
        "--dem_method",
        "--dsm_method",
        "--chm_method",
        "--jobs",
        "--joblib_backend",
        "--overwrite",
        "--overwrite_tiles",
        "--apply_fp_fix",
        "--no_fp_fix",
    }
    assert required <= options


def test_release_provenance_files_exist():
    assert (ROOT / "CITATION.cff").is_file()
    assert (ROOT / "RELEASE_PROVENANCE.md").is_file()
