from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]

def test_runtime_environment_contract():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["version"] == "0.2.1"
    assert data["project"]["requires-python"] == ">=3.12,<3.15"
    deps = "\n".join(data["project"]["dependencies"]).lower()
    assert "pyogrio" in deps
    assert "fiona" not in deps

def test_release_environment_assets_exist():
    required = [
        "pyproject.toml",
        "requirements.txt",
        "environment.yml",
        "constraints/runtime-0.2.1.txt",
        "INSTALLATION.md",
    ]
    for rel in required:
        assert (ROOT / rel).exists(), rel

def test_conda_environment_is_not_base():
    text = (ROOT / "environment.yml").read_text(encoding="utf-8")
    assert "name: fastgc" in text
    assert "-e . --no-deps" in text

