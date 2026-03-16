import subprocess
import sys


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "fastgc.cli", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0

    stdout = result.stdout
    assert "FAST-GC" in stdout
    assert "--workflow" in stdout
    assert "--products" in stdout
    assert "--jobs" in stdout
    assert "--joblib_backend" in stdout or "--joblib-backend" in stdout
    assert "--chm_method" in stdout or "--chm-method" in stdout
    assert "--terrain_products" in stdout or "--terrain-products" in stdout