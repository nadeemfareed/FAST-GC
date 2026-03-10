import subprocess
import sys

def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "fastgc.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0