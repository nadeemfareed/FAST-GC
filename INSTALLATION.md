# FAST-GC installation and environment isolation

FAST-GC 0.2.1 supports Python 3.12, 3.13, and 3.14.

## Important packaging rule

`pip install` installs into the Python environment from which pip is run. A Python
package cannot safely force pip to create and switch to a different interpreter
environment during its own installation.

FAST-GC therefore provides three supported isolated installation paths.

## 1. Editable source development (recommended for repository work)

Windows:

```powershell
Set-Location "C:\git\FAST-GC"
.\install_editable.ps1
```

This creates `C:\git\FAST-GC\.venv`, installs the validated dependency set
inside it, and installs the current checkout with `pip -e`.

Run FAST-GC without activation:

```powershell
.\fastgc-dev.ps1 --help
```

Ordinary edits under `src\fastgc` are immediately visible because the package is
editable. Re-run the installer only when `pyproject.toml`, dependency constraints,
or the Python interpreter changes.

Linux/macOS:

```bash
./install_editable.sh
./fastgc-dev.sh --help
```

## 2. Conda source development

```bash
conda env create -f environment.yml
conda activate fastgc
```

The environment file creates a clean Conda environment and installs this checkout
in editable mode.

## 3. PyPI application installation with isolation

FAST-GC is a command-line application. For a globally available command with an
isolated Python environment, use pipx:

```bash
pipx install fastgc
```

pipx creates a dedicated virtual environment for FAST-GC and exposes the `fastgc`
command on PATH.

Traditional:

```bash
pip install fastgc
```

remains supported, but it intentionally installs into the currently active Python
environment. Use it only inside a venv/Conda environment that you control.

## Release builds

Local `python -m build` is not the authoritative release path. Official PyPI
artifacts should be built by GitHub Actions in a clean hosted Python environment
after the complete test suite passes. This avoids contamination from developer
machines and produces a traceable release artifact.

## Broken bootstrap Python

Creating an isolated environment still requires a functional base Python runtime.
The bootstrap scripts test `ssl`, `ctypes`, and `venv` before installation. If the
base interpreter itself is damaged, FAST-GC cannot repair that Python installation;
use the Conda route or a functioning CPython installation.
