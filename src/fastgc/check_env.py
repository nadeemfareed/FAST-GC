"""FAST-GC runtime environment diagnostics.

This module intentionally avoids importing all FAST-GC submodules so it can report
missing or incompatible dependencies instead of failing at the first import.
"""
from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass
from importlib import metadata
from typing import Iterable

from packaging.specifiers import SpecifierSet
from packaging.version import Version


@dataclass(frozen=True)
class RequirementCheck:
    distribution: str
    import_name: str
    specifier: str
    required: bool = True


CORE_REQUIREMENTS: tuple[RequirementCheck, ...] = (
    RequirementCheck("numpy", "numpy", ">=2.0,<3.0"),
    RequirementCheck("scipy", "scipy", ">=1.15,<2.0"),
    RequirementCheck("laspy", "laspy", ">=2.5.4,<3.0"),
    RequirementCheck("lazrs", "lazrs", ">=0.7,<1.0"),
    RequirementCheck("rasterio", "rasterio", ">=1.5,<2.0"),
    RequirementCheck("pyogrio", "pyogrio", ">=0.13,<1.0"),
    RequirementCheck("shapely", "shapely", ">=2.0.6,<3.0"),
    RequirementCheck("tqdm", "tqdm", ">=4.66,<5.0"),
    RequirementCheck("joblib", "joblib", ">=1.4,<2.0"),
    RequirementCheck("psutil", "psutil", ">=6.0,<8.0"),
)

OPTIONAL_REQUIREMENTS: tuple[RequirementCheck, ...] = (
    RequirementCheck("scikit-learn", "sklearn", ">=1.6,<2.0", required=False),
    RequirementCheck("torch", "torch", ">=2.7,<3.0", required=False),
)


def _check_requirement(req: RequirementCheck) -> tuple[bool, str]:
    try:
        version_text = metadata.version(req.distribution)
    except metadata.PackageNotFoundError:
        status = "MISSING" if req.required else "not installed (optional)"
        return (not req.required, status)

    try:
        importlib.import_module(req.import_name)
    except Exception as exc:  # report binary/DLL errors as well as ImportError
        return False, f"{version_text}; import failed: {exc}"

    compatible = Version(version_text) in SpecifierSet(req.specifier)
    marker = "OK" if compatible else f"OUTSIDE {req.specifier}"
    return compatible, f"{version_text} ({marker})"


def _print_checks(title: str, checks: Iterable[RequirementCheck]) -> bool:
    print(f"\n{title}")
    print("-" * len(title))
    all_ok = True
    for req in checks:
        ok, detail = _check_requirement(req)
        all_ok = all_ok and ok
        print(f"{req.distribution:16s} {detail}")
    return all_ok


def main() -> int:
    print("FAST-GC environment report")
    print("==========================")
    print(f"Python      : {sys.version.split()[0]}")
    print(f"Executable  : {sys.executable}")
    print(f"Platform    : {platform.platform()}")
    print(f"Architecture: {platform.machine()}")

    py_ok = (3, 12) <= sys.version_info[:2] < (3, 15)
    if not py_ok:
        print("\nPython version is outside the supported range >=3.12,<3.15.")

    core_ok = _print_checks("Core dependencies", CORE_REQUIREMENTS)
    _print_checks("Optional ML dependencies", OPTIONAL_REQUIREMENTS)

    ok = py_ok and core_ok
    print("\nResult:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
