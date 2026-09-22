from __future__ import annotations

"""FAST-GC CHM method registry.

Existing v0.2.1 methods are declared here so the CLI and workflow retain the
same choices. New native methods can register themselves without modifying
``core.py`` or ``cli.py``.
"""

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
import pkgutil
from typing import Any

LEGACY_METHODS = {
    "p2r",
    "p99",
    "tin",
    "pitfree",
    "adaptive_pitfree",
    "csf_chm",
    "spikefree",
    "percentile",
    "percentile_top",
    "percentile_band",
}

LEGACY_SURFACE_METHODS = {
    "p2r",
    "p99",
    "tin",
    "pitfree",
    "adaptive_pitfree",
    "csf_chm",
}

CHM_SMOOTH_CHOICES = {"none", "median", "gaussian"}

# Mutable shared sets are intentional: argparse receives the same objects and
# future built-in registrations can extend them during package import.
CHM_METHOD_CHOICES = set(LEGACY_METHODS)
CHM_SURFACE_METHOD_CHOICES = set(LEGACY_SURFACE_METHODS)

_ROOT_BUILDERS: dict[str, Callable[..., str]] = {}
_DISCOVERED = False


def register_root_method(
    name: str,
    builder: Callable[..., str],
    *,
    surface_method: bool = True,
) -> None:
    """Register a new full-root CHM method.

    ``builder`` must accept the same keyword contract as
    :func:`fastgc.chm.api.build_chm_from_normalized_root` and return the output
    root path. Existing method names cannot be replaced accidentally.
    """
    key = str(name).strip().lower()
    if not key:
        raise ValueError("CHM method name cannot be empty.")
    if key in LEGACY_METHODS or key in _ROOT_BUILDERS:
        raise ValueError(f"CHM method already registered: {key}")
    _ROOT_BUILDERS[key] = builder
    CHM_METHOD_CHOICES.add(key)
    if surface_method:
        CHM_SURFACE_METHOD_CHOICES.add(key)


def get_root_builder(name: str) -> Callable[..., str] | None:
    return _ROOT_BUILDERS.get(str(name).strip().lower())


def registered_extension_methods() -> tuple[str, ...]:
    return tuple(sorted(_ROOT_BUILDERS))


def discover_builtin_methods() -> None:
    """Import modules under ``surfaces`` once.

    A future algorithm becomes discoverable by adding one module containing a
    ``register_root_method(...)`` call. No central switch statement is needed.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True

    package_name = f"{__package__}.surfaces"
    pkg = import_module(package_name)
    paths = getattr(pkg, "__path__", None)
    if paths is None:
        return

    for mod in pkgutil.iter_modules(paths):
        if mod.name.startswith("_"):
            continue
        import_module(f"{package_name}.{mod.name}")


__all__ = [
    "CHM_METHOD_CHOICES",
    "CHM_SURFACE_METHOD_CHOICES",
    "CHM_SMOOTH_CHOICES",
    "LEGACY_METHODS",
    "LEGACY_SURFACE_METHODS",
    "register_root_method",
    "get_root_builder",
    "registered_extension_methods",
    "discover_builtin_methods",
]
