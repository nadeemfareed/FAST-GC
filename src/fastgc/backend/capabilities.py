"""Runtime compute capability discovery for FAST-GC."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import platform
import sys


@dataclass(frozen=True)
class BackendCapabilities:
    platform: str
    machine: str
    python: str
    reference_available: bool
    rust_available: bool
    cuda_runtime_detected: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def detect_capabilities() -> BackendCapabilities:
    """
    Report available FAST-GC execution infrastructure.

    Detection only. This function does not select a scientific
    implementation and does not alter processing behavior.
    """

    # Use the same loader as the validated production native kernels.
    # This avoids capability reporting drifting from actual execution.
    try:
        from .native import native_available
        rust_available = bool(native_available())
    except Exception:
        rust_available = False

    # CUDA support has deliberately NOT been implemented yet.
    # Merely having a third-party CUDA package installed is not
    # sufficient to declare the FAST-GC CUDA backend available.
    cuda_runtime_detected = False

    return BackendCapabilities(
        platform=platform.system(),
        machine=platform.machine(),
        python=".".join(map(str, sys.version_info[:3])),
        reference_available=True,
        rust_available=rust_available,
        cuda_runtime_detected=cuda_runtime_detected,
    )
