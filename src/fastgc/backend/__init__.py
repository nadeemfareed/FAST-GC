"""
FAST-GC computational backend infrastructure.

This package does not define FAST-GC scientific algorithms.
It provides infrastructure for executing mathematically equivalent
implementations of existing FAST-GC computational kernels.

The existing Python/NumPy/SciPy implementation remains the
scientific reference.
"""

from .capabilities import BackendCapabilities, detect_capabilities

__all__ = [
    "BackendCapabilities",
    "detect_capabilities",
]
