"""Central FAST-GC compute-backend selection.

This module changes execution infrastructure only. Scientific behavior
continues to be defined by the validated FAST-GC reference algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

from .native import native_available


COMPUTE_BACKEND_CHOICES = ("auto", "native", "reference")
DEFAULT_COMPUTE_BACKEND = "auto"


@dataclass(frozen=True)
class ComputeBackend:
    requested: str
    resolved: str
    native_available: bool


def resolve_compute_backend(
    requested: str = DEFAULT_COMPUTE_BACKEND,
) -> ComputeBackend:
    requested = str(requested).strip().lower()

    if requested not in COMPUTE_BACKEND_CHOICES:
        raise ValueError(
            f"Unknown FAST-GC compute backend: {requested!r}. "
            f"Choose from {COMPUTE_BACKEND_CHOICES}."
        )

    available = bool(native_available())

    if requested == "reference":
        resolved = "reference"

    elif requested == "native":
        if not available:
            raise RuntimeError(
                "FAST-GC native compute backend was requested, "
                "but the validated native extension is unavailable."
            )
        resolved = "native"

    else:
        resolved = "native" if available else "reference"

    return ComputeBackend(
        requested=requested,
        resolved=resolved,
        native_available=available,
    )


def configure_compute_backend(
    requested: str = DEFAULT_COMPUTE_BACKEND,
) -> ComputeBackend:
    """Configure existing validated execution routes.

    No scientific thresholds, neighborhoods, stage ordering, model
    decisions, or classification rules are changed here.
    """

    state = resolve_compute_backend(requested)

    if state.resolved == "native":
        # Validated ALS native surface-plane route.
        os.environ["FASTGC_SURFACE_PLANE_BACKEND"] = "native"

        # Validated ULS native support route.
        os.environ["FASTGC_ULS_SUPPORT_BACKEND"] = "native_rayon"

        # Validated TLS native robust-vote route.
        os.environ["FASTGC_TLS_ROBUST_BACKEND"] = "native"

    else:
        os.environ["FASTGC_SURFACE_PLANE_BACKEND"] = "reference"
        os.environ["FASTGC_ULS_SUPPORT_BACKEND"] = "reference"
        os.environ["FASTGC_TLS_ROBUST_BACKEND"] = "reference"

    return state


def configure_sensor_compute_backend(sensor_mode: str) -> ComputeBackend:
    """Automatically configure only the validated native route for a sensor.

    This function changes execution routing only. It does not change
    scientific thresholds, neighborhoods, stage ordering, models, or
    classification decisions.
    """
    sensor = str(sensor_mode).strip().upper()
    if sensor not in {"ALS", "ULS", "TLS"}:
        raise ValueError(f"Unknown FAST-GC sensor mode: {sensor_mode!r}")

    state = resolve_compute_backend("auto")

    # Explicitly reset every sensor-specific optional route first so
    # inherited shell environment variables cannot leak across modes.
    os.environ["FASTGC_SURFACE_PLANE_BACKEND"] = "reference"
    os.environ["FASTGC_ULS_SUPPORT_BACKEND"] = "reference"
    os.environ["FASTGC_TLS_ROBUST_BACKEND"] = "reference"

    if state.resolved == "native":
        if sensor == "ALS":
            # Validated ALS native surface-plane backend.
            os.environ["FASTGC_SURFACE_PLANE_BACKEND"] = "native"
        elif sensor == "ULS":
            # Validated ULS native support backend.
            os.environ["FASTGC_ULS_SUPPORT_BACKEND"] = "native_rayon"
        elif sensor == "TLS":
            # Validated TLS native robust-vote backend.
            os.environ["FASTGC_TLS_ROBUST_BACKEND"] = "native"

    return state
