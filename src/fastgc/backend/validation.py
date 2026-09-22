"""Exact-equivalence utilities for future FAST-GC backends."""

from __future__ import annotations

import numpy as np


class BackendEquivalenceError(AssertionError):
    """Raised when an accelerated kernel differs from the reference."""


def assert_array_exact(
    reference,
    candidate,
    *,
    name: str = "array",
) -> None:
    """
    Require exact array equivalence including shape and dtype.

    Floating-point NaNs are considered equal only when they occur
    in identical locations.
    """

    ref = np.asarray(reference)
    got = np.asarray(candidate)

    if ref.shape != got.shape:
        raise BackendEquivalenceError(
            f"{name}: shape differs: {ref.shape} != {got.shape}"
        )

    if ref.dtype != got.dtype:
        raise BackendEquivalenceError(
            f"{name}: dtype differs: {ref.dtype} != {got.dtype}"
        )

    if np.issubdtype(ref.dtype, np.floating):
        equal = np.array_equal(ref, got, equal_nan=True)
    else:
        equal = np.array_equal(ref, got)

    if not equal:
        raise BackendEquivalenceError(
            f"{name}: values differ from FAST-GC reference"
        )


def assert_classification_exact(reference, candidate) -> None:
    """Require exact final LAS Classification equivalence."""

    assert_array_exact(
        reference,
        candidate,
        name="Classification",
    )
