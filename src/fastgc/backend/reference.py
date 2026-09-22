"""
Reference-backend marker.

FAST-GC's existing Python/NumPy/SciPy implementations remain
authoritative. Existing scientific functions are intentionally
NOT copied into this module.

Future accelerated kernels must be validated directly against
the corresponding existing implementation.
"""

BACKEND_NAME = "reference"
IS_SCIENTIFIC_REFERENCE = True
