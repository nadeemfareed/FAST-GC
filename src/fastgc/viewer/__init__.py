"""
Optional FAST-GC visualization infrastructure.

Viewer functionality consumes completed FAST-GC products only.
It has no dependency path back into scientific classification.
"""

from .manifest import (
    ProductRecord,
    ViewManifest,
)
from .discovery import (
    KNOWN_PRODUCTS,
    build_view_manifest_from_output,
    discover_product_files,
)

__all__ = [
    "ProductRecord",
    "ViewManifest",
    "KNOWN_PRODUCTS",
    "build_view_manifest_from_output",
    "discover_product_files",
    "RECIPE_SCHEMA",
    "RECIPE_SCHEMA_VERSION",
    "build_view_recipe",
    "write_view_recipe",
]

from .recipe import (
    RECIPE_SCHEMA,
    RECIPE_SCHEMA_VERSION,
    build_view_recipe,
    write_view_recipe,
)


from .server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    build_runtime_recipe,
    create_server,
)
