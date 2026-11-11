"""Single source of truth for `SurfaceFeature`.

Keep `SurfaceFeature` centralized here so planner code consumes one capability
vocabulary instead of inventing dialect flags in multiple places.
"""

from __future__ import annotations

import enum

import msgspec


class SurfaceFeature(enum.Enum):
    """Native dialect features. Add new variants here ONLY."""

    SCALAR_BTREE_FILTER = "scalar_btree_filter"
    SCALAR_RANGE_FILTER = "scalar_range_filter"
    VECTOR_EXACT_SCAN = "vector_exact_scan"
    AGGREGATION_GROUPBY = "aggregation_groupby"


class DialectCapabilities(msgspec.Struct, frozen=True, omit_defaults=True):
    """Per-dialect set of supported `SurfaceFeature`s."""

    dialect_name: str
    supported: tuple[SurfaceFeature, ...]
    query_file_extension: str = "sql"


__all__ = ["SurfaceFeature", "DialectCapabilities"]
