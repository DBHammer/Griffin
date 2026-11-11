"""Table/field-level exact vector index artifacts.

Artifacts are kept *typed* so downstream planners can pattern-match on them
instead of querying loose dicts. Each variant tags itself; the registry maps
tag → type for decoding.

"""

from __future__ import annotations

from typing import Union

import msgspec

from griffin.model_ids import ArtifactId


class VectorIndexArtifact(msgspec.Struct, frozen=True, tag="vector_index", tag_field="artifact_kind"):
    id: ArtifactId
    relative_path: str
    index_type: str  # current exact answer artifact: "exact"
    metric: str  # see VectorMetric.value
    dimension: int


Artifact = Union[
    VectorIndexArtifact,
]


__all__ = [
    "VectorIndexArtifact",
    "Artifact",
]
