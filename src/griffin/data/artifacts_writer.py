"""Real artifact writer for exact vector indexes."""

from __future__ import annotations

import pathlib

from griffin.data.entity_builders import ColumnMapping
from griffin.dataset.artifacts.base import Artifact, VectorIndexArtifact
from griffin.dataset.layout import artifact_path
from griffin.infra.errors import ContractViolation
from griffin.ir.data_type import VectorType
from griffin.ir.field import Field
from griffin.model_ids import ArtifactId, EntityId
from griffin.vector_hardness_core import build_exact_faiss_index, vector_column_matrix


def write_vector_index(
    owner: EntityId,
    field: Field,
    columns: ColumnMapping,
    dataset_root: pathlib.Path,
) -> Artifact:
    """Persist a faiss index for `field` (must be VectorType) to disk.

    Returns a VectorIndexArtifact pointing at the index file.
    """
    if not isinstance(field.type, VectorType):
        raise ContractViolation(
            f"write_vector_index called on non-vector field {field.id}"
        )
    import faiss

    raw = columns[field.id.name]
    arr = vector_column_matrix(raw, field)
    if arr.shape[0] == 0:
        raise ContractViolation(f"vector field {field.id} has no rows to index")
    index = build_exact_faiss_index(arr, metric=field.type.metric)
    out = artifact_path(dataset_root, "vector_index", owner.name, f"{field.id.name}.faiss")
    out.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out))

    return VectorIndexArtifact(
        id=ArtifactId(kind="vector_index", owner=owner, name=field.id.name),
        relative_path=str(out.relative_to(dataset_root)),
        index_type="exact",
        metric=field.type.metric.value,
        dimension=field.type.dimension,
    )


__all__ = ["write_vector_index"]
