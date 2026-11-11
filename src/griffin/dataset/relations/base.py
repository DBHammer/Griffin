"""Cross-field relations for relational tables."""

from __future__ import annotations

from typing import Union

import msgspec

from griffin.model_ids import FieldId, ParticipantRef, RelationId


class CorrelationRelation(msgspec.Struct, frozen=True, tag="correlation", tag_field="relation_kind"):
    id: RelationId
    field_a: FieldId
    field_b: FieldId
    coefficient: float

    def participants(self) -> tuple[ParticipantRef, ...]:
        return (self.field_a, self.field_b)


class CouplingRelation(msgspec.Struct, frozen=True, tag="coupling", tag_field="relation_kind"):
    id: RelationId
    fields: tuple[FieldId, ...]
    coupling_strength: float

    def participants(self) -> tuple[ParticipantRef, ...]:
        return tuple(self.fields)


Relation = Union[
    CorrelationRelation,
    CouplingRelation,
]


class RelationSpec(msgspec.Struct, frozen=True, omit_defaults=True):
    """Build-time request: 'I want a relation of this kind over these participants'.

    The data pipeline materializes the request into a concrete `Relation`.
    """

    kind: str  # registry key
    name: str
    participants: tuple[ParticipantRef, ...]


__all__ = [
    "CorrelationRelation",
    "CouplingRelation",
    "Relation",
    "RelationSpec",
]
