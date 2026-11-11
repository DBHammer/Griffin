"""Relation-kind registry — maps discriminator string to relation class."""

from __future__ import annotations

from griffin.infra.registry import FrozenRegistry, Registry

from .base import (
    CorrelationRelation,
    CouplingRelation,
    Relation,
)


def _build() -> FrozenRegistry[str, type[Relation]]:
    r: Registry[str, type[Relation]] = Registry()
    r.register("correlation", CorrelationRelation)
    r.register("coupling", CouplingRelation)
    return r.freeze()


RELATION_KIND_REGISTRY: FrozenRegistry[str, type[Relation]] = _build()


__all__ = ["RELATION_KIND_REGISTRY"]
