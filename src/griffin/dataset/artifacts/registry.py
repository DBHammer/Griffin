"""Artifact-kind registry — maps the discriminator string to the artifact class."""

from __future__ import annotations

from griffin.infra.registry import FrozenRegistry, Registry

from .base import (
    Artifact,
    VectorIndexArtifact,
)



def _build() -> FrozenRegistry[str, type[Artifact]]:
    r: Registry[str, type[Artifact]] = Registry()
    r.register("vector_index", VectorIndexArtifact)
    return r.freeze()



ARTIFACT_KIND_REGISTRY: FrozenRegistry[str, type[Artifact]] = _build()


__all__ = ["ARTIFACT_KIND_REGISTRY"]
