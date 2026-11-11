"""Dataset manifest types."""

from __future__ import annotations

import enum

import msgspec

from griffin.dataset.artifacts.base import Artifact
from griffin.dataset.entities.table import Table
from griffin.dataset.relations.base import Relation
from griffin.model_ids import EntityId


class ManifestFormat(enum.Enum):
    """On-disk encoding of the dataset manifest."""

    JSON_V3 = "json/v3"


class MetricsFileRef(msgspec.Struct, frozen=True, omit_defaults=True):
    owner: EntityId
    relative_path: str


class DatasetManifest(msgspec.Struct, frozen=True, omit_defaults=True):
    name: str
    format: ManifestFormat
    data_model: str
    output_format: str
    dialect: str
    tables: tuple[Table, ...] = ()
    metric_files: tuple[MetricsFileRef, ...] = ()
    artifacts: tuple[Artifact, ...] = ()
    relations: tuple[Relation, ...] = ()
    deterministic_table_files: tuple[str, ...] = ()


__all__ = ["ManifestFormat", "MetricsFileRef", "DatasetManifest"]
