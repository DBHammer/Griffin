"""Public dataset reader."""

from __future__ import annotations

import pathlib

import msgspec

from griffin.dataset.artifacts.base import Artifact
from griffin.dataset.deterministic import (
    TableDeterministicMetadata,
)
from griffin.dataset.deterministic_checks import (
    validate_deterministic_fields,
    validate_field_placements,
    validate_region_layout,
    validate_vector_attribute_couplings,
)
from griffin.dataset.entities.table import Table
from griffin.dataset.manifest import DatasetManifest
from griffin.dataset.relations.base import Relation
from griffin.dataset.statistics.base import EntityMetrics
from griffin.infra import json_io
from griffin.infra.errors import ContractViolation
from griffin.model_ids import EntityId

from .layout import manifest_path


class Dataset(msgspec.Struct, omit_defaults=True):
    """Read-only view of a persisted dataset."""

    root: str  # absolute path
    manifest: DatasetManifest
    _metrics_cache: tuple[EntityMetrics, ...] | None = None
    _deterministic_cache: tuple[TableDeterministicMetadata, ...] | None = None

    # ---------- public API ----------
    @classmethod
    def open(cls, root: pathlib.Path | str) -> "Dataset":
        root_path = pathlib.Path(root).resolve()
        m_path = manifest_path(root_path)
        if not m_path.exists():
            raise ContractViolation(f"manifest 文件缺失：{m_path}")
        manifest = json_io.read_json(m_path, DatasetManifest)
        dataset = cls(root=str(root_path), manifest=manifest)
        dataset._validate_contract()
        return dataset

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def data_model(self) -> str:
        return self.manifest.data_model

    @property
    def output_format(self) -> str:
        return self.manifest.output_format

    @property
    def dialect(self) -> str:
        return self.manifest.dialect

    def tables(self) -> tuple[Table, ...]:
        return self.manifest.tables

    def entity(self, owner: EntityId) -> Table:
        for entity in self.all_entities():
            if entity.id == owner:
                return entity
        raise ContractViolation(f"unknown entity: {owner}")

    def field_bearing_entity(self, owner: EntityId) -> Table:
        return self.entity(owner)

    def all_entities(self) -> tuple[Table, ...]:
        return self.manifest.tables

    def entity_count(self) -> int:
        return len(self.manifest.tables)

    def metrics(self) -> tuple[EntityMetrics, ...]:
        if self._metrics_cache is None:
            root_path = pathlib.Path(self.root)
            self._metrics_cache = tuple(
                json_io.read_json(root_path / metric_file.relative_path, EntityMetrics)
                for metric_file in self.manifest.metric_files
            )
        return self._metrics_cache

    def artifacts(self) -> tuple[Artifact, ...]:
        return self.manifest.artifacts

    def relations(self) -> tuple[Relation, ...]:
        return self.manifest.relations

    def deterministic_tables(self) -> tuple[TableDeterministicMetadata, ...]:
        if self._deterministic_cache is None:
            root_path = pathlib.Path(self.root)
            self._deterministic_cache = tuple(
                json_io.read_json(root_path / relative_path, TableDeterministicMetadata)
                for relative_path in self.manifest.deterministic_table_files
            )
        return self._deterministic_cache

    def deterministic_table(self, table_name: str) -> TableDeterministicMetadata | None:
        for metadata in self.deterministic_tables():
            if metadata.table_name == table_name:
                return metadata
        return None

    def _validate_contract(self) -> None:
        if self.manifest.data_model != "relational":
            raise ContractViolation("dataset data_model must be relational")
        if self.manifest.dialect != "postgresql":
            raise ContractViolation("dataset dialect must be postgresql")
        table_metadata = self.deterministic_tables()
        table_by_name = {table.id.name: table for table in self.manifest.tables}
        if self.manifest.tables and not self.manifest.deterministic_table_files:
            raise ContractViolation("relational dataset manifest must include deterministic_table_files")
        if len(table_metadata) != len(self.manifest.tables):
            raise ContractViolation(
                "relational dataset deterministic table metadata must cover every table exactly once"
            )
        seen_tables: set[str] = set()
        for metadata in table_metadata:
            table = table_by_name.get(metadata.table_name)
            if table is None:
                raise ContractViolation(
                    f"deterministic table metadata references unknown table: {metadata.table_name}"
                )
            if metadata.table_name in seen_tables:
                raise ContractViolation(
                    f"deterministic table metadata duplicated for: {metadata.table_name}"
                )
            seen_tables.add(metadata.table_name)
            validate_deterministic_fields(
                metadata.supported_fields,
                row_count=metadata.row_count,
                fields=table.fields,
                owner=f"table deterministic metadata {metadata.table_name}",
            )
            validate_field_placements(
                metadata.field_placements,
                row_count=metadata.row_count,
                models=metadata.supported_fields,
                owner=f"table deterministic placement {metadata.table_name}",
            )
            validate_region_layout(
                metadata.region_layout,
                row_count=metadata.row_count,
                fields=table.fields,
                owner=f"table deterministic region_layout {metadata.table_name}",
            )
            validate_vector_attribute_couplings(
                metadata.vector_attribute_couplings,
                row_count=metadata.row_count,
                models=metadata.supported_fields,
                placements=metadata.field_placements,
                region_layout=metadata.region_layout,
                fields=table.fields,
                owner=f"table deterministic vector_attribute_couplings {metadata.table_name}",
            )

__all__ = ["Dataset"]
