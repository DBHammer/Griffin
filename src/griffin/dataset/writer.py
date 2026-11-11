"""DatasetWriter — progressive build, atomic commit.

Usage:
    w = DatasetWriter.create(
        root,
        name="example",
        data_model="relational",
        output_format="parquet",
        dialect="postgresql",
    )
    w.add_table(table)
    w.add_metrics(entity_metrics)
    w.add_artifact(vector_artifact)
    w.add_relation(correlation_relation)
    w.commit()  # writes metadata/manifest.json
"""

from __future__ import annotations

import pathlib

from griffin.dataset.artifacts.base import Artifact
from griffin.dataset.deterministic import (
    TableDeterministicMetadata,
)
from griffin.dataset.deterministic_checks import (
    normalize_table_metadata,
    validate_deterministic_fields,
    validate_field_placements,
    validate_region_layout,
    validate_vector_attribute_couplings,
)
from griffin.dataset.entities.table import Table
from griffin.dataset.manifest import DatasetManifest, ManifestFormat, MetricsFileRef
from griffin.dataset.relations.base import Relation
from griffin.dataset.statistics.base import EntityMetrics
from griffin.infra import json_io
from griffin.infra.errors import ContractViolation
from griffin.model_ids import EntityId

from .layout import (
    deterministic_table_path,
    manifest_path,
    table_metrics_path,
)


class DatasetWriter:
    """Mutable builder for a persisted dataset.

    Not thread-safe. Construct, fill, commit, discard.
    """

    __slots__ = (
        "_root",
        "_name",
        "_data_model",
        "_output_format",
        "_dialect",
        "_tables",
        "_metric_files",
        "_artifacts",
        "_relations",
        "_deterministic_tables",
        "_committed",
    )

    def __init__(
        self,
        root: pathlib.Path,
        name: str,
        *,
        data_model: str,
        output_format: str,
        dialect: str,
    ) -> None:
        self._root = root
        self._name = name
        self._data_model = data_model
        self._output_format = output_format
        self._dialect = dialect
        self._tables: list[Table] = []
        self._metric_files: list[MetricsFileRef] = []
        self._artifacts: list[Artifact] = []
        self._relations: list[Relation] = []
        self._deterministic_tables: list[str] = []
        self._committed = False

    @classmethod
    def create(
        cls,
        root: pathlib.Path | str,
        *,
        name: str,
        data_model: str,
        output_format: str,
        dialect: str,
    ) -> "DatasetWriter":
        root_path = pathlib.Path(root).resolve()
        root_path.mkdir(parents=True, exist_ok=True)
        return cls(
            root_path,
            name,
            data_model=data_model,
            output_format=output_format,
            dialect=dialect,
        )

    @property
    def root(self) -> pathlib.Path:
        return self._root

    def add_table(self, table: Table) -> None:
        self._guard()
        self._ensure_unique(table.id)
        self._tables.append(table)

    def add_metrics(self, metrics: EntityMetrics) -> None:
        self._guard()
        if not self._has_owner(metrics.owner):
            raise ContractViolation(f"metrics references unknown owner: {metrics.owner}")
        path = self._metrics_path(metrics.owner)
        json_io.write_json(path, metrics)
        self._metric_files.append(
            MetricsFileRef(
                owner=metrics.owner,
                relative_path=str(path.relative_to(self._root)),
            )
        )

    def add_artifact(self, artifact: Artifact) -> None:
        self._guard()
        if not self._has_owner(artifact.id.owner):
            raise ContractViolation(f"artifact references unknown owner: {artifact.id.owner}")
        self._artifacts.append(artifact)

    def add_relation(self, relation: Relation) -> None:
        self._guard()
        self._relations.append(relation)

    def add_table_metadata(self, metadata: TableDeterministicMetadata) -> None:
        self._guard()
        metadata = normalize_table_metadata(metadata)
        table = next((table for table in self._tables if table.id.name == metadata.table_name), None)
        if table is None:
            raise ContractViolation(
                f"deterministic metadata references unknown table: {metadata.table_name}"
            )
        if metadata.row_ordinal_start != 0:
            raise ContractViolation("table deterministic metadata row_ordinal_start must be 0")
        validate_deterministic_fields(
            metadata.supported_fields,
            row_count=metadata.row_count,
            fields=table.fields,
            owner="table deterministic",
        )
        validate_field_placements(
            metadata.field_placements,
            row_count=metadata.row_count,
            models=metadata.supported_fields,
            owner="table deterministic placement",
        )
        validate_region_layout(
            metadata.region_layout,
            row_count=metadata.row_count,
            fields=table.fields,
            owner="table deterministic region_layout",
        )
        validate_vector_attribute_couplings(
            metadata.vector_attribute_couplings,
            row_count=metadata.row_count,
            models=metadata.supported_fields,
            placements=metadata.field_placements,
            region_layout=metadata.region_layout,
            fields=table.fields,
            owner="table deterministic vector_attribute_couplings",
        )
        path = deterministic_table_path(self._root, metadata.table_name)
        json_io.write_json(path, metadata)
        relative = str(path.relative_to(self._root))
        if relative not in self._deterministic_tables:
            self._deterministic_tables.append(relative)

    def commit(self) -> pathlib.Path:
        self._guard()
        self._validate_commit_contract()
        manifest = DatasetManifest(
            name=self._name,
            format=ManifestFormat.JSON_V3,
            data_model=self._data_model,
            output_format=self._output_format,
            dialect=self._dialect,
            tables=tuple(self._tables),
            metric_files=tuple(self._metric_files),
            artifacts=tuple(self._artifacts),
            relations=tuple(self._relations),
            deterministic_table_files=tuple(self._deterministic_tables),
        )
        path = manifest_path(self._root)
        json_io.write_json(path, manifest)
        self._committed = True
        return path

    def _guard(self) -> None:
        if self._committed:
            raise ContractViolation("writer already committed")

    def _ensure_unique(self, owner: EntityId) -> None:
        if self._has_owner(owner):
            raise ContractViolation(f"entity already added: {owner}")

    def _has_owner(self, owner: EntityId) -> bool:
        return any(table.id == owner for table in self._tables)

    def _metrics_path(self, owner: EntityId) -> pathlib.Path:
        return table_metrics_path(self._root, owner.name)

    def _validate_commit_contract(self) -> None:
        if self._data_model != "relational":
            raise ContractViolation("dataset data_model must be relational")
        if self._dialect != "postgresql":
            raise ContractViolation("dataset dialect must be postgresql")
        table_metadata = self._read_table_metadata()
        missing_tables = [
            table.id.name
            for table in self._tables
            if table.id.name not in table_metadata
        ]
        if missing_tables:
            raise ContractViolation(
                "relational dataset missing deterministic table metadata for: "
                + ", ".join(sorted(missing_tables))
            )

    def _read_table_metadata(self) -> dict[str, TableDeterministicMetadata]:
        metadata_by_name: dict[str, TableDeterministicMetadata] = {}
        for relative_path in self._deterministic_tables:
            path = self._root / relative_path
            if not path.exists():
                raise ContractViolation(f"deterministic table metadata file missing: {relative_path}")
            metadata = json_io.read_json(path, TableDeterministicMetadata)
            if metadata.table_name in metadata_by_name:
                raise ContractViolation(
                    f"duplicate deterministic table metadata for: {metadata.table_name}"
                )
            metadata_by_name[metadata.table_name] = metadata
        return metadata_by_name

__all__ = ["DatasetWriter"]
