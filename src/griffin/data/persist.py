"""Persistence helpers for materialized dataset entities."""

from __future__ import annotations

import pathlib

from griffin.data.deterministic import deterministic_table_metadata
from griffin.data.entity_builders import ColumnMapping
from griffin.dataset.data_files import write_data_file
from griffin.dataset.deterministic import (
    DeterministicFieldModel,
    FieldPlacementModel,
    RegionLayoutModel,
    VectorAttributeCouplingModel,
)
from griffin.dataset.entities import DataFileRef, Table
from griffin.dataset.layout import table_data_path
from griffin.dataset.row_identity import DEFAULT_ROW_ID_COLUMN
from griffin.dataset.writer import DatasetWriter


def persist_table(
    *,
    root_path: pathlib.Path,
    writer: DatasetWriter,
    materialized: dict[str, ColumnMapping],
    table: Table,
    columns: ColumnMapping,
    deterministic_models: tuple[DeterministicFieldModel, ...],
    row_count: int,
    output_format: str,
    region_layout: RegionLayoutModel | None = None,
    placements: tuple[FieldPlacementModel, ...] | None = None,
    vector_attribute_couplings: tuple[VectorAttributeCouplingModel, ...] = (),
) -> None:
    row_ids = row_ids_for_count(row_count)
    materialized[table.id.name] = {DEFAULT_ROW_ID_COLUMN: list(row_ids), **columns}
    written_path, written_format = write_data_file(
        table_data_path(root_path, table.id.name, output_format),
        table.fields,
        columns,
        output_format,
        leading_columns=((DEFAULT_ROW_ID_COLUMN, row_ids),),
    )
    writer.add_table(
        Table(
            id=table.id,
            fields=table.fields,
            primary_key=table.primary_key,
            data_file=DataFileRef(
                relative_path=str(written_path.relative_to(root_path)),
                format=written_format,
            ),
        )
    )
    writer.add_table_metadata(
        deterministic_table_metadata(
            table.id.name,
            row_count,
            deterministic_models,
            placements=placements,
            region_layout=region_layout,
            vector_attribute_couplings=vector_attribute_couplings,
        )
    )


def row_ids_for_count(row_count: int) -> list[int]:
    return list(range(row_count))


__all__ = ["persist_table", "row_ids_for_count"]
