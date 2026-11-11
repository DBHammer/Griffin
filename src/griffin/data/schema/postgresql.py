from __future__ import annotations

from griffin.dataset.entities.table import Table
from griffin.dataset.row_identity import DEFAULT_ROW_ID_COLUMN
from griffin.ir.data_type import DoubleType, IntType, TextType, VectorType
from griffin.ir.field import Field

from . import quote_identifier


def render_postgresql_schema(tables: tuple[Table, ...]) -> str:
    statements: list[str] = []
    if any(isinstance(field.type, VectorType) for table in tables for field in table.fields):
        statements.append("CREATE EXTENSION IF NOT EXISTS vector")
    for table in tables:
        statements.append(_create_table_sql(table))
    return ";\n".join(statements) + ";\n"


def _create_table_sql(table: Table) -> str:
    column_defs = [f"{quote_identifier(DEFAULT_ROW_ID_COLUMN)} BIGINT NOT NULL"]
    column_defs.extend(_field_sql(field) for field in table.fields)
    if table.primary_key:
        primary = ", ".join(quote_identifier(field.name) for field in table.primary_key)
        column_defs.append(f"PRIMARY KEY ({primary})")
    return f"CREATE TABLE {quote_identifier(table.id.name)} ({', '.join(column_defs)})"


def _field_sql(field: Field) -> str:
    parts = [quote_identifier(field.id.name), _type_sql(field)]
    if not field.nullable:
        parts.append("NOT NULL")
    return " ".join(parts)


def _type_sql(field: Field) -> str:
    data_type = field.type
    if isinstance(data_type, IntType):
        return "BIGINT"
    if isinstance(data_type, DoubleType):
        return "DOUBLE PRECISION"
    if isinstance(data_type, TextType):
        return "TEXT"
    if isinstance(data_type, VectorType):
        return f"vector({data_type.dimension})"
    raise TypeError(f"unsupported PostgreSQL schema field type: {type(data_type).__name__}")


__all__ = ["render_postgresql_schema"]
