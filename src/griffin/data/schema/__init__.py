from __future__ import annotations

from typing import Sequence

import msgspec

from griffin.dataset.entities.table import Table
from griffin.infra.errors import ContractViolation


class RelationalSchemaExport(msgspec.Struct, frozen=True, omit_defaults=True):
    dialect: str


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


from .postgresql import render_postgresql_schema  # noqa: E402


def render_relational_schema(
    export: RelationalSchemaExport,
    tables: Sequence[Table],
) -> str:
    if export.dialect != "postgresql":
        raise ContractViolation(f"unsupported relational schema export dialect: {export.dialect}")
    return render_postgresql_schema(tuple(tables))


__all__ = [
    "RelationalSchemaExport",
    "quote_identifier",
    "render_relational_schema",
]
