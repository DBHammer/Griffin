from __future__ import annotations

import msgspec

from griffin.ir.field import Field
from griffin.model_ids import FieldId, TableId

from .data_file import DataFileRef


class Table(msgspec.Struct, frozen=True, omit_defaults=True):
    id: TableId
    fields: tuple[Field, ...]
    primary_key: tuple[FieldId, ...] = ()
    data_file: DataFileRef | None = None


__all__ = ["Table"]
