"""Schema-aware dataset data file read/write helpers."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Iterable, Mapping, Sequence
from typing import SupportsFloat, SupportsInt

import numpy as np
import pandas as pd

from griffin.infra.errors import ContractViolation
from griffin.ir.data_type import DataType, DoubleType, IntType, ListType, TextType, VectorType
from griffin.ir.field import Field
from griffin.vector_hardness_core import vector_column_matrix

ColumnValues = Sequence[object] | np.ndarray

CSV_NULL_MARKER = r"\N"
CSV_ESCAPED_NULL_MARKER = r"\\N"


def write_data_file(
    path: pathlib.Path,
    fields: Sequence[Field],
    columns: Mapping[str, ColumnValues],
    format_name: str,
    *,
    leading_columns: Sequence[tuple[str, ColumnValues]] = (),
    trailing_columns: Sequence[tuple[str, ColumnValues]] = (),
) -> tuple[pathlib.Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered: dict[str, ColumnValues] = {name: values for name, values in leading_columns}
    ordered.update({field.id.name: columns[field.id.name] for field in fields})
    ordered.update({name: values for name, values in trailing_columns})
    if format_name == "parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception as exc:  # noqa: BLE001
            raise ContractViolation(f"pyarrow is required for dataset materialization: {exc}") from exc
        arrow_columns = {}
        field_lookup = {field.id.name: field for field in fields}
        for name, values in ordered.items():
            field = field_lookup.get(name)
            if field is not None and isinstance(field.type, VectorType):
                matrix = vector_column_matrix(values, field)
                flat = pa.array(matrix.ravel(), type=pa.float32())
                arrow_columns[name] = pa.FixedSizeListArray.from_arrays(flat, list_size=matrix.shape[1])
            else:
                arrow_columns[name] = pa.array(values)
        table = pa.table(arrow_columns)
        pq.write_table(table, path)
        return path, format_name
    if format_name == "csv":
        encoded: dict[str, Sequence[object]] = {
            name: [_encode_extra_csv_value(value) for value in values]
            for name, values in leading_columns
        }
        encoded.update(
            {
                field.id.name: [_encode_csv_value(value, field.type) for value in columns[field.id.name]]
                for field in fields
            }
        )
        encoded.update(
            {
                name: [_encode_extra_csv_value(value) for value in values]
                for name, values in trailing_columns
            }
        )
        pd.DataFrame(encoded).to_csv(path, index=False)
        return path, format_name
    raise ContractViolation(f"unsupported dataset data format: {format_name}")


def read_data_frame(
    path: pathlib.Path,
    fields: Sequence[Field],
    format_name: str,
    *,
    columns: Sequence[str] | None = None,
    extra_columns: Sequence[str] = (),
) -> pd.DataFrame:
    wanted = list(columns) if columns is not None else [field.id.name for field in fields]
    for extra in extra_columns:
        if extra not in wanted:
            wanted.append(extra)
    field_lookup = {field.id.name: field for field in fields}
    if format_name == "parquet":
        frame = pd.read_parquet(path, columns=wanted)
        for name in wanted:
            if name in field_lookup:
                field = field_lookup[name]
                frame[name] = _object_series(
                    _normalize_loaded_value(value, field.type)
                    for value in frame[name]
                )
            else:
                frame[name] = _object_series(_coerce_extra_column_value(value) for value in frame[name])
        return frame
    if format_name == "csv":
        frame = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            usecols=wanted,
        )
        for name in wanted:
            if name in field_lookup:
                field = field_lookup[name]
                frame[name] = frame[name].map(lambda raw: _decode_csv_value(raw, field.type))
            else:
                frame[name] = frame[name].map(_coerce_extra_column_raw)
        return frame
    raise ContractViolation(f"unsupported dataset data format: {format_name}")


def read_column_values(
    path: pathlib.Path,
    field: Field,
    format_name: str,
) -> list[object]:
    frame = read_data_frame(path, (field,), format_name, columns=(field.id.name,))
    return list(frame[field.id.name])


def _coerce_extra_column_value(value: object) -> object:
    if _is_nullish(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if value == "":
        return None
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise ContractViolation(f"extra column value must be an integer, got {type(value).__name__}")
    return int(value)


def _coerce_extra_column_raw(raw: str) -> object:
    if raw == "":
        return None
    return int(raw)


def _encode_csv_value(value: object, data_type: DataType) -> str:
    if value is None:
        return CSV_NULL_MARKER
    if isinstance(data_type, TextType):
        text = str(value)
        if text == CSV_NULL_MARKER:
            return CSV_ESCAPED_NULL_MARKER
        return text
    if isinstance(data_type, (IntType, DoubleType)):
        return str(value)
    if isinstance(data_type, VectorType):
        return json.dumps(_coerce_vector_components(value), separators=(",", ":"))
    if isinstance(data_type, ListType):
        return json.dumps(_coerce_list_for_json(value, data_type.element), separators=(",", ":"))
    raise ContractViolation(f"unsupported csv encoding type: {type(data_type).__name__}")


def _decode_csv_value(raw: str, data_type: DataType) -> object:
    if raw == CSV_NULL_MARKER:
        return None
    if isinstance(data_type, TextType):
        if raw == CSV_ESCAPED_NULL_MARKER:
            return CSV_NULL_MARKER
        return raw
    if isinstance(data_type, IntType):
        return int(raw)
    if isinstance(data_type, DoubleType):
        return float(raw)
    if isinstance(data_type, VectorType):
        payload = json.loads(raw)
        if not isinstance(payload, list):
            raise ContractViolation(f"invalid csv vector payload: {raw!r}")
        return [float(item) for item in payload]
    if isinstance(data_type, ListType):
        payload = json.loads(raw)
        if not isinstance(payload, list):
            raise ContractViolation(f"invalid csv list payload: {raw!r}")
        return [_coerce_list_item(item, data_type.element) for item in payload]
    raise ContractViolation(f"unsupported csv decoding type: {type(data_type).__name__}")


def _encode_extra_csv_value(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise ContractViolation(f"extra csv value must be an integer, got {type(value).__name__}")
    return str(int(value))


def _normalize_loaded_value(value: object, data_type: DataType) -> object:
    if _is_nullish(value):
        return None
    if isinstance(data_type, VectorType):
        if isinstance(value, list):
            return _coerce_number_sequence(value)
        if hasattr(value, "tolist"):
            payload = value.tolist()
            if isinstance(payload, list):
                return _coerce_number_sequence(payload)
        return _coerce_vector_components(value)
    if isinstance(data_type, ListType):
        if isinstance(value, list):
            return [_coerce_list_item(item, data_type.element) for item in value]
        if hasattr(value, "tolist"):
            payload = value.tolist()
            if isinstance(payload, list):
                return [_coerce_list_item(item, data_type.element) for item in payload]
        if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
            raise ContractViolation(f"loaded list value must be iterable, got {type(value).__name__}")
        return [_coerce_list_item(item, data_type.element) for item in value]
    if isinstance(data_type, IntType):
        return _coerce_int(value)
    if isinstance(data_type, DoubleType):
        return _coerce_float(value)
    if isinstance(data_type, TextType):
        return str(value)
    raise ContractViolation(f"unsupported loaded value type: {type(data_type).__name__}")


def _is_nullish(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return False
    if isinstance(value, (str, bytes, int, float, bool)):
        return bool(pd.isna(value))
    return False


def _object_series(values: Iterable[object]) -> pd.Series:
    return pd.Series(list(values), dtype=object)


def _coerce_list_for_json(value: object, element_type: DataType) -> list[object]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise ContractViolation(f"expected iterable list payload, got {type(value).__name__}")
    return [_coerce_list_item(item, element_type) for item in value]


def _coerce_list_item(value: object, element_type: DataType) -> object:
    if value is None:
        return None
    if isinstance(element_type, IntType):
        return _coerce_int(value)
    if isinstance(element_type, DoubleType):
        return _coerce_float(value)
    if isinstance(element_type, TextType):
        return str(value)
    if isinstance(element_type, VectorType):
        if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
            raise ContractViolation("nested vector list payload must be iterable")
        return _coerce_number_sequence(value)
    if isinstance(element_type, ListType):
        if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
            raise ContractViolation("nested list payload must be iterable")
        return [_coerce_list_item(item, element_type.element) for item in value]
    raise ContractViolation(f"unsupported list element type: {type(element_type).__name__}")


def _coerce_vector_components(value: object) -> list[float]:
    if isinstance(value, np.ndarray):
        return _coerce_number_sequence(value.tolist())
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise ContractViolation(f"vector value must be iterable, got {type(value).__name__}")
    return _coerce_number_sequence(value)


def _coerce_number_sequence(values: Iterable[object]) -> list[float]:
    return [_coerce_float(item) for item in values]


def _coerce_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, SupportsInt)):
        raise ContractViolation(f"expected integer-compatible value, got {type(value).__name__}")
    return int(value)


def _coerce_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, SupportsFloat)):
        raise ContractViolation(f"expected numeric value, got {type(value).__name__}")
    return float(value)


__all__ = [
    "CSV_NULL_MARKER",
    "read_column_values",
    "read_data_frame",
    "write_data_file",
]
