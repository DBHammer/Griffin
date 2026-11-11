"""Typed codec backed by msgspec.

We chose msgspec over a hand-rolled codec because it already handles:
    - frozen Struct round-trip
    - tagged unions via `Struct(tag=...)` discriminator
    - Enum encoding/decoding
    - tuple, list, dict, Optional, Union of typed Structs

If a future requirement forces hand-rolling (custom binary layout, schema
evolution beyond msgspec's tag mechanism), document the reason here before
removing this module.
"""

from __future__ import annotations

from typing import Type, TypeVar

import msgspec

from .errors import ContractViolation

T = TypeVar("T")


def encode_json(value: object) -> bytes:
    """Encode a typed structure (Struct, dataclass, Enum, primitive) to JSON bytes."""
    try:
        return msgspec.json.encode(value)
    except msgspec.EncodeError as exc:
        raise ContractViolation(f"encode failed: {exc}") from exc


def decode_json(data: bytes | str, target: Type[T]) -> T:
    """Decode JSON bytes/str into the typed structure `target`."""
    try:
        return msgspec.json.decode(data, type=target)
    except msgspec.DecodeError as exc:
        raise ContractViolation(f"decode failed: {exc}") from exc


def roundtrip(value: T, target: Type[T]) -> T:
    """Encode then decode `value` as `target`. Used by tests."""
    return decode_json(encode_json(value), target)


__all__ = [
    "encode_json",
    "decode_json",
    "roundtrip",
]
