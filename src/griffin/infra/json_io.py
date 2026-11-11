"""Filesystem JSON helpers using the typed codec."""

from __future__ import annotations

import pathlib
from typing import Type, TypeVar

from . import codec

T = TypeVar("T")


def write_json(path: pathlib.Path, value: object, *, mkdir: bool = True) -> None:
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(codec.encode_json(value))


def read_json(path: pathlib.Path, target: Type[T]) -> T:
    return codec.decode_json(path.read_bytes(), target)


__all__ = ["write_json", "read_json"]
