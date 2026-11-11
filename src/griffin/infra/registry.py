"""Typed registries.

`Registry[K, V]` is mutable during construction; call `.freeze()` to obtain a
`FrozenRegistry[K, V]` for use in production code.
"""

from __future__ import annotations

from typing import Generic, Iterator, Mapping, TypeVar

from .errors import ContractViolation

K = TypeVar("K")
V = TypeVar("V")


class FrozenRegistry(Generic[K, V]):
    __slots__ = ("_items",)

    def __init__(self, items: Mapping[K, V]) -> None:
        self._items: dict[K, V] = dict(items)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __getitem__(self, key: K) -> V:
        try:
            return self._items[key]
        except KeyError as exc:
            raise ContractViolation(f"unknown registry key: {key!r}") from exc

    def get(self, key: K) -> V | None:
        return self._items.get(key)

    def keys(self) -> Iterator[K]:
        return iter(self._items.keys())

    def values(self) -> Iterator[V]:
        return iter(self._items.values())

    def items(self) -> Iterator[tuple[K, V]]:
        return iter(self._items.items())

    def __len__(self) -> int:
        return len(self._items)


class Registry(Generic[K, V]):
    __slots__ = ("_items", "_frozen")

    def __init__(self) -> None:
        self._items: dict[K, V] = {}
        self._frozen = False

    def register(self, key: K, value: V) -> None:
        if self._frozen:
            raise ContractViolation("registry is frozen")
        if key in self._items:
            raise ContractViolation(f"duplicate registry key: {key!r}")
        self._items[key] = value

    def freeze(self) -> FrozenRegistry[K, V]:
        self._frozen = True
        return FrozenRegistry(self._items)


__all__ = ["Registry", "FrozenRegistry"]
