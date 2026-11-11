"""ID and short-hash helpers. Deterministic where possible."""

from __future__ import annotations

import secrets


def generate_id(*, prefix: str = "", length: int = 8) -> str:
    """Random opaque identifier; non-deterministic. Length is in hex chars."""
    raw = secrets.token_hex(max(length // 2, 1))[:length]
    return f"{prefix}{raw}" if prefix else raw


__all__ = ["generate_id"]
