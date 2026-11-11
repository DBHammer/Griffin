from __future__ import annotations

import msgspec


class DataFileRef(msgspec.Struct, frozen=True, omit_defaults=True):
    """Reference to a data file on disk relative to the dataset root."""

    relative_path: str
    format: str  # "parquet", "csv"


__all__ = ["DataFileRef"]
