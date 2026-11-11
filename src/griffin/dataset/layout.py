"""On-disk path conventions for a persisted dataset."""

from __future__ import annotations

import pathlib
from typing import Final

METADATA_DIR: Final = "metadata"
MANIFEST_FILENAME: Final = "manifest.json"
SCHEMA_SQL_FILENAME: Final = "schema.sql"
DATA_DIR: Final = "data"
TABLES_DIR: Final = "tables"
STATISTICS_DIR: Final = "statistics"
ARTIFACTS_DIR: Final = "artifacts"
RELATIONS_DIR: Final = "relations"
QUERY_RUNS_DIR: Final = "query_runs"
DETERMINISTIC_DIR: Final = "deterministic"


def metadata_dir(root: pathlib.Path) -> pathlib.Path:
    return root / METADATA_DIR


def manifest_path(root: pathlib.Path) -> pathlib.Path:
    return metadata_dir(root) / MANIFEST_FILENAME


def schema_sql_path(root: pathlib.Path) -> pathlib.Path:
    return root / SCHEMA_SQL_FILENAME


def table_data_path(
    root: pathlib.Path,
    owner_name: str,
    format_name: str,
) -> pathlib.Path:
    suffix = f".{format_name}"
    return root / DATA_DIR / TABLES_DIR / f"{owner_name}{suffix}"


def table_metrics_path(root: pathlib.Path, owner_name: str) -> pathlib.Path:
    return metadata_dir(root) / STATISTICS_DIR / TABLES_DIR / f"{owner_name}.json"


def artifact_path(root: pathlib.Path, kind: str, owner_name: str, name: str) -> pathlib.Path:
    return metadata_dir(root) / ARTIFACTS_DIR / kind / f"{owner_name}__{name}"


def relation_path(root: pathlib.Path, kind: str, name: str) -> pathlib.Path:
    return metadata_dir(root) / RELATIONS_DIR / kind / f"{name}.json"


def query_run_dir(root: pathlib.Path, run_id: str) -> pathlib.Path:
    return metadata_dir(root) / QUERY_RUNS_DIR / run_id


def query_run_path(root: pathlib.Path, run_id: str) -> pathlib.Path:
    return query_run_dir(root, run_id) / "run.json"


def deterministic_table_path(root: pathlib.Path, owner_name: str) -> pathlib.Path:
    return metadata_dir(root) / DETERMINISTIC_DIR / TABLES_DIR / f"{owner_name}.json"


__all__ = [
    "METADATA_DIR",
    "MANIFEST_FILENAME",
    "SCHEMA_SQL_FILENAME",
    "DATA_DIR",
    "TABLES_DIR",
    "STATISTICS_DIR",
    "ARTIFACTS_DIR",
    "RELATIONS_DIR",
    "QUERY_RUNS_DIR",
    "DETERMINISTIC_DIR",
    "metadata_dir",
    "manifest_path",
    "schema_sql_path",
    "table_data_path",
    "table_metrics_path",
    "artifact_path",
    "relation_path",
    "query_run_dir",
    "query_run_path",
    "deterministic_table_path",
]
