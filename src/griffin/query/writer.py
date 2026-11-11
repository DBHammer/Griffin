"""Persist a QueryWorkload to the public workload output layout."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import tempfile
from typing import Any

from griffin.dataset.layout import query_run_path
from griffin.dialects.base import RenderedQuery

from .spec import QueryWorkload, accepted_predicate_plan_to_public_dict, ground_truth_count


def invalidate_query_run(
    *,
    dataset_root: pathlib.Path | str,
    run_id: str,
    workload_dir: pathlib.Path | str,
) -> None:
    workload_path = pathlib.Path(workload_dir).resolve()
    _remove_known_workload_artifacts(workload_path)
    metadata_path = query_run_path(pathlib.Path(dataset_root).resolve(), run_id)
    if metadata_path.exists():
        metadata_path.unlink()


def publish_query_run(
    *,
    dataset_root: pathlib.Path | str,
    run_id: str,
    workload_dir: pathlib.Path | str,
    workload: QueryWorkload,
    dialect_name: str,
) -> pathlib.Path:
    root_path = pathlib.Path(workload_dir).resolve()
    root_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = query_run_path(pathlib.Path(dataset_root).resolve(), run_id)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    staging_root = pathlib.Path(tempfile.mkdtemp(prefix=f".{root_path.name}.", dir=root_path.parent))
    metadata_staging_path: pathlib.Path | None = None
    try:
        query_path = _write_workload_files(staging_root, workload, dialect_name=dialect_name)
        metadata_staging_path = _write_query_run_metadata_staging(
            metadata_path=metadata_path,
            run_id=run_id,
            workload_dir=root_path,
            workload=workload,
            dialect_name=dialect_name,
        )
        root_path.mkdir(parents=True, exist_ok=True)
        _remove_known_workload_artifacts(root_path)
        for staged_path in staging_root.iterdir():
            os.replace(staged_path, root_path / staged_path.name)
        os.replace(metadata_staging_path, metadata_path)
        metadata_staging_path = None
        return root_path / query_path.name
    except Exception:
        _remove_known_workload_artifacts(root_path)
        if metadata_path.exists():
            metadata_path.unlink()
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        if metadata_staging_path is not None:
            metadata_staging_path.unlink(missing_ok=True)


def _write_workload_files(
    root_path: pathlib.Path,
    workload: QueryWorkload,
    *,
    dialect_name: str,
) -> pathlib.Path:
    root_path.mkdir(parents=True, exist_ok=True)
    extension = query_file_extension_for_dialect(dialect_name)
    query_path = root_path / f"queries.{extension}"
    _write_query_text_file(query_path, workload, extension=extension)
    _write_ground_truth_file_if_needed(root_path / "ground_truth.jsonl", workload)
    return query_path


def _write_query_run_metadata_staging(
    *,
    metadata_path: pathlib.Path,
    run_id: str,
    workload_dir: pathlib.Path | str,
    workload: QueryWorkload,
    dialect_name: str,
) -> pathlib.Path:
    payload = {
        "run_id": run_id,
        "workload_dir": str(pathlib.Path(workload_dir).resolve()),
        "dialect": dialect_name,
        "query_count": workload.stats.accepted,
        "accepted": workload.stats.accepted,
        "rejected": workload.stats.rejected,
        "total_attempts": workload.stats.total_attempts,
        "canonical_evaluations": workload.stats.canonical_evaluations,
        "search_cost": _search_cost_record(workload),
        "ground_truth_count": ground_truth_count(workload),
        "queries": [_per_query_record(query) for query in workload.accepted],
    }
    descriptor, staging_name = tempfile.mkstemp(prefix=f".{metadata_path.name}.", dir=metadata_path.parent)
    os.close(descriptor)
    staging_path = pathlib.Path(staging_name)
    try:
        staging_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        staging_path.unlink(missing_ok=True)
        raise
    return staging_path


def _search_cost_record(workload: QueryWorkload) -> dict[str, int]:
    stats = workload.stats
    return {
        "canonical_evaluations": int(stats.canonical_evaluations),
        "hint_observations": int(stats.hint_observations),
        "seed_hint_observations": int(stats.seed_hint_observations),
        "local_search_hint_observations": int(stats.local_search_hint_observations),
        "full_search_calls": int(stats.full_search_calls),
        "cover_computations": int(stats.cover_computations),
        "domain_options_scored": int(stats.domain_options_scored),
    }


def _per_query_record(query: Any) -> dict[str, Any]:
    analysis = query.analysis
    record: dict[str, Any] = {
        "query_id": query.query_id,
        "accepted": analysis.accepted,
        "score": analysis.score,
    }
    report = analysis.public_report
    if report is not None:
        record["task_kind"] = report.kind.value
        record["achieved"] = report.achieved
        record["target"] = report.target
        if report.metadata_selectivity is not None:
            record["metadata_selectivity"] = report.metadata_selectivity
        if report.fields:
            record["fields"] = [
                {"owner": field.owner.name, "name": field.name}
                for field in report.fields
            ]
    if analysis.diagnostics:
        record["diagnostics"] = analysis.diagnostics
    if query.accepted_predicate_plan:
        record["accepted_predicate_plan"] = accepted_predicate_plan_to_public_dict(query.accepted_predicate_plan)
    return record


def _write_ground_truth_file_if_needed(path: pathlib.Path, workload: QueryWorkload) -> None:
    if ground_truth_count(workload) == 0:
        if path.exists():
            path.unlink()
        return
    with path.open("w", encoding="utf-8") as fh:
        for query in workload.accepted:
            if query.ground_truth is None:
                continue
            ground_truth: dict[str, Any] = {
                "query_id": query.query_id,
                "answer_kind": query.ground_truth.answer_kind.value,
                "metric": query.ground_truth.metric,
                "row_id_column": query.ground_truth.row_id_column,
                "k": query.ground_truth.k,
                "row_count": query.ground_truth.row_count,
                "answers": list(query.ground_truth.answers),
            }
            if query.ground_truth.answer_scope:
                ground_truth["answer_scope"] = dict(query.ground_truth.answer_scope)
            fh.write(json.dumps(ground_truth, ensure_ascii=False))
            fh.write("\n")


def _remove_known_workload_artifacts(root_path: pathlib.Path) -> None:
    for name in (
        "manifest.json",
        "queries.jsonl",
        "metadata.json",
        "stats.json",
        "queries.sql",
        "ground_truth.jsonl",
        "report.md",
    ):
        path = root_path / name
        if path.exists() and path.is_file():
            path.unlink()
    internal = root_path / ".internal"
    if internal.exists() and internal.is_dir():
        shutil.rmtree(internal)
    elif internal.exists():
        internal.unlink()


def query_file_extension_for_dialect(dialect_name: str | None) -> str:
    if dialect_name not in {None, "postgresql"}:
        raise ValueError(f"unsupported query dialect {dialect_name!r}")
    return "sql"


def _write_query_text_file(path: pathlib.Path, workload: QueryWorkload, *, extension: str) -> None:
    lines: list[str] = []
    for query in workload.accepted:
        lines.append(public_query_text(query.rendered, extension=extension).rstrip().removesuffix(";") + ";")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def public_query_text(rendered: RenderedQuery, *, extension: str) -> str:
    params = json.loads(rendered.parameters_json) if rendered.parameters_json else []
    if isinstance(params, list):
        return _inline_positional_parameters(rendered.text, params, extension=extension)
    if isinstance(params, dict):
        return _inline_named_parameters(rendered.text, params, extension=extension)
    return rendered.text


def _inline_positional_parameters(text: str, params: list[Any], *, extension: str) -> str:
    rendered = text
    for value in params:
        if "?" in rendered:
            rendered = rendered.replace("?", _literal(value, extension=extension), 1)
            continue
        if "%s" in rendered:
            rendered = rendered.replace("%s", _literal(value, extension=extension), 1)
            continue
    return rendered


def _inline_named_parameters(text: str, params: dict[str, Any], *, extension: str) -> str:
    rendered = text
    for name in sorted(params, key=len, reverse=True):
        rendered = rendered.replace(f"${name}", _literal(params[name], extension=extension))
    return rendered


def _literal(value: Any, *, extension: str) -> str:
    is_sql = extension == "sql"
    if value is None:
        return "NULL" if is_sql else "null"
    if isinstance(value, bool):
        if is_sql:
            return "TRUE" if value else "FALSE"
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_literal(item, extension=extension) for item in value) + "]"
    return _quoted_string(str(value))


def _quoted_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


__all__ = [
    "invalidate_query_run",
    "public_query_text",
    "publish_query_run",
    "query_file_extension_for_dialect",
]
