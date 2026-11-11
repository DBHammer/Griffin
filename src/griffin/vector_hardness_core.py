"""Shared FAISS-backed exact vector access and query proposal."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from griffin.infra.errors import ContractViolation
from griffin.ir.data_type import VectorMetric, VectorType
from griffin.ir.field import Field


def synthesize_calibration_query_from_base(
    base_vector: np.ndarray | Sequence[float],
    metric: VectorMetric,
    rng: np.random.Generator,
    *,
    attempt: int,
    noise_base: float,
) -> np.ndarray:
    base = np.asarray(base_vector, dtype="float32")
    if base.ndim != 1 or base.shape[0] == 0:
        raise ContractViolation("vector query synthesis requires a non-empty base vector")
    base = np.array(base, dtype="float32", copy=True)
    direction = rng.normal(0.0, 1.0, size=base.shape).astype("float32")
    norm = float(np.linalg.norm(direction))
    if norm > 0.0:
        direction = direction / norm
    scale = _query_noise_scale(noise_base, attempt=attempt)
    amount = float(rng.exponential(scale=max(scale, 1e-6)))
    query = base + direction * amount
    if metric is VectorMetric.COSINE:
        q_norm = float(np.linalg.norm(query))
        if q_norm > 0.0:
            query = query / q_norm
    result: np.ndarray = query.astype("float32")
    return result


def estimate_query_noise_base(matrix: np.ndarray) -> float:
    return _query_noise_base(matrix)


def build_exact_faiss_index(matrix: np.ndarray, *, metric: VectorMetric) -> Any:
    arr = metric_matrix(matrix, metric=metric)
    return _build_flat_l2_index(arr)


def vector_column_matrix(values: object, field: Field | None = None, *, dimension: int | None = None) -> np.ndarray:
    if field is not None:
        if not isinstance(field.type, VectorType):
            raise ContractViolation(f"vector matrix requires a vector field: {field.id}")
        dimension = field.type.dimension
        label = str(field.id)
    else:
        label = "vector column"
    if dimension is None:
        raise ContractViolation("vector matrix validation requires a dimension")
    if not isinstance(values, np.ndarray):
        raise ContractViolation(
            f"{label} must be materialized as a 2D float32 numpy matrix; "
            f"got {type(values).__name__}"
        )
    if values.ndim != 2 or values.shape[1] != dimension:
        raise ContractViolation(
            f"{label} has shape {values.shape}; expected (N, {dimension})"
        )
    return values.astype("float32", copy=False)


def exact_public_search(
    *,
    index: Any,
    query: np.ndarray,
    metric: VectorMetric,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    search_limit = index.ntotal if limit is None else min(max(int(limit), 0), index.ntotal)
    if search_limit <= 0:
        return np.asarray([], dtype="float32"), np.asarray([], dtype="int64")
    query_for_metric = metric_query(query, metric=metric)
    raw_distances, labels = index.search(np.asarray([query_for_metric], dtype="float32"), search_limit)
    public_distances: np.ndarray = faiss_distances_to_public(raw_distances[0], metric=metric)
    public_labels: np.ndarray = labels[0].astype("int64", copy=False)
    return public_distances, public_labels


def exact_self_first_neighbor_labels(
    matrix: np.ndarray | Sequence[Sequence[float]],
    *,
    metric: VectorMetric,
    limit: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    arr = np.asarray(matrix, dtype="float32")
    if arr.ndim != 2:
        raise ContractViolation("neighbor labels require a 2D vector matrix")
    if arr.shape[0] == 0:
        return ()
    neighbor_limit = arr.shape[0] if limit is None else min(max(int(limit), 1), arr.shape[0])
    index = build_exact_faiss_index(arr, metric=metric)
    query_matrix = metric_matrix(arr, metric=metric)
    raw_distances, all_labels = index.search(query_matrix, neighbor_limit)
    rows: list[tuple[int, ...]] = []
    del raw_distances
    for row_id, labels in enumerate(all_labels.astype("int64", copy=False)):
        ordered = [int(row_id)]
        seen = {int(row_id)}
        for label in labels.tolist():
            if len(ordered) == neighbor_limit:
                break
            neighbor_id = int(label)
            if neighbor_id < 0 or neighbor_id >= arr.shape[0] or neighbor_id in seen:
                continue
            ordered.append(neighbor_id)
            seen.add(neighbor_id)
        if len(ordered) != neighbor_limit:
            raise ContractViolation("neighbor labels must cover the requested window exactly once")
        rows.append(tuple(ordered))
    return tuple(rows)


def faiss_distances_to_public(distances: np.ndarray, *, metric: VectorMetric) -> np.ndarray:
    raw = np.array(distances, dtype="float32", copy=True)
    return _faiss_distances_to_public_inplace(raw, metric=metric)


def _faiss_distances_to_public_inplace(distances: np.ndarray, *, metric: VectorMetric) -> np.ndarray:
    raw = np.asarray(distances, dtype="float32")
    np.maximum(raw, 0.0, out=raw)
    if metric is VectorMetric.COSINE:
        raw /= 2.0
        return raw
    if metric is VectorMetric.L2:
        np.sqrt(raw, out=raw)
        return raw
    raise ContractViolation(f"unsupported vector metric: {metric.value}")


def metric_matrix(matrix: np.ndarray, *, metric: VectorMetric) -> np.ndarray:
    if metric is VectorMetric.COSINE:
        return normalized(matrix)
    arr: np.ndarray = np.asarray(matrix, dtype="float32")
    return arr


def metric_query(query: np.ndarray, *, metric: VectorMetric) -> np.ndarray:
    if metric is VectorMetric.COSINE:
        normalized_query: np.ndarray = normalized(np.asarray([query], dtype="float32"))[0]
        return normalized_query
    arr: np.ndarray = np.asarray(query, dtype="float32")
    return arr


def normalized(matrix: np.ndarray) -> np.ndarray:
    copy = np.array(matrix, dtype="float32", copy=True)
    if copy.ndim != 2:
        raise ContractViolation("vector matrix must be 2-dimensional")
    norms = np.linalg.norm(copy, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized_matrix: np.ndarray = copy / norms
    return normalized_matrix


def validate_exact_flat_l2_index(index: Any) -> None:
    import faiss

    if not isinstance(index, faiss.IndexFlatL2):
        raise ContractViolation(
            f"exact vector access requires FAISS IndexFlatL2; got {type(index).__name__}"
        )


def _faiss_index_dimension(index: Any) -> int:
    raw_dimension = getattr(index, "d", None)
    if raw_dimension is None:
        raise ContractViolation("vector index dimension is missing")
    dimension = int(raw_dimension)
    if dimension <= 0:
        raise ContractViolation(f"vector index dimension must be positive: {dimension}")
    return dimension


def _build_flat_l2_index(matrix: np.ndarray) -> Any:
    import faiss

    arr = np.ascontiguousarray(_validate_matrix(matrix), dtype="float32")
    index = faiss.IndexFlatL2(arr.shape[1])
    index.add(arr)
    return index


def _validate_matrix(matrix: np.ndarray) -> np.ndarray:
    arr: np.ndarray = np.asarray(matrix, dtype="float32")
    if arr.ndim != 2:
        raise ContractViolation("vector matrix must be 2-dimensional")
    if arr.shape[0] == 0:
        raise ContractViolation("vector matrix must be non-empty")
    return arr


def _query_noise_base(matrix: np.ndarray) -> float:
    base = float(np.mean(np.std(np.asarray(matrix, dtype="float32"), axis=0)))
    if not math.isfinite(base) or base <= 0.0:
        return 1e-3
    return base


def _query_noise_scale(base: float, *, attempt: int) -> float:
    factor = 0.25 * (1 + (attempt % 8))
    return base * factor


__all__ = [
    "build_exact_faiss_index",
    "estimate_query_noise_base",
    "exact_public_search",
    "exact_self_first_neighbor_labels",
    "faiss_distances_to_public",
    "metric_matrix",
    "metric_query",
    "normalized",
    "synthesize_calibration_query_from_base",
    "validate_exact_flat_l2_index",
    "vector_column_matrix",
]
