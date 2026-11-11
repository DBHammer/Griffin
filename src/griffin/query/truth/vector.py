"""FAISS-backed exact vector truth service."""

from __future__ import annotations

import math
import pathlib
import threading
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, SupportsFloat

import numpy as np

from griffin.dataset.artifacts.base import VectorIndexArtifact
from griffin.dataset.exact_cover import RowCover
from griffin.dataset.reader import Dataset
from griffin.dataset.row_identity import DEFAULT_ROW_ID_COLUMN
from griffin.infra.errors import ContractViolation
from griffin.ir.data_type import VectorMetric, VectorType
from griffin.ir.field import Field
from griffin.model_ids import FieldId
from griffin.vector_hardness_core import (
    estimate_query_noise_base,
    exact_public_search,
    validate_exact_flat_l2_index,
)


DEFAULT_FULL_SEARCH_CACHE_MAX_BYTES = 128 * 1024 * 1024
DEFAULT_FULL_SEARCH_CACHE_MAX_ENTRIES = 64


@dataclass(frozen=True)
class VectorKnnAnswer:
    row_id: int
    distance: float | None = None

    def to_json(self) -> dict[str, Any]:
        return {DEFAULT_ROW_ID_COLUMN: self.row_id}


@dataclass(frozen=True)
class VectorFullSearch:
    distances: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        if self.distances.ndim != 1 or self.labels.ndim != 1:
            raise ContractViolation("vector full-search payload must be one-dimensional")
        if self.distances.shape[0] != self.labels.shape[0]:
            raise ContractViolation("vector full-search distances and labels must have equal length")
        self.distances.setflags(write=False)
        self.labels.setflags(write=False)

    @property
    def row_count(self) -> int:
        return int(self.labels.size)

    def neighbors(
        self,
        *,
        k: int,
        row_filter: RowCover | None = None,
        exclude_rowid: int | None = None,
    ) -> tuple[int, ...]:
        if row_filter is not None and int(row_filter.total_count) != self.row_count:
            raise ContractViolation("vector full-search row filter total count does not match result row count")
        target_hits = _target_hit_count(
            k=k,
            row_filter=row_filter,
            exclude_rowid=exclude_rowid,
        )
        answers = _filter_answers(
            labels=self.labels,
            distances=self.distances,
            target_hits=target_hits,
            row_filter=row_filter,
            exclude_rowid=exclude_rowid,
        )
        return tuple(answer.row_id for answer in answers)


@dataclass(frozen=True)
class VectorSourceMetadata:
    row_count: int
    dimension: int
    metric: VectorMetric
    noise_base: float


@dataclass(frozen=True)
class VectorQueryObservation:
    answers: tuple[VectorKnnAnswer, ...]


class VectorKnnOracle:
    """Exact top-k, filtered top-k, and full search via FAISS flat index."""

    name = "faiss_vector_knn"

    def __init__(
        self,
        *,
        full_search_cache_max_bytes: int = DEFAULT_FULL_SEARCH_CACHE_MAX_BYTES,
        full_search_cache_max_entries: int = DEFAULT_FULL_SEARCH_CACHE_MAX_ENTRIES,
    ) -> None:
        self._index_cache: dict[tuple[str, object], Any] = {}
        self._source_metadata_cache: dict[tuple[str, object], VectorSourceMetadata] = {}
        self._full_search_cache: OrderedDict[tuple[object, ...], VectorFullSearch] = OrderedDict()
        self._full_search_cache_bytes = 0
        self._full_search_cache_max_bytes = max(0, int(full_search_cache_max_bytes))
        self._full_search_cache_max_entries = max(0, int(full_search_cache_max_entries))
        self._lock = threading.RLock()

    def _index(self, dataset: Dataset, artifact: VectorIndexArtifact) -> Any:
        key = (str(dataset.root), artifact.id)
        with self._lock:
            cached = self._index_cache.get(key)
            if cached is not None:
                return cached
        import faiss

        index_path = pathlib.Path(dataset.root) / artifact.relative_path
        if not index_path.exists():
            raise ContractViolation(f"missing FAISS vector index file: {index_path}")
        index = faiss.read_index(str(index_path))
        validate_exact_flat_l2_index(index)
        with self._lock:
            cached = self._index_cache.get(key)
            if cached is not None:
                return cached
            self._index_cache[key] = index
            return index

    def full_search(
        self,
        *,
        dataset: Dataset,
        field_id: FieldId,
        query_vector: Iterable[SupportsFloat],
        limit: int | None = None,
    ) -> VectorFullSearch:
        field = _vector_field(dataset, field_id)
        artifact = _vector_index_artifact(dataset, field_id)
        _validate_exact_artifact(artifact, field)
        source_index = self._index(dataset, artifact)
        query_tuple = _vector_tuple(query_vector)
        cache_key = _full_search_cache_key(
            dataset=dataset,
            artifact=artifact,
            query_vector=query_tuple,
            limit=limit,
        )
        with self._lock:
            cached = self._full_search_cache.get(cache_key)
            if cached is not None:
                self._full_search_cache.move_to_end(cache_key)
                return cached
        distances, labels = exact_public_search(
            index=source_index,
            query=np.asarray(query_tuple, dtype="float32"),
            metric=_vector_type(field).metric,
            limit=limit,
        )
        result = VectorFullSearch(distances=distances, labels=labels)
        self._cache_full_search(cache_key, result)
        return result

    def _cache_full_search(self, key: tuple[object, ...], result: VectorFullSearch) -> None:
        if self._full_search_cache_max_bytes <= 0 or self._full_search_cache_max_entries <= 0:
            return
        result_bytes = _full_search_payload_bytes(result)
        if result_bytes > self._full_search_cache_max_bytes:
            return
        with self._lock:
            replaced = self._full_search_cache.pop(key, None)
            if replaced is not None:
                self._full_search_cache_bytes -= _full_search_payload_bytes(replaced)
            self._full_search_cache[key] = result
            self._full_search_cache_bytes += result_bytes
            while (
                len(self._full_search_cache) > self._full_search_cache_max_entries
                or self._full_search_cache_bytes > self._full_search_cache_max_bytes
            ):
                _evicted_key, evicted = self._full_search_cache.popitem(last=False)
                self._full_search_cache_bytes -= _full_search_payload_bytes(evicted)

    def observe(
        self,
        *,
        dataset: Dataset,
        field_id: FieldId,
        query_vector: Iterable[SupportsFloat],
        k: int,
        row_filter: RowCover | None = None,
    ) -> VectorQueryObservation:
        query_tuple = _vector_tuple(query_vector)
        field = _vector_field(dataset, field_id)
        artifact = _vector_index_artifact(dataset, field_id)
        _validate_exact_artifact(artifact, field)
        source_index = self._index(dataset, artifact)
        if row_filter is not None and int(row_filter.total_count) != int(source_index.ntotal):
            raise ContractViolation("vector row filter total count does not match vector index")
        return _faiss_observe(
            index=source_index,
            query_vector=query_tuple,
            metric=_vector_type(field).metric,
            k=k,
            row_filter=row_filter,
            exclude_rowid=None,
        )

    def vector_rows(
        self,
        *,
        dataset: Dataset,
        field_id: FieldId,
        rowids: Iterable[int],
    ) -> np.ndarray:
        field = _vector_field(dataset, field_id)
        artifact = _vector_index_artifact(dataset, field_id)
        _validate_exact_artifact(artifact, field)
        index = self._index(dataset, artifact)
        ids = np.asarray(tuple(int(rowid) for rowid in rowids), dtype="int64")
        dimension = _vector_type(field).dimension
        if ids.size == 0:
            return np.empty((0, dimension), dtype="float32")
        for rowid in ids.tolist():
            if rowid < 0 or rowid >= int(index.ntotal):
                raise ContractViolation(f"vector rowid {rowid} is outside [0, {index.ntotal})")
        vectors = _reconstruct_vector_rows(index, ids)
        if vectors.shape != (int(ids.size), dimension):
            raise ContractViolation(
                f"vector row reconstruction shape mismatch: got {vectors.shape}, expected {(int(ids.size), dimension)}"
            )
        return vectors

    def vector_source_metadata(
        self,
        *,
        dataset: Dataset,
        field_id: FieldId,
    ) -> VectorSourceMetadata:
        field = _vector_field(dataset, field_id)
        artifact = _vector_index_artifact(dataset, field_id)
        _validate_exact_artifact(artifact, field)
        key = (str(dataset.root), artifact.id)
        with self._lock:
            cached = self._source_metadata_cache.get(key)
            if cached is not None:
                return cached
        index = self._index(dataset, artifact)
        vector_type = _vector_type(field)
        row_count = int(index.ntotal)
        sample_rowids = _source_sample_rowids(row_count, limit=1024)
        sample = self.vector_rows(dataset=dataset, field_id=field_id, rowids=sample_rowids)
        metadata = VectorSourceMetadata(
            row_count=row_count,
            dimension=vector_type.dimension,
            metric=vector_type.metric,
            noise_base=estimate_query_noise_base(sample) if sample.shape[0] > 0 else 1e-3,
        )
        with self._lock:
            cached = self._source_metadata_cache.get(key)
            if cached is not None:
                return cached
            self._source_metadata_cache[key] = metadata
            return metadata

def _vector_index_artifact(dataset: Dataset, field_id: FieldId) -> VectorIndexArtifact:
    for artifact in dataset.artifacts():
        if isinstance(artifact, VectorIndexArtifact) and artifact.id.owner == field_id.owner and artifact.id.name == field_id.name:
            return artifact
    raise ContractViolation(f"missing vector index artifact for {field_id}")


def _vector_field(dataset: Dataset, field_id: FieldId) -> Field:
    entity = dataset.entity(field_id.owner)
    field = _field_def(entity.fields, field_id)
    if not isinstance(field.type, VectorType):
        raise ContractViolation(f"field {field_id} is not a vector field")
    return field


def _validate_exact_artifact(artifact: VectorIndexArtifact, field: Field) -> None:
    vector_type = _vector_type(field)
    if artifact.index_type != "exact":
        raise ContractViolation(
            f"exact vector access requires exact FAISS index for {field.id}; "
            f"got {artifact.index_type}"
        )
    if artifact.metric != vector_type.metric.value:
        raise ContractViolation(f"vector index metric does not match field {field.id}")
    if artifact.dimension != vector_type.dimension:
        raise ContractViolation(f"vector index dimension does not match field {field.id}")


def _vector_type(field: Field) -> VectorType:
    if not isinstance(field.type, VectorType):
        raise ContractViolation(f"field {field.id} is not a vector field")
    return field.type


def _field_def(fields: tuple[Field, ...], field_id: FieldId) -> Field:
    for field in fields:
        if field.id == field_id:
            return field
    raise ContractViolation(f"unknown field: {field_id}")


def _faiss_observe(
    *,
    index: Any,
    query_vector: tuple[float, ...],
    metric: VectorMetric,
    k: int,
    row_filter: RowCover | None,
    exclude_rowid: int | None = None,
) -> VectorQueryObservation:
    if k <= 0:
        return VectorQueryObservation(answers=())
    if row_filter is not None and row_filter.count <= 0:
        return VectorQueryObservation(answers=())
    if row_filter is not None and int(row_filter.total_count) != int(index.ntotal):
        raise ContractViolation("vector row filter total count does not match vector index")
    query = np.asarray(query_vector, dtype="float32")
    target_hits = _target_hit_count(
        k=k,
        row_filter=row_filter,
        exclude_rowid=exclude_rowid,
    )
    if target_hits <= 0:
        return VectorQueryObservation(answers=())
    fetch_k = _initial_fetch_k(
        ntotal=index.ntotal,
        k=k,
        target_hits=target_hits,
        row_filter=row_filter,
        exclude_rowid=exclude_rowid,
    )
    seen_limit = 0
    while True:
        current_fetch = min(max(fetch_k, target_hits), index.ntotal)
        distances, labels = exact_public_search(
            index=index,
            query=query,
            metric=metric,
            limit=current_fetch,
        )
        answers = _filter_answers(
            labels=labels[:current_fetch],
            distances=distances[:current_fetch],
            target_hits=target_hits,
            row_filter=row_filter,
            exclude_rowid=exclude_rowid,
        )
        if len(answers) >= target_hits or current_fetch >= index.ntotal:
            return VectorQueryObservation(answers=tuple(answers))
        if current_fetch == seen_limit:
            return VectorQueryObservation(answers=tuple(answers))
        seen_limit = current_fetch
        fetch_k = min(index.ntotal, max(current_fetch + 1, current_fetch * 2))


def _filter_answers(
    *,
    labels: Iterable[int],
    distances: Iterable[float] | None = None,
    target_hits: int,
    row_filter: RowCover | None,
    exclude_rowid: int | None,
) -> list[VectorKnnAnswer]:
    answers: list[VectorKnnAnswer] = []
    distance_values = list(distances) if distances is not None else None
    for index, label in enumerate(labels):
        row_id = int(label)
        if row_id < 0:
            continue
        if exclude_rowid is not None and row_id == exclude_rowid:
            continue
        if row_filter is not None and not row_filter.contains(row_id):
            continue
        distance = None if distance_values is None else float(distance_values[index])
        answers.append(VectorKnnAnswer(row_id=row_id, distance=distance))
        if len(answers) >= target_hits:
            break
    return answers


def _target_hit_count(
    *,
    k: int,
    row_filter: RowCover | None,
    exclude_rowid: int | None,
) -> int:
    if row_filter is None:
        return k
    allowed_count = int(row_filter.count)
    if exclude_rowid is not None and row_filter.contains(exclude_rowid):
        allowed_count -= 1
    return min(k, max(allowed_count, 0))


def _initial_fetch_k(
    *,
    ntotal: int,
    k: int,
    target_hits: int,
    row_filter: RowCover | None,
    exclude_rowid: int | None,
) -> int:
    if row_filter is None:
        if exclude_rowid is not None:
            return min(ntotal, k + 1)
        return min(ntotal, k)
    allowed_count = int(row_filter.count)
    if exclude_rowid is not None and row_filter.contains(exclude_rowid):
        allowed_count -= 1
    if allowed_count <= 0:
        return min(ntotal, k)
    density = max(allowed_count / max(ntotal, 1), 1.0 / max(ntotal, 1))
    estimated = int(math.ceil(target_hits / density)) if target_hits > 0 else k
    slack = max(k, 8)
    capped = min(ntotal, max(target_hits, estimated + slack))
    if capped >= ntotal and allowed_count < ntotal:
        return min(ntotal - 1, max(target_hits, max(k, 8)))
    return capped


def _reconstruct_vector_rows(index: Any, rowids: np.ndarray) -> np.ndarray:
    if hasattr(index, "reconstruct_batch"):
        rows = index.reconstruct_batch(rowids)
    elif hasattr(index, "reconstruct"):
        rows = np.asarray([index.reconstruct(int(rowid)) for rowid in rowids.tolist()], dtype="float32")
    else:
        raise ContractViolation("exact vector row access requires FAISS reconstruct support")
    return np.asarray(rows, dtype="float32")


def _full_search_cache_key(
    *,
    dataset: Dataset,
    artifact: VectorIndexArtifact,
    query_vector: tuple[float, ...],
    limit: int | None,
) -> tuple[object, ...]:
    return (
        str(dataset.root),
        artifact.id.owner,
        artifact.id.name,
        tuple(float(component) for component in query_vector),
        limit,
    )


def _full_search_payload_bytes(result: VectorFullSearch) -> int:
    return int(result.distances.nbytes) + int(result.labels.nbytes)


def _source_sample_rowids(row_count: int, *, limit: int) -> tuple[int, ...]:
    total = int(row_count)
    if total <= 0 or limit <= 0:
        return ()
    sample_size = min(total, int(limit))
    if sample_size == total:
        return tuple(range(total))
    step = total / sample_size
    return tuple(dict.fromkeys(min(total - 1, int(index * step)) for index in range(sample_size)))


def _vector_tuple(query_vector: Iterable[SupportsFloat]) -> tuple[float, ...]:
    return tuple(float(component) for component in query_vector)


__all__ = [
    "DEFAULT_FULL_SEARCH_CACHE_MAX_BYTES",
    "DEFAULT_FULL_SEARCH_CACHE_MAX_ENTRIES",
    "VectorFullSearch",
    "VectorKnnAnswer",
    "VectorKnnOracle",
    "VectorQueryObservation",
    "VectorSourceMetadata",
]
