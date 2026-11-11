"""Observation request, cache, and cost-event contracts."""

from __future__ import annotations

import enum
import os
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import msgspec
import numpy as np

from griffin.dataset.exact_cover import OrdinalIntervalSet, RowCover
from griffin.infra.errors import ContractViolation
from griffin.infra.runtime_logging import RuntimeTaskLogger
from griffin.ir.data_type import VectorMetric
from griffin.model_ids import FieldId
from griffin.query.dataset_resources import DatasetResources
from griffin.query.filtered_ann_band import gls_loss
from griffin.query.model import (
    AnswerScope,
    HoleId,
    ParameterValue,
    StructuralSignature,
    TemplateNode,
    UnscopedAnswerScope,
    stable_signature,
)
from griffin.query.predicate_cover_search import compute_predicate_cover


class ObservationRequestKind(enum.Enum):
    PREDICATE_COVER = "predicate_cover"
    FILTERED_ANN_HIT = "filtered_ann_hit"
    VECTOR_SOURCE = "vector_source"
    VECTOR_TOPK = "vector_topk"
    VECTOR_FULL_SEARCH = "vector_full_search"


class HintCostStage(enum.Enum):
    FILTERED_ANN_SEED = "filtered_ann_seed"
    FILTERED_ANN_LOCAL_SEARCH = "filtered_ann_local_search"


class ObservationPurpose(enum.Enum):
    HINT = "hint"
    CANONICAL = "canonical"


class ObservationCacheStatus(enum.Enum):
    HIT = "hit"
    MISS = "miss"


@dataclass(frozen=True)
class CoverPayload:
    cover: RowCover
    answer_scope: AnswerScope = field(default_factory=UnscopedAnswerScope)

    def __post_init__(self) -> None:
        if not self.cover.fingerprint():
            raise ContractViolation("cover payload requires fingerprint")

    @property
    def fingerprint(self) -> str:
        return self.cover.fingerprint()

    @property
    def total_count(self) -> int:
        return self.cover.total_count

    @property
    def count(self) -> int:
        return self.cover.count

    @property
    def selectivity(self) -> float:
        return self.cover.selectivity

    @property
    def intervals(self) -> OrdinalIntervalSet:
        return self.cover.intervals

    @classmethod
    def from_cover(
        cls,
        *,
        cover: RowCover,
        answer_scope: AnswerScope | None = None,
    ) -> "CoverPayload":
        return cls(
            cover=cover,
            answer_scope=answer_scope or UnscopedAnswerScope(),
        )

    @classmethod
    def from_rowids(
        cls,
        *,
        rowids: frozenset[int],
        total_count: int,
        answer_scope: AnswerScope | None = None,
    ) -> "CoverPayload":
        return cls.from_cover(
            cover=RowCover.from_rowids(rowids, total_count=total_count),
            answer_scope=answer_scope,
        )


class ObservationRequest(msgspec.Struct, frozen=True, omit_defaults=True):
    kind: ObservationRequestKind
    field: FieldId
    query_vector: tuple[float, ...] = ()
    k: int | None = None
    limit: int | None = None
    template_signature: StructuralSignature | None = None
    cover_node: TemplateNode | None = None
    cover_values: Mapping[HoleId, ParameterValue] = msgspec.field(default_factory=dict)
    cover_payload: CoverPayload | None = None
    hit_band: tuple[int, int] | None = None


class ObservationRequestIdentity(msgspec.Struct, frozen=True, omit_defaults=True):
    purpose: ObservationPurpose
    kind: str
    request_fingerprint: str


class ObservationCostEvent(msgspec.Struct, frozen=True, omit_defaults=True):
    identity: ObservationRequestIdentity
    cache_status: ObservationCacheStatus = ObservationCacheStatus.MISS
    hint_stage: HintCostStage | None = None
    exact_calls: int = 0
    hint_observations: int = 0
    canonical_observations: int = 0
    cover_computations: int = 0
    full_search_calls: int = 0
    vector_observations: int = 0


class ObservationResult(msgspec.Struct, frozen=True, omit_defaults=True):
    identity: ObservationRequestIdentity
    observation: object | None = None
    cost_event: ObservationCostEvent | None = None


class ObservationScope(msgspec.Struct, frozen=True, omit_defaults=True):
    scope_id: str


@dataclass(frozen=True)
class VectorSourceHint:
    row_count: int
    dimension: int
    metric: VectorMetric
    noise_base: float


@dataclass(frozen=True)
class PredicateCoverHint:
    node_id: str
    scope_id: str
    cover: RowCover
    answer_scope: AnswerScope = field(default_factory=UnscopedAnswerScope)

    @property
    def total_count(self) -> int:
        return self.cover.total_count

    @property
    def cover_fingerprint(self) -> str:
        return self.cover.fingerprint()

    @property
    def selectivity(self) -> float:
        return self.cover.selectivity

    @property
    def payload(self) -> CoverPayload:
        return CoverPayload.from_cover(
            cover=self.cover,
            answer_scope=self.answer_scope,
        )


@dataclass(frozen=True)
class VectorTopKHint:
    rowids: tuple[int, ...]
    distances: tuple[float | None, ...] = ()


@dataclass(frozen=True)
class VectorFullSearchHint:
    distances: np.ndarray
    labels: np.ndarray
    row_count: int

    def __post_init__(self) -> None:
        if self.distances.ndim != 1 or self.labels.ndim != 1:
            raise ContractViolation("vector full-search payload must be one-dimensional")
        if self.distances.shape[0] != self.labels.shape[0]:
            raise ContractViolation("vector full-search distances and labels must have equal length")
        if int(self.row_count) != int(self.labels.shape[0]):
            raise ContractViolation("vector full-search row count must match labels")
        if self.distances.flags.writeable or self.labels.flags.writeable:
            raise ContractViolation("vector full-search hint requires read-only truth payload")


@dataclass(frozen=True)
class FilteredAnnHitHint:
    global_topk: VectorTopKHint
    selection: CoverPayload
    hit_count: int
    loss: float


@dataclass(frozen=True)
class ObservationSummary:
    kind: ObservationRequestKind
    predicate_cover: PredicateCoverHint | None = None
    source: VectorSourceHint | None = None
    topk: VectorTopKHint | None = None
    full_search: VectorFullSearchHint | None = None
    filtered_ann_hit: FilteredAnnHitHint | None = None
    cost: Mapping[str, int] = field(default_factory=dict)


class ObservationCostStats(msgspec.Struct, frozen=True, omit_defaults=True):
    exact_calls: int = 0
    hint_observations: int = 0
    seed_hint_observations: int = 0
    local_search_hint_observations: int = 0
    canonical_observations: int = 0
    full_search_calls: int = 0
    cover_computations: int = 0
    vector_observations: int = 0


@dataclass
class _InFlightObservation:
    condition: threading.Condition
    done: bool = False
    summary: ObservationSummary | None = None
    error: BaseException | None = None


DEFAULT_REQUEST_CACHE_LIMIT = 4096
DEFAULT_EXACT_CACHE_LIMIT = 4096
DEFAULT_RESULT_LOG_LIMIT = 4096
DEFAULT_COST_EVENT_LOG_LIMIT = 4096
OBSERVATION_CACHE_ENV = "MMBENCHGEN_QUERY_OBSERVATION_CACHE"


@runtime_checkable
class ObservationExecutionPort(Protocol):
    def observe(self, request: ObservationRequest) -> ObservationSummary: ...


class DatasetObservationExecutionPort:
    def __init__(self, resources: DatasetResources) -> None:
        self._resources = resources

    def observe(self, request: ObservationRequest) -> ObservationSummary:
        if request.kind is ObservationRequestKind.PREDICATE_COVER:
            if request.cover_node is None:
                raise ContractViolation("predicate cover observation requires cover node")
            cover = compute_predicate_cover(
                resources=self._resources,
                node=request.cover_node,
                values=request.cover_values,
            )
            return ObservationSummary(
                kind=request.kind,
                predicate_cover=PredicateCoverHint(
                    node_id=cover.node_id,
                    scope_id=cover.scope_id,
                    cover=cover.cover,
                    answer_scope=cover.answer_scope,
                ),
                cost={"cover_computations": 1},
            )
        if request.kind is ObservationRequestKind.FILTERED_ANN_HIT:
            if request.k is None or request.cover_payload is None or request.hit_band is None:
                raise ContractViolation("filtered ANN hit observation requires k, cover payload, and hit band")
            observation = self._resources.vector_oracle.observe(
                dataset=self._resources.dataset,
                field_id=request.field,
                query_vector=request.query_vector,
                k=request.k,
                row_filter=None,
            )
            topk_rowids = tuple(int(answer.row_id) for answer in observation.answers)
            hit_count = sum(1 for rowid in topk_rowids if request.cover_payload.cover.contains(rowid))
            return ObservationSummary(
                kind=request.kind,
                filtered_ann_hit=FilteredAnnHitHint(
                    global_topk=VectorTopKHint(
                        rowids=topk_rowids,
                        distances=tuple(
                            None
                            if getattr(answer, "distance", None) is None
                            else float(getattr(answer, "distance"))
                            for answer in observation.answers
                        ),
                    ),
                    selection=request.cover_payload,
                    hit_count=hit_count,
                    loss=gls_loss(
                        hit_count=hit_count,
                        band=request.hit_band,
                        total_count=request.cover_payload.total_count,
                        cover_count=request.cover_payload.count,
                        k=request.k,
                    ),
                ),
                cost={"hint_observations": 1},
            )
        if request.kind is ObservationRequestKind.VECTOR_SOURCE:
            source = self._resources.vector_oracle.vector_source_metadata(
                dataset=self._resources.dataset,
                field_id=request.field,
            )
            if not isinstance(source.metric, VectorMetric):
                raise ContractViolation("vector source hint requires a vector metric")
            return ObservationSummary(
                kind=request.kind,
                source=VectorSourceHint(
                    row_count=int(source.row_count),
                    dimension=int(source.dimension),
                    metric=source.metric,
                    noise_base=float(source.noise_base),
                ),
            )
        if request.kind is ObservationRequestKind.VECTOR_TOPK:
            if request.k is None:
                raise ContractViolation("vector top-k hint requires k")
            observation = self._resources.vector_oracle.observe(
                dataset=self._resources.dataset,
                field_id=request.field,
                query_vector=request.query_vector,
                k=request.k,
                row_filter=request.cover_payload.cover if request.cover_payload is not None else None,
            )
            return ObservationSummary(
                kind=request.kind,
                topk=VectorTopKHint(
                    rowids=tuple(int(answer.row_id) for answer in observation.answers),
                    distances=tuple(
                        None if getattr(answer, "distance", None) is None else float(getattr(answer, "distance"))
                        for answer in observation.answers
                    ),
                ),
                cost={"hint_observations": 1},
            )
        if request.kind is ObservationRequestKind.VECTOR_FULL_SEARCH:
            full_search = self._resources.vector_oracle.full_search(
                dataset=self._resources.dataset,
                field_id=request.field,
                query_vector=request.query_vector,
                limit=request.limit,
            )
            return ObservationSummary(
                kind=request.kind,
                full_search=VectorFullSearchHint(
                    distances=full_search.distances,
                    labels=full_search.labels,
                    row_count=int(full_search.row_count),
                ),
                cost={"full_search_calls": 1},
            )
        raise ContractViolation(f"unsupported observation kind: {request.kind.value}")


class ObservationSession:
    """Workload-run scoped observation execution shell."""

    def __init__(
        self,
        *,
        execution_port: object | None = None,
        task_logger: RuntimeTaskLogger | None = None,
        cache_enabled: bool | None = None,
        request_cache_limit: int = DEFAULT_REQUEST_CACHE_LIMIT,
        exact_cache_limit: int = DEFAULT_EXACT_CACHE_LIMIT,
        result_log_limit: int = DEFAULT_RESULT_LOG_LIMIT,
        cost_event_log_limit: int = DEFAULT_COST_EVENT_LOG_LIMIT,
    ) -> None:
        self._execution_port = execution_port
        self._task_logger = task_logger
        self._cache_enabled = _cache_enabled_from_env() if cache_enabled is None else bool(cache_enabled)
        self._request_cache_limit = max(int(request_cache_limit), 1)
        self._exact_cache_limit = max(int(exact_cache_limit), 1)
        self._request_cache: OrderedDict[str, ObservationSummary] = OrderedDict()
        self._exact_cache: OrderedDict[str, ObservationSummary] = OrderedDict()
        self._results: deque[ObservationResult] = deque(maxlen=max(int(result_log_limit), 1))
        self._cost_events: deque[ObservationCostEvent] = deque(maxlen=max(int(cost_event_log_limit), 1))
        self._total_cost_stats = ObservationCostStats()
        self._scope_cost_stats: dict[str, ObservationCostStats] = {}
        self._lock = threading.RLock()
        self._inflight: dict[str, _InFlightObservation] = {}

    @property
    def results(self) -> tuple[ObservationResult, ...]:
        with self._lock:
            return tuple(self._results)

    @property
    def cost_events(self) -> tuple[ObservationCostEvent, ...]:
        with self._lock:
            return tuple(self._cost_events)

    @property
    def total_cost_stats(self) -> ObservationCostStats:
        with self._lock:
            return self._total_cost_stats

    def scope_cost_stats(self, scope: ObservationScope) -> ObservationCostStats:
        with self._lock:
            return self._scope_cost_stats.get(scope.scope_id, ObservationCostStats())

    def measure_profile(self, name: str) -> AbstractContextManager[None]:
        if self._task_logger is None:
            return nullcontext()
        return self._task_logger.measure(name)

    def observe_hint(
        self,
        request: ObservationRequest,
        *,
        stage: HintCostStage | None = None,
        scope: ObservationScope | None = None,
    ) -> ObservationSummary:
        return self._observe(request, purpose=ObservationPurpose.HINT, stage=stage, scope=scope)

    def observe_canonical(
        self,
        request: ObservationRequest,
        *,
        scope: ObservationScope | None = None,
    ) -> ObservationSummary:
        return self._observe(request, purpose=ObservationPurpose.CANONICAL, stage=None, scope=scope)

    def _observe(
        self,
        request: ObservationRequest,
        *,
        purpose: ObservationPurpose,
        stage: HintCostStage | None,
        scope: ObservationScope | None,
    ) -> ObservationSummary:
        if not isinstance(self._execution_port, ObservationExecutionPort):
            raise ContractViolation("observation requires an ObservationExecutionPort")
        started_at = time.perf_counter()
        _validate_observation_request(request)
        identity = _observation_identity(request, purpose=purpose)
        request_cache_key = f"{identity.purpose.value}:{identity.request_fingerprint}"
        if not self._cache_enabled:
            return self._observe_uncached(
                request,
                identity=identity,
                stage=stage,
                scope=scope,
                started_at=started_at,
                purpose=purpose,
            )
        with self._lock:
            cached = self._cached_summary(request_cache_key, identity.request_fingerprint)
            if cached is not None:
                cached_summary = _summary_without_cost(cached)
                _cache_put(self._request_cache, request_cache_key, cached, limit=self._request_cache_limit)
                self._record_reuse_profile(request)
                cost_event = ObservationCostEvent(
                    identity=identity,
                    cache_status=ObservationCacheStatus.HIT,
                    hint_stage=stage,
                )
                self._record(cached_summary, cost_event, scope=scope)
                self._record_profile(
                    purpose=purpose,
                    request=request,
                    elapsed_sec=time.perf_counter() - started_at,
                )
                return cached_summary
            inflight = self._inflight.get(identity.request_fingerprint)
            if inflight is None:
                inflight = _InFlightObservation(condition=threading.Condition(self._lock))
                self._inflight[identity.request_fingerprint] = inflight
                owner = True
            else:
                owner = False
            if not owner:
                while not inflight.done:
                    inflight.condition.wait()
                self._record_profile(
                    purpose=purpose,
                    request=request,
                    elapsed_sec=time.perf_counter() - started_at,
                )
                if inflight.error is not None:
                    raise inflight.error
                if inflight.summary is None:
                    raise ContractViolation("inflight observation completed without summary")
                cached_summary = _summary_without_cost(inflight.summary)
                _cache_put(self._request_cache, request_cache_key, inflight.summary, limit=self._request_cache_limit)
                self._record_reuse_profile(request)
                self._record(
                    cached_summary,
                    ObservationCostEvent(
                        identity=identity,
                        cache_status=ObservationCacheStatus.HIT,
                        hint_stage=stage,
                    ),
                    scope=scope,
                )
                return cached_summary
        error: BaseException | None = None
        observed_summary: ObservationSummary | None = None
        try:
            observed_summary = self._execution_port.observe(request)
        except BaseException as exc:
            error = exc
        elapsed_sec = time.perf_counter() - started_at
        self._record_profile(
            purpose=purpose,
            request=request,
            elapsed_sec=elapsed_sec,
        )
        with self._lock:
            current = self._inflight.pop(identity.request_fingerprint, None)
            if current is None:
                raise ContractViolation("inflight observation owner missing")
            current.done = True
            current.summary = observed_summary
            current.error = error
            current.condition.notify_all()
            if error is not None:
                raise error
            if observed_summary is None:
                raise ContractViolation("observation execution returned no summary")
            if _cacheable_summary(observed_summary):
                _cache_put(self._request_cache, request_cache_key, observed_summary, limit=self._request_cache_limit)
                _cache_put(self._exact_cache, identity.request_fingerprint, observed_summary, limit=self._exact_cache_limit)
            cost_event = _observation_cost_event(identity, observed_summary, stage=stage)
            self._record(observed_summary, cost_event, scope=scope)
            return observed_summary

    def _observe_uncached(
        self,
        request: ObservationRequest,
        *,
        identity: ObservationRequestIdentity,
        stage: HintCostStage | None,
        scope: ObservationScope | None,
        started_at: float,
        purpose: ObservationPurpose,
    ) -> ObservationSummary:
        if not isinstance(self._execution_port, ObservationExecutionPort):
            raise ContractViolation("observation requires an ObservationExecutionPort")
        observed_summary = self._execution_port.observe(request)
        self._record_profile(
            purpose=purpose,
            request=request,
            elapsed_sec=time.perf_counter() - started_at,
        )
        cost_event = _observation_cost_event(identity, observed_summary, stage=stage)
        with self._lock:
            self._record(observed_summary, cost_event, scope=scope)
        return observed_summary

    def _cached_summary(self, request_cache_key: str, fingerprint: str) -> ObservationSummary | None:
        cached = self._request_cache.get(request_cache_key)
        if cached is not None:
            self._request_cache.move_to_end(request_cache_key)
            return cached
        cached = self._exact_cache.get(fingerprint)
        if cached is not None:
            self._exact_cache.move_to_end(fingerprint)
        return cached

    def _record(
        self,
        summary: ObservationSummary,
        cost_event: ObservationCostEvent,
        *,
        scope: ObservationScope | None,
    ) -> None:
        self._cost_events.append(cost_event)
        self._results.append(
            ObservationResult(
                identity=cost_event.identity,
                observation=_result_log_observation(summary),
                cost_event=cost_event,
            )
        )
        self._total_cost_stats = ObservationCostStats(
            exact_calls=int(self._total_cost_stats.exact_calls) + int(cost_event.exact_calls),
            hint_observations=int(self._total_cost_stats.hint_observations) + int(cost_event.hint_observations),
            seed_hint_observations=int(self._total_cost_stats.seed_hint_observations)
            + (int(cost_event.hint_observations) if cost_event.hint_stage is HintCostStage.FILTERED_ANN_SEED else 0),
            local_search_hint_observations=int(self._total_cost_stats.local_search_hint_observations)
            + (
                int(cost_event.hint_observations)
                if cost_event.hint_stage is HintCostStage.FILTERED_ANN_LOCAL_SEARCH
                else 0
            ),
            canonical_observations=int(self._total_cost_stats.canonical_observations)
            + int(cost_event.canonical_observations),
            full_search_calls=int(self._total_cost_stats.full_search_calls) + int(cost_event.full_search_calls),
            cover_computations=int(self._total_cost_stats.cover_computations) + int(cost_event.cover_computations),
            vector_observations=int(self._total_cost_stats.vector_observations) + int(cost_event.vector_observations),
        )
        if scope is not None:
            current = self._scope_cost_stats.get(scope.scope_id, ObservationCostStats())
            self._scope_cost_stats[scope.scope_id] = ObservationCostStats(
                exact_calls=int(current.exact_calls) + int(cost_event.exact_calls),
                hint_observations=int(current.hint_observations) + int(cost_event.hint_observations),
                seed_hint_observations=int(current.seed_hint_observations)
                + (int(cost_event.hint_observations) if cost_event.hint_stage is HintCostStage.FILTERED_ANN_SEED else 0),
                local_search_hint_observations=int(current.local_search_hint_observations)
                + (
                    int(cost_event.hint_observations)
                    if cost_event.hint_stage is HintCostStage.FILTERED_ANN_LOCAL_SEARCH
                    else 0
                ),
                canonical_observations=int(current.canonical_observations) + int(cost_event.canonical_observations),
                full_search_calls=int(current.full_search_calls) + int(cost_event.full_search_calls),
                cover_computations=int(current.cover_computations) + int(cost_event.cover_computations),
                vector_observations=int(current.vector_observations) + int(cost_event.vector_observations),
            )

    def _record_profile(
        self,
        *,
        purpose: ObservationPurpose,
        request: ObservationRequest,
        elapsed_sec: float,
    ) -> None:
        if self._task_logger is None or elapsed_sec <= 0.0:
            return
        names = [f"query.obs.{purpose.value}"]
        if request.kind is ObservationRequestKind.PREDICATE_COVER:
            names.append("query.obs.cover")
        elif request.kind is ObservationRequestKind.VECTOR_TOPK:
            names.append("query.obs.topk")
        elif request.kind is ObservationRequestKind.VECTOR_FULL_SEARCH:
            names.append("query.obs.full_search")
        elif request.kind is ObservationRequestKind.FILTERED_ANN_HIT:
            names.append("query.obs.hit")
        for index, name in enumerate(names):
            self._task_logger.record_duration(name, elapsed_sec, include_in_parent=index == 0)

    def record_profile_event(self, name: str, *, count: int = 1) -> None:
        if self._task_logger is None:
            return
        self._task_logger.record_event(name, count=count)

    def _record_reuse_profile(self, request: ObservationRequest) -> None:
        if self._task_logger is None:
            return
        self._task_logger.record_event("query.reuse.observation")
        self._task_logger.record_event(f"query.reuse.observation.{_observation_kind_profile_name(request.kind)}")


def summarize_observation_cost_events(
    events: tuple[ObservationCostEvent, ...],
) -> ObservationCostStats:
    hint_observations = 0
    exact_calls = 0
    seed_hint_observations = 0
    local_search_hint_observations = 0
    canonical_observations = 0
    full_search_calls = 0
    cover_computations = 0
    vector_observations = 0
    for event in events:
        hint_count = int(event.hint_observations)
        exact_calls += int(event.exact_calls)
        hint_observations += hint_count
        if event.hint_stage is HintCostStage.FILTERED_ANN_SEED:
            seed_hint_observations += hint_count
        elif event.hint_stage is HintCostStage.FILTERED_ANN_LOCAL_SEARCH:
            local_search_hint_observations += hint_count
        canonical_observations += int(event.canonical_observations)
        full_search_calls += int(event.full_search_calls)
        cover_computations += int(event.cover_computations)
        vector_observations += int(event.vector_observations)
    return ObservationCostStats(
        exact_calls=exact_calls,
        hint_observations=hint_observations,
        seed_hint_observations=seed_hint_observations,
        local_search_hint_observations=local_search_hint_observations,
        canonical_observations=canonical_observations,
        full_search_calls=full_search_calls,
        cover_computations=cover_computations,
        vector_observations=vector_observations,
    )


def _cache_put(
    cache: OrderedDict[str, ObservationSummary],
    key: str,
    value: ObservationSummary,
    *,
    limit: int,
) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)


def _summary_without_cost(summary: ObservationSummary) -> ObservationSummary:
    return ObservationSummary(
        kind=summary.kind,
        predicate_cover=summary.predicate_cover,
        source=summary.source,
        topk=summary.topk,
        full_search=summary.full_search,
        filtered_ann_hit=summary.filtered_ann_hit,
    )


def _cacheable_summary(summary: ObservationSummary) -> bool:
    return summary.kind not in {
        ObservationRequestKind.PREDICATE_COVER,
        ObservationRequestKind.VECTOR_FULL_SEARCH,
    }


def _cache_enabled_from_env() -> bool:
    value = os.environ.get(OBSERVATION_CACHE_ENV, "").strip().lower()
    if value in {"0", "false", "off", "no"}:
        return False
    return True


def _result_log_observation(summary: ObservationSummary) -> ObservationSummary:
    if summary.kind in {
        ObservationRequestKind.PREDICATE_COVER,
        ObservationRequestKind.VECTOR_FULL_SEARCH,
    }:
        return ObservationSummary(kind=summary.kind)
    return _summary_without_cost(summary)


def _observation_cost_event(
    identity: ObservationRequestIdentity,
    summary: ObservationSummary,
    *,
    stage: HintCostStage | None,
) -> ObservationCostEvent:
    cost = summary.cost
    vector_observations = int(cost.get("hint_observations", 0))
    hint_observations = vector_observations if identity.purpose is ObservationPurpose.HINT else 0
    canonical_observations = vector_observations if identity.purpose is ObservationPurpose.CANONICAL else 0
    return ObservationCostEvent(
        identity=identity,
        hint_stage=stage,
        exact_calls=int(cost.get("exact_calls", 0)),
        hint_observations=hint_observations,
        canonical_observations=canonical_observations,
        cover_computations=int(cost.get("cover_computations", 0)),
        full_search_calls=int(cost.get("full_search_calls", 0)),
        vector_observations=vector_observations,
    )


def _validate_observation_request(request: ObservationRequest) -> None:
    if request.kind is ObservationRequestKind.VECTOR_FULL_SEARCH and request.cover_payload is not None:
        raise ContractViolation("vector full-search observation must not include cover payload")
    if request.cover_payload is not None and request.kind not in {
        ObservationRequestKind.FILTERED_ANN_HIT,
        ObservationRequestKind.VECTOR_TOPK,
    }:
        raise ContractViolation("cover payload is only valid for filtered vector observations")
    if request.kind is ObservationRequestKind.VECTOR_TOPK and request.k is None:
        raise ContractViolation("vector top-k observation requires k")
    if request.kind is ObservationRequestKind.FILTERED_ANN_HIT:
        if request.k is None:
            raise ContractViolation("filtered ANN hit observation requires k")
        if request.cover_payload is None:
            raise ContractViolation("filtered ANN hit observation requires cover payload")
        if request.hit_band is None:
            raise ContractViolation("filtered ANN hit observation requires hit band")


def _observation_kind_profile_name(kind: ObservationRequestKind) -> str:
    if kind is ObservationRequestKind.PREDICATE_COVER:
        return "cover"
    if kind is ObservationRequestKind.VECTOR_TOPK:
        return "topk"
    if kind is ObservationRequestKind.VECTOR_FULL_SEARCH:
        return "full_search"
    if kind is ObservationRequestKind.FILTERED_ANN_HIT:
        return "hit"
    if kind is ObservationRequestKind.VECTOR_SOURCE:
        return "source"
    raise ContractViolation(f"unsupported observation kind: {kind.value}")


def _observation_identity(
    request: ObservationRequest,
    *,
    purpose: ObservationPurpose,
) -> ObservationRequestIdentity:
    return ObservationRequestIdentity(
        purpose=purpose,
        kind=request.kind.value,
        request_fingerprint=_observation_request_fingerprint(request),
    )


def _observation_request_fingerprint(request: ObservationRequest) -> str:
    if request.kind is ObservationRequestKind.PREDICATE_COVER:
        if request.cover_node is None:
            raise ContractViolation("predicate cover observation requires cover node")
        return stable_signature(
            "observation",
            {
                "kind": request.kind,
                "field": request.field,
                "template": request.template_signature,
                "cover_node_id": request.cover_node.node_id,
                "cover_values": dict(request.cover_values),
            },
        )
    if request.kind in {
        ObservationRequestKind.FILTERED_ANN_HIT,
        ObservationRequestKind.VECTOR_TOPK,
    }:
        cover_fingerprint = None
        if request.cover_payload is not None:
            cover_fingerprint = request.cover_payload.fingerprint
    if request.kind is ObservationRequestKind.FILTERED_ANN_HIT:
        if request.cover_payload is None:
            raise ContractViolation("filtered ANN hit observation requires cover payload")
        return stable_signature(
            "observation",
            {
                "kind": request.kind,
                "field": request.field,
                "query_vector": request.query_vector,
                "k": request.k,
                "cover_fingerprint": cover_fingerprint,
                "hit_band": request.hit_band,
            },
        )
    if request.kind is ObservationRequestKind.VECTOR_TOPK:
        return stable_signature(
            "observation",
            {
                "kind": request.kind,
                "field": request.field,
                "query_vector": request.query_vector,
                "k": request.k,
                "cover_fingerprint": cover_fingerprint,
            },
        )
    return stable_signature("observation", request)


__all__ = [
    "CoverPayload",
    "HintCostStage",
    "ObservationRequestKind",
    "ObservationExecutionPort",
    "ObservationRequest",
    "ObservationSummary",
    "ObservationCacheStatus",
    "ObservationCostEvent",
    "ObservationCostStats",
    "ObservationPurpose",
    "ObservationRequestIdentity",
    "ObservationResult",
    "ObservationSession",
    "FilteredAnnHitHint",
    "DatasetObservationExecutionPort",
    "PredicateCoverHint",
    "VectorFullSearchHint",
    "VectorSourceHint",
    "VectorTopKHint",
    "summarize_observation_cost_events",
]
