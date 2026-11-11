"""Attribute-vector coupling strategies and placement materialization."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from heapq import heapify, heappop, heappush
import enum
import logging
import os as _os
import random
import math
import time
from typing import Protocol, SupportsFloat, cast

from griffin.data.generation_request import VectorAttributeCouplingRequest
from griffin.dataset.deterministic import (
    CategoricalBlocksModel,
    CouplingGroup,
    DeterministicFieldModel,
    FieldPlacementModel,
    IntAffineBlocksModel,
    LognormalNumericModel,
    PlacementSegment,
    RegionLayoutModel,
    UniformNumericModel,
    VectorAttributeCouplingModel,
)
from griffin.infra.errors import ContractViolation
from griffin.ir.data_type import DoubleType, IntType, TextType
from griffin.ir.field import Field


_logger = logging.getLogger("griffin.task")


CountMatrix = tuple[tuple[int, ...], ...]
FractionalMatrix = tuple[tuple[float, ...], ...]
_DEFAULT_INDEPENDENT_MAX_GROUPS = 256
_COUPLING_DEBUG = _os.environ.get("GRIFFIN_COUPLING_DEBUG", "").strip() == "1"
_RHO_RELATIVE_TOLERANCE = 0.01
_ZERO_RHO_ABSOLUTE_TOLERANCE = 0.001


@dataclass(frozen=True, slots=True)
class CouplingTask:
    """Strategy-independent task for placing one attribute against vector regions."""

    name: str
    vector_field_name: str
    attribute_field_name: str
    target_rho: float
    max_groups: int | None
    field: Field
    model: DeterministicFieldModel
    local_values: tuple[object, ...]
    region_layout: RegionLayoutModel

    @property
    def row_count(self) -> int:
        return self.model.row_count


@dataclass(frozen=True, slots=True)
class GroupNumericStats:
    prefix_sum: tuple[float, ...]
    prefix_sq_sum: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class GroupingResult:
    groups: tuple[CouplingGroup, ...]
    categorical: bool
    numeric_group_stats: tuple[GroupNumericStats, ...] | None = None

    @property
    def counts(self) -> tuple[int, ...]:
        return tuple(group.local_end - group.local_start + 1 for group in self.groups)

    @property
    def scores(self) -> tuple[float, ...]:
        if not self.categorical:
            return tuple(
                float(group.representative_value) if group.representative_value is not None else float(group.group_id)
                for group in self.groups
            )
        # categorical: sorted label index orders the coupled-endpoint construction;
        # rho itself is NMI and no longer consumes these scores.
        sorted_labels = sorted(group.label for group in self.groups)
        label_to_index = {label: i for i, label in enumerate(sorted_labels)}
        return tuple(float(label_to_index[group.label]) for group in self.groups)


class LocalSearchDeltaMode(enum.Enum):
    """2×2 cycle-move step-size policy for local refinement."""

    ADAPTIVE = "adaptive"  # multi-scale delta, decays on stagnation (default)
    FIXED = "fixed"  # paper semantics: delta is always 1, stop on first stagnation


@dataclass(frozen=True, slots=True)
class CouplingStrategyConfig:
    iterations: int = 64
    matrix_search_enabled: bool = True   # False → 跳过插值矩阵搜索，直接从独立矩阵开始局部搜索
    local_search_enabled: bool = True
    local_search_max_steps: int = 2048
    local_search_initial_delta: int = 4
    local_search_delta_decay: float = 0.5
    local_search_delta_mode: LocalSearchDeltaMode = LocalSearchDeltaMode.ADAPTIVE
    local_search_swap_limit: int = 100_000
    local_search_time_budget_sec: float = float("inf")  # 局部搜索时间预算（秒），默认不限时


@dataclass(frozen=True, slots=True)
class CouplingMethodResult:
    placement: FieldPlacementModel
    region_group_counts: CountMatrix
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VectorCouplingPlan:
    field_name: str
    values: list[object]
    placement: FieldPlacementModel
    metadata: VectorAttributeCouplingModel
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _NumericSegmentSplit:
    left: int
    right: int
    split: int
    reduction: float


class AttributeGroupingStrategy(Protocol):
    name: str

    def group(self, task: CouplingTask) -> GroupingResult: ...


class CouplingPlanner(Protocol):
    name: str

    def plan(
        self,
        task: CouplingTask,
        grouping: GroupingResult,
        config: CouplingStrategyConfig,
        materializer: PlacementMaterializer,
    ) -> CouplingMethodResult: ...


class RhoMetric(Protocol):
    def from_values(
        self,
        values: Sequence[object],
        region_layout: RegionLayoutModel,
        *,
        categorical: bool,
    ) -> float: ...

    def from_matrix(self, matrix: Sequence[Sequence[float]], grouping: GroupingResult) -> float: ...


class PlacementMaterializer(Protocol):
    def placement_from_matrix(
        self,
        *,
        field_name: str,
        row_count: int,
        groups: tuple[CouplingGroup, ...],
        region_layout: RegionLayoutModel,
        matrix: CountMatrix,
    ) -> FieldPlacementModel: ...

    def apply(self, local_values: Sequence[object], placement: FieldPlacementModel) -> list[object]: ...


class DeterministicAttributeGroupingStrategy:
    """Build attribute groups from deterministic value models."""

    name = "deterministic_attribute_groups"

    def group(self, task: CouplingTask) -> GroupingResult:
        model = task.model
        if isinstance(model, CategoricalBlocksModel) or (
            isinstance(model, IntAffineBlocksModel) and model.distribution_kind == "zipf"
        ):
            groups = _categorical_ndv_groups(task, model)
            return GroupingResult(groups=groups, categorical=True)
        if isinstance(model, (UniformNumericModel, LognormalNumericModel, IntAffineBlocksModel)):
            groups = _numeric_sse_greedy_groups(task)
            return GroupingResult(
                groups=groups,
                categorical=False,
                numeric_group_stats=_build_numeric_group_stats(task.local_values, groups),
            )
        raise ContractViolation(f"vector coupling does not support model type: {type(model).__name__}")


class ExplainedVarianceRhoMetric:
    """Griffin dependency metric: NMI for categorical, explained variance for numeric.

    Categorical attributes use normalized mutual information
    rho_cat(A, Z) = I(A;Z) / H(A), matching the paper definition.
    Numeric attributes keep the fraction of explained variance
    rho_num(A, Z) = Var(E[A|Z]) / Var(A).
    """

    def from_values(
        self,
        values: Sequence[object],
        region_layout: RegionLayoutModel,
        *,
        categorical: bool,
    ) -> float:
        if categorical:
            return _categorical_value_rho(values, region_layout)
        return _numeric_value_rho(values, region_layout)

    def from_matrix(self, matrix: Sequence[Sequence[float]], grouping: GroupingResult) -> float:
        if grouping.categorical:
            return _categorical_matrix_nmi(matrix)
        integral = all(
            float(value).is_integer()
            for row in matrix
            for value in row
        )
        # exact per-row variance needs integral counts to index group segments;
        # fractional interpolation matrices are evaluated through group scores.
        if grouping.numeric_group_stats is not None and integral:
            return _numeric_matrix_rho_exact(cast(CountMatrix, matrix), grouping.numeric_group_stats)
        return _numeric_matrix_rho(matrix, grouping.scores)


def rho_target_satisfied(*, target_rho: float, observed_rho: float) -> bool:
    """Return whether observed rho satisfies the formal generation tolerance."""
    absolute_error = abs(observed_rho - target_rho)
    if target_rho == 0.0:
        return absolute_error <= _ZERO_RHO_ABSOLUTE_TOLERANCE
    return absolute_error / target_rho <= _RHO_RELATIVE_TOLERANCE


class StableSegmentPlacementMaterializer:
    """Materialize a region x group matrix into deterministic placement segments."""

    def placement_from_matrix(
        self,
        *,
        field_name: str,
        row_count: int,
        groups: tuple[CouplingGroup, ...],
        region_layout: RegionLayoutModel,
        matrix: CountMatrix,
    ) -> FieldPlacementModel:
        local_cursors = [group.local_start for group in groups]
        public_cursors = [region.public_start for region in region_layout.regions]
        segments: list[PlacementSegment] = []
        for region_index, region in enumerate(region_layout.regions):
            for group_index, count in enumerate(matrix[region_index]):
                if count <= 0:
                    continue
                local_start = local_cursors[group_index]
                local_end = local_start + count - 1
                public_start = public_cursors[region_index]
                public_end = public_start + count - 1
                segments.append(
                    PlacementSegment(
                        local_start=local_start,
                        local_end=local_end,
                        public_start=public_start,
                        public_end=public_end,
                        region_id=region.region_id,
                    )
                )
                local_cursors[group_index] = local_end + 1
                public_cursors[region_index] = public_end + 1
        segments.sort(key=lambda segment: segment.local_start)
        return FieldPlacementModel(field_name=field_name, row_count=row_count, segments=tuple(segments))

    def apply(self, local_values: Sequence[object], placement: FieldPlacementModel) -> list[object]:
        public_values: list[object] = [None] * placement.row_count
        for segment in placement.segments:
            for offset, local_ordinal in enumerate(range(segment.local_start, segment.local_end + 1)):
                public_values[segment.public_start + offset] = local_values[local_ordinal]
        if any(value is None for value in public_values):
            raise ContractViolation(
                f"field placement for {placement.field_name} does not cover the full public ordinal domain"
            )
        return public_values


class MatrixTransportCouplingPlanner:
    """Marginal-preserving matrix transport planner."""

    name = "matrix_transport"

    def __init__(self, metric: RhoMetric | None = None) -> None:
        self._metric = metric or ExplainedVarianceRhoMetric()

    def plan(
        self,
        task: CouplingTask,
        grouping: GroupingResult,
        config: CouplingStrategyConfig,
        materializer: PlacementMaterializer,
    ) -> CouplingMethodResult:
        region_counts = _region_counts(task.region_layout)
        group_counts = grouping.counts
        independent = _independent_matrix(region_counts, group_counts)
        coupled = _score_ordered_coupled_endpoint(region_counts, grouping)

        independent_rho = self._metric.from_matrix(independent, grouping)
        coupled_rho = self._metric.from_matrix(coupled, grouping)
        if _COUPLING_DEBUG:
            _logger.info(
                "[debug] %s: target=%.4f  independent_rho=%.6f  coupled_endpoint_rho=%.6f  regions=%d  groups=%d",
                task.name,
                task.target_rho,
                independent_rho,
                coupled_rho,
                len(region_counts),
                len(group_counts),
            )

        _t0 = time.perf_counter()
        if config.matrix_search_enabled:
            best_matrix, best_rho, best_mix, ms_timed_trajectory = _search_interpolated_matrix(
                target_rho=task.target_rho,
                independent=independent,
                coupled=coupled,
                grouping=grouping,
                metric=self._metric,
                iterations=config.iterations,
            )
        else:
            # only_local_search baseline：直接从独立矩阵开始，跳过矩阵搜索
            best_matrix = independent
            best_rho = self._metric.from_matrix(independent, grouping)
            best_mix = 0.0
            ms_timed_trajectory = [(0.0, best_rho)]
        t_matrix_search = time.perf_counter() - _t0
        if _COUPLING_DEBUG:
            _logger.info(
                "[debug] %s: matrix_search done  best_rho=%.6f  best_mix=%.4f  error=%.6f  elapsed=%.4f",
                task.name, best_rho, best_mix, abs(best_rho - task.target_rho), t_matrix_search,
            )
        local_search_diagnostics: dict[str, object] = {}
        t_local_search = 0.0
        if config.local_search_enabled and not rho_target_satisfied(
            target_rho=task.target_rho,
            observed_rho=best_rho,
        ):
            if _COUPLING_DEBUG:
                _logger.info(
                    "[debug] %s: local_search start  initial_rho=%.6f  target=%.4f  delta=%d  time_budget=%.1f",
                    task.name, best_rho, task.target_rho,
                    config.local_search_initial_delta, config.local_search_time_budget_sec,
                )
            _t1 = time.perf_counter()
            effective_max_steps, effective_swap_limit = _effective_local_search_budget(
                region_count=len(region_counts),
                group_count=len(group_counts),
                config=config,
            )
            best_matrix, best_rho, local_search_diagnostics = _refine_matrix_local_search(
                matrix=best_matrix,
                target_rho=task.target_rho,
                grouping=grouping,
                metric=self._metric,
                region_counts=region_counts,
                group_counts=group_counts,
                max_steps=effective_max_steps,
                initial_delta=config.local_search_initial_delta,
                delta_decay=config.local_search_delta_decay,
                swap_limit=effective_swap_limit,
                time_budget_sec=config.local_search_time_budget_sec,
                delta_mode=config.local_search_delta_mode,
            )
            t_local_search = time.perf_counter() - _t1
            if _COUPLING_DEBUG:
                ls_steps = local_search_diagnostics.get("local_search_steps", "?")
                ls_error = local_search_diagnostics.get("local_search_final_error", "?")
                _logger.info(
                    "[debug] %s: local_search done  final_rho=%.6f  steps=%s  error=%s  elapsed=%.4f",
                    task.name, best_rho, ls_steps, ls_error, t_local_search,
                )
        elif config.local_search_enabled:
            local_search_diagnostics = {
                "local_search_steps": 0,
                "local_search_skipped": True,
                "local_search_skip_reason": "target_tolerance_satisfied",
                "local_search_final_error": abs(best_rho - task.target_rho),
                "local_search_final_rho": best_rho,
            }
        placement = materializer.placement_from_matrix(
            field_name=task.attribute_field_name,
            row_count=task.row_count,
            groups=grouping.groups,
            region_layout=task.region_layout,
            matrix=best_matrix,
        )
        diagnostics: dict[str, object] = {
            "strategy": self.name,
            "grouping_strategy": DeterministicAttributeGroupingStrategy.name,
            "planned_rho": best_rho,
            "mix": best_mix,
            "iterations": config.iterations,
            "region_count": len(region_counts),
            "group_count": len(group_counts),
            "local_search_enabled": config.local_search_enabled,
            "local_search_delta_mode": config.local_search_delta_mode.value,
            "t_matrix_search": round(t_matrix_search, 4),
            "t_local_search": round(t_local_search, 4),
            "ms_timed_trajectory": ms_timed_trajectory,  # [(elapsed_sec, rho), ...] 矩阵搜索轨迹
        }
        diagnostics.update(local_search_diagnostics)
        import json as _json
        ls_traj = local_search_diagnostics.get("local_search_rho_trajectory_timed", [])
        _logger.info(
            "[coupling] %s  t_matrix_search=%.4f秒  t_local_search=%.4f秒"
            "  ms_trajectory=%s  ls_trajectory=%s",
            task.name,
            t_matrix_search,
            t_local_search,
            _json.dumps(ms_timed_trajectory),
            _json.dumps(ls_traj),
        )
        return CouplingMethodResult(
            placement=placement,
            region_group_counts=best_matrix,
            diagnostics=diagnostics,
        )


class AttributeVectorCouplingEngine:
    """Coordinates grouping, matrix planning, placement, and diagnostics."""

    def __init__(
        self,
        *,
        grouping: AttributeGroupingStrategy | None = None,
        planner: CouplingPlanner | None = None,
        metric: RhoMetric | None = None,
        materializer: PlacementMaterializer | None = None,
        config: CouplingStrategyConfig | None = None,
    ) -> None:
        self._grouping = grouping or DeterministicAttributeGroupingStrategy()
        self._metric = metric or ExplainedVarianceRhoMetric()
        self._planner = planner or MatrixTransportCouplingPlanner(self._metric)
        self._materializer = materializer or StableSegmentPlacementMaterializer()
        self._config = config

    def plan(self, task: CouplingTask, config: CouplingStrategyConfig | None = None) -> VectorCouplingPlan:
        _validate_task(task)
        resolved_config = config or self._config or CouplingStrategyConfig()
        grouping = self._grouping.group(task)
        if not grouping.groups:
            raise ContractViolation(f"vector coupling {task.name} requires at least one attribute group")
        result = self._planner.plan(
            task,
            grouping,
            resolved_config,
            self._materializer,
        )
        public_values = self._materializer.apply(task.local_values, result.placement)
        observed_rho = self._metric.from_values(
            public_values,
            task.region_layout,
            categorical=grouping.categorical,
        )
        metadata = VectorAttributeCouplingModel(
            name=task.name,
            vector_field_name=task.vector_field_name,
            attribute_field_name=task.attribute_field_name,
            target_rho=float(task.target_rho),
            observed_rho=observed_rho,
            max_groups=task.max_groups,
            groups=tuple(grouping.groups),
            region_group_counts=result.region_group_counts,
        )
        return VectorCouplingPlan(
            field_name=task.attribute_field_name,
            values=public_values,
            placement=result.placement,
            metadata=metadata,
            diagnostics=result.diagnostics,
        )


def plan_vector_attribute_coupling(
    *,
    request: VectorAttributeCouplingRequest,
    field: Field,
    model: DeterministicFieldModel,
    local_values: Sequence[object],
    region_layout: RegionLayoutModel,
    engine: AttributeVectorCouplingEngine | None = None,
) -> VectorCouplingPlan:
    task = CouplingTask(
        name=request.name,
        vector_field_name=request.vector_field,
        attribute_field_name=request.attribute_field,
        target_rho=request.target_rho,
        max_groups=request.max_groups,
        field=field,
        model=model,
        local_values=tuple(local_values),
        region_layout=region_layout,
    )
    return (engine or AttributeVectorCouplingEngine()).plan(task)


def plan_independent_attribute_placement(
    *,
    field: Field,
    model: DeterministicFieldModel,
    local_values: Sequence[object],
    region_layout: RegionLayoutModel,
    max_groups: int = _DEFAULT_INDEPENDENT_MAX_GROUPS,
) -> tuple[list[object], FieldPlacementModel]:
    task = CouplingTask(
        name=f"{field.id.name}_region_independent",
        vector_field_name=region_layout.vector_field_name,
        attribute_field_name=field.id.name,
        target_rho=0.0,
        max_groups=max_groups,
        field=field,
        model=model,
        local_values=tuple(local_values),
        region_layout=region_layout,
    )
    _validate_task(task)
    grouping = _independent_grouping(task)
    if not grouping.groups:
        raise ContractViolation(f"independent vector placement for {field.id.name} requires at least one attribute group")
    matrix = _independent_matrix(_region_counts(region_layout), grouping.counts)
    materializer = StableSegmentPlacementMaterializer()
    placement = materializer.placement_from_matrix(
        field_name=field.id.name,
        row_count=task.row_count,
        groups=grouping.groups,
        region_layout=region_layout,
        matrix=matrix,
    )
    public_values = materializer.apply(task.local_values, placement)
    return public_values, placement


def _independent_grouping(task: CouplingTask) -> GroupingResult:
    model = task.model
    if isinstance(model, CategoricalBlocksModel) or (
        isinstance(model, IntAffineBlocksModel) and model.distribution_kind == "zipf"
    ):
        return DeterministicAttributeGroupingStrategy().group(task)
    if isinstance(model, (UniformNumericModel, LognormalNumericModel, IntAffineBlocksModel)):
        return GroupingResult(groups=_numeric_equal_row_groups(task), categorical=False)
    raise ContractViolation(f"independent vector placement does not support model type: {type(model).__name__}")


def _numeric_equal_row_groups(task: CouplingTask) -> tuple[CouplingGroup, ...]:
    model = task.model
    if not isinstance(model, (UniformNumericModel, LognormalNumericModel, IntAffineBlocksModel)):
        raise ContractViolation(
            f"numeric independent vector placement requires a numeric model, got {type(model).__name__}"
        )
    group_count = min(max(1, task.max_groups or _DEFAULT_INDEPENDENT_MAX_GROUPS), model.non_null_row_count)
    if group_count <= 0:
        return ()
    base = model.non_null_row_count // group_count
    remainder = model.non_null_row_count % group_count
    groups: list[CouplingGroup] = []
    cursor = model.nulls_first
    for group_id in range(group_count):
        count = base + (1 if group_id < remainder else 0)
        if count <= 0:
            continue
        start = cursor
        end = cursor + count - 1
        values = [_coerce_float(value) for value in task.local_values[start : end + 1]]
        groups.append(
            CouplingGroup(
                group_id=group_id,
                label=f"g{group_id}",
                local_start=start,
                local_end=end,
                representative_value=sum(values) / len(values),
            )
        )
        cursor = end + 1
    return tuple(groups)


def _categorical_ndv_groups(
    task: CouplingTask,
    model: CategoricalBlocksModel | IntAffineBlocksModel,
) -> tuple[CouplingGroup, ...]:
    groups: list[CouplingGroup] = []
    seen_labels: set[str] = set()
    current_label: str | None = None
    current_start: int | None = None
    current_end: int | None = None

    def flush() -> None:
        nonlocal current_label, current_start, current_end
        if current_label is None or current_start is None or current_end is None:
            return
        if current_label in seen_labels:
            raise ContractViolation(
                f"categorical vector coupling for {task.attribute_field_name} requires contiguous NDV groups; "
                f"label {current_label!r} appears in multiple local ranges"
            )
        seen_labels.add(current_label)
        groups.append(
            CouplingGroup(
                group_id=len(groups),
                label=current_label,
                local_start=current_start,
                local_end=current_end,
                representative_value=_representative_for_label(task.field, current_label),
            )
        )
        current_label = None
        current_start = None
        current_end = None

    for block in model.blocks:
        block_start = model.nulls_first + block.start
        block_end = model.nulls_first + block.end
        if current_label == block.label and current_end is not None and block_start == current_end + 1:
            current_end = block_end
            continue
        flush()
        current_label = block.label
        current_start = block_start
        current_end = block_end
    flush()
    return tuple(groups)


def observed_rho_for_values(
    values: Sequence[object],
    region_layout: RegionLayoutModel,
    *,
    categorical: bool | None = None,
) -> float:
    if not values or region_layout.row_count <= 0:
        return 0.0
    if len(values) != region_layout.row_count:
        raise ContractViolation("rho calculation requires values length to match region_layout.row_count")
    non_null = [value for value in values if value is not None]
    if not non_null:
        return 0.0
    if categorical is True:
        return _categorical_value_rho(values, region_layout)
    if categorical is False:
        return _numeric_value_rho(values, region_layout)
    if all(_is_numeric(value) for value in non_null):
        return _numeric_value_rho(values, region_layout)
    return _categorical_value_rho(values, region_layout)


def _validate_task(task: CouplingTask) -> None:
    if task.model.row_count != task.region_layout.row_count:
        raise ContractViolation(f"vector coupling {task.name} row_count mismatch between attribute and region layout")
    if task.model.row_count != len(task.local_values):
        raise ContractViolation(f"vector coupling {task.name} local value length does not match model row_count")
    if getattr(task.model, "nulls_first", 0) != 0:
        raise ContractViolation(f"vector coupling {task.name} does not support nullable attributes")
    if not 0.0 <= task.target_rho <= 1.0:
        raise ContractViolation(f"vector coupling {task.name} target_rho must be within [0,1]")


def _numeric_sse_greedy_groups(task: CouplingTask) -> tuple[CouplingGroup, ...]:
    model = task.model
    if not isinstance(model, (UniformNumericModel, LognormalNumericModel, IntAffineBlocksModel)):
        raise ContractViolation(
            f"numeric vector coupling requires a numeric model, got {type(model).__name__}"
        )
    group_count = min(task.max_groups or 512, model.non_null_row_count)
    if group_count < 2 and model.non_null_row_count > 1:
        group_count = 2
    if group_count <= 0:
        return ()
    values = [
        _coerce_float(v)
        for v in task.local_values[model.nulls_first : model.nulls_first + model.non_null_row_count]
    ]
    n = len(values)
    if n <= 1:
        return ()
    prefix_sum = [0.0]
    prefix_sum_sq = [0.0]
    for v in values:
        prefix_sum.append(prefix_sum[-1] + v)
        prefix_sum_sq.append(prefix_sum_sq[-1] + v * v)

    def _range_sse(left: int, right: int) -> float:
        if right - left <= 1:
            return 0.0
        s = prefix_sum[right] - prefix_sum[left]
        sq = prefix_sum_sq[right] - prefix_sum_sq[left]
        count = right - left
        return sq - s * s / count

    def _best_split(left: int, right: int) -> _NumericSegmentSplit | None:
        if right - left < 2:
            return None
        full_sse = _range_sse(left, right)
        best_reduction = -1.0
        best_split = -1
        for split in range(left + 1, right):
            reduction = full_sse - (_range_sse(left, split) + _range_sse(split, right))
            if reduction > best_reduction:
                best_reduction = reduction
                best_split = split
        if best_split < 0 or best_reduction <= 1e-12:
            return None
        return _NumericSegmentSplit(left=left, right=right, split=best_split, reduction=best_reduction)

    active_segments: set[tuple[int, int]] = {(0, n)}
    heap: list[tuple[float, int, int, int]] = []

    def _push_split(left: int, right: int) -> None:
        split = _best_split(left, right)
        if split is not None:
            heappush(heap, (-split.reduction, split.left, split.split, split.right))

    _push_split(0, n)
    while len(active_segments) < group_count and heap:
        negative_reduction, left, split, right = heappop(heap)
        if (left, right) not in active_segments:
            continue
        reduction = -negative_reduction
        if reduction <= 1e-12:
            break
        active_segments.remove((left, right))
        active_segments.add((left, split))
        active_segments.add((split, right))
        _push_split(left, split)
        _push_split(split, right)

    segments = sorted(active_segments)

    groups: list[CouplingGroup] = []
    cursor = model.nulls_first
    for group_id, (left, right) in enumerate(segments):
        count = right - left
        if count <= 0:
            continue
        start = cursor
        end = cursor + count - 1
        seg_values = values[left:right]
        representative = sum(seg_values) / len(seg_values) if seg_values else 0.0
        groups.append(
            CouplingGroup(
                group_id=group_id,
                label=f"g{group_id}",
                local_start=start,
                local_end=end,
                representative_value=representative,
            )
        )
        cursor = end + 1
    return tuple(groups)


def _build_numeric_group_stats(
    local_values: Sequence[object],
    groups: tuple[CouplingGroup, ...],
) -> tuple[GroupNumericStats, ...]:
    stats: list[GroupNumericStats] = []
    for group in groups:
        group_values = [_coerce_float(value) for value in local_values[group.local_start : group.local_end + 1]]
        prefix_sum: list[float] = [0.0]
        prefix_sq_sum: list[float] = [0.0]
        for value in group_values:
            prefix_sum.append(prefix_sum[-1] + value)
            prefix_sq_sum.append(prefix_sq_sum[-1] + value * value)
        stats.append(
            GroupNumericStats(
                prefix_sum=tuple(prefix_sum),
                prefix_sq_sum=tuple(prefix_sq_sum),
            )
        )
    return tuple(stats)


def _representative_for_label(field: Field, label: str) -> float | None:
    if isinstance(field.type, IntType):
        return float(int(float(label)))
    if isinstance(field.type, DoubleType):
        return float(label)
    if isinstance(field.type, TextType):
        return None
    return None


def _region_counts(region_layout: RegionLayoutModel) -> tuple[int, ...]:
    return tuple(region.public_end - region.public_start + 1 for region in region_layout.regions)


def _independent_matrix(
    region_counts: tuple[int, ...],
    group_counts: tuple[int, ...],
) -> CountMatrix:
    total = sum(region_counts)
    if total != sum(group_counts):
        raise ContractViolation("matrix margins do not agree")
    if total <= 0:
        result = tuple(tuple(0 for _ in group_counts) for _ in region_counts)
        _validate_matrix(result, region_counts, group_counts)
        return result

    matrix = [[0 for _ in group_counts] for _ in region_counts]
    assigned_by_region = [0 for _ in region_counts]
    processed = 0
    for group_index, group_count in enumerate(group_counts):
        next_processed = processed + group_count
        column = _independent_column_counts(
            region_counts=region_counts,
            assigned_by_region=assigned_by_region,
            total=total,
            next_processed=next_processed,
            group_count=group_count,
        )
        for region_index, count in enumerate(column):
            matrix[region_index][group_index] = count
            assigned_by_region[region_index] += count
        processed = next_processed

    result = tuple(tuple(row) for row in matrix)
    _validate_matrix(result, region_counts, group_counts)
    return result


def _independent_column_counts(
    *,
    region_counts: tuple[int, ...],
    assigned_by_region: list[int],
    total: int,
    next_processed: int,
    group_count: int,
) -> tuple[int, ...]:
    column: list[int] = []
    remainders: list[float] = []
    for region_index, region_count in enumerate(region_counts):
        capacity = region_count - assigned_by_region[region_index]
        quota = next_processed * region_count / total - assigned_by_region[region_index]
        base = max(0, min(capacity, math.floor(quota)))
        column.append(base)
        remainders.append(quota - base)

    remaining = group_count - sum(column)
    if remaining > 0:
        # 余数分配改用堆：每次取「remainder 最大（同分取容量大、index 小）」的
        # region，与原逐轮 max 扫描等价，但每列从 O(k²) 降到 O((k+remaining)·log k)。
        heap = [
            (
                -remainders[i],
                -(region_counts[i] - assigned_by_region[i] - column[i]),
                i,
            )
            for i in range(len(region_counts))
            if assigned_by_region[i] + column[i] < region_counts[i]
        ]
        if not heap:
            raise ContractViolation("independent matrix could not satisfy row margin")
        heapify(heap)
        while remaining > 0:
            _neg_remainder, _neg_capacity, region_index = heappop(heap)
            column[region_index] += 1
            remainders[region_index] -= 1.0
            remaining -= 1
            capacity = region_counts[region_index] - assigned_by_region[region_index] - column[region_index]
            if capacity > 0:
                heappush(
                    heap,
                    (-remainders[region_index], -capacity, region_index),
                )

    elif remaining < 0:
        # 退化方向同理：每次取「remainder 最小（同分取已分配少、index 小）」。
        heap = [
            (remainders[i], column[i], i)
            for i in range(len(column))
            if column[i] > 0
        ]
        if not heap:
            raise ContractViolation("independent matrix over-allocated a group")
        heapify(heap)
        while remaining < 0:
            _remainder, _count, region_index = heappop(heap)
            column[region_index] -= 1
            remainders[region_index] += 1.0
            remaining += 1
            if column[region_index] > 0:
                heappush(
                    heap,
                    (remainders[region_index], column[region_index], region_index),
                )

    return tuple(column)


def _score_ordered_coupled_endpoint(
    region_counts: tuple[int, ...],
    grouping: GroupingResult,
) -> CountMatrix:
    """Construct a high-coupling endpoint without changing matrix column identity."""
    group_counts = grouping.counts
    scores = grouping.scores
    matrix = [[0 for _ in group_counts] for _ in region_counts]
    remaining = list(group_counts)
    group_order = tuple(
        sorted(
            range(len(group_counts)),
            key=lambda index: (scores[index], index),
        )
    )
    order_index = 0
    for region_index, region_count in enumerate(region_counts):
        need = region_count
        while need > 0 and order_index < len(group_order):
            group_index = group_order[order_index]
            take = min(need, remaining[group_index])
            matrix[region_index][group_index] += take
            remaining[group_index] -= take
            need -= take
            if remaining[group_index] == 0:
                order_index += 1
        if need != 0:
            raise ContractViolation("score-ordered coupled endpoint could not satisfy row margin")
    endpoint = tuple(tuple(row) for row in matrix)
    _validate_matrix(endpoint, region_counts, group_counts)
    return endpoint


def _search_interpolated_matrix(
    *,
    target_rho: float,
    independent: CountMatrix,
    coupled: CountMatrix,
    grouping: GroupingResult,
    metric: RhoMetric,
    iterations: int,
) -> tuple[CountMatrix, float, float, list[tuple[float, float]]]:
    """二分插值搜索，返回 (matrix, rho, mix, timed_trajectory)。

    迭代在纯分数插值矩阵上评估 ρ（插值路径的单调性更纯粹）；搜索结束后
    对最优 mix 做一轮统一整数化，返回整数矩阵及其 ρ。整数化引入的偏移
    由后续 local search 精修；命中容差时提前停止。
    """
    import time as _time
    t0_ms = _time.perf_counter()

    region_counts = tuple(sum(row) for row in independent)
    group_counts = tuple(sum(independent[row][column] for row in range(len(independent))) for column in range(len(grouping.groups)))

    best_values = _fractional_interpolated_matrix(independent=independent, coupled=coupled, mix=0.0)
    best_mix = 0.0
    best_rho = metric.from_matrix(best_values, grouping)
    best_error = abs(best_rho - target_rho)
    timed_trajectory: list[tuple[float, float]] = [(0.0, best_rho)]

    coupled_rho = metric.from_matrix(coupled, grouping)
    if abs(coupled_rho - target_rho) < best_error:
        best_values = _fractional_interpolated_matrix(independent=independent, coupled=coupled, mix=1.0)
        best_mix = 1.0
        best_rho = coupled_rho
        best_error = abs(coupled_rho - target_rho)
        timed_trajectory.append((_time.perf_counter() - t0_ms, best_rho))

    # 提前停止：若初始误差已满足阈值则整数化后直接返回
    if rho_target_satisfied(target_rho=target_rho, observed_rho=best_rho):
        return _finalize_interpolated_matrix(
            best_values,
            region_counts=region_counts,
            group_counts=group_counts,
            metric=metric,
            grouping=grouping,
            best_mix=best_mix,
            timed_trajectory=timed_trajectory,
        )

    low = 0.0
    high = 1.0
    for _iter_idx in range(max(1, iterations)):
        mix = (low + high) / 2.0
        values = _fractional_interpolated_matrix(independent=independent, coupled=coupled, mix=mix)
        rho = metric.from_matrix(values, grouping)
        error = abs(rho - target_rho)
        if _COUPLING_DEBUG:
            _logger.info(
                "[debug]   ms iter=%d  mix=%.4f  rho=%.6f  error=%.6f  best_error=%.6f",
                _iter_idx, mix, rho, error, best_error,
            )
        if error < best_error:
            best_values = values
            best_mix = mix
            best_rho = rho
            best_error = error
            timed_trajectory.append((_time.perf_counter() - t0_ms, best_rho))
        if rho < target_rho:
            low = mix
        else:
            high = mix
        if rho_target_satisfied(target_rho=target_rho, observed_rho=best_rho):
            break
    return _finalize_interpolated_matrix(
        best_values,
        region_counts=region_counts,
        group_counts=group_counts,
        metric=metric,
        grouping=grouping,
        best_mix=best_mix,
        timed_trajectory=timed_trajectory,
    )


def _fractional_interpolated_matrix(
    *,
    independent: CountMatrix,
    coupled: CountMatrix,
    mix: float,
) -> FractionalMatrix:
    values = [
        [
            (1.0 - mix) * independent[row_index][column_index] + mix * coupled[row_index][column_index]
            for column_index in range(len(independent[row_index]))
        ]
        for row_index in range(len(independent))
    ]
    return tuple(tuple(row) for row in values)


def _finalize_interpolated_matrix(
    best_values: FractionalMatrix,
    *,
    region_counts: tuple[int, ...],
    group_counts: tuple[int, ...],
    metric: RhoMetric,
    grouping: GroupingResult,
    best_mix: float,
    timed_trajectory: list[tuple[float, float]],
) -> tuple[CountMatrix, float, float, list[tuple[float, float]]]:
    """搜索结束后的一轮统一整数化；对已经是整数的输入恒等。"""
    matrix = _integerize_matrix(best_values, region_counts, group_counts)
    rho = metric.from_matrix(matrix, grouping)
    return matrix, rho, best_mix, timed_trajectory


def _integerize_matrix(
    values: Sequence[Sequence[float]],
    region_counts: tuple[int, ...],
    group_counts: tuple[int, ...],
) -> CountMatrix:
    matrix = [[int(math.floor(value)) for value in row] for row in values]
    row_deficit = [region_counts[row] - sum(matrix[row]) for row in range(len(region_counts))]
    column_deficit = [
        group_counts[column] - sum(matrix[row][column] for row in range(len(region_counts)))
        for column in range(len(group_counts))
    ]
    cells = sorted(
        (
            (
                float(values[row][column]) - matrix[row][column],
                row,
                column,
            )
            for row in range(len(region_counts))
            for column in range(len(group_counts))
        ),
        reverse=True,
    )
    remaining = sum(row_deficit)
    for _, row, column in cells:
        if remaining == 0:
            break
        if row_deficit[row] <= 0 or column_deficit[column] <= 0:
            continue
        delta = min(row_deficit[row], column_deficit[column])
        matrix[row][column] += delta
        row_deficit[row] -= delta
        column_deficit[column] -= delta
        remaining -= delta
    result = tuple(tuple(row) for row in matrix)
    _validate_matrix(result, region_counts, group_counts)
    return result


def _numeric_matrix_rho(matrix: Sequence[Sequence[float]], group_scores: tuple[float, ...]) -> float:
    # 直接用加权统计量计算，避免将矩阵展开为 O(N) 的列表
    total = sum(sum(row) for row in matrix)
    if total <= 0:
        return 0.0
    # 全局加权均值
    mean = sum(count * score for row in matrix for count, score in zip(row, group_scores, strict=True)) / total
    # 全局加权方差
    variance = sum(count * (score - mean) ** 2 for row in matrix for count, score in zip(row, group_scores, strict=True)) / total
    if variance <= 0.0:
        return 0.0
    # 组间方差（between-region variance）
    between = 0.0
    for row in matrix:
        region_count = sum(row)
        if region_count <= 0:
            continue
        region_mean = sum(count * score for count, score in zip(row, group_scores, strict=True)) / region_count
        between += region_count * (region_mean - mean) ** 2 / total
    return max(0.0, min(1.0, between / variance))


def _categorical_matrix_nmi(matrix: Sequence[Sequence[float]]) -> float:
    """Normalized mutual information rho_cat = I(A;Z) / H(A).

    Rows index vector regions Z, columns index attribute groups A.
    Both marginals stay fixed during coupling search, so H(A) is a
    constant of the assignment matrix; only the joint counts vary.
    """
    if not matrix:
        return 0.0
    total = sum(sum(row) for row in matrix)
    if total <= 0:
        return 0.0
    region_totals = [sum(row) for row in matrix]
    n_groups = len(matrix[0])
    group_totals = [sum(matrix[r][g] for r in range(len(matrix))) for g in range(n_groups)]

    # H(A): entropy of the attribute-group marginal (fixed during search)
    entropy = 0.0
    for group_total in group_totals:
        if group_total <= 0:
            continue
        p = group_total / total
        entropy -= p * math.log(p)
    if entropy <= 0.0:
        return 0.0

    # I(A;Z) = sum p(a,z) log(p(a,z) / (p(a) p(z)))
    info = 0.0
    for region_index, row in enumerate(matrix):
        region_total = region_totals[region_index]
        if region_total <= 0:
            continue
        p_z = region_total / total
        for group_index, count in enumerate(row):
            if count <= 0:
                continue
            group_total = group_totals[group_index]
            p_joint = count / total
            p_a = group_total / total
            if p_a <= 0.0 or p_z <= 0.0:
                continue
            info += p_joint * math.log(p_joint / (p_a * p_z))
    info = max(0.0, info)
    return max(0.0, min(1.0, info / entropy))


def _numeric_matrix_rho_exact(
    matrix: CountMatrix,
    group_stats: tuple[GroupNumericStats, ...],
) -> float:
    if not matrix:
        return 0.0
    n_groups = len(matrix[0])
    if n_groups != len(group_stats):
        raise ContractViolation("matrix group count does not match numeric group stats")

    group_cursors = [0 for _ in range(n_groups)]
    total_count = 0
    total_sum = 0.0
    total_sq_sum = 0.0
    region_counts: list[int] = []
    region_sums: list[float] = []

    for row in matrix:
        if len(row) != n_groups:
            raise ContractViolation("matrix column count mismatch in numeric exact rho")
        region_count = 0
        region_sum = 0.0
        region_sq_sum = 0.0
        for group_index, raw_count in enumerate(row):
            count = int(raw_count)  # mathematically integral counts may arrive as floats
            if count <= 0:
                continue
            stats = group_stats[group_index]
            start = group_cursors[group_index]
            end = start + count
            if end >= len(stats.prefix_sum):
                raise ContractViolation("matrix count exceeds numeric group capacity")
            region_sum += stats.prefix_sum[end] - stats.prefix_sum[start]
            region_sq_sum += stats.prefix_sq_sum[end] - stats.prefix_sq_sum[start]
            group_cursors[group_index] = end
            region_count += count
        total_count += region_count
        total_sum += region_sum
        total_sq_sum += region_sq_sum
        region_counts.append(region_count)
        region_sums.append(region_sum)

    if total_count <= 1:
        return 0.0
    global_mean = total_sum / total_count
    global_variance = total_sq_sum / total_count - global_mean * global_mean
    if global_variance <= 0.0:
        return 0.0

    between = 0.0
    for region_count, region_sum in zip(region_counts, region_sums, strict=True):
        if region_count <= 0:
            continue
        region_mean = region_sum / region_count
        between += region_count * (region_mean - global_mean) ** 2 / total_count
    return max(0.0, min(1.0, between / global_variance))


def _categorical_value_rho(
    values: Sequence[object],
    region_layout: RegionLayoutModel,
) -> float:
    labels = sorted({str(value) for value in values if value is not None})
    if len(labels) <= 1:
        return 0.0
    label_index = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in region_layout.regions]
    for region_index, region in enumerate(region_layout.regions):
        for ordinal in range(region.public_start, region.public_end + 1):
            value = values[ordinal]
            if value is None:
                continue
            matrix[region_index][label_index[str(value)]] += 1
    return _categorical_matrix_nmi(tuple(tuple(row) for row in matrix))


def _numeric_value_rho(
    values: Sequence[object],
    region_layout: RegionLayoutModel,
) -> float:
    numeric = [_coerce_float(value) for value in values if value is not None]
    if len(numeric) <= 1:
        return 0.0
    mean = sum(numeric) / len(numeric)
    variance = sum((value - mean) ** 2 for value in numeric) / len(numeric)
    if variance <= 0.0:
        return 0.0
    between = 0.0
    total = len(numeric)
    for region in region_layout.regions:
        region_values = [
            _coerce_float(values[ordinal])
            for ordinal in range(region.public_start, region.public_end + 1)
            if values[ordinal] is not None
        ]
        if not region_values:
            continue
        region_mean = sum(region_values) / len(region_values)
        between += len(region_values) * (region_mean - mean) ** 2 / total
    return max(0.0, min(1.0, between / variance))


def _is_numeric(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (str, SupportsFloat)):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _coerce_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, SupportsFloat)):
        raise ContractViolation(f"vector coupling numeric value must be int or float, got {type(value).__name__}")
    return float(value)


def _validate_matrix(
    matrix: CountMatrix,
    region_counts: tuple[int, ...],
    group_counts: tuple[int, ...],
) -> None:
    if len(matrix) != len(region_counts):
        raise ContractViolation("matrix row count does not match regions")
    for row, expected in zip(matrix, region_counts, strict=True):
        if len(row) != len(group_counts):
            raise ContractViolation("matrix column count does not match groups")
        if sum(row) != expected:
            raise ContractViolation("matrix row sums do not match region counts")
    for column_index, expected in enumerate(group_counts):
        observed = sum(row[column_index] for row in matrix)
        if observed != expected:
            raise ContractViolation("matrix column sums do not match group counts")


def _effective_local_search_budget(
    *,
    region_count: int,
    group_count: int,
    config: CouplingStrategyConfig,
) -> tuple[int, int]:
    """Local-search budget scaled by matrix size.

    High-NDV categorical couplings build large region×group matrices whose
    2×2 cycle-move candidate space is huge; the default step/swap budgets
    under-cover it and the refinement becomes unstable (observed rho drifts
    with tiny target changes).  Scale both budgets by the matrix cell count
    so the search converges to within tolerance.
    """
    matrix_cells = region_count * group_count
    swap_limit = max(config.local_search_swap_limit, matrix_cells * 2)
    max_steps = max(config.local_search_max_steps, matrix_cells // 40)
    return max_steps, swap_limit


def _refine_matrix_local_search(
    *,
    matrix: CountMatrix,
    target_rho: float,
    grouping: GroupingResult,
    metric: RhoMetric,
    region_counts: tuple[int, ...],
    group_counts: tuple[int, ...],
    max_steps: int,
    initial_delta: int,
    delta_decay: float,
    swap_limit: int,
    time_budget_sec: float = float("inf"),  # 时间预算（秒），默认不限时
    delta_mode: LocalSearchDeltaMode = LocalSearchDeltaMode.ADAPTIVE,
) -> tuple[CountMatrix, float, dict[str, object]]:
    """Griffin-style 2×2 local search refinement.

    Starting from an integerized matrix produced by interpolation search, this
    function performs fine-grained 2×2 swaps to reduce |ρ(M) - target_ρ|.

    Each step picks two region indices (r1, r2) and two group indices (g1, g2)
    and attempts to move `delta` assignment units among the four cells
    (r1,g1), (r1,g2), (r2,g1), (r2,g2) in the direction that reduces the
    dependency error.  The swap preserves all row and column marginals.

    Delta policy:
    - adaptive (default): multi-scale delta from `initial_delta`, decaying by
      `delta_decay` on stagnation, down to delta=1.
    - fixed: delta is always 1 (the paper semantics); stagnation ends the
      refinement immediately.
    """
    n_regions = len(region_counts)
    n_groups = len(group_counts)

    # 跳过退化情况
    if n_regions < 2 or n_groups < 2:
        return matrix, metric.from_matrix(matrix, grouping), {"local_search_steps": 0, "local_search_skipped": True}

    total = sum(sum(row) for row in matrix)
    if total <= 0:
        return matrix, 0.0, {"local_search_steps": 0, "local_search_skipped": True}

    best: list[list[int]] = [list(row) for row in matrix]
    region_counts_list = [sum(row) for row in best]
    numeric_stats = grouping.numeric_group_stats if not grouping.categorical else None
    numeric_group_region_sums: list[list[float]] | None = None

    if numeric_stats is not None:
        global_weighted_sum = sum(stats.prefix_sum[-1] for stats in numeric_stats)
        global_weighted_sq_sum = sum(stats.prefix_sq_sum[-1] for stats in numeric_stats)
        region_sums = [0.0] * n_regions
        numeric_group_region_sums = []
        for group_index, stats in enumerate(numeric_stats):
            cursor = 0
            group_region_sums: list[float] = []
            for region_index in range(n_regions):
                end = cursor + best[region_index][group_index]
                group_region_sum = stats.prefix_sum[end] - stats.prefix_sum[cursor]
                group_region_sums.append(group_region_sum)
                region_sums[region_index] += group_region_sum
                cursor = end
            numeric_group_region_sums.append(group_region_sums)

        global_mean = global_weighted_sum / total
        total_variance = global_weighted_sq_sum / total - global_mean * global_mean
        if total_variance <= 0.0:
            return matrix, 0.0, {"local_search_steps": 0, "local_search_skipped": True}

        def _compute_rho_from_sums(candidate_region_sums: Sequence[float]) -> float:
            between = 0.0
            for r in range(n_regions):
                cnt = region_counts_list[r]
                if cnt <= 0:
                    continue
                rmean = candidate_region_sums[r] / cnt
                between += cnt * (rmean - global_mean) ** 2 / total
            return max(0.0, min(1.0, between / total_variance))
    else:
        # categorical：rho = NMI = I(A;Z)/H(A)。行列和固定 → H(A) 常数，
        # 2×2 cycle move 只改 4 个 cell 的联合计数，I 可增量更新。
        # H(A) 必须与 _categorical_matrix_nmi 同源：用 matrix 的实际列和，
        # 而不是传入的 group_counts（整数化后二者可能不一致）。
        group_totals = [sum(row[g] for row in matrix) for g in range(n_groups)]
        nmi_entropy = 0.0
        for group_total in group_totals:
            if group_total <= 0:
                continue
            p_a = group_total / total
            nmi_entropy -= p_a * math.log(p_a)
        if nmi_entropy <= 0.0:
            return matrix, 0.0, {"local_search_steps": 0, "local_search_skipped": True}

        def _nmi_cell_contrib(count: int, region_index: int, group_index: int) -> float:
            if count <= 0:
                return 0.0
            p_joint = count / total
            p_a = group_totals[group_index] / total
            p_z = region_counts_list[region_index] / total
            return p_joint * math.log(p_joint / (p_a * p_z))

        current_info = 0.0
        for region_index, row in enumerate(best):
            for group_index, count in enumerate(row):
                current_info += _nmi_cell_contrib(count, region_index, group_index)

    def _numeric_candidate_group_sums(
        group_index: int,
        first_region: int,
        first_delta: int,
        second_region: int,
        second_delta: int,
    ) -> list[float]:
        assert numeric_stats is not None
        stats = numeric_stats[group_index]
        cursor = 0
        candidate_sums: list[float] = []
        for region_index in range(n_regions):
            count = best[region_index][group_index]
            if region_index == first_region:
                count += first_delta
            elif region_index == second_region:
                count += second_delta
            end = cursor + count
            candidate_sums.append(stats.prefix_sum[end] - stats.prefix_sum[cursor])
            cursor = end
        return candidate_sums

    if numeric_stats is not None:
        best_rho = _compute_rho_from_sums(region_sums)
    else:
        best_rho = max(0.0, min(1.0, current_info / nmi_entropy))
    best_error = abs(best_rho - target_rho)

    if delta_mode is LocalSearchDeltaMode.FIXED:
        delta = 1
    else:
        delta = max(1, initial_delta)
    total_steps = 0
    rng = random.Random(42)
    rho_trajectory: list[float] = [best_rho]
    import time as _time
    t0_ls = _time.perf_counter()
    rho_trajectory_timed: list[tuple[float, float]] = [(0.0, best_rho)]
    sample_interval_sec = 10.0
    next_sample_time = sample_interval_sec

    for step in range(max_steps):
        if rho_target_satisfied(target_rho=target_rho, observed_rho=best_rho):
            break
        now = _time.perf_counter() - t0_ls
        if now >= time_budget_sec:
            if _COUPLING_DEBUG:
                _logger.info(
                    "[debug]   ls budget exhausted: elapsed=%.3f  budget=%.3f  delta=%d  best_error=%.6f",
                    now, time_budget_sec, delta, best_error,
                )
            break
        if now >= next_sample_time:
            rho_trajectory_timed.append((now, best_rho))
            while next_sample_time <= now:
                next_sample_time += sample_interval_sec
        improved = False

        # 随机采样 swap_limit 个 (r1,r2,g1,g2) 四元组
        for _ in range(swap_limit):
            r1, r2 = rng.randrange(n_regions), rng.randrange(n_regions)
            if r1 == r2:
                continue
            g1, g2 = rng.randrange(n_groups), rng.randrange(n_groups)
            if g1 == g2:
                continue

            for sign in (1, -1):
                d = sign * delta
                # (r1,g1)+=d, (r1,g2)-=d, (r2,g1)-=d, (r2,g2)+=d
                if (best[r1][g1] + d < 0 or best[r1][g2] - d < 0 or
                        best[r2][g1] - d < 0 or best[r2][g2] + d < 0):
                    continue

                if numeric_group_region_sums is not None:
                    new_g1_sums = _numeric_candidate_group_sums(g1, r1, d, r2, -d)
                    new_g2_sums = _numeric_candidate_group_sums(g2, r1, -d, r2, d)
                    candidate_region_sums = [
                        region_sums[region_index]
                        - numeric_group_region_sums[g1][region_index]
                        - numeric_group_region_sums[g2][region_index]
                        + new_g1_sums[region_index]
                        + new_g2_sums[region_index]
                        for region_index in range(n_regions)
                    ]
                    new_rho = _compute_rho_from_sums(candidate_region_sums)
                else:
                    # categorical NMI：候选只改 4 个 cell，I 的增量为新旧贡献之差
                    moved_info_delta = (
                        _nmi_cell_contrib(best[r1][g1] + d, r1, g1)
                        + _nmi_cell_contrib(best[r1][g2] - d, r1, g2)
                        + _nmi_cell_contrib(best[r2][g1] - d, r2, g1)
                        + _nmi_cell_contrib(best[r2][g2] + d, r2, g2)
                        - _nmi_cell_contrib(best[r1][g1], r1, g1)
                        - _nmi_cell_contrib(best[r1][g2], r1, g2)
                        - _nmi_cell_contrib(best[r2][g1], r2, g1)
                        - _nmi_cell_contrib(best[r2][g2], r2, g2)
                    )
                    new_rho = max(0.0, min(1.0, (current_info + moved_info_delta) / nmi_entropy))
                new_error = abs(new_rho - target_rho)

                if new_error < best_error:
                    best[r1][g1] += d
                    best[r1][g2] -= d
                    best[r2][g1] -= d
                    best[r2][g2] += d
                    best_rho = new_rho
                    best_error = new_error
                    if numeric_group_region_sums is not None:
                        region_sums[:] = candidate_region_sums
                        numeric_group_region_sums[g1] = new_g1_sums
                        numeric_group_region_sums[g2] = new_g2_sums
                    else:
                        current_info += moved_info_delta
                    total_steps += 1
                    improved = True
                    rho_trajectory.append(best_rho)
                    rho_trajectory_timed.append((_time.perf_counter() - t0_ls, best_rho))
                    break

            if improved:
                break

        if not improved:
            if delta_mode is LocalSearchDeltaMode.FIXED:
                # paper semantics: δ=1 is the smallest nonzero move; stagnation ends refinement
                break
            new_delta = max(1, int(delta * delta_decay))
            if _COUPLING_DEBUG:
                _logger.info(
                    "[debug]   ls delta=%d → %d  step=%d  best_error=%.6f  elapsed=%.3f  budget=%.3f",
                    delta, new_delta, total_steps, best_error,
                    _time.perf_counter() - t0_ls, time_budget_sec,
                )
            if new_delta == delta:
                break
            delta = new_delta

    # 最终采样点
    final_elapsed = _time.perf_counter() - t0_ls
    if not rho_trajectory_timed or rho_trajectory_timed[-1][0] < final_elapsed - 0.5:
        rho_trajectory_timed.append((final_elapsed, best_rho))

    result_matrix = tuple(tuple(row) for row in best)
    exact_rho = metric.from_matrix(result_matrix, grouping)
    if not math.isclose(exact_rho, best_rho, rel_tol=0.0, abs_tol=1e-6):
        raise ContractViolation(
            f"local search rho state diverged from exact metric: incremental={best_rho}, exact={exact_rho}"
        )
    best_rho = exact_rho
    best_error = abs(best_rho - target_rho)
    return result_matrix, best_rho, {
        "local_search_steps": total_steps,
        "local_search_final_error": best_error,
        "local_search_final_rho": best_rho,
        "local_search_skipped": False,
        "local_search_rho_trajectory": rho_trajectory,
        "local_search_rho_trajectory_timed": rho_trajectory_timed,
    }


__all__ = [
    "AttributeGroupingStrategy",
    "AttributeVectorCouplingEngine",
    "CouplingMethodResult",
    "CouplingPlanner",
    "CouplingStrategyConfig",
    "CouplingTask",
    "DeterministicAttributeGroupingStrategy",
    "ExplainedVarianceRhoMetric",
    "GroupingResult",
    "LocalSearchDeltaMode",
    "MatrixTransportCouplingPlanner",
    "PlacementMaterializer",
    "RhoMetric",
    "StableSegmentPlacementMaterializer",
    "VectorCouplingPlan",
    "plan_independent_attribute_placement",
    "plan_vector_attribute_coupling",
    "rho_target_satisfied",
]
