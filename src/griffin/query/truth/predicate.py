"""Exact metadata substrate accessors shared across workload families."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import threading
from statistics import NormalDist
from types import MappingProxyType
from griffin.dataset.deterministic import (
    CategoricalBlocksModel,
    DeterministicFieldModel,
    IntAffineBlocksModel,
    LognormalNumericModel,
    RegionLayoutModel,
    TableDeterministicMetadata,
    UniformNumericModel,
)
from griffin.dataset.exact_cover import (
    OrdinalInterval,
    OrdinalIntervalSet,
    field_model_for_name,
    intersect_intervals,
    project_local_cover,
    region_counts,
    union_intervals,
)
from griffin.dataset.reader import Dataset
from griffin.infra.errors import ContractViolation
from griffin.model_ids import FieldId
from griffin.query.model import (
    BucketSetValue,
    BucketValue,
    NumericComparisonValue,
    NumericRangeValue,
    ScalarValue,
    TextValue,
)

OwnerDeterministicMetadata = TableDeterministicMetadata

_STANDARD_NORMAL = NormalDist()
_ORDINAL_TOLERANCE = 1.0e-12


@dataclass(frozen=True, slots=True)
class _CategoricalLocalCoverIndex:
    covers_by_label: Mapping[str, OrdinalIntervalSet]


class MetadataPredicateAccess:
    """Translate a field predicate into public ordinal cover via deterministic metadata."""

    def __init__(self) -> None:
        self._categorical_indexes: dict[
            int,
            tuple[CategoricalBlocksModel, _CategoricalLocalCoverIndex],
        ] = {}
        self._lock = threading.RLock()

    def public_cover(
        self,
        owner: OwnerDeterministicMetadata,
        field_name: str,
        predicate: object,
    ) -> OrdinalIntervalSet:
        model = field_model_for_name(owner, field_name)
        local_cover = self.local_cover(model, predicate)
        return project_local_cover(owner, field_name, local_cover)

    def local_cover(
        self,
        model: DeterministicFieldModel,
        predicate: object,
    ) -> OrdinalIntervalSet:
        if isinstance(model, CategoricalBlocksModel):
            if (
                model.value_type == "int"
                and isinstance(predicate, NumericRangeValue | NumericComparisonValue)
            ):
                return _categorical_int_range_intervals(model, predicate)
            return _categorical_intervals_from_index(
                self._categorical_index(model),
                predicate,
            )
        return local_cover_for_predicate(model, predicate)

    def _categorical_index(
        self,
        model: CategoricalBlocksModel,
    ) -> _CategoricalLocalCoverIndex:
        key = id(model)
        with self._lock:
            cached = self._categorical_indexes.get(key)
            if cached is not None:
                return cached[1]
            index = _build_categorical_local_cover_index(model)
            self._categorical_indexes[key] = (model, index)
            return index


class HybridPredicateAccess:
    """Exact predicate access over deterministic metadata."""

    def __init__(self) -> None:
        self._metadata_access = MetadataPredicateAccess()

    def local_cover(
        self,
        model: DeterministicFieldModel,
        predicate: object,
    ) -> OrdinalIntervalSet:
        return self._metadata_access.local_cover(model, predicate)

    def public_cover(
        self,
        owner: OwnerDeterministicMetadata,
        field_name: str,
        predicate: object,
        *,
        dataset: Dataset,
        field_id: FieldId,
    ) -> OrdinalIntervalSet:
        del dataset, field_id
        return self._metadata_access.public_cover(owner, field_name, predicate)


class IntervalBooleanAccess:
    """Boolean algebra over ordinal interval sets."""

    def combine(
        self,
        covers: tuple[OrdinalIntervalSet, ...],
        *,
        logic: str,
        domain_row_count: int,
    ) -> OrdinalIntervalSet:
        if not covers:
            if domain_row_count <= 0:
                return ()
            return ((0, domain_row_count - 1),)
        if logic == "and":
            combined = covers[0]
            for candidate in covers[1:]:
                combined = intersect_intervals(combined, candidate)
                if not combined:
                    return ()
            return combined
        if logic == "or":
            combined = covers[0]
            for candidate in covers[1:]:
                combined = union_intervals(combined, candidate)
            return combined
        raise ContractViolation(f"unsupported boolean logic for exact access: {logic}")


class LayoutRegionAccess:
    """Count interval hits per contiguous public region."""

    def region_counts(
        self,
        hit_set: OrdinalIntervalSet,
        region_layout: RegionLayoutModel,
    ) -> tuple[int, ...]:
        return region_counts(hit_set, region_layout)


def local_cover_for_predicate(
    model: DeterministicFieldModel,
    predicate: object,
) -> OrdinalIntervalSet:
    if isinstance(model, IntAffineBlocksModel):
        return _int_affine_intervals(model, predicate)
    if isinstance(model, CategoricalBlocksModel):
        return _categorical_intervals(model, predicate)
    if isinstance(model, UniformNumericModel):
        return _numeric_intervals(model, predicate)
    if isinstance(model, LognormalNumericModel):
        return _numeric_intervals(model, predicate)
    raise ContractViolation(f"unsupported deterministic model type: {type(model).__name__}")


def _categorical_intervals(
    model: CategoricalBlocksModel,
    predicate: object,
) -> OrdinalIntervalSet:
    if (
        model.value_type == "int"
        and isinstance(predicate, NumericRangeValue | NumericComparisonValue)
    ):
        return _categorical_int_range_intervals(model, predicate)
    return _categorical_intervals_from_index(
        _build_categorical_local_cover_index(model),
        predicate,
    )


def _parse_int_label(label: str) -> int | None:
    try:
        value = int(label)
    except ValueError:
        return None
    return value if str(value) == label else None


def _categorical_int_range_intervals(
    model: CategoricalBlocksModel,
    predicate: NumericRangeValue | NumericComparisonValue,
) -> OrdinalIntervalSet:
    """int 分类模型（n05~n07/n10/n11 等 int64 列）的 range cover。

    这些列的确定性模型是 CategoricalBlocksModel(value_type="int")，此前只支持
    eq/in（按 label 求 cover）；range 谓词直接 ContractViolation。range cover =
    所有 int 值落在 [low, high] 的 block 区间之并（与数据实际语义一致，
    由 duckdb 实扫可验证）。
    """
    value = _coerce_numeric_range(predicate)
    if not model.blocks or model.non_null_row_count <= 0:
        return ()
    intervals: list[OrdinalInterval] = []
    for block in model.blocks:
        v = _parse_int_label(block.label)
        if v is None:
            continue
        lo_ok = v >= value.low if value.inclusive_low else v > value.low
        hi_ok = v <= value.high if value.inclusive_high else v < value.high
        if lo_ok and hi_ok:
            intervals.append(
                (model.nulls_first + block.start, model.nulls_first + block.end)
            )
    return tuple(sorted(intervals))


def _build_categorical_local_cover_index(
    model: CategoricalBlocksModel,
) -> _CategoricalLocalCoverIndex:
    covers_by_label: dict[str, list[OrdinalInterval]] = {}
    for block in model.blocks:
        covers_by_label.setdefault(block.label, []).append(
            (
                model.nulls_first + block.start,
                model.nulls_first + block.end,
            )
        )
    return _CategoricalLocalCoverIndex(
        covers_by_label=MappingProxyType(
            {
                label: tuple(intervals)
                for label, intervals in covers_by_label.items()
            }
        ),
    )


def _categorical_intervals_from_index(
    index: _CategoricalLocalCoverIndex,
    predicate: object,
) -> OrdinalIntervalSet:
    labels = set(_categorical_labels(predicate))
    return tuple(
        sorted(
            interval
            for label in labels
            for interval in index.covers_by_label.get(label, ())
        )
    )


def _int_affine_intervals(
    model: IntAffineBlocksModel,
    predicate: object,
) -> OrdinalIntervalSet:
    if isinstance(predicate, NumericRangeValue | NumericComparisonValue):
        return _int_affine_range_intervals(model, predicate)
    ranks = sorted(
        rank
        for label in set(_categorical_labels(predicate))
        if (rank := _int_affine_index(model, label)) is not None
    )
    return tuple(
        (
            model.nulls_first + model.blocks[rank].start,
            model.nulls_first + model.blocks[rank].end,
        )
        for rank in ranks
    )


def _int_affine_range_intervals(
    model: IntAffineBlocksModel,
    predicate: NumericRangeValue | NumericComparisonValue,
) -> OrdinalIntervalSet:
    value = _coerce_numeric_range(predicate)
    if not model.blocks or model.non_null_row_count <= 0:
        return ()
    if model.step == 0:
        if not _range_contains_value(model.bias, value):
            return ()
        block = model.blocks[0]
        return ((model.nulls_first + block.start, model.nulls_first + block.end),)
    first_rank = _int_affine_first_index(model, value.low, inclusive=value.inclusive_low)
    last_rank = _int_affine_last_index(model, value.high, inclusive=value.inclusive_high)
    if first_rank is None or last_rank is None or first_rank > last_rank:
        return ()
    first = model.blocks[first_rank]
    last = model.blocks[last_rank]
    return ((model.nulls_first + first.start, model.nulls_first + last.end),)


def _int_affine_first_index(
    model: IntAffineBlocksModel,
    boundary: int | float,
    *,
    inclusive: bool,
) -> int | None:
    if isinstance(boundary, float) and math.isinf(boundary):
        return 0 if boundary < 0.0 else None
    integer = _int_affine_integer_boundary(boundary)
    delta = integer - model.bias
    rank = _ceil_div(delta, model.step) if inclusive else delta // model.step + 1
    rank = max(rank, 0)
    return rank if rank < len(model.blocks) else None


def _int_affine_last_index(
    model: IntAffineBlocksModel,
    boundary: int | float,
    *,
    inclusive: bool,
) -> int | None:
    if isinstance(boundary, float) and math.isinf(boundary):
        return len(model.blocks) - 1 if boundary > 0.0 else None
    integer = _int_affine_integer_boundary(boundary)
    delta = integer - model.bias
    rank = delta // model.step if inclusive else _ceil_div(delta, model.step) - 1
    rank = min(rank, len(model.blocks) - 1)
    return rank if rank >= 0 else None


def _int_affine_integer_boundary(value: int | float) -> int:
    if isinstance(value, bool):
        raise ContractViolation("Int affine range boundary must be an integer")
    if isinstance(value, int):
        return value
    if math.isfinite(value) and value.is_integer():
        return int(value)
    raise ContractViolation("Int affine range boundary must be an integer")


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _int_affine_index(model: IntAffineBlocksModel, label: str) -> int | None:
    try:
        value = int(label)
    except ValueError:
        return None
    if str(value) != label:
        return None
    if model.step == 0:
        return 0 if value == model.bias and len(model.blocks) == 1 else None
    delta = value - model.bias
    if delta < 0 or delta % model.step != 0:
        return None
    rank = delta // model.step
    return rank if rank < len(model.blocks) else None


def _categorical_labels(predicate: object) -> tuple[str, ...]:
    if isinstance(predicate, BucketValue):
        return (predicate.label,)
    if isinstance(predicate, BucketSetValue):
        if not predicate.labels:
            raise ContractViolation("categorical IN predicate requires at least one label")
        return predicate.labels
    if isinstance(predicate, TextValue):
        return (predicate.text,)
    if isinstance(predicate, ScalarValue):
        import json

        payload = json.loads(predicate.json_payload)
        return (str(payload),)
    raise ContractViolation(f"categorical predicate requires bucket-like value, got {type(predicate).__name__}")


def _numeric_intervals(
    model: UniformNumericModel | LognormalNumericModel,
    predicate: object,
) -> OrdinalIntervalSet:
    value = _coerce_numeric_range(predicate)
    if model.non_null_row_count <= 0:
        return ()
    if _numeric_model_is_constant(model):
        observed = _constant_numeric_value(model)
        if _range_contains_value(observed, value):
            ordinal = model.nulls_first
            return ((ordinal, ordinal + model.non_null_row_count - 1),)
        return ()
    lower = _first_matching_ordinal(model, value)
    upper = _last_matching_ordinal(model, value)
    if lower is None or upper is None or lower > upper:
        return ()
    return ((model.nulls_first + lower, model.nulls_first + upper),)


def _coerce_numeric_range(predicate: object) -> NumericRangeValue:
    if isinstance(predicate, NumericRangeValue):
        return predicate
    if isinstance(predicate, NumericComparisonValue):
        if predicate.operator == "lt":
            return NumericRangeValue(
                low=-math.inf,
                high=predicate.value,
                inclusive_low=True,
                inclusive_high=False,
            )
        if predicate.operator == "le":
            return NumericRangeValue(
                low=-math.inf,
                high=predicate.value,
                inclusive_low=True,
                inclusive_high=True,
            )
        if predicate.operator == "gt":
            return NumericRangeValue(
                low=predicate.value,
                high=math.inf,
                inclusive_low=False,
                inclusive_high=True,
            )
        if predicate.operator == "ge":
            return NumericRangeValue(
                low=predicate.value,
                high=math.inf,
                inclusive_low=True,
                inclusive_high=True,
            )
        raise ContractViolation(f"unsupported numeric comparison operator: {predicate.operator}")
    raise ContractViolation(f"numeric predicate requires range value, got {type(predicate).__name__}")


def _first_matching_ordinal(
    model: UniformNumericModel | LognormalNumericModel,
    value: NumericRangeValue,
) -> int | None:
    if isinstance(model, LognormalNumericModel) and model.value_type == "int":
        return _integer_first_matching_ordinal(model, value)
    if isinstance(model, UniformNumericModel):
        return _uniform_lower_ordinal(model, value)
    return _lognormal_lower_ordinal(model, value)


def _last_matching_ordinal(
    model: UniformNumericModel | LognormalNumericModel,
    value: NumericRangeValue,
) -> int | None:
    if isinstance(model, LognormalNumericModel) and model.value_type == "int":
        return _integer_last_matching_ordinal(model, value)
    if isinstance(model, UniformNumericModel):
        return _uniform_upper_ordinal(model, value)
    return _lognormal_upper_ordinal(model, value)


def _uniform_lower_ordinal(
    model: UniformNumericModel,
    value: NumericRangeValue,
) -> int | None:
    return _uniform_raw_lower_ordinal(model, value.low, inclusive=value.inclusive_low)


def _uniform_upper_ordinal(
    model: UniformNumericModel,
    value: NumericRangeValue,
) -> int | None:
    return _uniform_raw_upper_ordinal(model, value.high, inclusive=value.inclusive_high)


def _uniform_value_at_ordinal(model: UniformNumericModel, ordinal: int) -> float:
    """与数据生成完全相同的正向公式，用于校正反变换的浮点误差。"""
    if model.non_null_row_count <= 1:
        return model.linear_b
    return model.linear_k * ordinal + model.linear_b


def _correct_uniform_lower(
    model: UniformNumericModel, ordinal: int, boundary: float, *, inclusive: bool
) -> int | None:
    """用正向公式把下界挪到真正最小的匹配 ordinal（最多挪几格）。"""
    count = model.non_null_row_count

    def matches(index: int) -> bool:
        value = _uniform_value_at_ordinal(model, index)
        return value >= boundary if inclusive else value > boundary

    for _ in range(8):
        if ordinal > 0 and matches(ordinal - 1):
            ordinal -= 1
        elif ordinal < count and not matches(ordinal):
            ordinal += 1
        else:
            break
    return None if ordinal >= count else max(0, ordinal)


def _correct_uniform_upper(
    model: UniformNumericModel, ordinal: int, boundary: float, *, inclusive: bool
) -> int | None:
    """用正向公式把上界挪到真正最大的匹配 ordinal。"""
    count = model.non_null_row_count

    def matches(index: int) -> bool:
        value = _uniform_value_at_ordinal(model, index)
        return value <= boundary if inclusive else value < boundary

    for _ in range(8):
        if ordinal < count - 1 and matches(ordinal + 1):
            ordinal += 1
        elif ordinal >= 0 and not matches(ordinal):
            ordinal -= 1
        else:
            break
    return None if ordinal < 0 else min(count - 1, ordinal)


def _uniform_raw_lower_ordinal(
    model: UniformNumericModel,
    boundary: float,
    *,
    inclusive: bool,
) -> int | None:
    if math.isinf(boundary):
        return 0 if boundary < 0 else None
    if math.isclose(model.linear_k, 0.0):
        return None
    ratio = (boundary - model.linear_b) / model.linear_k
    ordinal = _ordinal_lower_bound(ratio, inclusive=inclusive)
    clamped = _clamp_lower_ordinal(ordinal, model.non_null_row_count)
    if clamped is None:
        return None
    return _correct_uniform_lower(model, clamped, boundary, inclusive=inclusive)


def _uniform_raw_upper_ordinal(
    model: UniformNumericModel,
    boundary: float,
    *,
    inclusive: bool,
) -> int | None:
    if math.isinf(boundary):
        return model.non_null_row_count - 1 if boundary > 0 else None
    if math.isclose(model.linear_k, 0.0):
        return None
    ratio = (boundary - model.linear_b) / model.linear_k
    ordinal = _ordinal_upper_bound(ratio, inclusive=inclusive)
    clamped = _clamp_upper_ordinal(ordinal, model.non_null_row_count)
    if clamped is None:
        return None
    return _correct_uniform_upper(model, clamped, boundary, inclusive=inclusive)


def _lognormal_lower_ordinal(
    model: LognormalNumericModel,
    value: NumericRangeValue,
) -> int | None:
    return _lognormal_raw_lower_ordinal(model, value.low, inclusive=value.inclusive_low)


def _lognormal_upper_ordinal(
    model: LognormalNumericModel,
    value: NumericRangeValue,
) -> int | None:
    return _lognormal_raw_upper_ordinal(model, value.high, inclusive=value.inclusive_high)


def _lognormal_raw_lower_ordinal(
    model: LognormalNumericModel,
    boundary: float,
    *,
    inclusive: bool,
) -> int | None:
    if math.isinf(boundary):
        return 0 if boundary < 0 else None
    base_boundary = _inverse_linear_value(model.linear_k, model.linear_b, boundary)
    if base_boundary is None:
        return None
    q = _lognormal_cdf(model, base_boundary)
    ratio = q * model.non_null_row_count - 0.5
    ordinal = _ordinal_lower_bound(ratio, inclusive=inclusive)
    return _clamp_lower_ordinal(ordinal, model.non_null_row_count)


def _lognormal_raw_upper_ordinal(
    model: LognormalNumericModel,
    boundary: float,
    *,
    inclusive: bool,
) -> int | None:
    if math.isinf(boundary):
        return model.non_null_row_count - 1 if boundary > 0 else None
    base_boundary = _inverse_linear_value(model.linear_k, model.linear_b, boundary)
    if base_boundary is None:
        return None
    q = _lognormal_cdf(model, base_boundary)
    ratio = q * model.non_null_row_count - 0.5
    ordinal = _ordinal_upper_bound(ratio, inclusive=inclusive)
    return _clamp_upper_ordinal(ordinal, model.non_null_row_count)


def _lognormal_cdf(model: LognormalNumericModel, value: float) -> float:
    if value <= 0.0:
        return 0.0
    z_score = (math.log(value) - model.mean) / model.sigma
    return min(max(_STANDARD_NORMAL.cdf(z_score), 0.0), 1.0)


def _apply_linear_value(k: float, b: float, base: float) -> float:
    return k * base + b


def _inverse_linear_value(k: float, b: float, observed: float) -> float | None:
    if math.isclose(k, 0.0):
        return None
    return (observed - b) / k


def _ordinal_tolerance(ratio: float) -> float:
    """边界容差随 ratio 量级放大。

    ratio 可达成 10^6（row_count），浮点反变换的绝对误差约 |ratio| * 1e-16；
    固定 1e-12 会在 ratio 接近整数时把 floor/ceil 推错一格（曾导致 ``<=`` 边界
    少算 1 行）。改用相对容差，最大 1e-6，仍远小于 1，不会跨过相邻 ordinal。
    """
    return max(_ORDINAL_TOLERANCE, abs(ratio) * 1.0e-12)


def _ordinal_lower_bound(ratio: float, *, inclusive: bool) -> int:
    tolerance = _ordinal_tolerance(ratio)
    if inclusive:
        return math.ceil(ratio - tolerance)
    return math.floor(ratio + tolerance) + 1


def _ordinal_upper_bound(ratio: float, *, inclusive: bool) -> int:
    tolerance = _ordinal_tolerance(ratio)
    if inclusive:
        return math.floor(ratio + tolerance)
    return math.ceil(ratio - tolerance) - 1


def _clamp_lower_ordinal(ordinal: int, non_null_row_count: int) -> int | None:
    if ordinal >= non_null_row_count:
        return None
    return max(0, ordinal)


def _clamp_upper_ordinal(ordinal: int, non_null_row_count: int) -> int | None:
    if ordinal < 0:
        return None
    return min(non_null_row_count - 1, ordinal)


def _numeric_model_is_constant(model: UniformNumericModel | LognormalNumericModel) -> bool:
    if isinstance(model, UniformNumericModel):
        return (
            model.non_null_row_count <= 1
            or math.isclose(model.minimum, model.maximum)
            or math.isclose(model.linear_k, 0.0)
        )
    return model.non_null_row_count <= 1 or model.sigma <= 0.0 or math.isclose(model.linear_k, 0.0)


def _constant_numeric_value(model: UniformNumericModel | LognormalNumericModel) -> float:
    if isinstance(model, UniformNumericModel):
        base = float(model.minimum)
        raw = _apply_linear_value(model.linear_k, model.linear_b, base)
        return raw
    base = math.exp(model.mean)
    raw = _apply_linear_value(model.linear_k, model.linear_b, base)
    if model.value_type == "int":
        return float(max(0, int(round(raw))))
    return raw


def _range_contains_value(observed: float, value: NumericRangeValue) -> bool:
    lower_ok = observed >= value.low if value.inclusive_low else observed > value.low
    upper_ok = observed <= value.high if value.inclusive_high else observed < value.high
    return lower_ok and upper_ok


def _integer_first_matching_ordinal(
    model: LognormalNumericModel,
    value: NumericRangeValue,
) -> int | None:
    low = 0
    high = model.non_null_row_count
    while low < high:
        mid = (low + high) // 2
        observed = _integer_value_at(model, mid)
        if _integer_below_range(observed, value):
            low = mid + 1
        else:
            high = mid
    if low >= model.non_null_row_count:
        return None
    return low if _range_contains_value(float(_integer_value_at(model, low)), value) else None


def _integer_last_matching_ordinal(
    model: LognormalNumericModel,
    value: NumericRangeValue,
) -> int | None:
    low = 0
    high = model.non_null_row_count
    while low < high:
        mid = (low + high) // 2
        observed = _integer_value_at(model, mid)
        if _integer_above_range(observed, value):
            high = mid
        else:
            low = mid + 1
    ordinal = low - 1
    if ordinal < 0:
        return None
    return ordinal if _range_contains_value(float(_integer_value_at(model, ordinal)), value) else None


def _integer_value_at(model: LognormalNumericModel, ordinal: int) -> int:
    if model.non_null_row_count <= 1:
        quantile = 0.5
    else:
        quantile = (ordinal + 0.5) / model.non_null_row_count
    z_score = _STANDARD_NORMAL.inv_cdf(min(max(quantile, 1.0e-9), 1.0 - 1.0e-9))
    base = math.exp(model.mean + model.sigma * z_score)
    raw = _apply_linear_value(model.linear_k, model.linear_b, base)
    return max(0, int(round(raw)))


def _integer_below_range(observed: int, value: NumericRangeValue) -> bool:
    return observed < value.low or (observed == value.low and not value.inclusive_low)


def _integer_above_range(observed: int, value: NumericRangeValue) -> bool:
    return observed > value.high or (observed == value.high and not value.inclusive_high)


__all__ = [
    "HybridPredicateAccess",
    "IntervalBooleanAccess",
    "LayoutRegionAccess",
    "MetadataPredicateAccess",
    "local_cover_for_predicate",
]
