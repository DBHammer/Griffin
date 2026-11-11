"""Deterministic relational field models and materialization helpers."""

from __future__ import annotations

import hashlib
import math
import random
import string
from collections.abc import Iterable

from statistics import NormalDist
from griffin.data.generation_request import FieldGenerationRequest
from griffin.dataset.deterministic import (
    CategoryBlock,
    CategoricalBlocksModel,
    DeterministicFieldModel,
    FieldPlacementModel,
    IntAffineBlocksModel,
    LognormalNumericModel,
    RegionLayoutModel,
    TableDeterministicMetadata,
    UniformNumericModel,
    VectorAttributeCouplingModel,
    identity_field_placements,
)
from griffin.infra.errors import ContractViolation
from griffin.ir.data_type import DoubleType, IntType, TextType
from griffin.ir.field import Field, FieldRole


_STANDARD_NORMAL = NormalDist()
_ZIPF_TEXT_LABEL_LENGTH = 16


def deterministic_model_for_field(
    field: Field,
    row_count: int,
    *,
    request: FieldGenerationRequest | None,
    seed: int = 0,
) -> DeterministicFieldModel | None:
    if field.role in {FieldRole.PRIMARY_KEY, FieldRole.VECTOR}:
        return None
    distribution = request.distribution if request is not None else None
    if distribution is None:
        return None
    nulls_first = 0
    non_null_row_count = row_count
    kind = distribution.kind

    if kind == "uniform" and isinstance(field.type, IntType):
        if distribution.ndv is None or distribution.value_domain is None:
            raise ContractViolation(
                f"int uniform field {field.id} requires ndv and value_domain"
            )
        domain = distribution.value_domain
        counts = uniform_counts(row_count=non_null_row_count, ndv=distribution.ndv)
        step, bias, labels = _int_affine_support(
            ndv=distribution.ndv,
            minimum=domain.min,
            maximum=domain.max,
            seed=seed,
            salt=f"{field.id.owner.name}.{field.id.name}:uniform:int",
        )
        return IntAffineBlocksModel(
            field_name=field.id.name,
            distribution_kind="uniform",
            row_count=row_count,
            non_null_row_count=non_null_row_count,
            nulls_first=nulls_first,
            domain_minimum=domain.min,
            domain_maximum=domain.max,
            step=step,
            bias=bias,
            blocks=_category_blocks_from_counts(labels, counts),
        )
    if kind == "uniform" and isinstance(field.type, DoubleType):
        minimum = distribution.min if distribution.min is not None else 0.0
        maximum = distribution.max if distribution.max is not None else 1.0
        linear_k, linear_b = _choose_numeric_linear_transform(
            row_count=non_null_row_count,
            target_min=minimum,
            target_max=maximum,
            seed=seed,
            salt=f"{field.id.owner.name}.{field.id.name}:uniform",
        )
        return UniformNumericModel(
            field_name=field.id.name,
            value_type="double",
            row_count=row_count,
            non_null_row_count=non_null_row_count,
            nulls_first=nulls_first,
            minimum=minimum,
            maximum=maximum,
            linear_k=linear_k,
            linear_b=linear_b,
        )
    if kind == "lognormal" and isinstance(field.type, (IntType, DoubleType)):
        mean = distribution.mean if distribution.mean is not None else 0.0
        sigma = distribution.sigma if distribution.sigma is not None else 1.0
        return LognormalNumericModel(
            field_name=field.id.name,
            value_type="int" if isinstance(field.type, IntType) else "double",
            row_count=row_count,
            non_null_row_count=non_null_row_count,
            nulls_first=nulls_first,
            mean=mean,
            sigma=sigma,
        )
    if kind == "categorical" and isinstance(field.type, (IntType, TextType)):
        values = distribution.values
        if not values:
            raise ContractViolation(f"categorical field {field.id} requires non-empty values")
        blocks = _categorical_blocks(values, distribution.weights, non_null_row_count)
        return CategoricalBlocksModel(
            field_name=field.id.name,
            value_type="int" if isinstance(field.type, IntType) else "text",
            row_count=row_count,
            non_null_row_count=non_null_row_count,
            nulls_first=nulls_first,
            blocks=blocks,
        )
    if kind == "zipf" and isinstance(field.type, (IntType, TextType)):
        if distribution.ndv is None or distribution.s is None:
            raise ContractViolation(f"zipf field {field.id} requires ndv and s")
        counts = zipf_counts(row_count=non_null_row_count, ndv=distribution.ndv, slope=distribution.s)
        salt = f"{field.id.owner.name}.{field.id.name}"
        if isinstance(field.type, IntType):
            zipf_domain = distribution.value_domain
            if zipf_domain is None:
                raise ContractViolation(f"int zipf field {field.id} requires value_domain")
            step, bias, labels = _int_affine_support(
                ndv=distribution.ndv,
                minimum=zipf_domain.min,
                maximum=zipf_domain.max,
                seed=seed,
                salt=f"{salt}:zipf:int",
            )
            return IntAffineBlocksModel(
                field_name=field.id.name,
                distribution_kind="zipf",
                row_count=row_count,
                non_null_row_count=non_null_row_count,
                nulls_first=nulls_first,
                domain_minimum=zipf_domain.min,
                domain_maximum=zipf_domain.max,
                step=step,
                bias=bias,
                blocks=_category_blocks_from_counts(labels, counts),
            )
        labels = _random_text_labels(ndv=distribution.ndv, seed=seed, salt=salt)
        return CategoricalBlocksModel(
            field_name=field.id.name,
            value_type="text",
            row_count=row_count,
            non_null_row_count=non_null_row_count,
            nulls_first=nulls_first,
            blocks=_category_blocks_from_counts(labels, counts),
        )
    return None


def materialize_supported_field(
    field: Field,
    row_count: int,
    *,
    model: DeterministicFieldModel,
) -> list[object]:
    nulls_first = getattr(model, "nulls_first", 0)
    values: list[object] = [None] * nulls_first
    non_null_row_count = getattr(model, "non_null_row_count", row_count - nulls_first)
    if non_null_row_count <= 0:
        return values + [None] * max(0, row_count - len(values))

    if isinstance(model, UniformNumericModel):
        for index in range(non_null_row_count):
            values.append(_uniform_value_at(field, model, index))
        return values
    if isinstance(model, LognormalNumericModel):
        for index in range(non_null_row_count):
            values.append(_lognormal_value_at(field, model, index))
        return values
    if isinstance(model, IntAffineBlocksModel):
        for block in model.blocks:
            values.extend([int(block.label)] * (block.end - block.start + 1))
        return values
    if isinstance(model, CategoricalBlocksModel):
        for block in model.blocks:
            materialized = _coerce_categorical_label(field, block.label)
            values.extend([materialized] * (block.end - block.start + 1))
        return values
    raise ContractViolation(f"unsupported deterministic model type: {type(model).__name__}")


def deterministic_table_metadata(
    table_name: str,
    row_count: int,
    models: Iterable[DeterministicFieldModel],
    *,
    placements: Iterable[FieldPlacementModel] | None = None,
    region_layout: RegionLayoutModel | None = None,
    vector_attribute_couplings: Iterable[VectorAttributeCouplingModel] | None = None,
) -> TableDeterministicMetadata:
    supported_fields = tuple(models)
    return TableDeterministicMetadata(
        table_name=table_name,
        row_ordinal_start=0,
        row_count=row_count,
        supported_fields=supported_fields,
        field_placements=tuple(placements) if placements is not None else identity_field_placements(supported_fields),
        region_layout=region_layout,
        vector_attribute_couplings=tuple(vector_attribute_couplings or ()),
    )


def _categorical_blocks(
    labels: tuple[str, ...],
    weights: tuple[float, ...],
    row_count: int,
) -> tuple[CategoryBlock, ...]:
    counts = categorical_counts(labels=labels, weights=weights, row_count=row_count)
    return _category_blocks_from_counts(labels, counts)


def _category_blocks_from_counts(
    labels: tuple[str, ...],
    counts: tuple[int, ...],
) -> tuple[CategoryBlock, ...]:
    if len(labels) != len(counts):
        raise ContractViolation("category labels and counts must have the same length")
    blocks: list[CategoryBlock] = []
    cursor = 0
    for label, count in zip(labels, counts, strict=True):
        if count <= 0:
            continue
        blocks.append(CategoryBlock(label=label, start=cursor, end=cursor + count - 1))
        cursor += count
    return tuple(blocks)


def _random_text_labels(*, ndv: int, seed: int, salt: str) -> tuple[str, ...]:
    digest = hashlib.sha256(f"{seed}:{salt}:{ndv}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big", signed=False))
    alphabet = string.ascii_lowercase + string.digits
    labels: list[str] = []
    seen: set[str] = set()
    while len(labels) < ndv:
        first = rng.choice(string.ascii_lowercase)
        rest = "".join(rng.choice(alphabet) for _ in range(_ZIPF_TEXT_LABEL_LENGTH - 1))
        label = first + rest
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return tuple(labels)


def _int_affine_support(
    *,
    ndv: int,
    minimum: int,
    maximum: int,
    seed: int,
    salt: str,
) -> tuple[int, int, tuple[str, ...]]:
    if ndv < 1:
        raise ContractViolation("int affine ndv must be >= 1")
    if minimum > maximum:
        raise ContractViolation("int affine value_domain min must be <= max")
    rng = _stable_rng(seed=seed, salt=salt)
    if ndv == 1:
        bias = rng.randint(minimum, maximum)
        return 0, bias, (str(bias),)
    max_step = (maximum - minimum) // (ndv - 1)
    if max_step < 1:
        raise ContractViolation(
            "int affine value_domain must contain at least ndv distinct integers"
        )
    step = rng.randint(1, max_step)
    max_bias = maximum - step * (ndv - 1)
    bias = rng.randint(minimum, max_bias)
    return step, bias, tuple(str(bias + step * index) for index in range(ndv))


def zipf_counts(*, row_count: int, ndv: int, slope: float) -> tuple[int, ...]:
    if ndv < 1:
        raise ContractViolation("zipf ndv must be >= 1")
    if row_count < ndv:
        raise ContractViolation(f"zipf ndv must be <= row_count; got ndv={ndv}, row_count={row_count}")
    if slope <= 0.0:
        raise ContractViolation("zipf s must be > 0")
    remaining = row_count - ndv
    weights = tuple(1.0 / ((index + 1) ** slope) for index in range(ndv))
    extras = categorical_counts(
        labels=tuple(str(index) for index in range(ndv)),
        weights=weights,
        row_count=remaining,
    )
    return tuple(1 + count for count in extras)


def uniform_counts(*, row_count: int, ndv: int) -> tuple[int, ...]:
    if ndv < 1:
        raise ContractViolation("uniform ndv must be >= 1")
    if row_count < ndv:
        raise ContractViolation(
            f"uniform ndv must be <= row_count; got ndv={ndv}, row_count={row_count}"
        )
    base, remainder = divmod(row_count, ndv)
    return tuple(base + (1 if index < remainder else 0) for index in range(ndv))


def categorical_counts(
    *,
    labels: tuple[str, ...],
    weights: tuple[float, ...],
    row_count: int,
) -> tuple[int, ...]:
    raw_weights = weights or tuple(1.0 for _ in labels)
    total = sum(raw_weights)
    if total <= 0.0:
        raise ContractViolation("categorical weights must sum to a positive value")
    exact = [row_count * weight / total for weight in raw_weights]
    counts = [int(math.floor(value)) for value in exact]
    remainder = row_count - sum(counts)
    order = sorted(
        range(len(labels)),
        key=lambda index: exact[index] - counts[index],
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    return tuple(counts)


def _stable_rng(*, seed: int, salt: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{salt}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big", signed=False))


def _choose_numeric_linear_transform(
    *,
    row_count: int,
    target_min: float,
    target_max: float,
    seed: int,
    salt: str,
) -> tuple[float, float]:
    lower = min(float(target_min), float(target_max))
    upper = max(float(target_min), float(target_max))
    if row_count <= 1 or math.isclose(lower, upper):
        return 0.0, lower
    span = upper - lower
    max_k = span / float(row_count - 1)
    rng = _stable_rng(seed=seed, salt=f"{salt}:linear")
    k = max_k * (0.1 + 0.9 * rng.random())
    b_low = lower
    b_high = upper - k * float(row_count - 1)
    b = b_low if b_high <= b_low else rng.uniform(b_low, b_high)
    return k, b


def _apply_linear_transform(value: float, *, k: float, b: float) -> float:
    return k * value + b


def _lognormal_base_value(*, mean: float, sigma: float, ordinal: int, row_count: int) -> float:
    if row_count <= 1:
        quantile = 0.5
    else:
        quantile = (ordinal + 0.5) / row_count
    z_score = _STANDARD_NORMAL.inv_cdf(min(max(quantile, 1.0e-9), 1.0 - 1.0e-9))
    return math.exp(mean + sigma * z_score)


def _uniform_value_at(
    field: Field,
    model: UniformNumericModel,
    ordinal: int,
) -> object:
    if model.non_null_row_count <= 1:
        transformed = model.linear_b
    else:
        transformed = _apply_linear_transform(
            float(ordinal), k=model.linear_k, b=model.linear_b
        )
    if isinstance(field.type, IntType):
        return int(round(transformed))
    if isinstance(field.type, DoubleType):
        return float(transformed)
    raise ContractViolation(f"uniform model requires numeric field, got {type(field.type).__name__}")


def _lognormal_value_at(
    field: Field,
    model: LognormalNumericModel,
    ordinal: int,
) -> object:
    base = _lognormal_base_value(
        mean=model.mean,
        sigma=model.sigma,
        ordinal=ordinal,
        row_count=max(model.non_null_row_count, 1),
    )
    if isinstance(field.type, IntType):
        return max(0, int(round(base)))
    if isinstance(field.type, DoubleType):
        return float(base)
    raise ContractViolation(f"lognormal model requires numeric field, got {type(field.type).__name__}")


def _coerce_categorical_label(field: Field, label: str) -> object:
    if isinstance(field.type, IntType):
        return int(float(label))
    if isinstance(field.type, TextType):
        return label
    raise ContractViolation(
        f"categorical deterministic model requires int/text field, got {type(field.type).__name__}"
    )


__all__ = [
    "categorical_counts",
    "deterministic_model_for_field",
    "deterministic_table_metadata",
    "materialize_supported_field",
    "uniform_counts",
    "zipf_counts",
]
