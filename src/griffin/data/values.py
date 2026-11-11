"""Typed value generation for dataset materialization."""

from __future__ import annotations

import math
import random
import logging
from collections.abc import Iterable
from statistics import NormalDist
from typing import NamedTuple, Protocol, assert_never, cast

import numpy as np
from scipy.special import erf as _vectorized_erf

from griffin.data.generation_request import (
    CorrelationBlockRequest,
    DistributionRequest,
    FieldGenerationRequest,
    VectorGenerationRequest,
)
from griffin.data.deterministic import materialize_supported_field
from griffin.dataset.deterministic import (
    DeterministicFieldModel,
    IntAffineBlocksModel,
    LognormalNumericModel,
    UniformNumericModel,
)
from griffin.infra.errors import ContractViolation
from griffin.ir.data_type import DataType, DoubleType, IntType, ListType, TextType, VectorMetric, VectorType
from griffin.ir.field import Field

_LOGGER = logging.getLogger("griffin.task")
_STANDARD_NORMAL = NormalDist()
_GAUSS_HERMITE_ORDER = 48
_GAUSS_HERMITE_NODES, _GAUSS_HERMITE_WEIGHTS = np.polynomial.hermite.hermgauss(_GAUSS_HERMITE_ORDER)
_GAUSS_HERMITE_NORMAL_NODES = np.sqrt(2.0) * _GAUSS_HERMITE_NODES
_GAUSS_HERMITE_NORMAL_WEIGHTS = _GAUSS_HERMITE_WEIGHTS / math.sqrt(math.pi)


class ValueGenerator(Protocol):
    def generate(self, n: int) -> Iterable[object] | np.ndarray: ...


class _CorrelationMarginal(NamedTuple):
    field_name: str
    kind: str
    value_type: str
    lower: float
    upper: float
    mean: float
    sigma: float


class IntGenerator:
    __slots__ = ("_low", "_high", "_rng")

    def __init__(self, *, low: int = 0, high: int = 1_000_000, seed: int = 0) -> None:
        self._low = low
        self._high = high
        self._rng = random.Random(seed)

    def generate(self, n: int) -> list[int]:
        return [self._rng.randint(self._low, self._high) for _ in range(n)]


class DoubleGenerator:
    __slots__ = ("_low", "_high", "_rng")

    def __init__(self, *, low: float = 0.0, high: float = 1.0, seed: int = 0) -> None:
        self._low = low
        self._high = high
        self._rng = random.Random(seed)

    def generate(self, n: int) -> list[float]:
        return [self._rng.uniform(self._low, self._high) for _ in range(n)]


class TextGenerator:
    __slots__ = ("_corpus", "_rng")

    def __init__(self, *, corpus: tuple[str, ...] | None = None, seed: int = 0) -> None:
        self._corpus = corpus or (
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "zeta",
            "eta",
            "theta",
            "iota",
            "kappa",
        )
        self._rng = random.Random(seed)

    def generate(self, n: int) -> list[str]:
        rng = self._rng
        return [
            " ".join(rng.choice(self._corpus) for _ in range(rng.randint(2, 6)))
            for _ in range(n)
        ]


class VectorGenerator:
    __slots__ = ("_dim", "_metric", "_rng")

    def __init__(
        self,
        *,
        dimension: int,
        metric: VectorMetric = VectorMetric.COSINE,
        seed: int = 0,
    ) -> None:
        self._dim = dimension
        self._metric = metric
        self._rng = np.random.default_rng(seed)

    def generate(self, n: int) -> np.ndarray:
        out = self._rng.normal(0.0, 1.0, size=(n, self._dim)).astype("float32")
        if self._metric == VectorMetric.COSINE:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            out = out / norms
        return out.astype("float32", copy=False)


def generator_for(dt: DataType, *, seed: int = 0) -> ValueGenerator:
    match dt:
        case IntType():
            return IntGenerator(seed=seed)
        case DoubleType():
            return DoubleGenerator(seed=seed)
        case TextType():
            return TextGenerator(seed=seed)
        case VectorType():
            return VectorGenerator(dimension=dt.dimension, metric=dt.metric, seed=seed)
        case ListType():
            inner = generator_for(dt.element, seed=seed + 1)

            class _ListGen:
                def generate(self, n: int) -> list[list[object]]:
                    rng = random.Random(seed + 2)
                    out: list[list[object]] = []
                    for _ in range(n):
                        out.append(list(inner.generate(rng.randint(1, 3))))
                    return out

            return _ListGen()
        case _:
            assert_never(dt)


def generate_field_values(
    field: Field,
    row_count: int,
    *,
    seed: int,
    request: FieldGenerationRequest | None = None,
) -> list[object]:
    values: list[object]
    distribution = request.distribution if request is not None else None
    if distribution is None:
        if isinstance(field.type, VectorType):
            raise ContractViolation(
                f"vector field {field.id} must be materialized by the vector column path"
            )
        values = list(generator_for(field.type, seed=seed).generate(row_count))
    elif distribution.kind == "uniform":
        values = _uniform_values(field, row_count, distribution, seed)
    elif distribution.kind == "lognormal":
        values = _lognormal_values(field, row_count, distribution, seed)
    elif distribution.kind == "categorical":
        values = _categorical_values(field, row_count, distribution, seed)
    else:
        raise ContractViolation(f"unsupported distribution kind: {distribution.kind}")
    return values


def generate_correlation_block(
    fields: tuple[Field, ...],
    row_count: int,
    *,
    block: CorrelationBlockRequest,
    models: tuple[DeterministicFieldModel, ...],
    seed: int,
    anchor_field: str | None = None,
) -> dict[str, list[object]]:
    if len(fields) != len(block.columns) or len(models) != len(block.columns):
        raise ContractViolation(f"correlation block {block.name} does not align with field models")
    size = len(block.columns)
    target_matrix = np.asarray(block.matrix, dtype=float)
    if target_matrix.shape != (size, size):
        raise ContractViolation(
            f"correlation block {block.name} expected a {size}x{size} matrix; got {target_matrix.shape}"
        )
    marginals = tuple(_correlation_marginal(model) for model in models)
    latent_matrix = _latent_correlation_matrix(
        block_name=block.name,
        target_matrix=target_matrix,
        marginals=marginals,
    )
    sample = _correlated_normal_sample(
        latent_matrix,
        row_count=row_count,
        seed=seed,
        anchor_index=block.columns.index(anchor_field) if anchor_field is not None else None,
    )

    generated: dict[str, list[object]] = {}
    for index, field in enumerate(fields):
        ordered_values = materialize_supported_field(
            field,
            row_count,
            model=models[index],
        )
        generated[field.id.name] = _place_ordered_values_by_sample_rank(
            sample[:, index],
            ordered_values,
        )
    return generated


def _correlated_normal_sample(
    matrix: np.ndarray,
    *,
    row_count: int,
    seed: int,
    anchor_index: int | None,
) -> np.ndarray:
    if anchor_index is None:
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(np.zeros(matrix.shape[0]), matrix, size=row_count)
    if row_count <= 0:
        return np.empty((0, matrix.shape[0]), dtype=float)
    anchor_values = np.asarray(
        [
            _STANDARD_NORMAL.inv_cdf(
                min(max((ordinal + 0.5) / row_count, 1.0e-9), 1.0 - 1.0e-9)
            )
            for ordinal in range(row_count)
        ],
        dtype=float,
    )
    size = matrix.shape[0]
    if size == 1:
        return anchor_values.reshape(row_count, 1)
    other_indexes = [index for index in range(size) if index != anchor_index]
    cov_oo = matrix[np.ix_(other_indexes, other_indexes)]
    cov_oa = matrix[np.ix_(other_indexes, [anchor_index])].reshape(len(other_indexes))
    conditional_cov = _stabilize_covariance(cov_oo - np.outer(cov_oa, cov_oa))
    rng = np.random.default_rng(seed)
    residual = rng.multivariate_normal(
        np.zeros(len(other_indexes)),
        conditional_cov,
        size=row_count,
    )
    sample = np.empty((row_count, size), dtype=float)
    sample[:, anchor_index] = anchor_values
    for local_index, field_index in enumerate(other_indexes):
        sample[:, field_index] = cov_oa[local_index] * anchor_values + residual[:, local_index]
    return sample


def _correlation_marginal(
    model: DeterministicFieldModel,
) -> _CorrelationMarginal:
    if isinstance(model, IntAffineBlocksModel):
        if model.distribution_kind != "uniform":
            raise ContractViolation(
                "correlation blocks support int affine models only for uniform distributions"
            )
        return _CorrelationMarginal(
            field_name=model.field_name,
            kind="uniform",
            value_type="int",
            lower=float(model.bias),
            upper=float(model.bias + model.step * (len(model.blocks) - 1)),
            mean=0.0,
            sigma=1.0,
        )
    if isinstance(model, UniformNumericModel):
        return _CorrelationMarginal(
            field_name=model.field_name,
            kind="uniform",
            value_type=model.value_type,
            lower=float(model.minimum),
            upper=float(model.maximum),
            mean=0.0,
            sigma=1.0,
        )
    if isinstance(model, LognormalNumericModel):
        return _CorrelationMarginal(
            field_name=model.field_name,
            kind="lognormal",
            value_type=model.value_type,
            lower=0.0,
            upper=1.0,
            mean=float(model.mean),
            sigma=float(model.sigma),
        )
    raise ContractViolation(
        f"correlation blocks do not support deterministic model {type(model).__name__}"
    )


def _latent_correlation_matrix(
    *,
    block_name: str,
    target_matrix: np.ndarray,
    marginals: tuple[_CorrelationMarginal, ...],
) -> np.ndarray:
    size = target_matrix.shape[0]
    matrix = np.eye(size, dtype=float)
    for row in range(size):
        for col in range(row + 1, size):
            latent = _latent_correlation_for_target(
                float(target_matrix[row, col]),
                marginals[row],
                marginals[col],
                block_name=block_name,
            )
            matrix[row, col] = latent
            matrix[col, row] = latent
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    if float(np.min(eigenvalues)) < -1.0e-8:
        _warn_correlation_calibration(
            f"numeric correlation block {block_name} calibrated latent matrix is not PSD; "
            "using the nearest stabilized latent correlation matrix and leaving final error "
            "to observed Pearson validation"
        )
        matrix = _stabilize_correlation(matrix)
    return matrix


def _latent_correlation_for_target(
    target: float,
    left: _CorrelationMarginal,
    right: _CorrelationMarginal,
    *,
    block_name: str,
) -> float:
    if not -1.0 <= target <= 1.0:
        raise ContractViolation(
            f"numeric correlation block {block_name} target rho must be within [-1,1]: {target:g}"
        )
    if left.kind == "uniform" and right.kind == "uniform" and left.value_type == "double" and right.value_type == "double":
        return _checked_latent_rho(2.0 * math.sin(math.pi * target / 6.0), target, left, right, block_name)
    if left.kind == "lognormal" and right.kind == "lognormal" and left.value_type == "double" and right.value_type == "double":
        scale = math.sqrt(math.expm1(left.sigma * left.sigma) * math.expm1(right.sigma * right.sigma))
        if 1.0 + target * scale <= 0.0:
            return _boundary_latent_rho(
                target,
                left,
                right,
                block_name=block_name,
                reason="is outside the closed-form lognormal attainable range",
            )
        numerator = math.log1p(target * scale)
        latent = numerator / (left.sigma * right.sigma)
        return _checked_latent_rho(latent, target, left, right, block_name)
    return _solve_latent_rho(target, left, right, block_name=block_name)


def _checked_latent_rho(
    latent: float,
    target: float,
    left: _CorrelationMarginal,
    right: _CorrelationMarginal,
    block_name: str,
) -> float:
    if not math.isfinite(latent) or latent < -1.0 - 1.0e-10 or latent > 1.0 + 1.0e-10:
        return _boundary_latent_rho(
            target,
            left,
            right,
            block_name=block_name,
            reason=f"requires latent rho {latent:g} outside [-1,1]",
        )
    return min(1.0, max(-1.0, latent))


def _solve_latent_rho(
    target: float,
    left: _CorrelationMarginal,
    right: _CorrelationMarginal,
    *,
    block_name: str,
) -> float:
    low = -1.0 + 1.0e-10
    high = 1.0 - 1.0e-10
    low_value = _final_pearson_for_latent(low, left, right)
    high_value = _final_pearson_for_latent(high, left, right)
    if target < low_value - 1.0e-8 or target > high_value + 1.0e-8:
        boundary = low if target < low_value else high
        _warn_correlation_calibration(
            f"numeric correlation block {block_name} target rho {target:g} for "
            f"{left.field_name}/{right.field_name} is outside attainable final Pearson range "
            f"[{low_value:.6g},{high_value:.6g}]; using latent rho {boundary:.6g}"
        )
        return boundary
    for _ in range(72):
        mid = (low + high) / 2.0
        value = _final_pearson_for_latent(mid, left, right)
        if value < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def _boundary_latent_rho(
    target: float,
    left: _CorrelationMarginal,
    right: _CorrelationMarginal,
    *,
    block_name: str,
    reason: str,
) -> float:
    low = -1.0 + 1.0e-10
    high = 1.0 - 1.0e-10
    low_error = abs(_final_pearson_for_latent(low, left, right) - target)
    high_error = abs(_final_pearson_for_latent(high, left, right) - target)
    latent = low if low_error <= high_error else high
    _warn_correlation_calibration(
        f"numeric correlation block {block_name} target rho {target:g} for "
        f"{left.field_name}/{right.field_name} {reason}; using latent rho {latent:.6g}"
    )
    return latent


def _warn_correlation_calibration(message: str) -> None:
    _LOGGER.warning("[data] %s", message)


def _final_pearson_for_latent(
    latent: float,
    left: _CorrelationMarginal,
    right: _CorrelationMarginal,
) -> float:
    x = _marginal_values_for_nodes(left, _GAUSS_HERMITE_NORMAL_NODES)
    conditional = latent * _GAUSS_HERMITE_NORMAL_NODES[:, None] + math.sqrt(max(0.0, 1.0 - latent * latent)) * _GAUSS_HERMITE_NORMAL_NODES[None, :]
    y = _marginal_values_for_nodes(right, conditional)
    weights = _GAUSS_HERMITE_NORMAL_WEIGHTS
    x_mean = float(np.sum(weights * x))
    x_var = float(np.sum(weights * (x - x_mean) ** 2))
    y_mean = float(np.sum(weights[:, None] * weights[None, :] * y))
    y_var = float(np.sum(weights[:, None] * weights[None, :] * (y - y_mean) ** 2))
    if x_var <= 0.0 or y_var <= 0.0:
        return 0.0
    cov = float(np.sum(weights[:, None] * weights[None, :] * (x[:, None] - x_mean) * (y - y_mean)))
    return cov / math.sqrt(x_var * y_var)


def _marginal_values_for_nodes(
    marginal: _CorrelationMarginal,
    nodes: np.ndarray,
) -> np.ndarray:
    if marginal.kind == "uniform":
        values = marginal.lower + (marginal.upper - marginal.lower) * _normal_to_uniform(nodes)
    elif marginal.kind == "lognormal":
        values = np.exp(marginal.mean + marginal.sigma * nodes)
    else:
        raise ContractViolation(f"unsupported correlation marginal: {marginal.kind}")
    if marginal.value_type == "int":
        return np.round(values)
    return values


def generate_vector_values(
    row_count: int,
    *,
    request: VectorGenerationRequest,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cluster_count = max(1, request.cluster_count or 1)
    separation = request.cluster_separation or 1.0
    spread = request.cluster_spread or 1.0
    centers = rng.normal(0.0, separation, size=(cluster_count, request.dimension)).astype("float32")

    counts = vector_cluster_counts(
        row_count=row_count,
        cluster_count=cluster_count,
        cluster_size_distribution=request.cluster_size_distribution,
    )
    assignments = np.repeat(np.arange(cluster_count, dtype=np.int64), np.asarray(counts, dtype=np.int64))
    vectors = centers[assignments] + rng.normal(
        0.0,
        spread,
        size=(row_count, request.dimension),
    ).astype("float32")
    vectors *= _length_scales(rng, row_count, request.vector_length_variation)[:, np.newaxis].astype("float32")

    normalize_mode = (request.normalize or "").strip().lower()
    if normalize_mode == "l2":
        norms = np.linalg.norm(vectors, axis=1)
        norms[norms == 0.0] = 1.0
        vectors = vectors / norms[:, np.newaxis]

    result: np.ndarray = vectors.astype("float32", copy=False)
    return result


def _uniform_values(
    field: Field,
    row_count: int,
    distribution: DistributionRequest,
    seed: int,
) -> list[object]:
    rng = random.Random(seed)
    lower = distribution.min if distribution.min is not None else 0.0
    upper = distribution.max if distribution.max is not None else 1.0
    if upper < lower:
        lower, upper = upper, lower
    if isinstance(field.type, IntType):
        low_int = int(math.floor(lower))
        high_int = int(math.ceil(upper))
        return [rng.randint(low_int, high_int) for _ in range(row_count)]
    if isinstance(field.type, DoubleType):
        return [rng.uniform(lower, upper) for _ in range(row_count)]
    raise ContractViolation(f"uniform distribution requires numeric field, got {type(field.type).__name__}")


def _lognormal_values(
    field: Field,
    row_count: int,
    distribution: DistributionRequest,
    seed: int,
) -> list[object]:
    rng = random.Random(seed)
    mean = distribution.mean if distribution.mean is not None else 0.0
    sigma = distribution.sigma if distribution.sigma is not None else 1.0
    values = [rng.lognormvariate(mean, sigma) for _ in range(row_count)]
    if isinstance(field.type, IntType):
        return [max(0, int(round(v))) for v in values]
    if isinstance(field.type, DoubleType):
        float_values: list[object] = [float(value) for value in values]
        return float_values
    raise ContractViolation(f"lognormal distribution requires numeric field, got {type(field.type).__name__}")


def _categorical_values(
    field: Field,
    row_count: int,
    distribution: DistributionRequest,
    seed: int,
) -> list[object]:
    if not distribution.values:
        raise ContractViolation("categorical distribution requires values")
    rng = random.Random(seed)
    raw_weights = distribution.weights or tuple(1.0 for _ in distribution.values)
    if len(raw_weights) != len(distribution.values):
        raise ContractViolation("categorical distribution weights must match values")
    chosen = rng.choices(distribution.values, weights=raw_weights, k=row_count)
    return [_coerce_scalar(field, value) for value in chosen]


def _place_ordered_values_by_sample_rank(
    sample: np.ndarray,
    ordered_values: list[object],
) -> list[object]:
    if len(sample) != len(ordered_values):
        raise ContractViolation(
            "correlation sample and ordered values must have equal length"
        )
    ranked_rows = np.argsort(sample, kind="stable")
    result: list[object] = [None] * len(ordered_values)
    for ordinal, row in enumerate(ranked_rows.tolist()):
        result[int(row)] = ordered_values[ordinal]
    return result


def _coerce_scalar(field: Field, value: str) -> object:
    if isinstance(field.type, IntType):
        return int(float(value))
    if isinstance(field.type, DoubleType):
        return float(value)
    return value


def _normal_to_uniform(sample: np.ndarray) -> np.ndarray:
    result: np.ndarray = 0.5 * (1.0 + _vectorized_erf(sample / math.sqrt(2.0)))
    return result


def _stabilize_correlation(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    np.fill_diagonal(symmetric, 1.0)
    eigvals = np.linalg.eigvalsh(symmetric)
    minimum = float(np.min(eigvals))
    if minimum <= 0.0:
        symmetric = symmetric + np.eye(symmetric.shape[0]) * (abs(minimum) + 1.0e-6)
        diagonal = np.sqrt(np.diag(symmetric))
        symmetric = symmetric / np.outer(diagonal, diagonal)
        np.fill_diagonal(symmetric, 1.0)
    return cast(np.ndarray, symmetric)


def _stabilize_covariance(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    eigvals = np.linalg.eigvalsh(symmetric)
    minimum = float(np.min(eigvals))
    if minimum < 0.0:
        symmetric = symmetric + np.eye(symmetric.shape[0]) * (abs(minimum) + 1e-9)
    return cast(np.ndarray, symmetric)


def _length_scales(
    rng: np.random.Generator,
    row_count: int,
    variation: str | None,
) -> np.ndarray:
    normalized = (variation or "none").strip().lower()
    if normalized in {"none", ""}:
        return np.ones(row_count, dtype=float)
    if normalized == "low":
        return rng.lognormal(mean=0.0, sigma=0.08, size=row_count)
    if normalized == "medium":
        return rng.lognormal(mean=0.0, sigma=0.18, size=row_count)
    if normalized == "high":
        return rng.lognormal(mean=0.0, sigma=0.35, size=row_count)
    return rng.lognormal(mean=0.0, sigma=0.18, size=row_count)

def _zipf_weights(size: int, slope: float) -> list[float]:
    alpha = max(0.1, slope)
    weights = [1.0 / ((index + 1) ** alpha) for index in range(size)]
    total = sum(weights) or 1.0
    return [weight / total for weight in weights]


def vector_cluster_counts(
    *,
    row_count: int,
    cluster_count: int | None,
    cluster_size_distribution: str | None,
) -> tuple[int, ...]:
    if row_count <= 0:
        return ()
    count = max(1, cluster_count or 1)
    weights = _vector_cluster_weights(count, cluster_size_distribution)
    return rounded_counts(row_count, weights)


def _vector_cluster_weights(count: int, mode: str | None) -> tuple[float, ...]:
    if count <= 0:
        return ()
    normalized = (mode or "uniform").strip().lower()
    if normalized == "uniform":
        return tuple(1.0 for _ in range(count))
    if normalized == "zipf_0_5":
        return tuple(1.0 / ((index + 1) ** 0.5) for index in range(count))
    if normalized == "zipf_1_0":
        return tuple(1.0 / (index + 1) for index in range(count))
    return tuple(1.0 for _ in range(count))


def rounded_counts(total: int, weights: tuple[float, ...]) -> tuple[int, ...]:
    if total <= 0 or not weights:
        return ()
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        raise ContractViolation("weights must sum to a positive value")
    exact = [total * weight / weight_sum for weight in weights]
    counts = [int(math.floor(value)) for value in exact]
    remainder = total - sum(counts)
    order = sorted(
        range(len(weights)),
        key=lambda index: exact[index] - counts[index],
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    return tuple(counts)


__all__ = [
    "ValueGenerator",
    "IntGenerator",
    "DoubleGenerator",
    "TextGenerator",
    "VectorGenerator",
    "generate_correlation_block",
    "generate_field_values",
    "generate_vector_values",
    "generator_for",
    "rounded_counts",
    "vector_cluster_counts",
]
