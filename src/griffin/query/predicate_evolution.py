"""Typed candidate evolution for fixed predicate templates."""

from __future__ import annotations

import hashlib
import math
import random
from bisect import bisect_left
from collections.abc import Mapping
from functools import lru_cache
from dataclasses import dataclass, field, replace
from typing import Protocol, TypeAlias

from griffin.infra.errors import ContractViolation
from griffin.ir.expr import (
    BinaryOp,
    BinOp,
    Expr,
    HoleRef,
    RangePredicateTemplate,
    SetMembership,
)
from griffin.query.domain import parameter_space_is_empty
from griffin.query.model import (
    AggregateNode,
    BindingFingerprint,
    BucketSetValue,
    CategoricalSetParameterSpace,
    EnumeratedParameterSpace,
    HoleId,
    LimitNode,
    NumericRangeValue,
    OrderedRangeBoundary,
    OrderedRangeParameterSpace,
    ParameterSpace,
    ParameterValue,
    QueryTemplate,
    RankNode,
    SelectNode,
    TemplateNode,
    stable_signature,
)
from griffin.query.predicate_cover import (
    PredicateCoverTargetMetric,
    predicate_target_distance,
)
from griffin.query.task import PredicateEvolutionStrategyKind


@dataclass(frozen=True)
class PredicateAxisValueProfile:
    value: ParameterValue
    selectivity: float


@dataclass(frozen=True)
class PredicateEvolutionPrior:
    marginal_minimum: float = 0.0
    marginal_center: float = 0.5
    marginal_maximum: float = 1.0
    axis_profiles: tuple[tuple[PredicateAxisValueProfile, ...], ...] = ()


@dataclass(frozen=True)
class PredicateEvolutionProblem:
    template: QueryTemplate
    cover_root: TemplateNode
    predicate_domains: tuple[ParameterSpace, ...]
    target_metric: PredicateCoverTargetMetric
    target_minimum: float
    target_maximum: float
    strategy: PredicateEvolutionStrategyKind = PredicateEvolutionStrategyKind.STANDARD
    prior: PredicateEvolutionPrior = field(default_factory=PredicateEvolutionPrior)


PredicateEvolutionCandidate: TypeAlias = tuple[ParameterValue, ...]


_STANDARD_ORDERED_RANGE_GLOBAL_MUTATION_RATE = 0.2


@dataclass(frozen=True)
class PredicateEvolutionFeedback:
    candidate: PredicateEvolutionCandidate
    binding_fingerprint: BindingFingerprint
    observed_value: float
    objective_distance: float
    usable: bool


@dataclass(frozen=True)
class PredicateEvolutionScore:
    candidate: PredicateEvolutionCandidate
    generation: int
    binding_fingerprint: BindingFingerprint
    observed_value: float
    objective_distance: float
    usable: bool


@dataclass
class PredicateEvolutionState:
    generation: int = 0
    pending: list[PredicateEvolutionCandidate] = field(default_factory=list)
    scores: list[PredicateEvolutionScore] = field(default_factory=list)
    observed_candidates: set[PredicateEvolutionCandidate] = field(default_factory=set)


class PredicateEvolutionStrategy(Protocol):
    kind: PredicateEvolutionStrategyKind

    def candidate_frontier_floor(self, axis_count: int) -> int: ...

    def propose(
        self,
        problem: PredicateEvolutionProblem,
        state: PredicateEvolutionState,
        *,
        seed: int,
        candidate_limit: int,
    ) -> PredicateEvolutionCandidate | None: ...

    def observe(
        self,
        state: PredicateEvolutionState,
        problem: PredicateEvolutionProblem,
        feedback: PredicateEvolutionFeedback,
    ) -> None: ...


class GuidedPredicateEvolutionStrategy:
    """Deterministic bounded beam over complete typed predicate candidates."""

    kind = PredicateEvolutionStrategyKind.GUIDED

    def candidate_frontier_floor(self, axis_count: int) -> int:
        return _predicate_beam_limit(axis_count)

    def propose(
        self,
        problem: PredicateEvolutionProblem,
        state: PredicateEvolutionState,
        *,
        seed: int,
        candidate_limit: int,
    ) -> PredicateEvolutionCandidate | None:
        if any(parameter_space_is_empty(space) for space in problem.predicate_domains):
            return None
        while True:
            if not state.pending:
                if state.generation == 0 and not state.scores:
                    proposals = _initial_predicate_candidates(
                        problem,
                        seed=seed,
                        candidate_limit=candidate_limit,
                    )
                else:
                    proposals = _next_predicate_candidates(
                        state,
                        problem,
                    )
                state.pending = [
                    candidate
                    for candidate in proposals
                    if candidate not in state.observed_candidates
                ]
                state.generation += 1
                if not state.pending:
                    return _probe_unobserved_enumerated_candidate(problem, state)
            candidate = state.pending.pop(0)
            if candidate not in state.observed_candidates:
                return candidate

    def observe(
        self,
        state: PredicateEvolutionState,
        problem: PredicateEvolutionProblem,
        feedback: PredicateEvolutionFeedback,
    ) -> None:
        score = PredicateEvolutionScore(
            candidate=feedback.candidate,
            generation=state.generation,
            binding_fingerprint=feedback.binding_fingerprint,
            observed_value=feedback.observed_value,
            objective_distance=feedback.objective_distance,
            usable=feedback.usable,
        )
        state.scores.append(score)
        immediate = _predicate_candidate_neighbors(problem, score)
        pending: list[PredicateEvolutionCandidate] = []
        for candidate in (*immediate, *state.pending):
            if candidate in state.observed_candidates:
                continue
            _append_unique_candidate(
                pending,
                candidate,
                limit=_predicate_beam_limit(len(problem.predicate_domains)) * 2,
            )
        state.pending = pending


class StandardPredicateEvolutionStrategy:
    """Seeded population GA using exact predicate-cover feedback as fitness."""

    kind = PredicateEvolutionStrategyKind.STANDARD

    def candidate_frontier_floor(self, axis_count: int) -> int:
        return _standard_population_limit(axis_count, candidate_limit=64)

    def propose(
        self,
        problem: PredicateEvolutionProblem,
        state: PredicateEvolutionState,
        *,
        seed: int,
        candidate_limit: int,
    ) -> PredicateEvolutionCandidate | None:
        if any(parameter_space_is_empty(space) for space in problem.predicate_domains):
            return None
        while True:
            if not state.pending:
                inherited_scores: tuple[PredicateEvolutionScore, ...] = ()
                if state.generation == 0 and not state.scores:
                    proposals = _standard_initial_population(
                        problem,
                        seed=seed,
                        candidate_limit=candidate_limit,
                    )
                else:
                    proposals, inherited_scores = _standard_next_population(
                        problem,
                        state,
                        seed=seed,
                        candidate_limit=candidate_limit,
                    )
                pending = [
                    candidate
                    for candidate in proposals
                    if candidate not in state.observed_candidates
                ]
                if not pending:
                    return None
                state.generation += 1
                state.pending = pending
                state.scores.extend(
                    replace(score, generation=state.generation)
                    for score in inherited_scores
                )
            candidate = state.pending.pop(0)
            if candidate not in state.observed_candidates:
                return candidate

    def observe(
        self,
        state: PredicateEvolutionState,
        problem: PredicateEvolutionProblem,
        feedback: PredicateEvolutionFeedback,
    ) -> None:
        del problem
        state.scores.append(
            PredicateEvolutionScore(
                candidate=feedback.candidate,
                generation=state.generation,
                binding_fingerprint=feedback.binding_fingerprint,
                observed_value=feedback.observed_value,
                objective_distance=feedback.objective_distance,
                usable=feedback.usable,
            )
        )


_GUIDED_PREDICATE_EVOLUTION_STRATEGY = GuidedPredicateEvolutionStrategy()
_STANDARD_PREDICATE_EVOLUTION_STRATEGY = StandardPredicateEvolutionStrategy()


def predicate_evolution_strategy(
    kind: PredicateEvolutionStrategyKind,
) -> PredicateEvolutionStrategy:
    if kind is PredicateEvolutionStrategyKind.GUIDED:
        return _GUIDED_PREDICATE_EVOLUTION_STRATEGY
    if kind is PredicateEvolutionStrategyKind.STANDARD:
        return _STANDARD_PREDICATE_EVOLUTION_STRATEGY
    raise ContractViolation(f"unsupported predicate evolution strategy {kind!r}")


def _standard_initial_population(
    problem: PredicateEvolutionProblem,
    *,
    seed: int,
    candidate_limit: int,
) -> list[PredicateEvolutionCandidate]:
    limit = _standard_population_limit(
        len(problem.predicate_domains),
        candidate_limit=candidate_limit,
    )
    rng = _standard_generation_rng(seed, generation=0)
    population: list[PredicateEvolutionCandidate] = []
    # GA 初始化保留单谓词 selectivity 排序：前半个体取自按预测
    # selectivity 排序的确定性初始候选（与 GUIDED 初始化同源），
    # 后半随机采样保持种群多样性。
    ordered = _initial_predicate_candidates(problem, seed=seed)
    seeded_cap = limit // 2
    for candidate in ordered:
        if len(population) >= seeded_cap:
            break
        _append_unique_candidate(population, candidate, limit=limit)
    for _attempt in range(limit * 64):
        candidate = tuple(
            _standard_random_axis_value(space, rng)
            for space in problem.predicate_domains
        )
        _append_unique_candidate(population, candidate, limit=limit)
        if len(population) >= limit:
            break
    return population


def _standard_next_population(
    problem: PredicateEvolutionProblem,
    state: PredicateEvolutionState,
    *,
    seed: int,
    candidate_limit: int,
) -> tuple[
    list[PredicateEvolutionCandidate],
    tuple[PredicateEvolutionScore, ...],
]:
    generation_scores = [
        score for score in state.scores if score.generation == state.generation
    ]
    if not generation_scores:
        return [], ()
    ranked = sorted(generation_scores, key=_standard_score_key)
    population_limit = _standard_population_limit(
        len(problem.predicate_domains),
        candidate_limit=candidate_limit,
    )
    elite_count = min(2, len(ranked), population_limit - 1)
    offspring_limit = max(population_limit - elite_count, 1)
    rng = _standard_generation_rng(seed, generation=state.generation)
    offspring: list[PredicateEvolutionCandidate] = []
    for _attempt in range(population_limit * 128):
        first = _standard_tournament_select(ranked, rng)
        second = _standard_tournament_select(ranked, rng)
        child = _standard_crossover(
            problem,
            first.candidate,
            second.candidate,
            rng,
        )
        child = _standard_mutate(problem, child, rng)
        if child in state.observed_candidates:
            continue
        _append_unique_candidate(offspring, child, limit=offspring_limit)
        if len(offspring) >= offspring_limit:
            break
    return offspring, tuple(ranked[:elite_count])


def _standard_population_limit(axis_count: int, *, candidate_limit: int) -> int:
    bounded = min(max(16, max(axis_count, 1) * 8), 64)
    return min(max(int(candidate_limit), 1), bounded)


def _standard_generation_rng(seed: int, *, generation: int) -> random.Random:
    return random.Random(_stable_seed(f"{seed}:standard:{generation}"))


def _standard_score_key(
    score: PredicateEvolutionScore,
) -> tuple[int, float, str]:
    return (
        0 if score.usable else 1,
        score.objective_distance,
        score.binding_fingerprint,
    )


def _standard_tournament_select(
    scores: list[PredicateEvolutionScore],
    rng: random.Random,
) -> PredicateEvolutionScore:
    competitors = [scores[rng.randrange(len(scores))] for _index in range(3)]
    return min(competitors, key=_standard_score_key)


def _standard_random_axis_value(
    space: ParameterSpace,
    rng: random.Random,
) -> ParameterValue:
    if isinstance(space, EnumeratedParameterSpace):
        return space.values[rng.randrange(len(space.values))]
    if isinstance(space, CategoricalSetParameterSpace):
        return _standard_random_categorical_set_value(space, rng)
    if isinstance(space, OrderedRangeParameterSpace):
        pair = _standard_random_range_pair(len(space.boundaries), rng)
        return _ordered_range_value(space, pair)
    raise ContractViolation(f"unsupported predicate parameter space {type(space).__name__}")


def _standard_random_categorical_set_value(
    space: CategoricalSetParameterSpace,
    rng: random.Random,
) -> BucketSetValue:
    atom_count = len(space.atoms)
    max_cardinality = min(space.max_size, atom_count)
    if max_cardinality <= 0:
        raise ContractViolation("categorical set parameter space is empty")
    cardinality_counts = tuple(
        math.comb(atom_count, cardinality)
        for cardinality in range(1, max_cardinality + 1)
    )
    ticket = rng.randrange(sum(cardinality_counts))
    for cardinality, value_count in enumerate(
        cardinality_counts,
        start=1,
    ):
        if ticket < value_count:
            indices = frozenset(rng.sample(range(atom_count), cardinality))
            return _categorical_set_value(space, indices)
        ticket -= value_count
    raise AssertionError("categorical set sample ticket is outside parameter space")


def _standard_random_range_pair(
    boundary_count: int,
    rng: random.Random,
) -> tuple[int, int]:
    pair_index = rng.randrange(boundary_count * (boundary_count + 1) // 2)
    lower_index = 0
    row_size = boundary_count
    while pair_index >= row_size:
        pair_index -= row_size
        lower_index += 1
        row_size -= 1
    return lower_index, lower_index + pair_index


def _standard_crossover(
    problem: PredicateEvolutionProblem,
    first: PredicateEvolutionCandidate,
    second: PredicateEvolutionCandidate,
    rng: random.Random,
) -> PredicateEvolutionCandidate:
    child: list[ParameterValue] = []
    for space, first_value, second_value in zip(
        problem.predicate_domains,
        first,
        second,
        strict=True,
    ):
        if isinstance(space, EnumeratedParameterSpace):
            child.append(first_value if rng.randrange(2) == 0 else second_value)
        elif isinstance(space, CategoricalSetParameterSpace):
            child.append(
                _standard_categorical_crossover(
                    space,
                    first_value,
                    second_value,
                    rng,
                )
            )
        elif isinstance(space, OrderedRangeParameterSpace):
            first_pair = _ordered_range_indices(space, first_value)
            second_pair = _ordered_range_indices(space, second_value)
            endpoints = sorted(
                (
                    first_pair[0] if rng.randrange(2) == 0 else second_pair[0],
                    first_pair[1] if rng.randrange(2) == 0 else second_pair[1],
                )
            )
            child.append(_ordered_range_value(space, (endpoints[0], endpoints[1])))
        else:
            raise ContractViolation(
                f"unsupported predicate parameter space {type(space).__name__}"
            )
    return tuple(child)


def _standard_categorical_crossover(
    space: CategoricalSetParameterSpace,
    first: ParameterValue,
    second: ParameterValue,
    rng: random.Random,
) -> BucketSetValue:
    _validate_categorical_set_value(space, first)
    _validate_categorical_set_value(space, second)
    assert isinstance(first, BucketSetValue)
    assert isinstance(second, BucketSetValue)
    first_labels = frozenset(first.labels)
    second_labels = frozenset(second.labels)
    union_indices = [
        index
        for index, atom in enumerate(space.atoms)
        if atom.label in first_labels or atom.label in second_labels
    ]
    selected = [
        index
        for index in union_indices
        if (
            space.atoms[index].label in first_labels
            and space.atoms[index].label in second_labels
        )
        or rng.randrange(2) == 0
    ]
    if not selected:
        selected = [union_indices[rng.randrange(len(union_indices))]]
    if len(selected) > space.max_size:
        selected = rng.sample(selected, space.max_size)
    return _categorical_set_value(space, frozenset(selected))


def _standard_mutate(
    problem: PredicateEvolutionProblem,
    candidate: PredicateEvolutionCandidate,
    rng: random.Random,
) -> PredicateEvolutionCandidate:
    child = list(candidate)
    axis_count = len(problem.predicate_domains)
    selected_axes = {rng.randrange(axis_count)}
    for axis in range(axis_count):
        if rng.random() < (1.0 / axis_count):
            selected_axes.add(axis)
    for axis in selected_axes:
        child[axis] = _standard_mutate_axis_value(
            problem.predicate_domains[axis],
            child[axis],
            rng,
        )
    return tuple(child)


def _standard_mutate_axis_value(
    space: ParameterSpace,
    current: ParameterValue,
    rng: random.Random,
) -> ParameterValue:
    if isinstance(space, EnumeratedParameterSpace):
        alternatives = [value for value in space.values if value != current]
        return alternatives[rng.randrange(len(alternatives))] if alternatives else current
    if isinstance(space, CategoricalSetParameterSpace):
        return _standard_mutate_categorical(space, current, rng)
    if isinstance(space, OrderedRangeParameterSpace):
        return _standard_mutate_ordered_range(space, current, rng)
    raise ContractViolation(f"unsupported predicate parameter space {type(space).__name__}")


def _standard_mutate_ordered_range(
    space: OrderedRangeParameterSpace,
    current: ParameterValue,
    rng: random.Random,
) -> NumericRangeValue:
    pair = _ordered_range_indices(space, current)
    if len(space.boundaries) <= 1:
        return _ordered_range_value(space, pair)
    if rng.random() >= _STANDARD_ORDERED_RANGE_GLOBAL_MUTATION_RATE:
        return _ordered_range_value(
            space,
            _standard_ordered_range_local_mutation(space, pair, rng),
        )
    return _ordered_range_value(
        space,
        _standard_ordered_range_global_mutation(space, pair, rng),
    )


def _standard_ordered_range_local_mutation(
    space: OrderedRangeParameterSpace,
    pair: tuple[int, int],
    rng: random.Random,
) -> tuple[int, int]:
    lower_index, upper_index = pair
    radius = max(math.isqrt(len(space.boundaries)), 1)
    endpoint_ranges: list[tuple[bool, int, int, int]] = []
    lower_start = max(lower_index - radius, 0)
    lower_stop = min(lower_index + radius + 1, upper_index + 1)
    if lower_stop - lower_start > 1:
        endpoint_ranges.append((True, lower_start, lower_stop, lower_index))
    upper_start = max(upper_index - radius, lower_index)
    upper_stop = min(upper_index + radius + 1, len(space.boundaries))
    if upper_stop - upper_start > 1:
        endpoint_ranges.append((False, upper_start, upper_stop, upper_index))
    mutate_lower, start, stop, current = endpoint_ranges[
        rng.randrange(len(endpoint_ranges))
    ]
    replacement = _standard_random_other_index(
        start=start,
        stop=stop,
        current=current,
        rng=rng,
    )
    if mutate_lower:
        return replacement, upper_index
    return lower_index, replacement


def _standard_ordered_range_global_mutation(
    space: OrderedRangeParameterSpace,
    pair: tuple[int, int],
    rng: random.Random,
) -> tuple[int, int]:
    lower_index, upper_index = pair
    mutable_endpoints: list[bool] = []
    if upper_index > 0:
        mutable_endpoints.append(True)
    if lower_index < len(space.boundaries) - 1:
        mutable_endpoints.append(False)
    mutate_lower = mutable_endpoints[rng.randrange(len(mutable_endpoints))]
    if mutate_lower:
        return (
            _standard_random_other_index(
                start=0,
                stop=upper_index + 1,
                current=lower_index,
                rng=rng,
            ),
            upper_index,
        )
    return (
        lower_index,
        _standard_random_other_index(
            start=lower_index,
            stop=len(space.boundaries),
            current=upper_index,
            rng=rng,
        ),
    )


def _standard_random_other_index(
    *,
    start: int,
    stop: int,
    current: int,
    rng: random.Random,
) -> int:
    alternative_count = stop - start - 1
    if alternative_count <= 0:
        return current
    offset = rng.randrange(alternative_count)
    candidate = start + offset
    return candidate if candidate < current else candidate + 1


def _standard_mutate_categorical(
    space: CategoricalSetParameterSpace,
    current: ParameterValue,
    rng: random.Random,
) -> BucketSetValue:
    _validate_categorical_set_value(space, current)
    assert isinstance(current, BucketSetValue)
    by_label = {atom.label: index for index, atom in enumerate(space.atoms)}
    selected = {by_label[label] for label in current.labels}
    unselected = set(range(len(space.atoms))) - selected
    operations: list[str] = []
    if unselected and len(selected) < space.max_size:
        operations.append("add")
    if len(selected) > 1:
        operations.append("remove")
    if selected and unselected:
        operations.append("swap")
    if not operations:
        return current
    operation = operations[rng.randrange(len(operations))]
    if operation in {"remove", "swap"}:
        selected.remove(rng.choice(sorted(selected)))
    if operation in {"add", "swap"}:
        selected.add(rng.choice(sorted(unselected)))
    return _categorical_set_value(space, frozenset(selected))


def predicate_evolution_values(
    problem: PredicateEvolutionProblem,
    candidate: PredicateEvolutionCandidate,
) -> dict[HoleId, ParameterValue]:
    if len(candidate) != len(problem.predicate_domains):
        raise ContractViolation(
            "predicate search candidate arity does not match parameter spaces"
        )
    values: dict[HoleId, ParameterValue] = {}
    for value, space in zip(candidate, problem.predicate_domains, strict=True):
        if isinstance(space, EnumeratedParameterSpace):
            if value not in space.values:
                raise ContractViolation(
                    "predicate candidate value is outside enumerated space"
                )
        elif isinstance(space, CategoricalSetParameterSpace):
            _validate_categorical_set_value(space, value)
        else:
            _validate_ordered_range_value(space, value)
        values[space.hole_id] = value
    return values


def predicate_marginal_band(
    root: TemplateNode,
    spaces: tuple[ParameterSpace, ...],
    *,
    target_minimum: float,
    target_maximum: float,
) -> tuple[float, float, float]:
    hole_ids = frozenset(space.hole_id for space in spaces)
    expressions = _predicate_expressions_for_holes(root, hole_ids)
    targets = (
        min(max(float(target_minimum), 0.0), 1.0),
        min(max((float(target_minimum) + float(target_maximum)) / 2.0, 0.0), 1.0),
        min(max(float(target_maximum), 0.0), 1.0),
    )
    if not expressions:
        axis_count = max(len(spaces), 1)
        return (
            targets[0] ** (1.0 / axis_count),
            targets[1] ** (1.0 / axis_count),
            targets[2] ** (1.0 / axis_count),
        )
    return (
        _solve_equal_axis_selectivity(expressions, hole_ids, targets[0]),
        _solve_equal_axis_selectivity(expressions, hole_ids, targets[1]),
        _solve_equal_axis_selectivity(expressions, hole_ids, targets[2]),
    )


def _solve_equal_axis_selectivity(
    expressions: tuple[Expr, ...],
    hole_ids: frozenset[HoleId],
    target: float,
) -> float:
    low = 0.0
    high = 1.0
    for _ in range(64):
        middle = (low + high) / 2.0
        predicted = _equal_axis_selectivity(expressions, hole_ids, middle)
        if predicted < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _equal_axis_selectivity(
    expressions: tuple[Expr, ...],
    hole_ids: frozenset[HoleId],
    value: float,
) -> float:
    result = 1.0
    for expression in expressions:
        result *= _equal_expr_selectivity(expression, hole_ids, value)
    return min(max(result, 0.0), 1.0)


def _equal_expr_selectivity(
    expression: Expr,
    hole_ids: frozenset[HoleId],
    value: float,
) -> float:
    if not (_expr_hole_ids(expression) & hole_ids):
        return 1.0
    if isinstance(expression, BinaryOp) and expression.operator in {
        BinOp.AND,
        BinOp.OR,
    }:
        left = _equal_expr_selectivity(expression.left, hole_ids, value)
        right = _equal_expr_selectivity(expression.right, hole_ids, value)
        if expression.operator is BinOp.AND:
            return left * right
        return 1.0 - ((1.0 - left) * (1.0 - right))
    return value


def _predicate_expressions_for_holes(
    node: TemplateNode,
    hole_ids: frozenset[HoleId],
) -> tuple[Expr, ...]:
    expressions: list[Expr] = []
    if isinstance(node, SelectNode):
        if _expr_hole_ids(node.predicate_expr) & hole_ids:
            expressions.append(node.predicate_expr)
        expressions.extend(_predicate_expressions_for_holes(node.input, hole_ids))
    elif isinstance(node, (RankNode, AggregateNode, LimitNode)):
        expressions.extend(_predicate_expressions_for_holes(node.input, hole_ids))
    return tuple(expressions)


def _expr_hole_ids(expression: Expr) -> frozenset[HoleId]:
    if isinstance(expression, HoleRef):
        return frozenset((expression.hole_id,))
    if isinstance(expression, BinaryOp):
        return _expr_hole_ids(expression.left) | _expr_hole_ids(expression.right)
    if isinstance(expression, SetMembership):
        result = _expr_hole_ids(expression.expr)
        for value in expression.values:
            result |= _expr_hole_ids(value)
        return result
    if isinstance(expression, RangePredicateTemplate):
        return frozenset((expression.range_value.hole_id,))
    return frozenset()


def _initial_predicate_candidates(
    problem: PredicateEvolutionProblem,
    *,
    seed: int,
    candidate_limit: int | None = None,
) -> list[PredicateEvolutionCandidate]:
    if not problem.predicate_domains:
        return [()]
    limit = _predicate_beam_limit(len(problem.predicate_domains))
    if candidate_limit is not None:
        limit = min(limit, max(int(candidate_limit), 1))
    axis_values = tuple(
        _axis_initial_values(problem, axis, space, seed=seed)
        for axis, space in enumerate(problem.predicate_domains)
    )
    if any(not values for values in axis_values):
        return []
    beam: list[PredicateEvolutionCandidate] = [()]
    for values in axis_values:
        expanded = [candidate + (value,) for candidate in beam for value in values]
        beam = _rank_predicate_candidates(problem, expanded, limit=limit)
    if any(_axis_profiles(problem, axis) for axis in range(len(problem.predicate_domains))):
        ranked = _rank_predicate_candidates(problem, beam, limit=limit)
    else:
        stratified = _stratified_enumerated_candidates(
            problem.predicate_domains,
            seed=seed,
            limit=limit,
        )
        ranked = _rank_predicate_candidates(problem, [*stratified, *beam], limit=limit)
    return _shuffle_in_band_candidates(problem, ranked, seed=seed)


def _shuffle_in_band_candidates(
    problem: PredicateEvolutionProblem,
    candidates: list[PredicateEvolutionCandidate],
    *,
    seed: int,
) -> list[PredicateEvolutionCandidate]:
    """对预测 selectivity 落在 target band 内的候选做 seed 依赖 shuffle。

    band 内候选在 selectivity 语义上等价，但确定性排序会让不同 query 的
    首个 usable cover 固定在同一个 selectivity 值上。shuffle 让不同 query
    产出不同 selectivity 的 cover，实现 range 内多样性。
    """
    if len(candidates) < 2:
        return candidates
    in_band: list[PredicateEvolutionCandidate] = []
    out_band: list[PredicateEvolutionCandidate] = []
    for candidate in candidates:
        distance = predicate_target_distance(
            _predicted_candidate_selectivity(problem, candidate),
            problem.target_minimum,
            problem.target_maximum,
        )
        (in_band if distance == 0.0 else out_band).append(candidate)
    if len(in_band) < 2:
        return candidates
    rng = random.Random(seed ^ 0xD1A1B1E5)
    rng.shuffle(in_band)
    return [*in_band, *out_band]


def _stratified_enumerated_candidates(
    spaces: tuple[ParameterSpace, ...],
    *,
    seed: int,
    limit: int,
) -> list[PredicateEvolutionCandidate]:
    if any(not isinstance(space, EnumeratedParameterSpace) for space in spaces):
        return []
    enumerated_spaces = tuple(
        space for space in spaces if isinstance(space, EnumeratedParameterSpace)
    )
    product_size = _finite_product_size(
        tuple(len(space.values) for space in enumerated_spaces)
    )
    if product_size <= 0:
        return []
    offset = seed % product_size
    stride = _coprime_stride(
        product_size,
        _stable_seed(f"enumerated-stride|{seed}"),
    )
    return [
        _enumerated_candidate_from_linear_index(
            (offset + ordinal * stride) % product_size,
            enumerated_spaces,
        )
        for ordinal in range(min(limit, product_size))
    ]


def _coprime_stride(modulus: int, seed: int) -> int:
    if modulus <= 1:
        return 1
    stride = seed % modulus
    if stride == 0:
        stride = 1
    while math.gcd(stride, modulus) != 1:
        stride += 1
        if stride >= modulus:
            stride = 1
    return stride


def _next_predicate_candidates(
    state: PredicateEvolutionState,
    problem: PredicateEvolutionProblem,
) -> list[PredicateEvolutionCandidate]:
    if not state.scores:
        return []
    limit = _predicate_beam_limit(len(problem.predicate_domains))
    elites = sorted(
        state.scores,
        key=lambda score: (
            0 if score.usable else 1,
            score.objective_distance,
            _candidate_categorical_cardinality(problem, score.candidate),
            score.binding_fingerprint,
        ),
    )[: min(8, limit)]
    proposals: list[PredicateEvolutionCandidate] = []
    for score in elites:
        for neighbor in _predicate_candidate_neighbors(
            problem,
            score,
            include_neutral=True,
        ):
            _append_unique_candidate(proposals, neighbor, limit=limit * 4)
    for left_index, left in enumerate(elites):
        for right in elites[left_index + 1 :]:
            for axis in range(len(problem.predicate_domains)):
                child = list(left.candidate)
                child[axis] = right.candidate[axis]
                _append_unique_candidate(proposals, tuple(child), limit=limit * 4)
                child = list(right.candidate)
                child[axis] = left.candidate[axis]
                _append_unique_candidate(proposals, tuple(child), limit=limit * 4)
    novel = [
        candidate
        for candidate in proposals
        if candidate not in state.observed_candidates
    ]
    return _rank_predicate_candidates(problem, novel, limit=limit)


def _predicate_candidate_neighbors(
    problem: PredicateEvolutionProblem,
    score: PredicateEvolutionScore,
    *,
    include_neutral: bool = False,
) -> tuple[PredicateEvolutionCandidate, ...]:
    if score.observed_value < problem.target_minimum:
        direction = 1
    elif score.observed_value > problem.target_maximum:
        direction = -1
    else:
        direction = 0
    if direction == 0 and not include_neutral:
        return ()
    candidates: list[PredicateEvolutionCandidate] = []
    for axis, space in enumerate(problem.predicate_domains):
        for value in _axis_neighbor_values(
            problem,
            axis,
            space,
            score.candidate[axis],
            direction=direction,
            observed_value=score.observed_value,
        ):
            neighbor = list(score.candidate)
            neighbor[axis] = value
            _append_unique_candidate(
                candidates,
                tuple(neighbor),
                limit=max(len(problem.predicate_domains) * 8, 16),
            )
    return tuple(candidates)


def _axis_initial_values(
    problem: PredicateEvolutionProblem,
    axis: int,
    space: ParameterSpace,
    *,
    seed: int,
) -> tuple[ParameterValue, ...]:
    if isinstance(space, CategoricalSetParameterSpace):
        return _categorical_set_initial_values(
            space,
            targets=(
                problem.prior.marginal_minimum,
                problem.prior.marginal_center,
                problem.prior.marginal_maximum,
            ),
        )
    if isinstance(space, OrderedRangeParameterSpace):
        return _ordered_range_initial_values(
            space,
            targets=(
                problem.prior.marginal_minimum,
                problem.prior.marginal_center,
                problem.prior.marginal_maximum,
            ),
        )
    profiles = _axis_profiles(problem, axis)
    selected: list[ParameterValue] = []
    if profiles:
        ranked = sorted(
            profiles,
            key=lambda profile: (
                predicate_target_distance(
                    profile.selectivity,
                    problem.prior.marginal_minimum,
                    problem.prior.marginal_maximum,
                ),
                abs(profile.selectivity - problem.prior.marginal_center),
                _parameter_value_key(profile.value),
            ),
        )
        for profile in ranked[:8]:
            _append_unique_value(selected, profile.value, limit=12)
    values = space.values
    if values:
        offset = seed % len(values)
        positions = (
            0,
            len(values) // 4,
            len(values) // 2,
            (3 * len(values)) // 4,
            len(values) - 1,
        )
        for position in positions:
            _append_unique_value(
                selected, values[(position + offset) % len(values)], limit=12
            )
    return tuple(selected)


def _axis_neighbor_values(
    problem: PredicateEvolutionProblem,
    axis: int,
    space: ParameterSpace,
    current: ParameterValue,
    *,
    direction: int,
    observed_value: float,
) -> tuple[ParameterValue, ...]:
    if isinstance(space, CategoricalSetParameterSpace):
        return _categorical_set_neighbors(
            space,
            current,
            direction=direction,
            target=problem.prior.marginal_center,
            exact_target=(problem.target_minimum + problem.target_maximum) / 2.0,
            observed_value=observed_value,
            axis_count=len(problem.predicate_domains),
        )
    if isinstance(space, OrderedRangeParameterSpace):
        return _ordered_range_neighbors(
            space,
            current,
            direction=direction,
            target=problem.prior.marginal_center,
            exact_target=(problem.target_minimum + problem.target_maximum) / 2.0,
            observed_value=observed_value,
            axis_count=len(problem.predicate_domains),
        )
    profiles = _axis_profiles(problem, axis)
    if profiles:
        current_profile = next(
            (profile for profile in profiles if profile.value == current), None
        )
        if current_profile is not None:
            if direction > 0:
                eligible = [
                    profile
                    for profile in profiles
                    if profile.selectivity > current_profile.selectivity
                ]
            elif direction < 0:
                eligible = [
                    profile
                    for profile in profiles
                    if profile.selectivity < current_profile.selectivity
                ]
            else:
                eligible = [
                    profile
                    for profile in profiles
                    if profile.value != current
                    and predicate_target_distance(
                        profile.selectivity,
                        problem.prior.marginal_minimum,
                        problem.prior.marginal_maximum,
                    )
                    == 0.0
                ]
            eligible.sort(
                key=lambda profile: (
                    abs(profile.selectivity - current_profile.selectivity),
                    abs(profile.selectivity - problem.prior.marginal_center),
                    _parameter_value_key(profile.value),
                )
            )
            return tuple(profile.value for profile in eligible[:4])
    try:
        index = space.values.index(current)
    except ValueError:
        return ()
    positions = (
        index + direction if direction else index - 1,
        index + (2 * direction) if direction else index + 1,
        0,
        len(space.values) - 1,
    )
    values: list[ParameterValue] = []
    for position in positions:
        if 0 <= position < len(space.values) and position != index:
            _append_unique_value(values, space.values[position], limit=4)
    return tuple(values)


def _categorical_set_initial_values(
    space: CategoricalSetParameterSpace,
    *,
    targets: tuple[float, ...],
) -> tuple[ParameterValue, ...]:
    if not space.atoms:
        return ()
    row_count = max(space.owner_row_count, 1)
    index_sets: list[frozenset[int]] = []
    for target in targets:
        target_count = min(max(float(target), 0.0), 1.0) * row_count
        for indices in _near_count_subsets(
            space, tuple(range(len(space.atoms))), target_count
        ):
            if indices and len(indices) <= space.max_size and indices not in index_sets:
                index_sets.append(indices)
        complement_target = row_count - target_count
        if 0.0 < complement_target < row_count:
            for excluded in _near_count_subsets(
                space,
                tuple(range(len(space.atoms))),
                complement_target,
            ):
                complement = (
                    frozenset(index for index, _atom in enumerate(space.atoms))
                    - excluded
                )
                if (
                    complement
                    and len(complement) <= space.max_size
                    and complement not in index_sets
                ):
                    index_sets.append(complement)
    index_sets.sort(
        key=lambda indices: (
            predicate_target_distance(
                _categorical_set_selectivity(space, indices),
                min(targets),
                max(targets),
            ),
            len(indices),
            abs(
                _categorical_set_selectivity(space, indices)
                - targets[len(targets) // 2]
            ),
            tuple(sorted(indices)),
        )
    )
    return tuple(_categorical_set_value(space, indices) for indices in index_sets[:12])


def _near_count_subsets(
    space: CategoricalSetParameterSpace,
    available: tuple[int, ...],
    target_count: float,
    *,
    max_items: int | None = None,
) -> tuple[frozenset[int], ...]:
    if not available:
        return ()
    item_limit = min(
        len(available),
        space.max_size if max_items is None else max(int(max_items), 0),
    )
    if item_limit <= 0:
        return ()
    atoms = space.atoms
    descending = tuple(
        sorted(available, key=lambda index: (-atoms[index].local_count, index))
    )
    ascending = tuple(reversed(descending))
    candidates: list[frozenset[int]] = []
    for order in (descending, ascending):
        total = 0
        best_distance = float("inf")
        best_length = 0
        crossing_length = 0
        for position, index in enumerate(order, start=1):
            if position > item_limit:
                break
            total += atoms[index].local_count
            distance = abs(total - target_count)
            if distance < best_distance:
                best_distance = distance
                best_length = position
            if total >= target_count:
                crossing_length = position
                break
        if crossing_length:
            _append_unique_index_set(
                candidates,
                frozenset(order[:crossing_length]),
            )
        if best_length:
            _append_unique_index_set(
                candidates,
                frozenset(order[:best_length]),
            )
        greedy: set[int] = set()
        greedy_total = 0
        for index in order:
            if len(greedy) >= item_limit:
                break
            count = atoms[index].local_count
            if greedy_total + count <= target_count:
                greedy.add(index)
                greedy_total += count
        if greedy:
            _append_unique_index_set(candidates, frozenset(greedy))
        remaining = [index for index in available if index not in greedy]
        if remaining and len(greedy) < item_limit:
            repair = min(
                remaining,
                key=lambda index: (
                    abs((greedy_total + atoms[index].local_count) - target_count),
                    index,
                ),
            )
            _append_unique_index_set(candidates, frozenset((*greedy, repair)))
    closest = min(
        available,
        key=lambda index: (abs(space.atoms[index].local_count - target_count), index),
    )
    _append_unique_index_set(candidates, frozenset((closest,)))
    return tuple(candidate for candidate in candidates if len(candidate) <= item_limit)


def _categorical_set_neighbors(
    space: CategoricalSetParameterSpace,
    current: ParameterValue,
    *,
    direction: int,
    target: float,
    exact_target: float,
    observed_value: float,
    axis_count: int,
) -> tuple[ParameterValue, ...]:
    _validate_categorical_set_value(space, current)
    assert isinstance(current, BucketSetValue)
    by_label = {atom.label: index for index, atom in enumerate(space.atoms)}
    selected = frozenset(by_label[label] for label in current.labels)
    unselected = frozenset(index for index, _atom in enumerate(space.atoms)) - selected
    current_count = sum(space.atoms[index].local_count for index in selected)
    row_count = max(space.owner_row_count, 1)
    current_selectivity = current_count / float(row_count)
    local_targets = _feedback_local_targets(
        current_selectivity=current_selectivity,
        marginal_target=target,
        exact_target=exact_target,
        observed_value=observed_value,
        axis_count=axis_count,
        direction=direction,
    )
    candidates: list[frozenset[int]] = []
    remaining_capacity = space.max_size - len(selected)
    if direction >= 0 and unselected and remaining_capacity > 0:
        for local_target in local_targets:
            gap = max((local_target * row_count) - current_count, 1.0)
            for added in _near_count_subsets(
                space,
                tuple(unselected),
                gap,
                max_items=remaining_capacity,
            ):
                _append_unique_index_set(candidates, selected | added)
    if direction <= 0 and len(selected) > 1:
        for local_target in local_targets:
            gap = max(current_count - (local_target * row_count), 1.0)
            for removed_set in _near_count_subsets(
                space,
                tuple(selected),
                gap,
                max_items=len(selected) - 1,
            ):
                remaining = selected - removed_set
                if remaining:
                    _append_unique_index_set(candidates, remaining)
    desired_delta = (local_targets[0] * row_count) - current_count
    if direction > 0:
        desired_delta = max(desired_delta, 1.0)
    elif direction < 0:
        desired_delta = min(desired_delta, -1.0)
    else:
        desired_delta = 0.0
    for removed_index, added_index in _closest_categorical_swaps(
        space,
        selected,
        unselected,
        desired_delta=desired_delta,
        direction=direction,
    ):
        _append_unique_index_set(
            candidates,
            (selected - frozenset((removed_index,))) | frozenset((added_index,)),
        )
    candidates = [
        indices
        for indices in candidates
        if indices and indices != selected and len(indices) <= space.max_size
    ]
    candidates.sort(
        key=lambda indices: (
            min(
                abs(_categorical_set_selectivity(space, indices) - local_target)
                for local_target in local_targets
            ),
            tuple(sorted(indices)),
        )
    )
    return tuple(_categorical_set_value(space, indices) for indices in candidates[:12])


def _feedback_local_targets(
    *,
    current_selectivity: float,
    marginal_target: float,
    exact_target: float,
    observed_value: float,
    axis_count: int,
    direction: int,
) -> tuple[float, ...]:
    targets: list[float] = []

    def append(value: float) -> None:
        bounded = min(max(value, 0.0), 1.0)
        if direction > 0 and bounded <= current_selectivity:
            return
        if direction < 0 and bounded >= current_selectivity:
            return
        if bounded not in targets:
            targets.append(bounded)

    if observed_value > 0.0 and exact_target > 0.0:
        ratio = exact_target / observed_value
        append(current_selectivity * ratio)
        append(current_selectivity * (ratio ** (1.0 / max(axis_count, 1))))
    exact_gap = abs(exact_target - observed_value)
    append(current_selectivity + (direction * exact_gap))
    append(marginal_target)
    append(current_selectivity + (direction * 0.05))
    append(current_selectivity + (direction * 0.10))
    if not targets:
        append(current_selectivity + (direction * 1e-9))
    return tuple(targets) or (current_selectivity,)


def _closest_categorical_swaps(
    space: CategoricalSetParameterSpace,
    selected: frozenset[int],
    unselected: frozenset[int],
    *,
    desired_delta: float,
    direction: int,
) -> tuple[tuple[int, int], ...]:
    if not selected or not unselected:
        return ()
    additions = sorted(
        ((space.atoms[index].local_count, index) for index in unselected),
        key=lambda item: (item[0], item[1]),
    )
    addition_counts = [item[0] for item in additions]
    scored: list[tuple[float, int, int]] = []
    for removed in selected:
        removed_count = space.atoms[removed].local_count
        position = bisect_left(addition_counts, removed_count + desired_delta)
        for candidate_position in (position - 1, position, position + 1):
            if not 0 <= candidate_position < len(additions):
                continue
            added_count, added = additions[candidate_position]
            delta = added_count - removed_count
            if direction > 0 and delta <= 0:
                continue
            if direction < 0 and delta >= 0:
                continue
            scored.append((abs(delta - desired_delta), removed, added))
    scored.sort()
    result: list[tuple[int, int]] = []
    for _distance, removed, added in scored:
        pair = (removed, added)
        if pair not in result:
            result.append(pair)
            if len(result) >= 4:
                break
    return tuple(result)


def _categorical_set_value(
    space: CategoricalSetParameterSpace,
    indices: frozenset[int],
) -> BucketSetValue:
    if not indices or len(indices) > space.max_size:
        raise ContractViolation(
            "categorical set value cardinality is outside parameter space"
        )
    return BucketSetValue(
        labels=tuple(space.atoms[index].label for index in sorted(indices)),
    )


def _categorical_set_selectivity(
    space: CategoricalSetParameterSpace,
    indices: frozenset[int],
) -> float:
    if space.owner_row_count <= 0:
        return 0.0
    count = sum(space.atoms[index].local_count for index in indices)
    return min(max(count / float(space.owner_row_count), 0.0), 1.0)


def _ordered_range_initial_values(
    space: OrderedRangeParameterSpace,
    *,
    targets: tuple[float, ...],
) -> tuple[ParameterValue, ...]:
    if not space.boundaries:
        return ()
    row_count = max(space.owner_row_count, 1)
    upper_counts = tuple(boundary.upper_count for boundary in space.boundaries)
    pairs: set[tuple[int, int]] = set()
    for target in targets:
        target_count = min(max(float(target), 0.0), 1.0) * row_count
        for lower_index, boundary in enumerate(space.boundaries):
            desired_upper_count = boundary.lower_count + target_count
            for upper_index in _ordered_count_positions(
                upper_counts,
                desired_upper_count,
                start=lower_index,
                stop=len(space.boundaries),
            ):
                pairs.add((lower_index, upper_index))
    center = targets[len(targets) // 2]
    pairs_sorted = sorted(
        pairs,
        key=lambda pair: (
            predicate_target_distance(
                _ordered_range_selectivity(space, pair),
                min(targets),
                max(targets),
            ),
            abs(_ordered_range_selectivity(space, pair) - center),
            pair[1] - pair[0],
            pair,
        ),
    )
    return tuple(_ordered_range_value(space, pair) for pair in pairs_sorted[:12])


def _ordered_range_neighbors(
    space: OrderedRangeParameterSpace,
    current: ParameterValue,
    *,
    direction: int,
    target: float,
    exact_target: float,
    observed_value: float,
    axis_count: int,
) -> tuple[ParameterValue, ...]:
    current_pair = _ordered_range_indices(space, current)
    lower_index, upper_index = current_pair
    boundary_count = len(space.boundaries)
    row_count = max(space.owner_row_count, 1)
    current_selectivity = _ordered_range_selectivity(space, current_pair)
    current_count = current_selectivity * row_count
    local_targets = _feedback_local_targets(
        current_selectivity=current_selectivity,
        marginal_target=target,
        exact_target=exact_target,
        observed_value=observed_value,
        axis_count=axis_count,
        direction=direction,
    )
    lower_counts = tuple(boundary.lower_count for boundary in space.boundaries)
    upper_counts = tuple(boundary.upper_count for boundary in space.boundaries)
    pairs: set[tuple[int, int]] = set()

    def add(candidate_lower: int, candidate_upper: int) -> None:
        if (
            0 <= candidate_lower <= candidate_upper < boundary_count
            and (candidate_lower, candidate_upper) != current_pair
        ):
            pairs.add((candidate_lower, candidate_upper))

    if direction > 0:
        add(lower_index - 1, upper_index)
        add(0, upper_index)
        add(lower_index, upper_index + 1)
        add(lower_index, boundary_count - 1)
        for local_target in local_targets:
            target_count = local_target * row_count
            for candidate_lower in _ordered_count_positions(
                lower_counts,
                space.boundaries[upper_index].upper_count - target_count,
                start=0,
                stop=lower_index,
            ):
                add(candidate_lower, upper_index)
            for candidate_upper in _ordered_count_positions(
                upper_counts,
                space.boundaries[lower_index].lower_count + target_count,
                start=upper_index + 1,
                stop=boundary_count,
            ):
                add(lower_index, candidate_upper)
    elif direction < 0:
        add(lower_index + 1, upper_index)
        add(upper_index, upper_index)
        add(lower_index, upper_index - 1)
        add(lower_index, lower_index)
        for local_target in local_targets:
            target_count = local_target * row_count
            for candidate_lower in _ordered_count_positions(
                lower_counts,
                space.boundaries[upper_index].upper_count - target_count,
                start=lower_index + 1,
                stop=upper_index + 1,
            ):
                add(candidate_lower, upper_index)
            for candidate_upper in _ordered_count_positions(
                upper_counts,
                space.boundaries[lower_index].lower_count + target_count,
                start=lower_index,
                stop=upper_index,
            ):
                add(lower_index, candidate_upper)
    else:
        width = upper_index - lower_index
        for delta in {1, max(width // 2, 1), max(boundary_count // 8, 1)}:
            add(lower_index - delta, upper_index - delta)
            add(lower_index + delta, upper_index + delta)
        for candidate_lower in {
            0,
            max(boundary_count - width - 1, 0),
            lower_index - 1,
            lower_index + 1,
        }:
            if not 0 <= candidate_lower < boundary_count:
                continue
            for local_target in local_targets:
                for candidate_upper in _ordered_count_positions(
                    upper_counts,
                    space.boundaries[candidate_lower].lower_count
                    + local_target * row_count,
                    start=candidate_lower,
                    stop=boundary_count,
                ):
                    add(candidate_lower, candidate_upper)

    filtered_pairs = []
    for pair in pairs:
        candidate_count = _ordered_range_selectivity(space, pair) * row_count
        if direction > 0 and candidate_count <= current_count:
            continue
        if direction < 0 and candidate_count >= current_count:
            continue
        filtered_pairs.append(pair)
    filtered_pairs.sort(
        key=lambda pair: (
            min(
                abs(_ordered_range_selectivity(space, pair) - local_target)
                for local_target in local_targets
            ),
            abs(pair[0] - lower_index) + abs(pair[1] - upper_index),
            pair,
        )
    )
    return tuple(_ordered_range_value(space, pair) for pair in filtered_pairs[:12])


def _ordered_count_positions(
    counts: tuple[int, ...],
    target_count: float,
    *,
    start: int,
    stop: int,
) -> tuple[int, ...]:
    if start >= stop:
        return ()
    crossing = bisect_left(counts, target_count, start, stop)
    positions: list[int] = []
    for position in (crossing - 1, crossing):
        if start <= position < stop and position not in positions:
            positions.append(position)
    return tuple(positions)


def _ordered_range_value(
    space: OrderedRangeParameterSpace,
    pair: tuple[int, int],
) -> NumericRangeValue:
    lower_index, upper_index = pair
    if not 0 <= lower_index <= upper_index < len(space.boundaries):
        raise ContractViolation("ordered range indices are outside parameter space")
    return NumericRangeValue(
        low=space.boundaries[lower_index].value,
        high=space.boundaries[upper_index].value,
    )


def _ordered_range_indices(
    space: OrderedRangeParameterSpace,
    value: ParameterValue,
) -> tuple[int, int]:
    _validate_ordered_range_value(space, value)
    assert isinstance(value, NumericRangeValue)
    return (
        _ordered_range_boundary_index(space.boundaries, value.low),
        _ordered_range_boundary_index(space.boundaries, value.high),
    )


def _ordered_range_boundary_index(
    boundaries: tuple[OrderedRangeBoundary, ...],
    boundary_value: float,
) -> int:
    # boundaries 按 value 排序（domain 构建时 sorted(set(cut_points))），二分定位。
    # 不再每次重建 frozenset/dict（O(N)），切点加到 10000 后该路径曾是
    # count>1 高选择率查询初始种群生成超时的根因。
    index = bisect_left(boundaries, boundary_value, key=lambda boundary: boundary.value)
    if index >= len(boundaries) or boundaries[index].value != boundary_value:
        raise ContractViolation(
            "ordered range candidate contains boundary outside parameter space"
        )
    return index


def _validate_ordered_range_value(
    space: OrderedRangeParameterSpace,
    value: ParameterValue,
) -> None:
    if not isinstance(value, NumericRangeValue):
        raise ContractViolation("ordered range candidate requires NumericRangeValue")
    if not value.inclusive_low or not value.inclusive_high:
        raise ContractViolation("ordered range candidate requires inclusive boundaries")
    if value.low > value.high:
        raise ContractViolation(
            "ordered range candidate lower boundary exceeds upper boundary"
        )
    _ordered_range_boundary_index(space.boundaries, value.low)
    _ordered_range_boundary_index(space.boundaries, value.high)


def _ordered_range_selectivity(
    space: OrderedRangeParameterSpace,
    pair: tuple[int, int],
) -> float:
    if space.owner_row_count <= 0:
        return 0.0
    lower_index, upper_index = pair
    count = (
        space.boundaries[upper_index].upper_count
        - space.boundaries[lower_index].lower_count
    )
    return min(max(count / float(space.owner_row_count), 0.0), 1.0)


def _axis_profiles(
    problem: PredicateEvolutionProblem,
    axis: int,
) -> tuple[PredicateAxisValueProfile, ...]:
    if axis >= len(problem.prior.axis_profiles):
        return ()
    return problem.prior.axis_profiles[axis]


@lru_cache(maxsize=2048)
def _axis_profile_selectivity_index(
    profiles: tuple[PredicateAxisValueProfile, ...],
) -> dict[ParameterValue, float]:
    """profile.value -> selectivity 的缓存字典，避免每个候选线性扫描 profile 列表。"""
    return {profile.value: profile.selectivity for profile in profiles}


def _rank_predicate_candidates(
    problem: PredicateEvolutionProblem,
    candidates: list[PredicateEvolutionCandidate],
    *,
    limit: int,
) -> list[PredicateEvolutionCandidate]:
    unique = list(dict.fromkeys(candidates))
    center = (problem.target_minimum + problem.target_maximum) / 2.0
    scored: list[tuple[float, int, float, int, PredicateEvolutionCandidate]] = []
    for ordinal, candidate in enumerate(unique):
        predicted = _predicted_candidate_selectivity(problem, candidate)
        scored.append(
            (
                predicate_target_distance(
                    predicted,
                    problem.target_minimum,
                    problem.target_maximum,
                ),
                _candidate_categorical_cardinality(problem, candidate),
                abs(predicted - center),
                ordinal,
                candidate,
            )
        )
    scored.sort(key=lambda item: item[:3])
    return [
        candidate
        for _distance, _cardinality, _center, _ordinal, candidate in scored[:limit]
    ]


def _candidate_categorical_cardinality(
    problem: PredicateEvolutionProblem,
    candidate: PredicateEvolutionCandidate,
) -> int:
    total = 0
    for space, value in zip(problem.predicate_domains, candidate):
        if not isinstance(space, CategoricalSetParameterSpace):
            continue
        if not isinstance(value, BucketSetValue):
            raise ContractViolation(
                "categorical set space requires BucketSetValue candidate"
            )
        total += len(value.labels)
    return total


def _predicted_candidate_selectivity(
    problem: PredicateEvolutionProblem,
    candidate: PredicateEvolutionCandidate,
) -> float:
    selectivities: dict[HoleId, float] = {}
    for axis, space in enumerate(problem.predicate_domains):
        if axis < len(candidate):
            value = candidate[axis]
            if isinstance(space, CategoricalSetParameterSpace):
                assert isinstance(value, BucketSetValue)
                selectivity = _categorical_value_selectivity(space, value)
            elif isinstance(space, OrderedRangeParameterSpace):
                selectivity = _ordered_range_selectivity(
                    space,
                    _ordered_range_indices(space, value),
                )
            else:
                index = _axis_profile_selectivity_index(_axis_profiles(problem, axis))
                selectivity = index.get(
                    value,
                    problem.prior.marginal_center,
                )
        else:
            selectivity = problem.prior.marginal_center
        selectivities[space.hole_id] = selectivity
    hole_ids = frozenset(selectivities)
    expressions = _predicate_expressions_for_holes(problem.cover_root, hole_ids)
    if not expressions:
        result = 1.0
        for selectivity in selectivities.values():
            result *= selectivity
        return min(max(result, 0.0), 1.0)
    result = 1.0
    for expression in expressions:
        result *= _predicted_expr_selectivity(expression, selectivities)
    return min(max(result, 0.0), 1.0)


def _predicted_expr_selectivity(
    expression: Expr,
    selectivities: Mapping[HoleId, float],
) -> float:
    relevant = _expr_hole_ids(expression) & frozenset(selectivities)
    if not relevant:
        return 1.0
    if isinstance(expression, BinaryOp) and expression.operator in {
        BinOp.AND,
        BinOp.OR,
    }:
        left = _predicted_expr_selectivity(expression.left, selectivities)
        right = _predicted_expr_selectivity(expression.right, selectivities)
        if expression.operator is BinOp.AND:
            return left * right
        return 1.0 - ((1.0 - left) * (1.0 - right))
    result = 1.0
    for hole_id in sorted(relevant):
        result *= selectivities[hole_id]
    return result


def _probe_unobserved_enumerated_candidate(
    problem: PredicateEvolutionProblem,
    state: PredicateEvolutionState,
) -> PredicateEvolutionCandidate | None:
    if any(
        not isinstance(space, EnumeratedParameterSpace)
        for space in problem.predicate_domains
    ):
        return None
    spaces = tuple(
        space
        for space in problem.predicate_domains
        if isinstance(space, EnumeratedParameterSpace)
    )
    sizes = tuple(len(space.values) for space in spaces)
    product_size = _finite_product_size(sizes)
    if product_size <= len(state.observed_candidates):
        return None
    for linear_index in range(product_size):
        candidate = _enumerated_candidate_from_linear_index(linear_index, spaces)
        if candidate not in state.observed_candidates:
            return candidate
    return None


def _enumerated_candidate_from_linear_index(
    index: int,
    spaces: tuple[EnumeratedParameterSpace, ...],
) -> PredicateEvolutionCandidate:
    values: list[ParameterValue] = [None for _space in spaces]
    remaining = index
    for position in range(len(spaces) - 1, -1, -1):
        space = spaces[position]
        value_index = remaining % len(space.values)
        values[position] = space.values[value_index]
        remaining //= len(space.values)
    return tuple(values)


def _finite_product_size(sizes: tuple[int, ...]) -> int:
    result = 1
    for size in sizes:
        result *= max(size, 0)
    return result


def _predicate_beam_limit(axis_count: int) -> int:
    if axis_count <= 0:
        return 1
    return min(max(24, axis_count * 12), 96)


def _append_unique_candidate(
    candidates: list[PredicateEvolutionCandidate],
    candidate: PredicateEvolutionCandidate,
    *,
    limit: int,
) -> None:
    if len(candidates) < limit and candidate not in candidates:
        candidates.append(candidate)


def _append_unique_value(
    values: list[ParameterValue],
    value: ParameterValue,
    *,
    limit: int,
) -> None:
    if len(values) < limit and value not in values:
        values.append(value)


def _append_unique_index_set(
    values: list[frozenset[int]],
    value: frozenset[int],
) -> None:
    if value and value not in values:
        values.append(value)


def _parameter_value_key(value: ParameterValue) -> str:
    return stable_signature("predicate_fragment", {"value": value})


def _categorical_value_selectivity(
    space: CategoricalSetParameterSpace,
    value: BucketSetValue,
) -> float:
    counts = {atom.label: atom.local_count for atom in space.atoms}
    if space.owner_row_count <= 0:
        return 0.0
    count = sum(counts[label] for label in value.labels)
    return min(max(count / float(space.owner_row_count), 0.0), 1.0)


def _validate_categorical_set_value(
    space: CategoricalSetParameterSpace,
    value: ParameterValue,
) -> None:
    if not isinstance(value, BucketSetValue) or not value.labels:
        raise ContractViolation(
            "categorical set candidate requires a non-empty BucketSetValue"
        )
    atom_order = {atom.label: index for index, atom in enumerate(space.atoms)}
    try:
        indices = tuple(atom_order[label] for label in value.labels)
    except KeyError as exc:
        raise ContractViolation(
            "categorical set candidate contains label outside parameter space"
        ) from exc
    if indices != tuple(sorted(set(indices))):
        raise ContractViolation(
            "categorical set candidate labels must use canonical atom order"
        )
    if len(indices) > space.max_size:
        raise ContractViolation(
            "categorical set candidate exceeds parameter space max size"
        )


def _stable_seed(payload: str) -> int:
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


__all__ = [
    "GuidedPredicateEvolutionStrategy",
    "PredicateAxisValueProfile",
    "PredicateEvolutionCandidate",
    "PredicateEvolutionFeedback",
    "PredicateEvolutionPrior",
    "PredicateEvolutionProblem",
    "PredicateEvolutionScore",
    "PredicateEvolutionState",
    "PredicateEvolutionStrategy",
    "StandardPredicateEvolutionStrategy",
    "predicate_evolution_strategy",
    "predicate_evolution_values",
    "predicate_marginal_band",
]
