"""Parameter solver session and built-in strategy router."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from griffin.infra.errors import ContractViolation
from griffin.query.vector_search import VectorParameterSearchStrategy
from griffin.query.search_core import (
    BindingProblem,
    CandidateProposal,
    CandidateProposalKind,
    FixedBindingStrategy,
    ParameterFeedback,
    ParameterSolverKind,
    ParameterSearchState,
    ParameterSearchStrategy,
    PredicateCoverFillStrategy,
)


@dataclass
class ParameterSolverSession:
    problem: BindingProblem
    strategy: ParameterSearchStrategy
    state: ParameterSearchState

    def next(self) -> CandidateProposal:
        if self.state.parameter_attempts >= self.problem.budget:
            return CandidateProposal.no_proposal("parameter budget exhausted")
        proposal = self.strategy.next(self.problem, self.state)
        if proposal.kind is CandidateProposalKind.NO_PROPOSAL:
            return proposal
        if proposal.binding is None:
            return CandidateProposal.no_proposal("strategy returned candidate without binding")
        self.state.attempted_bindings.add(proposal.binding.binding_signature)
        self.state.parameter_attempts += 1
        return proposal

    def observe(self, feedback: ParameterFeedback) -> None:
        self.state = self.strategy.observe(self.problem, self.state, feedback)


class ParameterSolver:
    def __init__(
        self,
        strategy: ParameterSearchStrategy | None = None,
    ) -> None:
        self._strategy = strategy

    def start(
        self,
        problem: BindingProblem,
        *,
        state: ParameterSearchState | None = None,
    ) -> ParameterSolverSession:
        return ParameterSolverSession(
            problem=problem,
            strategy=self._strategy or route_strategy(problem),
            state=state or ParameterSearchState(),
        )

    def solve(
        self,
        problem: BindingProblem,
        *,
        state: ParameterSearchState | None = None,
    ) -> Iterator[CandidateProposal]:
        session = self.start(problem, state=state)
        while True:
            proposal = session.next()
            yield proposal
            if proposal.kind is CandidateProposalKind.NO_PROPOSAL:
                return


def route_strategy(problem: BindingProblem) -> ParameterSearchStrategy:
    if problem.solver_kind is ParameterSolverKind.VECTOR:
        return VectorParameterSearchStrategy()
    if problem.solver_kind is ParameterSolverKind.PREDICATE_COVER:
        return PredicateCoverFillStrategy()
    if problem.solver_kind is ParameterSolverKind.FIXED:
        return FixedBindingStrategy()
    raise ContractViolation(f"unsupported parameter search kind {problem.solver_kind.value}")


__all__ = [
    "BindingProblem",
    "CandidateProposal",
    "CandidateProposalKind",
    "FixedBindingStrategy",
    "ParameterFeedback",
    "ParameterSearchState",
    "ParameterSearchStrategy",
    "ParameterSolver",
    "ParameterSolverSession",
    "PredicateCoverFillStrategy",
    "VectorParameterSearchStrategy",
    "route_strategy",
]
