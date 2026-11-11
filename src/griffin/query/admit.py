"""Accepted query binding contract."""

from __future__ import annotations

import msgspec

from griffin.infra.errors import ContractViolation
from griffin.query.evaluate import CandidateJudgement, SearchEvidence
from griffin.query.model import MaterializedQuery


class AcceptedBinding(msgspec.Struct, frozen=True, omit_defaults=True):
    materialized: MaterializedQuery
    evidence: SearchEvidence
    judgement: CandidateJudgement


def accepted_binding(
    *,
    materialized: MaterializedQuery,
    evidence: SearchEvidence,
    judgement: CandidateJudgement,
) -> AcceptedBinding:
    if not judgement.accepted:
        raise ContractViolation("accepted candidate requires an accepted judgement")
    return AcceptedBinding(
        materialized=materialized,
        evidence=evidence,
        judgement=judgement,
    )


__all__ = [
    "AcceptedBinding",
    "accepted_binding",
]
