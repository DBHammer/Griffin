"""Materialize typed dataset relations from generated data."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from griffin.dataset.relations.base import (
    CorrelationRelation,
    CouplingRelation,
    Relation,
)
from griffin.infra.errors import ContractViolation
from griffin.model_ids import FieldId, RelationId


def _coerce_numeric(values: list[Any]) -> list[float]:
    out: list[float] = []
    for v in values:
        if v is None:
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def _coerce_numeric_pairs(a_values: list[Any], b_values: list[Any]) -> tuple[list[float], list[float]]:
    a_out: list[float] = []
    b_out: list[float] = []
    for a_value, b_value in zip(a_values, b_values):
        if a_value is None or b_value is None:
            continue
        try:
            a_num = float(a_value)
            b_num = float(b_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(a_num) or not math.isfinite(b_num):
            continue
        a_out.append(a_num)
        b_out.append(b_num)
    return a_out, b_out


def materialize_correlation(
    name: str,
    a: FieldId,
    b: FieldId,
    columns: Mapping[str, list[Any]],
) -> Relation:
    """Pearson correlation between two numeric columns.

    Returns a CorrelationRelation with the coefficient inline (no on-disk file
    needed — a single float fits in the manifest).
    """
    a_vals, b_vals = _coerce_numeric_pairs(columns[a.name], columns[b.name])
    if len(a_vals) < 2:
        coeff = 0.0
    else:
        a_arr = np.asarray(a_vals, dtype=float)
        b_arr = np.asarray(b_vals, dtype=float)
        var_a = float(np.var(a_arr))
        var_b = float(np.var(b_arr))
        if var_a == 0.0 or var_b == 0.0:
            coeff = 0.0
        else:
            coeff = float(np.corrcoef(a_arr, b_arr)[0, 1])
            if not math.isfinite(coeff):
                coeff = 0.0
    return CorrelationRelation(
        id=RelationId(kind="correlation", name=name),
        field_a=a,
        field_b=b,
        coefficient=coeff,
    )


def materialize_coupling(
    name: str,
    fields: tuple[FieldId, ...],
    columns: Mapping[str, list[Any]],
) -> Relation:
    """Coupling strength = average pairwise |correlation| over numeric columns.

    Strength in [0, 1]; 0 = independent, 1 = perfectly coupled.
    """
    if len(fields) < 2:
        raise ContractViolation("coupling requires at least two fields")
    cols = [_coerce_numeric(columns[f.name]) for f in fields]
    n = min(len(c) for c in cols)
    if n < 2:
        return CouplingRelation(
            id=RelationId(kind="coupling", name=name),
            fields=fields,
            coupling_strength=0.0,
        )
    pair_correlations: list[float] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i][:n], cols[j][:n]
            ma, mb = sum(a) / n, sum(b) / n
            cov = sum((a[k] - ma) * (b[k] - mb) for k in range(n))
            va = sum((x - ma) ** 2 for x in a)
            vb = sum((x - mb) ** 2 for x in b)
            denom = math.sqrt(va * vb)
            if denom > 0:
                pair_correlations.append(abs(cov / denom))
    strength = sum(pair_correlations) / len(pair_correlations) if pair_correlations else 0.0
    return CouplingRelation(
        id=RelationId(kind="coupling", name=name),
        fields=fields,
        coupling_strength=strength,
    )


__all__ = [
    "materialize_correlation",
    "materialize_coupling",
]
