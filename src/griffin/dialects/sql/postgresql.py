"""PostgreSQL renderer — closed visitor over RelationalAst.

PostgreSQL-specific bits:
    - Vector distance via pgvector operators (<=> cosine, <-> l2)
    - Vector literal needs ::vector cast
    - Param style: %s
"""

from __future__ import annotations

import json

from griffin.dialects._param_builder import ParamBuilder, ParamStyle
from griffin.dialects.capabilities import DialectCapabilities, SurfaceFeature
from griffin.ir.data_type import VectorMetric
from griffin.ir.expr import VectorDistance

from ._renderer_base import SqlRendererBase


POSTGRESQL_CAPABILITIES = DialectCapabilities(
    dialect_name="postgresql",
    supported=(
        SurfaceFeature.SCALAR_BTREE_FILTER,
        SurfaceFeature.SCALAR_RANGE_FILTER,
        SurfaceFeature.VECTOR_EXACT_SCAN,
        SurfaceFeature.AGGREGATION_GROUPBY,
    ),
)


class PostgreSQLRenderer(SqlRendererBase):
    name = "postgresql"
    capabilities = POSTGRESQL_CAPABILITIES

    def _param_style(self) -> ParamStyle:
        return ParamStyle.PERCENT_S

    def _render_vector_distance(self, node: VectorDistance, params: ParamBuilder) -> str:
        col = f"{node.variable}.{node.field.name}"
        qv = json.loads(node.query_vector_literal.value_json)
        # Format vector as PG literal '[…]' so the cast works.
        qv_str = "[" + ",".join(repr(float(x)) for x in qv) + "]"
        placeholder = params.add(qv_str)
        op = {
            VectorMetric.COSINE: "<=>",
            VectorMetric.L2: "<->",
        }[node.metric]
        return f"({col} {op} {placeholder}::vector)"

    def _format_vector_literal(self, value_json: str, params: ParamBuilder) -> str:
        # Render '[…]'::vector form so PG can cast it.
        vals = json.loads(value_json)
        text = "[" + ",".join(repr(float(x)) for x in vals) + "]"
        return f"{params.add(text)}::vector"


__all__ = [
    "PostgreSQLRenderer",
    "POSTGRESQL_CAPABILITIES",
]
