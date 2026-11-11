from __future__ import annotations

import enum
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, TypeVar

import yaml

from griffin.data.generation_request import (
    VectorRegionAssignmentBackend,
    VectorRegionClusteringBackend,
)
from griffin.infra.errors import ContractViolation
from griffin.query.shape import PredicateOperator
from .contract import (
    validate_distribution_contract,
    validate_relational_logical_type,
    validate_vector_contract,
)
from .models import (
    ConstraintConfig,
    ConstraintMetric,
    DataJobConfig,
    DataModelKind,
    DatasetOutputFormat,
    Difficulty,
    DistributionConfig,
    DistributionKind,
    JobConfig,
    JobType,
    IntValueDomainConfig,
    CategoricalAssociationBlockConfig,
    NumericCorrelationBlockConfig,
    NumericCorrelationPairConfig,
    PredicateCountRangeConfig,
    PredicateEvolutionStrategyKind,
    QueryFamily,
    QueryFormConfig,
    QueryJobConfig,
    QueryOutputConfig,
    QueryOutputKind,
    QuerySpecConfig,
    RelationalColumnConfig,
    RelationalDataJobConfig,
    RelationalDialect,
    RelationalDataSpecConfig,
    RelationalTableConfig,
    SearchConfig,
    VectorColumnSpec,
    VectorAttributeCouplingConfig,
    DEFAULT_QUERY_MAX_ATTEMPTS_PER_QUERY,
)

EnumT = TypeVar("EnumT", bound=enum.Enum)

_DATA_TOP_LEVEL_KEYS = frozenset(
    {"type", "data_model", "seed", "output_dir", "output_format", "dialect", "spec"}
)
_RELATIONAL_SPEC_KEYS = frozenset({"tables"})
_TABLE_KEYS = frozenset(
    {
        "name",
        "rows",
        "columns",
        "numeric_correlation_blocks",
        "categorical_association_blocks",
        "vector_attribute_couplings",
    }
)
_COLUMN_KEYS = frozenset({"name", "type", "primary_key", "distribution", "vector_spec"})
_VECTOR_SPEC_KEYS = frozenset(
    {
        "dim",
        "metric",
        "normalize",
        "cluster_count",
        "cluster_size_distribution",
        "cluster_separation",
        "cluster_spread",
        "vector_length_variation",
        "region_count",
        "region_assignment_backend",
        "region_clustering_backend",
    }
)


def _default_vector_normalize(metric: str, *, field: str) -> str:
    normalized = metric.strip().lower()
    if normalized == "cosine":
        return "l2"
    if normalized == "l2":
        return "none"
    raise ContractViolation(f"{field}.metric must be one of: cosine, l2")
_CORRELATION_BLOCK_KEYS = frozenset({"name", "columns", "pairs"})
_CATEGORICAL_ASSOCIATION_KEYS = frozenset({"name", "columns", "target_cramers_v"})
_VECTOR_ATTRIBUTE_COUPLING_KEYS = frozenset({"name", "vector", "attribute", "target_rho", "max_groups"})
_QUERY_TOP_LEVEL_KEYS = frozenset({"type", "seed", "input_dir", "output_dir", "spec"})
_QUERY_SPEC_KEYS = frozenset({"family", "count", "target_tables", "form", "constraints", "output", "search"})
_QUERY_FORM_KEYS = frozenset(
    {
        "kind",
        "predicate_count",
        "operators",
        "predicate_fields",
        "max_in_values",
        "difficulty",
    }
)
_QUERY_OUTPUT_KEYS = frozenset({"kind", "top_k"})
_QUERY_CONSTRAINT_KEYS = frozenset({"metric", "min", "max"})
_QUERY_SEARCH_KEYS = frozenset(
    {"max_attempts_per_query", "predicate_evolution_strategy", "best_so_far_fallback"}
)
_DISTRIBUTION_KEYS = frozenset(
    {
        "type",
        "min",
        "max",
        "mean",
        "sigma",
        "values",
        "weights",
        "ndv",
        "s",
        "value_domain",
    }
)
_INT_ZIPF_VALUE_DOMAIN_KEYS = frozenset({"min", "max"})
def load_job_config(path: str | Path) -> JobConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    payload = _decode_payload(text=text, suffix=config_path.suffix.lower())
    config = _load_job_config_payload(payload)
    return _resolve_corpus_paths(config_path, config)


def load_job_config_payload(payload: Mapping[str, Any]) -> JobConfig:
    return _load_job_config_payload(payload)


def _load_job_config_payload(payload: Mapping[str, Any]) -> JobConfig:
    task_type = str(payload.get("type", "")).strip().lower()
    if task_type == JobType.DATA.value:
        return _load_data_job(payload)
    if task_type == JobType.QUERY.value:
        return _load_query_job(payload)
    raise ContractViolation("config must declare type: data|query")


def _decode_payload(*, text: str, suffix: str) -> Mapping[str, Any]:
    if suffix == ".json":
        raw = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        raw = yaml.safe_load(text)
    else:
        raise ContractViolation(f"unsupported config format: {suffix or '<unknown>'}")
    if not isinstance(raw, dict):
        raise ContractViolation("config root must be a mapping")
    return raw


def _load_data_job(payload: Mapping[str, Any]) -> DataJobConfig:
    _reject_unknown_keys(payload, allowed=_DATA_TOP_LEVEL_KEYS, field="data config")
    data_model = _enum(DataModelKind, payload.get("data_model"), field="data_model")
    spec_raw = payload.get("spec")
    if not isinstance(spec_raw, Mapping):
        raise ContractViolation("spec must be a mapping")
    spec_payload = spec_raw
    output_format = _enum(
        DatasetOutputFormat,
        payload.get("output_format", DatasetOutputFormat.PARQUET.value),
        field="output_format",
    )
    return RelationalDataJobConfig(
        type=JobType.DATA,
        data_model=data_model,
        seed=_int(payload.get("seed", 42), field="seed"),
        output_dir=_required_text(payload.get("output_dir"), field="output_dir"),
        output_format=output_format,
        dialect=_enum(RelationalDialect, payload.get("dialect"), field="dialect"),
        spec=_load_relational_data_spec(spec_payload),
    )


def _load_query_job(payload: Mapping[str, Any]) -> QueryJobConfig:
    _reject_unknown_keys(payload, allowed=_QUERY_TOP_LEVEL_KEYS, field="query config")
    spec_payload = _mapping(payload, "spec")
    return QueryJobConfig(
        type=JobType.QUERY,
        seed=_int(payload.get("seed"), field="seed"),
        input_dir=_required_text(payload.get("input_dir"), field="input_dir"),
        output_dir=_optional_text(payload.get("output_dir")),
        spec=_load_query_spec(spec_payload),
    )


def _resolve_corpus_paths(config_path: Path, config: JobConfig) -> JobConfig:
    if not isinstance(config, QueryJobConfig):
        return config
    input_dir = Path(config.input_dir)
    candidate_data_files = [config_path.with_name("data.yaml")]
    if "high_dim" in config_path.stem:
        candidate_data_files.insert(0, config_path.with_name("data_high_dim.yaml"))
    sibling_data = next((path for path in candidate_data_files if path.exists()), None)
    if sibling_data is None:
        return config
    if input_dir.exists() and "outputs/examples/" not in config.input_dir:
        return config
    data_payload = _decode_payload(text=sibling_data.read_text(encoding="utf-8"), suffix=sibling_data.suffix.lower())
    output_dir = _optional_text(data_payload.get("output_dir"))
    if output_dir is None:
        return config
    return QueryJobConfig(
        type=config.type,
        seed=config.seed,
        input_dir=output_dir,
        spec=config.spec,
        output_dir=config.output_dir,
    )


def _load_relational_data_spec(payload: Mapping[str, Any]) -> RelationalDataSpecConfig:
    _reject_unknown_keys(payload, allowed=_RELATIONAL_SPEC_KEYS, field="spec")
    tables = tuple(_load_relational_table(item) for item in _sequence(payload, "tables"))
    if not tables:
        raise ContractViolation("relational data config requires spec.tables")
    return RelationalDataSpecConfig(tables=tables)


def _load_relational_table(raw: Any) -> RelationalTableConfig:
    payload = _ensure_mapping(raw, field="table")
    _reject_unknown_keys(payload, allowed=_TABLE_KEYS, field="table")
    return RelationalTableConfig(
        name=_required_text(payload.get("name"), field="table.name"),
        rows=_int(payload.get("rows"), field="table.rows"),
        columns=tuple(_load_relational_column(item) for item in _sequence(payload, "columns")),
        numeric_correlation_blocks=tuple(
            _load_numeric_correlation_block(item)
            for item in _sequence(payload, "numeric_correlation_blocks", default=())
        ),
        categorical_association_blocks=tuple(
            _load_categorical_association_block(item)
            for item in _sequence(payload, "categorical_association_blocks", default=())
        ),
        vector_attribute_couplings=tuple(
            _load_vector_attribute_coupling(item)
            for item in _sequence(payload, "vector_attribute_couplings", default=())
        ),
    )


def _load_relational_column(raw: Any) -> RelationalColumnConfig:
    payload = _ensure_mapping(raw, field="column")
    _reject_unknown_keys(payload, allowed=_COLUMN_KEYS, field="column")
    distribution = payload.get("distribution")
    if distribution is not None and not isinstance(distribution, Mapping):
        raise ContractViolation("column.distribution must be a mapping")
    vector_spec = payload.get("vector_spec")
    if vector_spec is not None and not isinstance(vector_spec, Mapping):
        raise ContractViolation("column.vector_spec must be a mapping")
    type_name = _required_text(payload.get("type"), field="column.type")
    normalized_type = validate_relational_logical_type(type_name, field="column.type")
    primary_key = _bool(payload.get("primary_key", False), field="column.primary_key")
    if primary_key and "distribution" in payload:
        raise ContractViolation("column.distribution is not allowed when column.primary_key=true")
    if primary_key and "vector_spec" in payload:
        raise ContractViolation("column.vector_spec is not allowed when column.primary_key=true")
    if normalized_type == "vector":
        if primary_key:
            raise ContractViolation("column.primary_key is not allowed when column.type=vector")
        if distribution is not None:
            raise ContractViolation("column.distribution is not allowed when column.type=vector")
        if vector_spec is None:
            raise ContractViolation("column.vector_spec must be set when column.type=vector")
    elif vector_spec is not None:
        raise ContractViolation("column.vector_spec is only allowed when column.type=vector")
    return RelationalColumnConfig(
        name=_required_text(payload.get("name"), field="column.name"),
        type_name=type_name,
        primary_key=primary_key,
        distribution=_load_distribution(distribution) if distribution is not None else None,
        vector_spec=_load_vector_spec(vector_spec, field="column.vector_spec") if vector_spec is not None else None,
    )


def _load_vector_spec(raw: Mapping[str, Any], *, field: str) -> VectorColumnSpec:
    _reject_unknown_keys(raw, allowed=_VECTOR_SPEC_KEYS, field=field)
    metric = _required_text(raw.get("metric"), field=f"{field}.metric")
    normalize = _optional_text(raw.get("normalize"))
    if normalize is None:
        normalize = _default_vector_normalize(metric, field=field)
    spec = VectorColumnSpec(
        dim=_int(raw.get("dim"), field=f"{field}.dim"),
        metric=metric,
        normalize=normalize,
        cluster_count=_optional_int(raw.get("cluster_count")),
        cluster_size_distribution=_optional_text(raw.get("cluster_size_distribution")),
        cluster_separation=_optional_float(raw.get("cluster_separation")),
        cluster_spread=_optional_float(raw.get("cluster_spread")),
        vector_length_variation=_optional_text(raw.get("vector_length_variation")),
        region_count=_optional_int(raw.get("region_count")),
        region_assignment_backend=(
            VectorRegionAssignmentBackend.SKLEARN
            if raw.get("region_assignment_backend") is None
            else _enum(
                VectorRegionAssignmentBackend,
                raw.get("region_assignment_backend"),
                field=f"{field}.region_assignment_backend",
            )
        ),
        region_clustering_backend=(
            VectorRegionClusteringBackend.SKLEARN
            if raw.get("region_clustering_backend") is None
            else _enum(
                VectorRegionClusteringBackend,
                raw.get("region_clustering_backend"),
                field=f"{field}.region_clustering_backend",
            )
        ),
    )
    validate_vector_contract(spec, field=field)
    return spec

def _load_numeric_correlation_block(raw: Any) -> NumericCorrelationBlockConfig:
    payload = _ensure_mapping(raw, field="numeric_correlation_blocks[]")
    _reject_unknown_keys(
        payload,
        allowed=_CORRELATION_BLOCK_KEYS,
        field="numeric_correlation_blocks[]",
    )
    return NumericCorrelationBlockConfig(
        name=_required_text(payload.get("name"), field="numeric_correlation_blocks[].name"),
        columns=_text_tuple(payload.get("columns")),
        pairs=tuple(
            _load_numeric_correlation_pair(item, index=index)
            for index, item in enumerate(_sequence(payload, "pairs"))
        ),
    )


def _load_numeric_correlation_pair(raw: Any, *, index: int) -> NumericCorrelationPairConfig:
    field = f"numeric_correlation_blocks[].pairs[{index}]"
    if not isinstance(raw, list) or len(raw) != 3:
        raise ContractViolation(f"{field} must be [left, right, rho]")
    return NumericCorrelationPairConfig(
        left=_required_text(raw[0], field=f"{field}[0]"),
        right=_required_text(raw[1], field=f"{field}[1]"),
        rho=_float_with_range(
            raw[2],
            field=f"{field}[2]",
            minimum=-1.0,
            maximum=1.0,
        ),
    )


def _load_categorical_association_block(raw: Any) -> CategoricalAssociationBlockConfig:
    payload = _ensure_mapping(raw, field="categorical_association_blocks[]")
    _reject_unknown_keys(
        payload,
        allowed=_CATEGORICAL_ASSOCIATION_KEYS,
        field="categorical_association_blocks[]",
    )
    return CategoricalAssociationBlockConfig(
        name=_required_text(payload.get("name"), field="categorical_association_blocks[].name"),
        columns=_text_tuple(payload.get("columns")),
        target_cramers_v=_float_with_range(
            payload.get("target_cramers_v"),
            field="categorical_association_blocks[].target_cramers_v",
            minimum=0.0,
            maximum=1.0,
        ),
    )


def _load_vector_attribute_coupling(raw: Any) -> VectorAttributeCouplingConfig:
    payload = _ensure_mapping(raw, field="vector_attribute_couplings[]")
    _reject_unknown_keys(
        payload,
        allowed=_VECTOR_ATTRIBUTE_COUPLING_KEYS,
        field="vector_attribute_couplings[]",
    )
    target_rho = _float_with_range(
        payload.get("target_rho"),
        field="vector_attribute_couplings[].target_rho",
        minimum=0.0,
        maximum=1.0,
    )
    max_groups = _optional_int(payload.get("max_groups"))
    return VectorAttributeCouplingConfig(
        name=_required_text(payload.get("name"), field="vector_attribute_couplings[].name"),
        vector=_required_text(payload.get("vector"), field="vector_attribute_couplings[].vector"),
        attribute=_required_text(payload.get("attribute"), field="vector_attribute_couplings[].attribute"),
        target_rho=target_rho,
        max_groups=max_groups,
    )


def _load_distribution(raw: Mapping[str, Any]) -> DistributionConfig:
    _reject_unknown_keys(raw, allowed=_DISTRIBUTION_KEYS, field="distribution")
    kind = _enum(DistributionKind, raw.get("type"), field="distribution.type")
    allowed_fields = {
        DistributionKind.UNIFORM: {"type", "min", "max", "ndv", "value_domain"},
        DistributionKind.LOGNORMAL: {"type", "mean", "sigma"},
        DistributionKind.CATEGORICAL: {"type", "values", "weights"},
        DistributionKind.ZIPF: {"type", "ndv", "s", "value_domain"},
    }[kind]
    for key in sorted(raw):
        if key not in allowed_fields:
            raise ContractViolation(f"distribution.{key} is not allowed for {kind.value}")
    min_value = _optional_float(raw.get("min"))
    max_value = _optional_float(raw.get("max"))
    mean = _optional_float(raw.get("mean"))
    sigma = _optional_float(raw.get("sigma"))
    values = _text_tuple(raw.get("values"))
    weights = _float_tuple(raw.get("weights"))
    ndv = _optional_int(raw.get("ndv"))
    s = _optional_float(raw.get("s"))
    value_domain_raw = raw.get("value_domain")
    if value_domain_raw is not None and not isinstance(value_domain_raw, Mapping):
        raise ContractViolation("distribution.value_domain must be a mapping")
    distribution = DistributionConfig(
        kind=kind,
        min=min_value,
        max=max_value,
        mean=mean,
        sigma=sigma,
        values=values,
        weights=weights,
        ndv=ndv,
        s=s,
        value_domain=(
            _load_int_value_domain(value_domain_raw)
            if isinstance(value_domain_raw, Mapping)
            else None
        ),
    )
    validate_distribution_contract(distribution, field="distribution")
    return distribution


def _load_int_value_domain(raw: Mapping[str, Any]) -> IntValueDomainConfig:
    _reject_unknown_keys(
        raw,
        allowed=_INT_ZIPF_VALUE_DOMAIN_KEYS,
        field="distribution.value_domain",
    )
    return IntValueDomainConfig(
        min=_strict_int(raw.get("min"), field="distribution.value_domain.min"),
        max=_strict_int(raw.get("max"), field="distribution.value_domain.max"),
    )


def _load_query_spec(payload: Mapping[str, Any]) -> QuerySpecConfig:
    _reject_unknown_keys(payload, allowed=_QUERY_SPEC_KEYS, field="query spec")
    form_payload = _mapping(payload.get("form"), "spec.form")
    output_payload = payload.get("output")
    if output_payload is not None and not isinstance(output_payload, Mapping):
        raise ContractViolation("spec.output must be a mapping")
    search_payload = payload.get("search")
    family = _enum(QueryFamily, payload.get("family"), field="spec.family")
    return QuerySpecConfig(
        family=family,
        count=_int(payload.get("count"), field="spec.count"),
        target_tables=_text_tuple(payload.get("target_tables")),
        form=_load_query_form(form_payload),
        constraints=tuple(_load_constraint(item) for item in _sequence(payload, "constraints", default=())),
        output=_load_query_output(output_payload) if isinstance(output_payload, Mapping) else None,
        search=_load_search(search_payload) if isinstance(search_payload, Mapping) else None,
    )


def _load_query_form(raw: Mapping[str, Any]) -> QueryFormConfig:
    _reject_unknown_keys(raw, allowed=_QUERY_FORM_KEYS, field="spec.form")
    difficulty = raw.get("difficulty")
    return QueryFormConfig(
        kind=_enum(QueryFamily, raw.get("kind"), field="spec.form.kind"),
        predicate_count=_load_predicate_count(raw.get("predicate_count")),
        operators=tuple(_enum(PredicateOperator, item, field="spec.form.operators[]") for item in _sequence(raw, "operators", default=())),
        predicate_fields=_load_predicate_fields(raw.get("predicate_fields")),
        max_in_values=_optional_int(raw.get("max_in_values")),
        difficulty=_enum(Difficulty, difficulty, field="spec.form.difficulty") if difficulty is not None else None,
    )


def _load_predicate_count(raw: Any) -> PredicateCountRangeConfig | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or isinstance(raw, int):
        raise ContractViolation("spec.form.predicate_count must be an object with min and max")
    payload = _ensure_mapping(raw, field="spec.form.predicate_count")
    _reject_unknown_keys(payload, allowed=frozenset({"min", "max"}), field="spec.form.predicate_count")
    return PredicateCountRangeConfig(
        min=_int(payload.get("min"), field="spec.form.predicate_count.min"),
        max=_int(payload.get("max"), field="spec.form.predicate_count.max"),
    )


def _load_predicate_fields(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    result: list[str] = []
    for index, item in enumerate(_sequence({"predicate_fields": raw}, "predicate_fields", default=())):
        field_label = f"spec.form.predicate_fields[{index}]"
        if not isinstance(item, str):
            raise ContractViolation(f"{field_label} must be a field name string")
        result.append(item)
    return tuple(result)


def _load_constraint(raw: Any) -> ConstraintConfig:
    payload = _ensure_mapping(raw, field="constraints[]")
    _reject_unknown_keys(payload, allowed=_QUERY_CONSTRAINT_KEYS, field="constraints[]")
    return ConstraintConfig(
        metric=_enum(ConstraintMetric, payload.get("metric"), field="constraints[].metric"),
        min=float(payload.get("min") or 0.0),
        max=float(payload.get("max") or 0.0),
    )


def _load_query_output(raw: Mapping[str, Any]) -> QueryOutputConfig:
    _reject_unknown_keys(raw, allowed=_QUERY_OUTPUT_KEYS, field="spec.output")
    kind = _enum(QueryOutputKind, raw.get("kind"), field="spec.output.kind")
    top_k = _optional_int(raw.get("top_k"))
    if top_k is None or top_k < 1:
        raise ContractViolation("spec.output.top_k must be >= 1 when spec.output.kind=top_k")
    return QueryOutputConfig(
        kind=kind,
        top_k=top_k,
    )


def _load_search(raw: Mapping[str, Any]) -> SearchConfig:
    _reject_unknown_keys(raw, allowed=_QUERY_SEARCH_KEYS, field="spec.search")
    attempts = _int(
        raw.get("max_attempts_per_query", DEFAULT_QUERY_MAX_ATTEMPTS_PER_QUERY),
        field="spec.search.max_attempts_per_query",
    )
    if attempts <= 0:
        raise ContractViolation("spec.search.max_attempts_per_query must be positive")
    strategy_raw = raw.get("predicate_evolution_strategy")
    strategy = (
        None
        if strategy_raw is None
        else _enum(
            PredicateEvolutionStrategyKind,
            strategy_raw,
            field="spec.search.predicate_evolution_strategy",
        )
    )
    best_so_far_fallback = bool(raw.get("best_so_far_fallback", False))
    return SearchConfig(
        max_attempts_per_query=attempts,
        predicate_evolution_strategy=strategy,
        best_so_far_fallback=best_so_far_fallback,
    )


def _mapping(payload: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ContractViolation(f"{field} must be a mapping")
    nested = payload.get(field)
    if isinstance(nested, Mapping):
        return nested
    return payload


def _ensure_mapping(payload: Any, *, field: str) -> Mapping[str, Any]:
    return _mapping(payload, field)


def _sequence(payload: Mapping[str, Any], field: str, default: Iterable[Any] | None = None) -> tuple[Any, ...]:
    raw = payload.get(field)
    if raw is None:
        return tuple(default or ())
    if not isinstance(raw, list):
        raise ContractViolation(f"{field} must be a list")
    return tuple(raw)


def _required_text(value: Any, *, field: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ContractViolation(f"{field} must be non-empty")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ContractViolation(f"{field} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ContractViolation(f"{field} must be an integer") from exc


def _strict_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractViolation(f"{field} must be an integer")
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ContractViolation(f"expected integer, got {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ContractViolation(f"expected integer, got {value!r}") from exc


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ContractViolation(f"expected float, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ContractViolation(f"expected float, got {value!r}") from exc


def _bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ContractViolation(f"{field} must be a boolean")


def _float_with_range(
    value: Any,
    *,
    field: str,
    minimum: float,
    maximum: float,
) -> float:
    number = _optional_float(value)
    if number is None:
        raise ContractViolation(f"{field} must be set")
    if not minimum <= number <= maximum:
        raise ContractViolation(f"{field} must be within [{minimum:g},{maximum:g}]")
    return number


def _text_tuple(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ContractViolation("expected list of strings")
    return tuple(_required_text(item, field="list[]") for item in raw)


def _float_tuple(raw: Any) -> tuple[float, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ContractViolation("expected list of floats")
    return tuple(float(item) for item in raw)


def _reject_unknown_keys(
    payload: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    field: str,
) -> None:
    unknown = sorted(str(key) for key in payload.keys() if key not in allowed)
    if unknown:
        raise ContractViolation(f"{field} contains unsupported keys: {', '.join(unknown)}")


def _enum(enum_type: type[EnumT], value: Any, *, field: str) -> EnumT:
    text = _optional_text(value)
    if text is None:
        raise ContractViolation(f"{field} must be non-empty")
    try:
        return enum_type(text)
    except ValueError as exc:
        allowed = ", ".join(str(item.value) for item in enum_type)
        raise ContractViolation(f"{field} must be one of: {allowed}") from exc


__all__ = ["load_job_config"]
