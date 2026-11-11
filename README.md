<div align="center">
  <img src="assets/logo.svg" width="560" alt="Griffin logo">
</div>

<h3 align="center">Customizable attribute--vector data and FANNS query workload generation</h3>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="Artifact" src="https://img.shields.io/badge/Artifact-ICDE%202027-7B4FBF">
  <img alt="Backends" src="https://img.shields.io/badge/Backends-faiss%20%7C%20numpy%20%7C%20scikit--learn-2F6EC4">
  <img alt="CLI" src="https://img.shields.io/badge/CLI-griffin-1F2D44">
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#configuration">Configuration</a> ·
  <a href="#outputs">Outputs</a> ·
  <a href="#repository-layout">Layout</a>
</p>

---

**Griffin** is the generator behind the ICDE submission *Griffin: Customizable
Attribute--Vector Data and Query Workload Generation for Filtered ANNS Benchmarking*.
It produces, from a single YAML file per stage, (i) attribute--vector datasets whose
cross-modal dependency `rho` is user-controlled and (ii) filtered-ANNS query workloads
whose selectivity and query--filter correlation (GLS) hit specified targets.

No external DBMS is needed: datasets are written to Parquet/CSV plus metadata, and
queries are validated against an exact FAISS index built from the generated vectors.

## Why Griffin

| Stage | What a benchmark needs | What Griffin does |
| --- | --- | --- |
| Data | Datasets with the same schema but different attribute--vector dependencies, so that performance is not tied to one layout | Attribute groups + vector regions, joined by a capacity-constrained assignment that reaches a target `rho` |
| Query | Workloads whose hardness is an input rather than an outcome | Deterministic key-based surrogates score candidates; predicate values and query vectors are searched until the target selectivity and GLS are met |

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Requirements: Python >= 3.11 and the packages in `pyproject.toml`
(`faiss-cpu`, `numpy`, `scikit-learn`, `ortools`, `pandas`, `pyarrow`, `pyyaml`,
`scipy`, `msgspec`).

## Quick start

```bash
# 1) attribute--vector dataset: 10K rows, 96-d vectors, 24 regions, target rho = 0.55
griffin --config configs/data_10k.yaml

# 2) 5 filtered-ANN queries on top of it
griffin --config configs/query_filtered_ann.yaml
```

Without installing, use the module form:

```bash
PYTHONPATH=src python -m griffin --config configs/data_10k.yaml
```

A run takes a few seconds on a laptop. The data stage logs the achieved dependency:

```text
[data] 耦合质量：events.embedding~category target_rho=0.550, observed_rho=0.554, relative_error=0.79%
```

Scaling up to the paper settings is a matter of raising `rows`, `dim`,
`cluster_count`, `region_count`, and `spec.count` (paper: N=1M rows, d=192, R=32,
workloads of about 100 queries).

## How it works

```text
data config ──► attribute groups + vector regions ──► capacity-constrained assignment
                                                              │
                                                              ▼
                                      dataset  (parquet + metadata + FAISS index)

query config ─► surrogate scoring ──► predicate instantiation ──► region-guided query vector
                                                              │
                                                              ▼
                                       queries.sql + ground_truth.jsonl
```

## Configuration

### Data stage (`configs/data_10k.yaml`)

| Field | Meaning |
| --- | --- |
| `type`, `data_model` | task type (`data`) and data model (`relational`) |
| `seed` | RNG seed; identical seeds reproduce identical datasets |
| `output_dir` | output root |
| `output_format` | `parquet` or `csv` |
| `dialect` | SQL dialect for `schema.sql` (currently `postgresql`) |
| `spec.tables[].rows` | row count |
| `spec.tables[].columns[]` | `int`/`text`/`double` with a `distribution`, or `vector` with `vector_spec` |
| `vector_spec.dim`, `metric`, `normalize` | vector dimension, distance metric, normalization |
| `vector_spec.cluster_count`, `cluster_size_distribution`, `cluster_spread` | vector cluster structure |
| `vector_spec.region_count` | number of vector regions used for dependency control |
| `vector_attribute_couplings[].target_rho` | target attribute--vector dependency strength |

### Query stage (`configs/query_filtered_ann.yaml`)

| Field | Meaning |
| --- | --- |
| `input_dir` | dataset produced by the data stage |
| `spec.family` | `filtered_ann` |
| `spec.count` | number of queries to instantiate |
| `spec.form.predicate_count`, `operators`, `predicate_fields` | predicate template space |
| `spec.constraints[].metric/min/max` | target global selectivity range |
| `spec.output.kind`, `top_k` | ranked output and K |
| `spec.search.max_attempts_per_query` | instantiation budget |

## Outputs

```text
outputs/data_10k/
├── schema.sql
├── data/tables/events.parquet
└── metadata/
    ├── manifest.json
    ├── statistics/tables/events.json
    ├── deterministic/tables/events.json
    └── artifacts/vector_index/       # exact FAISS index for top-K ground truth

outputs/query_filtered_ann/
├── queries.sql                       # one executable statement per query
└── ground_truth.jsonl                # filtered_ann only; _rowid answers
```

## Repository layout

```text
.
├── assets/logo.svg
├── configs/
│   ├── data_10k.yaml
│   └── query_filtered_ann.yaml
├── src/griffin/                      # core package
│   ├── adapters/                     # CLI entry point
│   ├── config/                       # typed config, load, compile
│   ├── data/                         # generation, regions, assignment
│   ├── dataset/                      # dataset reader, metadata, statistics
│   ├── dialects/                     # SQL rendering
│   ├── ir/                           # relational IR
│   ├── query/                        # selectivity, GLS, search, ground truth
│   └── vector_hardness_core.py
└── pyproject.toml
```

## Paper mapping

| Paper | Code |
| --- | --- |
| Sec. 4, attribute--vector data generation | `src/griffin/data/` (`vector_regions.py`, `vector_coupling.py`) |
| Sec. 4, dependency control (assignment) | `src/griffin/data/entity_builders.py` |
| Sec. 5, surrogate-based query generation | `src/griffin/query/` (`predicate_cover_search.py`, `filtered_ann_band.py`, `vector_search.py`) |
| Metadata / dataset reader | `src/griffin/dataset/` |

## Technical report

The full version, including the proofs of the lemmas and theorems and the
complexity analysis of both generation stages, is available as
[`technical-report.pdf`](technical-report.pdf).

