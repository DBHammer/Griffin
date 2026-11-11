"""Vector region partitioning and layout materialization."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    pairwise_distances,
    pairwise_distances_argmin_min,
)

from griffin.data.generation_request import (
    VectorRegionAssignmentBackend,
    VectorRegionClusteringBackend,
)
from griffin.dataset.deterministic import RegionLayoutModel, RegionRange
from griffin.infra.errors import ContractViolation
from griffin.ir.data_type import VectorMetric
from griffin.vector_hardness_core import metric_matrix


_DEFAULT_KMEANS_MAX_ITERATIONS = 64
_DEFAULT_KMEANS_MAX_ATTEMPTS = 8
DEFAULT_KMEANS_MAX_POINTS_PER_CENTROID = 256
_REGION_NEIGHBOR_LIMIT = 64


@dataclass(frozen=True, slots=True)
class KMeansRegionConfig:
    max_iterations: int = _DEFAULT_KMEANS_MAX_ITERATIONS
    # None preserves the established 256-points-per-centroid sampling policy.
    max_points_per_centroid: int | None = None
    max_attempts: int = _DEFAULT_KMEANS_MAX_ATTEMPTS
    assignment_backend: VectorRegionAssignmentBackend = (
        VectorRegionAssignmentBackend.SKLEARN
    )
    clustering_backend: VectorRegionClusteringBackend = (
        VectorRegionClusteringBackend.SKLEARN
    )


@dataclass(frozen=True, slots=True)
class KMeansRegionEvidence:
    training_sample_count: int
    effective_max_points_per_centroid: int | None
    assignment_backend: VectorRegionAssignmentBackend
    clustering_backend: VectorRegionClusteringBackend
    attempts: int
    total_iterations: int
    kmeans_and_assignment_time_sec: float


@dataclass(frozen=True, slots=True)
class KMeansRegionResult:
    labels: np.ndarray
    centers: np.ndarray
    counts: np.ndarray
    assigned_squared_distances: np.ndarray
    inertia: float
    iterations: int
    evidence: KMeansRegionEvidence


@dataclass(frozen=True, slots=True)
class KMeansRegionAssignment:
    labels: np.ndarray
    squared_distances: np.ndarray
    counts: np.ndarray
    inertia: float


@dataclass(frozen=True, slots=True)
class KMeansCenterTraining:
    centers: np.ndarray
    iterations: int
    training_sample_count: int
    effective_max_points_per_centroid: int


def materialize_vector_region_layout(
    vectors: np.ndarray,
    *,
    vector_field_name: str,
    metric: VectorMetric,
    requested_region_count: int | None,
    seed: int,
    config: KMeansRegionConfig | None = None,
) -> tuple[np.ndarray, RegionLayoutModel]:
    matrix = _vector_matrix(vectors)
    row_count = int(matrix.shape[0])
    if row_count <= 0:
        return matrix, RegionLayoutModel(
            vector_field_name=vector_field_name, row_count=0
        )

    region_count = _resolve_region_count(row_count, requested_region_count)
    metric_vectors = metric_matrix(matrix, metric=metric)
    partition = solve_kmeans_regions(
        metric_vectors,
        region_count=region_count,
        seed=seed,
        config=config,
    )
    del metric_vectors

    order = _region_contiguous_order(partition.labels)
    reordered_vectors = np.ascontiguousarray(matrix[order], dtype="float32")
    ordered_labels = partition.labels[order]
    ordered_distances = partition.assigned_squared_distances[order]
    starts = _region_starts_from_counts(partition.counts)
    regions = _region_ranges(
        labels=ordered_labels,
        assigned_squared_distances=ordered_distances,
        counts=partition.counts,
        starts=starts,
    )
    layout = RegionLayoutModel(
        vector_field_name=vector_field_name,
        row_count=row_count,
        region_method="kmeans",
        region_clustering_backend=partition.evidence.clustering_backend.value,
        region_assignment_backend=partition.evidence.assignment_backend.value,
        requested_region_count=region_count
        if requested_region_count is None
        else requested_region_count,
        resolved_region_count=len(regions),
        region_centers=_centers_payload(partition.centers),
        region_neighbor_limit=_region_neighbor_limit(len(regions)),
        region_neighbors=_region_neighbors_payload(partition.centers),
        region_inertia=float(partition.inertia),
        region_iterations=int(partition.iterations),
        regions=regions,
    )
    return reordered_vectors, layout


def solve_kmeans_regions(
    points: np.ndarray,
    *,
    region_count: int,
    seed: int,
    config: KMeansRegionConfig | None = None,
) -> KMeansRegionResult:
    data = _vector_matrix(points)
    resolved_config = config or KMeansRegionConfig()
    _validate_problem(data, region_count=region_count, config=resolved_config)

    if data.shape[0] == region_count:
        labels = np.arange(region_count, dtype=np.int64)
        centers = np.array(data, dtype="float32", copy=True, order="C")
        counts = np.ones(region_count, dtype=np.int64)
        return KMeansRegionResult(
            labels=labels,
            centers=centers,
            counts=counts,
            assigned_squared_distances=np.zeros(region_count, dtype="float32"),
            inertia=0.0,
            iterations=0,
            evidence=KMeansRegionEvidence(
                training_sample_count=region_count,
                effective_max_points_per_centroid=None,
                assignment_backend=resolved_config.assignment_backend,
                clustering_backend=resolved_config.clustering_backend,
                attempts=0,
                total_iterations=0,
                kmeans_and_assignment_time_sec=0.0,
            ),
        )

    rng = np.random.default_rng(seed)
    kmeans_and_assignment_time_sec = 0.0
    total_iterations = 0
    attempts = min(region_count, resolved_config.max_attempts)
    last_empty_regions: tuple[int, ...] = ()

    for attempt in range(attempts):
        attempt_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        started = time.perf_counter()
        training = train_kmeans_centers(
            data,
            region_count=region_count,
            seed=attempt_seed,
            config=resolved_config,
        )
        centers = training.centers
        attempt_iterations = training.iterations
        total_iterations += attempt_iterations

        assignment = assign_to_centers(
            data,
            centers,
            backend=resolved_config.assignment_backend,
        )
        kmeans_and_assignment_time_sec += time.perf_counter() - started
        empty_regions = tuple(
            int(value) for value in np.flatnonzero(assignment.counts == 0)
        )
        if not empty_regions:
            return KMeansRegionResult(
                labels=assignment.labels,
                centers=centers,
                counts=assignment.counts,
                assigned_squared_distances=assignment.squared_distances,
                inertia=assignment.inertia,
                iterations=attempt_iterations,
                evidence=KMeansRegionEvidence(
                    training_sample_count=training.training_sample_count,
                    effective_max_points_per_centroid=(
                        training.effective_max_points_per_centroid
                    ),
                    assignment_backend=resolved_config.assignment_backend,
                    clustering_backend=resolved_config.clustering_backend,
                    attempts=attempt + 1,
                    total_iterations=total_iterations,
                    kmeans_and_assignment_time_sec=(kmeans_and_assignment_time_sec),
                ),
            )
        last_empty_regions = empty_regions

    raise ContractViolation(
        "vector region KMeans could not produce the requested number of non-empty regions: "
        f"requested={region_count}, empty_region_ids={list(last_empty_regions)}, attempts={attempts}"
    )


def _validate_problem(
    points: np.ndarray,
    *,
    region_count: int,
    config: KMeansRegionConfig,
) -> None:
    if region_count < 1:
        raise ContractViolation("region_count must be >= 1")
    if points.shape[0] < region_count:
        raise ContractViolation("region_count must be <= row_count")
    if points.shape[1] < 1:
        raise ContractViolation("vector region dimension must be positive")
    if config.max_iterations < 1:
        raise ContractViolation("KMeans max_iterations must be >= 1")
    if (
        config.max_points_per_centroid is not None
        and config.max_points_per_centroid < 1
    ):
        raise ContractViolation("KMeans max_points_per_centroid must be >= 1")
    if config.max_attempts < 1:
        raise ContractViolation("KMeans max_attempts must be >= 1")


def train_kmeans_centers(
    points: np.ndarray,
    *,
    region_count: int,
    seed: int,
    config: KMeansRegionConfig,
) -> KMeansCenterTraining:
    data = _vector_matrix(points)
    if config.clustering_backend is VectorRegionClusteringBackend.SKLEARN:
        return _train_kmeans_centers_sklearn(
            data,
            region_count=region_count,
            seed=seed,
            config=config,
        )
    raise ContractViolation(
        f"unsupported KMeans clustering backend: {config.clustering_backend!r}"
    )


def _train_kmeans_centers_sklearn(
    points: np.ndarray,
    *,
    region_count: int,
    seed: int,
    config: KMeansRegionConfig,
) -> KMeansCenterTraining:
    effective_cap = _resolved_training_cap(config.max_points_per_centroid)
    training_sample_count = min(
        int(points.shape[0]),
        region_count * effective_cap,
    )
    if training_sample_count < points.shape[0]:
        sample_indices = np.random.default_rng(seed).choice(
            points.shape[0],
            size=training_sample_count,
            replace=False,
        )
        training_points = np.ascontiguousarray(points[sample_indices], dtype="float32")
    else:
        training_points = np.array(points, dtype="float32", copy=True, order="C")
    model = KMeans(
        n_clusters=region_count,
        init="k-means++",
        n_init=1,
        max_iter=config.max_iterations,
        tol=0.0,
        verbose=0,
        random_state=seed,
        copy_x=False,
        algorithm="lloyd",
    )
    model.fit(training_points)
    return KMeansCenterTraining(
        centers=np.ascontiguousarray(model.cluster_centers_, dtype="float32"),
        iterations=int(model.n_iter_),
        training_sample_count=training_sample_count,
        effective_max_points_per_centroid=effective_cap,
    )


def _resolved_training_cap(configured: int | None) -> int:
    if configured is not None:
        return configured
    return DEFAULT_KMEANS_MAX_POINTS_PER_CENTROID


def assign_to_centers(
    points: np.ndarray,
    centers: np.ndarray,
    *,
    backend: VectorRegionAssignmentBackend,
) -> KMeansRegionAssignment:
    data = _vector_matrix(points)
    center_matrix = _vector_matrix(centers)
    if center_matrix.shape[0] < 1:
        raise ContractViolation("KMeans assignment requires at least one center")
    if data.shape[1] != center_matrix.shape[1]:
        raise ContractViolation(
            "KMeans assignment points and centers must have the same dimension"
        )
    if backend is VectorRegionAssignmentBackend.SKLEARN:
        return _assign_to_centers_sklearn(data, center_matrix)
    raise ContractViolation(f"unsupported KMeans assignment backend: {backend!r}")


def _assign_to_centers_sklearn(
    points: np.ndarray,
    centers: np.ndarray,
) -> KMeansRegionAssignment:
    raw_labels, raw_distances = pairwise_distances_argmin_min(
        points,
        centers,
        metric="euclidean",
        axis=1,
    )
    labels = raw_labels.astype("int64", copy=False)
    assigned_distances = np.asarray(raw_distances, dtype="float32")
    np.square(assigned_distances, out=assigned_distances)
    counts = np.bincount(labels, minlength=centers.shape[0]).astype(np.int64)
    inertia = float(np.sum(assigned_distances, dtype="float64"))
    return KMeansRegionAssignment(
        labels=labels,
        squared_distances=assigned_distances,
        counts=counts,
        inertia=inertia,
    )


def _vector_matrix(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim != 2:
        raise ContractViolation("vector region layout requires a 2D vector matrix")
    return np.ascontiguousarray(matrix, dtype="float32")


def _resolve_region_count(row_count: int, requested_region_count: int | None) -> int:
    if requested_region_count is not None:
        if requested_region_count < 1:
            raise ContractViolation("region_count must be >= 1")
        if requested_region_count > row_count:
            raise ContractViolation("region_count must be <= row_count")
        return requested_region_count
    return min(row_count, max(8, row_count // 1000))


def _region_contiguous_order(labels: np.ndarray) -> np.ndarray:
    return np.argsort(labels, kind="stable").astype(np.int64, copy=False)


def _region_starts_from_counts(counts: np.ndarray) -> np.ndarray:
    return np.cumsum(
        np.concatenate(
            (np.asarray([0], dtype=np.int64), counts[:-1].astype(np.int64, copy=False))
        ),
        dtype=np.int64,
    )


def _region_ranges(
    *,
    labels: np.ndarray,
    assigned_squared_distances: np.ndarray,
    counts: np.ndarray,
    starts: np.ndarray,
) -> tuple[RegionRange, ...]:
    regions: list[RegionRange] = []
    for region_id, start in enumerate(starts.tolist()):
        count = int(counts[region_id])
        end = start + count
        if count <= 0:
            raise ContractViolation("vector region layout cannot contain empty regions")
        member_labels = labels[start:end]
        if not np.all(member_labels == region_id):
            raise ContractViolation(
                "vector region layout order is not region-contiguous"
            )
        member_distances = assigned_squared_distances[start:end]
        representative = int(start + int(np.argmin(member_distances)))
        regions.append(
            RegionRange(
                region_id=region_id,
                public_start=int(start),
                public_end=int(end - 1),
                representative_public_ordinal=representative,
            )
        )
    return tuple(regions)


def _centers_payload(centers: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(value) for value in row)
        for row in np.asarray(centers, dtype="float32")
    )


def _region_neighbors_payload(centers: np.ndarray) -> tuple[tuple[int, ...], ...]:
    center_matrix = _vector_matrix(centers)
    limit = _region_neighbor_limit(center_matrix.shape[0])
    distances = pairwise_distances(center_matrix, metric="euclidean")
    rows: list[tuple[int, ...]] = []
    for row_id, row in enumerate(distances):
        nearest = np.argsort(row, kind="stable")
        neighbors = [int(row_id)]
        neighbors.extend(int(index) for index in nearest if int(index) != row_id)
        rows.append(tuple(neighbors[:limit]))
    return tuple(rows)


def _region_neighbor_limit(region_count: int) -> int:
    return min(max(int(region_count), 1), _REGION_NEIGHBOR_LIMIT)


__all__ = [
    "KMeansCenterTraining",
    "KMeansRegionAssignment",
    "KMeansRegionConfig",
    "KMeansRegionEvidence",
    "KMeansRegionResult",
    "DEFAULT_KMEANS_MAX_POINTS_PER_CENTROID",
    "assign_to_centers",
    "materialize_vector_region_layout",
    "solve_kmeans_regions",
    "train_kmeans_centers",
]
