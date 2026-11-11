"""Shared interval, cover, and placement helpers for deterministic dataset metadata."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from griffin.dataset.deterministic import (
    CouplingGroup,
    DeterministicFieldModel,
    FieldPlacementModel,
    PlacementSegment,
    RegionLayoutModel,
    TableDeterministicMetadata,
    VectorAttributeCouplingModel,
)
from griffin.infra.errors import ContractViolation
from griffin.model_ids import TableId

if TYPE_CHECKING:
    from griffin.dataset.reader import Dataset

OrdinalInterval = tuple[int, int]
OrdinalIntervalSet = tuple[OrdinalInterval, ...]
OwnerDeterministicMetadata = TableDeterministicMetadata


@dataclass(frozen=True, slots=True)
class RowCover:
    intervals: OrdinalIntervalSet
    total_count: int
    _count: int = field(init=False, repr=False)
    _fingerprint: str | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        total_count = int(self.total_count)
        if total_count < 0:
            raise ContractViolation("row cover total count must be non-negative")
        normalized = normalize_intervals(self.intervals)
        for start, end in normalized:
            if start < 0 or end >= total_count:
                raise ContractViolation("row cover interval is outside total count")
        object.__setattr__(self, "intervals", normalized)
        object.__setattr__(self, "total_count", total_count)
        count = intervals_row_count(normalized)
        object.__setattr__(self, "_count", count)
        object.__setattr__(self, "_fingerprint", None)

    @classmethod
    def _from_normalized(
        cls,
        intervals: OrdinalIntervalSet,
        total_count: int,
    ) -> "RowCover":
        cover = object.__new__(cls)
        object.__setattr__(cover, "intervals", intervals)
        object.__setattr__(cover, "total_count", total_count)
        object.__setattr__(cover, "_count", intervals_row_count(intervals))
        object.__setattr__(cover, "_fingerprint", None)
        return cover

    @classmethod
    def empty(cls, total_count: int) -> "RowCover":
        total = int(total_count)
        if total < 0:
            raise ContractViolation("row cover total count must be non-negative")
        return cls._from_normalized((), total)

    @classmethod
    def full(cls, total_count: int) -> "RowCover":
        total = int(total_count)
        if total <= 0:
            return cls.empty(total)
        return cls._from_normalized(((0, total - 1),), total)

    @classmethod
    def from_intervals(cls, intervals: OrdinalIntervalSet, *, total_count: int) -> "RowCover":
        return cls(intervals, int(total_count))

    @classmethod
    def from_normalized_intervals(
        cls,
        intervals: OrdinalIntervalSet,
        *,
        total_count: int,
    ) -> "RowCover":
        total = int(total_count)
        if total < 0:
            raise ContractViolation("row cover total count must be non-negative")
        if normalize_intervals(intervals) is not intervals:
            raise ContractViolation("row cover intervals must be normalized")
        if intervals and (intervals[0][0] < 0 or intervals[-1][1] >= total):
            raise ContractViolation("row cover interval is outside total count")
        return cls._from_normalized(intervals, total)

    @classmethod
    def from_rowids(cls, rowids: Iterable[int], *, total_count: int) -> "RowCover":
        return cls(rowids_to_intervals(rowids), int(total_count))

    @property
    def count(self) -> int:
        return self._count

    @property
    def selectivity(self) -> float:
        return self.count / self.total_count if self.total_count > 0 else 0.0

    def contains(self, rowid: int) -> bool:
        return contains_ordinal(self.intervals, int(rowid))

    def rowids(self) -> tuple[int, ...]:
        return rowids_from_intervals(self.intervals)

    def rowid_set(self) -> frozenset[int]:
        return frozenset(self.rowids())

    def union(self, other: "RowCover") -> "RowCover":
        self._require_same_total_count(other)
        return RowCover._from_normalized(
            union_intervals(self.intervals, other.intervals),
            self.total_count,
        )

    def intersect(self, other: "RowCover") -> "RowCover":
        self._require_same_total_count(other)
        return RowCover._from_normalized(
            intersect_intervals(self.intervals, other.intervals),
            self.total_count,
        )

    def difference(self, other: "RowCover") -> "RowCover":
        self._require_same_total_count(other)
        return RowCover._from_normalized(
            subtract_intervals(self.intervals, other.intervals),
            self.total_count,
        )

    def fingerprint(self) -> str:
        fingerprint = self._fingerprint
        if fingerprint is None:
            fingerprint = _row_cover_fingerprint_normalized(
                self.intervals,
                total_count=self.total_count,
            )
            object.__setattr__(self, "_fingerprint", fingerprint)
        return fingerprint

    def _require_same_total_count(self, other: "RowCover") -> None:
        if self.total_count != other.total_count:
            raise ContractViolation("row cover set algebra requires matching total count")


@dataclass(frozen=True, slots=True)
class DeterministicHitProfile:
    cover: OrdinalIntervalSet
    row_count: int
    hit_count: int
    global_selectivity: float
    region_counts: tuple[int, ...] = ()
    region_hit_rates: tuple[float, ...] = ()


def field_model_for_name(
    metadata: OwnerDeterministicMetadata,
    field_name: str,
) -> DeterministicFieldModel:
    for model in _supported_models(metadata):
        if model.field_name == field_name:
            return model
    raise ContractViolation(f"deterministic model missing for field {field_name}")


def deterministic_owner_metadata(dataset: Dataset, owner: object) -> OwnerDeterministicMetadata | None:
    if isinstance(owner, TableId):
        return dataset.deterministic_table(owner.name)
    return None


def field_placement_for_name(
    metadata: OwnerDeterministicMetadata,
    field_name: str,
) -> FieldPlacementModel:
    for placement in _field_placements(metadata):
        if placement.field_name == field_name:
            return placement
    raise ContractViolation(f"deterministic placement missing for field {field_name}")


def project_intervals(
    placement: FieldPlacementModel,
    intervals: OrdinalIntervalSet,
) -> OrdinalIntervalSet:
    projected: list[OrdinalInterval] = []
    segments = placement.segments
    segment_index = 0
    for interval_start, interval_end in normalize_intervals(intervals):
        segment_index = _first_segment_ending_at_or_after(
            segments,
            interval_start,
            minimum=segment_index,
        )
        scan_index = segment_index
        while scan_index < len(segments):
            segment = segments[scan_index]
            if segment.local_start > interval_end:
                break
            overlap_start = (
                interval_start
                if interval_start >= segment.local_start
                else segment.local_start
            )
            overlap_end = (
                interval_end
                if interval_end <= segment.local_end
                else segment.local_end
            )
            start_offset = overlap_start - segment.local_start
            end_offset = overlap_end - segment.local_start
            projected.append(
                (
                    segment.public_start + start_offset,
                    segment.public_start + end_offset,
                )
            )
            scan_index += 1
    return union_intervals(tuple(projected), ())


def _first_segment_ending_at_or_after(
    segments: tuple[PlacementSegment, ...],
    local_ordinal: int,
    *,
    minimum: int,
) -> int:
    low = minimum
    high = len(segments)
    while low < high:
        middle = (low + high) // 2
        if segments[middle].local_end < local_ordinal:
            low = middle + 1
        else:
            high = middle
    return low


def project_intervals_via_coupling(
    coupling: VectorAttributeCouplingModel,
    region_layout: RegionLayoutModel,
    intervals: OrdinalIntervalSet,
) -> OrdinalIntervalSet:
    return project_intervals(
        coupling_replay_placement(coupling, region_layout),
        intervals,
    )


def coupling_replay_placement(
    coupling: VectorAttributeCouplingModel,
    region_layout: RegionLayoutModel,
) -> FieldPlacementModel:
    """Rebuild the stable attribute placement encoded by a coupling artifact."""
    _validate_replay_shape(coupling, region_layout)
    group_cursors = [group.local_start for group in coupling.groups]
    segments: list[PlacementSegment] = []
    for region_index, region in enumerate(region_layout.regions):
        public_cursor = region.public_start
        for group_index, count in enumerate(coupling.region_group_counts[region_index]):
            if count == 0:
                continue
            local_start = group_cursors[group_index]
            local_end = local_start + count - 1
            public_start = public_cursor
            public_end = public_cursor + count - 1
            segments.append(
                PlacementSegment(
                    local_start=local_start,
                    local_end=local_end,
                    public_start=public_start,
                    public_end=public_end,
                    region_id=region.region_id,
                )
            )
            group_cursors[group_index] = local_end + 1
            public_cursor = public_end + 1
        if public_cursor != region.public_end + 1:
            raise ContractViolation("coupling replay row does not cover region")
    _validate_replay_consumed_groups(coupling.groups, group_cursors)
    segments.sort(key=lambda segment: segment.local_start)
    return FieldPlacementModel(
        field_name=coupling.attribute_field_name,
        row_count=region_layout.row_count,
        segments=tuple(segments),
    )


def project_local_cover(
    metadata: OwnerDeterministicMetadata,
    field_name: str,
    local_cover: OrdinalIntervalSet,
) -> OrdinalIntervalSet:
    """Project a deterministic field local cover into public ``_rowid`` coordinates."""
    placement = field_placement_for_name(metadata, field_name)
    return project_intervals(placement, local_cover)


def project_local_cover_profile(
    metadata: OwnerDeterministicMetadata,
    field_name: str,
    local_cover: OrdinalIntervalSet,
) -> DeterministicHitProfile:
    return hit_profile(metadata, project_local_cover(metadata, field_name, local_cover))


def hit_profile(
    metadata: OwnerDeterministicMetadata,
    hit_set: OrdinalIntervalSet,
) -> DeterministicHitProfile:
    row_count = metadata.row_count
    hit_count = intervals_row_count(hit_set)
    selectivity = global_selectivity(hit_set, row_count=row_count)
    layout = metadata.region_layout
    counts = region_counts(hit_set, layout) if layout is not None else ()
    rates = _region_hit_rates_from_counts(counts, layout) if layout is not None else ()
    return DeterministicHitProfile(
        cover=hit_set,
        row_count=row_count,
        hit_count=hit_count,
        global_selectivity=selectivity,
        region_counts=counts,
        region_hit_rates=rates,
    )


def intervals_row_count(intervals: OrdinalIntervalSet) -> int:
    return sum(end - start + 1 for start, end in intervals)


def rowids_from_intervals(intervals: OrdinalIntervalSet) -> tuple[int, ...]:
    rowids: list[int] = []
    for start, end in intervals:
        rowids.extend(range(start, end + 1))
    return tuple(rowids)


def rowids_to_intervals(rowids: Iterable[int]) -> OrdinalIntervalSet:
    ordered = sorted({int(rowid) for rowid in rowids})
    if not ordered:
        return ()
    result: list[OrdinalInterval] = []
    start = ordered[0]
    end = ordered[0]
    for rowid in ordered[1:]:
        if rowid == end + 1:
            end = rowid
            continue
        result.append((start, end))
        start = rowid
        end = rowid
    result.append((start, end))
    return tuple(result)


def expand_cover(intervals: OrdinalIntervalSet) -> set[int]:
    rowids: set[int] = set()
    for start, end in intervals:
        rowids.update(range(start, end + 1))
    return rowids


def global_selectivity(hit_set: OrdinalIntervalSet, *, row_count: int) -> float:
    if row_count <= 0:
        return 0.0
    return intervals_row_count(hit_set) / row_count


def region_counts(
    hit_set: OrdinalIntervalSet,
    region_layout: RegionLayoutModel,
) -> tuple[int, ...]:
    if not region_layout.regions:
        return ()
    if not hit_set:
        return tuple(0 for _ in region_layout.regions)
    counts: list[int] = []
    hit_index = 0
    hit_count = len(hit_set)
    for region in region_layout.regions:
        region_start = region.public_start
        region_end = region.public_end
        while hit_index < hit_count and hit_set[hit_index][1] < region_start:
            hit_index += 1
        cursor = hit_index
        total = 0
        while cursor < hit_count:
            hit_start, hit_end = hit_set[cursor]
            if hit_start > region_end:
                break
            overlap_start = max(hit_start, region_start)
            overlap_end = min(hit_end, region_end)
            if overlap_start <= overlap_end:
                total += overlap_end - overlap_start + 1
            if hit_end <= region_end:
                cursor += 1
                continue
            break
        hit_index = cursor
        counts.append(total)
    return tuple(counts)


def region_hit_rates(
    hit_set: OrdinalIntervalSet,
    region_layout: RegionLayoutModel,
) -> tuple[float, ...]:
    counts = region_counts(hit_set, region_layout)
    return _region_hit_rates_from_counts(counts, region_layout)


def _region_hit_rates_from_counts(
    counts: tuple[int, ...],
    region_layout: RegionLayoutModel,
) -> tuple[float, ...]:
    rates: list[float] = []
    for count, region in zip(counts, region_layout.regions, strict=True):
        region_size = region.public_end - region.public_start + 1
        rates.append(0.0 if region_size <= 0 else count / region_size)
    return tuple(rates)


def intersect_intervals(
    left: OrdinalIntervalSet,
    right: OrdinalIntervalSet,
) -> OrdinalIntervalSet:
    result: list[OrdinalInterval] = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        overlap = _intersect_one(left[i], right[j])
        if overlap is not None:
            result.append(overlap)
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return tuple(result)


def subtract_intervals(
    left: OrdinalIntervalSet,
    right: OrdinalIntervalSet,
) -> OrdinalIntervalSet:
    if not left or not right:
        return left
    result: list[OrdinalInterval] = []
    right_index = 0
    right_count = len(right)
    for start, end in left:
        cursor = start
        while right_index < right_count and right[right_index][1] < cursor:
            right_index += 1
        scan_index = right_index
        while scan_index < right_count and right[scan_index][0] <= end:
            cut_start, cut_end = right[scan_index]
            if cut_start > cursor:
                result.append((cursor, min(end, cut_start - 1)))
            cursor = max(cursor, cut_end + 1)
            if cursor > end:
                break
            scan_index += 1
        if cursor <= end:
            result.append((cursor, end))
    return tuple(result)


def union_intervals(
    left: OrdinalIntervalSet,
    right: OrdinalIntervalSet,
) -> OrdinalIntervalSet:
    normalized_left = normalize_intervals(left)
    normalized_right = normalize_intervals(right)
    if not normalized_left:
        return normalized_right
    if not normalized_right:
        return normalized_left

    merged: list[OrdinalInterval] = []
    left_index = 0
    right_index = 0
    while left_index < len(normalized_left) or right_index < len(normalized_right):
        if right_index >= len(normalized_right) or (
            left_index < len(normalized_left)
            and normalized_left[left_index][0] <= normalized_right[right_index][0]
        ):
            interval = normalized_left[left_index]
            left_index += 1
        else:
            interval = normalized_right[right_index]
            right_index += 1
        _append_merged_interval(merged, interval)
    return tuple(merged)


def normalize_intervals(
    intervals: OrdinalIntervalSet,
) -> OrdinalIntervalSet:
    if not intervals:
        return ()

    ordered = True
    canonical = True
    previous_start: int | None = None
    previous_end: int | None = None
    for raw_start, raw_end in intervals:
        start = int(raw_start)
        end = int(raw_end)
        if start > end:
            raise ContractViolation("interval start must not exceed interval end")
        if type(raw_start) is not int or type(raw_end) is not int:
            canonical = False
        if previous_start is not None and start < previous_start:
            ordered = False
            canonical = False
        if previous_end is not None and start <= previous_end + 1:
            canonical = False
        previous_start = start
        previous_end = end

    if canonical:
        return intervals

    converted = [(int(start), int(end)) for start, end in intervals]
    if not ordered:
        converted.sort()

    merged: list[OrdinalInterval] = []
    for interval in converted:
        _append_merged_interval(merged, interval)
    return tuple(merged)


def _append_merged_interval(
    merged: list[OrdinalInterval],
    interval: OrdinalInterval,
) -> None:
    start, end = interval
    if not merged:
        merged.append(interval)
        return
    last_start, last_end = merged[-1]
    if start <= last_end + 1:
        if end > last_end:
            merged[-1] = (last_start, end)
        return
    merged.append(interval)


def contains_ordinal(intervals: OrdinalIntervalSet, ordinal: int) -> bool:
    value = int(ordinal)
    lo = 0
    hi = len(intervals)
    while lo < hi:
        mid = (lo + hi) // 2
        start, end = intervals[mid]
        if value < start:
            hi = mid
        elif value > end:
            lo = mid + 1
        else:
            return True
    return False


def row_cover_fingerprint(intervals: OrdinalIntervalSet, *, total_count: int) -> str:
    normalized = normalize_intervals(intervals)
    return _row_cover_fingerprint_normalized(normalized, total_count=total_count)


def _row_cover_fingerprint_normalized(
    intervals: OrdinalIntervalSet,
    *,
    total_count: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(int(total_count).to_bytes(8, "big", signed=False))
    digest.update(intervals_row_count(intervals).to_bytes(8, "big", signed=False))
    for start, end in intervals:
        digest.update(int(start).to_bytes(8, "big", signed=True))
        digest.update(int(end).to_bytes(8, "big", signed=True))
    return digest.hexdigest()


def _intersect_one(
    left: OrdinalInterval,
    right: OrdinalInterval,
) -> OrdinalInterval | None:
    start = max(left[0], right[0])
    end = min(left[1], right[1])
    if start > end:
        return None
    return (start, end)


def _validate_replay_shape(
    coupling: VectorAttributeCouplingModel,
    region_layout: RegionLayoutModel,
) -> None:
    if len(coupling.region_group_counts) != len(region_layout.regions):
        raise ContractViolation("coupling replay matrix row count does not match region layout")
    if not coupling.groups:
        raise ContractViolation("coupling replay requires groups")
    expected_local_start = 0
    seen_group_ids: set[int] = set()
    for group in coupling.groups:
        if group.group_id in seen_group_ids:
            raise ContractViolation("coupling replay group ids must be unique")
        seen_group_ids.add(group.group_id)
        if group.local_start != expected_local_start:
            raise ContractViolation("coupling replay groups must be contiguous from local ordinal 0")
        if group.local_end < group.local_start:
            raise ContractViolation("coupling replay group ranges must be non-empty")
        expected_local_start = group.local_end + 1
    for row in coupling.region_group_counts:
        if len(row) != len(coupling.groups):
            raise ContractViolation("coupling replay matrix column count does not match groups")
        if any(count < 0 for count in row):
            raise ContractViolation("coupling replay matrix counts must be non-negative")


def _validate_replay_consumed_groups(
    groups: tuple[CouplingGroup, ...],
    group_cursors: list[int],
) -> None:
    for group, cursor in zip(groups, group_cursors, strict=True):
        if cursor != group.local_end + 1:
            raise ContractViolation("coupling replay column does not cover group")


def _supported_models(metadata: OwnerDeterministicMetadata) -> tuple[DeterministicFieldModel, ...]:
    return metadata.supported_fields


def _field_placements(metadata: OwnerDeterministicMetadata) -> tuple[FieldPlacementModel, ...]:
    return metadata.field_placements


__all__ = [
    "coupling_replay_placement",
    "DeterministicHitProfile",
    "OrdinalInterval",
    "OrdinalIntervalSet",
    "RowCover",
    "contains_ordinal",
    "deterministic_owner_metadata",
    "expand_cover",
    "field_model_for_name",
    "field_placement_for_name",
    "hit_profile",
    "intersect_intervals",
    "intervals_row_count",
    "global_selectivity",
    "normalize_intervals",
    "project_intervals",
    "project_intervals_via_coupling",
    "project_local_cover",
    "project_local_cover_profile",
    "region_counts",
    "region_hit_rates",
    "row_cover_fingerprint",
    "rowids_from_intervals",
    "rowids_to_intervals",
    "subtract_intervals",
    "union_intervals",
]
