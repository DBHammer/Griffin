"""Filtered ANN hit-band math."""

from __future__ import annotations

import hashlib
import math

import msgspec

from griffin.dataset.exact_cover import RowCover
from griffin.infra.errors import ContractViolation
from griffin.query.model import (
    QueryTargetSeedStage,
    SearchIdentity,
    derive_query_target_seed,
)
from griffin.query.task import FilteredAnnGLSTargetMode, Hardness


class FilteredAnnHitBandDiagnostic(msgspec.Struct, frozen=True, omit_defaults=True):
    total_count: int
    cover_count: int
    k: int
    difficulty: str
    support_min: int
    support_max: int
    reason: str


def hit_band(
    *,
    total_count: int,
    cover_count: int,
    k: int,
    hardness: Hardness,
) -> tuple[int, int] | None:
    band, _ = resolve_hit_band(
        total_count=total_count,
        cover_count=cover_count,
        k=k,
        hardness=hardness,
    )
    return band


def topk_hit_count(topk_rowids: tuple[int, ...], row_filter: RowCover) -> int:
    return sum(1 for rowid in topk_rowids if row_filter.contains(int(rowid)))


def hit_loss(hit_count: int, band: tuple[int, int]) -> float:
    minimum, maximum = band
    if minimum <= hit_count <= maximum:
        return 0.0
    return float(min(abs(int(hit_count) - minimum), abs(int(hit_count) - maximum)))


class ReachableGLSPoint(msgspec.Struct, frozen=True, omit_defaults=True):
    hit_count: int
    gls: float


def gls_from_hit_count(
    *,
    hit_count: int,
    total_count: int,
    cover_count: int,
    k: int,
) -> float | None:
    n = int(total_count)
    m = int(cover_count)
    k_int = int(k)
    h = int(hit_count)
    if n <= 0 or m <= 0 or k_int <= 0:
        return None
    global_selectivity = m / float(n)
    if global_selectivity <= 0.0:
        return None
    local_selectivity = h / float(k_int)
    ratio = local_selectivity / global_selectivity
    if ratio < 0.0:
        return None
    return float((ratio - 1.0) / (ratio + 1.0))


def gls_loss(
    *,
    hit_count: int,
    band: tuple[int, int],
    total_count: int,
    cover_count: int,
    k: int,
) -> float:
    """GLS-value loss toward a hit-count band, per paper §V-C.

    `band` is a hit-count interval [lo, hi]; its GLS image is the target GLS
    interval (GLS is monotonically increasing in hit count). The loss is the
    distance of the observed GLS to that interval (0 inside, |distance| outside),
    matching errGLS(q) for scalar and range targets. It is NOT the integer
    hit-count distance, which can order candidates differently from GLS error.
    """
    lo, hi = band
    actual = gls_from_hit_count(
        hit_count=hit_count, total_count=total_count, cover_count=cover_count, k=k,
    )
    if actual is None:
        return float("inf")
    lo_gls = gls_from_hit_count(
        hit_count=lo, total_count=total_count, cover_count=cover_count, k=k,
    )
    hi_gls = gls_from_hit_count(
        hit_count=hi, total_count=total_count, cover_count=cover_count, k=k,
    )
    if lo_gls is None or hi_gls is None:
        return float("inf")
    if lo_gls > hi_gls:
        lo_gls, hi_gls = hi_gls, lo_gls
    if actual < lo_gls:
        return float(lo_gls - actual)
    if actual > hi_gls:
        return float(actual - hi_gls)
    return 0.0


def reachable_gls_points(*, total_count: int, cover_count: int, k: int) -> tuple[ReachableGLSPoint, ...]:
    support = _hit_support_range(total_count=total_count, cover_count=cover_count, k=k)
    if support is None:
        return ()
    support_min, support_max = support
    points: list[ReachableGLSPoint] = []
    for hit in range(support_min, support_max + 1):
        gls = gls_from_hit_count(hit_count=hit, total_count=total_count, cover_count=cover_count, k=k)
        if gls is None:
            continue
        points.append(ReachableGLSPoint(hit_count=hit, gls=float(gls)))
    return tuple(points)


def reachable_gls_band(*, total_count: int, cover_count: int, k: int) -> tuple[float, float] | None:
    points = reachable_gls_points(total_count=total_count, cover_count=cover_count, k=k)
    if not points:
        return None
    return float(points[0].gls), float(points[-1].gls)


# 可插拔的几何可达边界提供器：(total_count, cover_count, k) -> (h_min, h_max) | None。
# 基数算术定义的"可达"不保证几何可达：1% 谓词的 gls_max(10/10) 可能无解
# （实测合格点 10-NN 合格率最大仅 0.50）。注册提供器后，目标会被钳制到
# 实测几何可达的 hit 区间；未注册或返回 None 时保持原有基数口径。
_GLS_HIT_BOUNDS_PROVIDER = None


def register_gls_hit_bounds_provider(fn) -> None:
    """注册 (total_count, cover_count, k) -> (h_min, h_max) | None 的探测函数。"""
    global _GLS_HIT_BOUNDS_PROVIDER
    _GLS_HIT_BOUNDS_PROVIDER = fn


def _resolve_hit_bounds(total_count: int, cover_count: int, k: int) -> tuple[int, int] | None:
    if _GLS_HIT_BOUNDS_PROVIDER is None:
        return None
    try:
        bounds = _GLS_HIT_BOUNDS_PROVIDER(int(total_count), int(cover_count), int(k))
    except Exception:  # 探测失败按无边界处理，绝不阻断搜索
        return None
    if not isinstance(bounds, tuple) or len(bounds) != 2:
        return None
    lo, hi = int(bounds[0]), int(bounds[1])
    if lo > hi:
        lo, hi = hi, lo
    return max(0, lo), min(int(k), hi)


def resolve_gls_target_for_mode(
    *,
    mode: FilteredAnnGLSTargetMode,
    total_count: int,
    cover_count: int,
    k: int,
    random_seed: int,
) -> ReachableGLSPoint | None:
    points = reachable_gls_points(total_count=total_count, cover_count=cover_count, k=k)
    if not points:
        return None
    bounds = _resolve_hit_bounds(total_count, cover_count, k)
    if bounds is not None:
        lo, hi = bounds
        clamped = [p for p in points if lo <= int(p.hit_count) <= hi]
        if not clamped:
            # 几何区间与基数点集不相交：退回最近端点，避免无解
            clamped = [min(points, key=lambda p: min(abs(int(p.hit_count) - lo), abs(int(p.hit_count) - hi)))]
        points = clamped
    if mode is FilteredAnnGLSTargetMode.MIN:
        for p in points:
            if int(p.hit_count) >= 1:
                return p
        return points[0]
    if mode is FilteredAnnGLSTargetMode.MAX:
        return points[-1]
    if mode is not FilteredAnnGLSTargetMode.RANDOM:
        raise ContractViolation(f"unsupported filtered ANN GLS target mode {mode.value}")
    candidates = points[1:-1] if len(points) > 2 else points
    token = (
        f"filtered-ann-gls-random|{int(random_seed)}|{int(total_count)}|"
        f"{int(cover_count)}|{int(k)}"
    ).encode("utf-8")
    index = int.from_bytes(hashlib.sha256(token).digest()[:8], "big", signed=False) % len(candidates)
    return candidates[index]


def resolve_search_gls_target_for_mode(
    *,
    mode: FilteredAnnGLSTargetMode,
    total_count: int,
    cover_count: int,
    k: int,
    search_identity: SearchIdentity,
) -> ReachableGLSPoint | None:
    """Resolve one GLS target that remains fixed while its search continues."""
    return resolve_gls_target_for_mode(
        mode=mode,
        total_count=total_count,
        cover_count=cover_count,
        k=k,
        random_seed=derive_query_target_seed(
            search_identity,
            QueryTargetSeedStage.FILTERED_ANN_GLS,
            total_count,
            cover_count,
            k,
        ),
    )


def gls_target_hit_band(*, target_hit_count: int, k: int) -> tuple[int, int]:
    """Return the hit-count acceptance band for a GLS mode target.

    Paper §V-C (Alg. 3): errGLS(q) = |GLS(q) - GLS*| must reach 0, which
    happens iff the exact top-K hit count equals the target hit count.
    The band is therefore the singleton [target, target]; the former
    ±tolerance relaxation let the search accept band-edge hit counts
    whose GLS-value error exceeds the 10% evaluation threshold.
    """
    k_int = int(k)
    target = int(target_hit_count)
    if k_int <= 0:
        raise ContractViolation("filtered ANN GLS hit band requires positive k")
    if target < 1 or target > k_int:
        raise ContractViolation(
            f"filtered ANN GLS target hit count must be within [1, {k_int}], got {target}"
        )
    return target, target


def hit_band_for_gls(
    *,
    total_count: int,
    cover_count: int,
    k: int,
    target_gls_min: float,
    target_gls_max: float,
) -> tuple[int, int] | None:
    support = _hit_support_range(total_count=total_count, cover_count=cover_count, k=k)
    if support is None:
        return None
    support_min, support_max = support
    n = int(total_count)
    m = int(cover_count)
    k_int = int(k)
    global_selectivity = m / float(n)
    if global_selectivity <= 0.0:
        return None
    gls_min = max(float(target_gls_min), -1.0)
    gls_max = min(float(target_gls_max), 0.999999999)
    if gls_min > gls_max:
        return None
    local_sel_min = 0.0 if gls_min <= -1.0 else global_selectivity * ((1.0 + gls_min) / (1.0 - gls_min))
    local_sel_max = global_selectivity * ((1.0 + gls_max) / (1.0 - gls_max))
    raw_hit_min = int(math.ceil(local_sel_min * float(k_int) - 1e-12))
    raw_hit_max = int(math.floor(local_sel_max * float(k_int) + 1e-12))
    low = max(support_min, raw_hit_min)
    high = min(support_max, raw_hit_max)
    if low > high:
        return None
    return low, high


def nearest_reachable_gls(
    *,
    target_gls: float,
    total_count: int,
    cover_count: int,
    k: int,
) -> tuple[int, float] | None:
    points = reachable_gls_points(total_count=total_count, cover_count=cover_count, k=k)
    if not points:
        return None
    best = min(points, key=lambda point: abs(point.gls - float(target_gls)))
    return best.hit_count, float(best.gls)


def _hit_support_range(*, total_count: int, cover_count: int, k: int) -> tuple[int, int] | None:
    n = int(total_count)
    m = int(cover_count)
    k_int = int(k)
    if n <= 0 or m < 0 or k_int <= 0:
        return None
    support_min = max(0, k_int - (n - m))
    support_max = min(k_int, m)
    if support_min > support_max:
        return None
    return support_min, support_max


def resolve_hit_band(
    *,
    total_count: int,
    cover_count: int,
    k: int,
    hardness: Hardness,
) -> tuple[tuple[int, int] | None, FilteredAnnHitBandDiagnostic]:
    difficulty = hardness.value
    n = int(total_count)
    m = int(cover_count)
    k_int = int(k)
    support_min = max(0, k_int - (n - m)) if n > 0 and k_int > 0 else 0
    support_max = min(k_int, m) if k_int > 0 and m > 0 else 0
    if n <= 0 or k <= 0 or m < k:
        reason = "non_positive_total_count" if n <= 0 else "non_positive_top_k"
        if m < k and n > 0 and k > 0:
            reason = "cover_count_below_top_k"
        return None, FilteredAnnHitBandDiagnostic(
            total_count=n,
            cover_count=m,
            k=k_int,
            difficulty=difficulty,
            support_min=support_min,
            support_max=support_max,
            reason=reason,
        )
    if m == k:
        return None, FilteredAnnHitBandDiagnostic(
            total_count=n,
            cover_count=m,
            k=k_int,
            difficulty=difficulty,
            support_min=support_min,
            support_max=support_max,
            reason="cover_count_equals_top_k",
        )
    if m / k <= 1.2:
        return None, FilteredAnnHitBandDiagnostic(
            total_count=n,
            cover_count=m,
            k=k_int,
            difficulty=difficulty,
            support_min=support_min,
            support_max=support_max,
            reason="cover_count_too_close_to_top_k",
        )
    if support_min > support_max:
        return None, FilteredAnnHitBandDiagnostic(
            total_count=n,
            cover_count=m,
            k=k_int,
            difficulty=difficulty,
            support_min=support_min,
            support_max=support_max,
            reason="empty_support_range",
        )
    mu = k * m / n
    if mu < 0.5:
        bands = {
            Hardness.HARD: (0, 0),
            Hardness.MEDIUM: (1, 1),
            Hardness.EASY: (2, support_max),
        }
    else:
        q25 = _hypergeom_quantile(n=n, m=m, k=k, alpha=0.25)
        q35 = _hypergeom_quantile(n=n, m=m, k=k, alpha=0.35)
        q65 = _hypergeom_quantile(n=n, m=m, k=k, alpha=0.65)
        q75 = _hypergeom_quantile(n=n, m=m, k=k, alpha=0.75)
        q95 = _hypergeom_quantile(n=n, m=m, k=k, alpha=0.95)
        if mu < 2:
            bands = {
                Hardness.HARD: (support_min, q25),
                Hardness.MEDIUM: (q25 + 1, q75),
                Hardness.EASY: (max(q95, q75 + 1, 2), support_max),
            }
        elif mu < 5:
            bands = {
                Hardness.HARD: (support_min, q25),
                Hardness.MEDIUM: (q25 + 1, q75),
                Hardness.EASY: (q75 + 1, support_max),
            }
        else:
            bands = {
                Hardness.HARD: (support_min, q25),
                Hardness.MEDIUM: (q35, q65),
                Hardness.EASY: (q75, support_max),
            }
    if hardness not in bands:
        raise ContractViolation(f"unknown filtered ANN difficulty {difficulty}")
    low, high = bands[hardness]
    low = max(support_min, int(low))
    high = min(support_max, int(high))
    diagnostic = FilteredAnnHitBandDiagnostic(
        total_count=n,
        cover_count=m,
        k=k_int,
        difficulty=difficulty,
        support_min=support_min,
        support_max=support_max,
        reason="ok",
    )
    if low > high:
        return None, FilteredAnnHitBandDiagnostic(
            total_count=n,
            cover_count=m,
            k=k_int,
            difficulty=difficulty,
            support_min=support_min,
            support_max=support_max,
            reason="empty_difficulty_band",
        )
    return (low, high), diagnostic


def format_hit_band_diagnostic(diagnostic: FilteredAnnHitBandDiagnostic) -> str:
    return (
        "filtered_ann hit band infeasible: "
        f"N={diagnostic.total_count}, "
        f"M={diagnostic.cover_count}, "
        f"k={diagnostic.k}, "
        f"difficulty={diagnostic.difficulty}, "
        f"support_min={diagnostic.support_min}, "
        f"support_max={diagnostic.support_max}, "
        f"reason={diagnostic.reason}"
    )


def hit_band_diagnostics(diagnostic: FilteredAnnHitBandDiagnostic) -> dict[str, str]:
    return {
        "filtered_ann_infeasible": "true",
        "filtered_ann_N": str(diagnostic.total_count),
        "filtered_ann_M": str(diagnostic.cover_count),
        "filtered_ann_k": str(diagnostic.k),
        "filtered_ann_difficulty": diagnostic.difficulty,
        "filtered_ann_support_min": str(diagnostic.support_min),
        "filtered_ann_support_max": str(diagnostic.support_max),
        "filtered_ann_infeasible_reason": diagnostic.reason,
    }


def _hypergeom_quantile(*, n: int, m: int, k: int, alpha: float) -> int:
    support_min = max(0, k - (n - m))
    support_max = min(k, m)
    cumulative = 0.0
    denominator = math.comb(n, k)
    for h in range(support_min, support_max + 1):
        probability = math.comb(m, h) * math.comb(n - m, k - h) / denominator
        cumulative += probability
        if cumulative >= alpha:
            return h
    return support_max


__all__ = [
    "FilteredAnnHitBandDiagnostic",
    "ReachableGLSPoint",
    "format_hit_band_diagnostic",
    "gls_from_hit_count",
    "gls_target_hit_band",
    "hit_band",
    "hit_band_diagnostics",
    "hit_band_for_gls",
    "hit_loss",
    "nearest_reachable_gls",
    "reachable_gls_band",
    "reachable_gls_points",
    "resolve_gls_target_for_mode",
    "resolve_search_gls_target_for_mode",
    "resolve_hit_band",
    "topk_hit_count",
]
