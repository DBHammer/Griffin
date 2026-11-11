"""Dataset 级共享 runtime 资源宿主。

本模块定义：

- `DatasetResources`：dataset identity 绑定的 typed service 集合。workload
  task 持有它，query session 与所有 long-lived holders 通过参数借用，
  不直接构造内部 service。

字段约束：

- `vector_oracle` 是唯一的 `VectorKnnOracle` 实例，自带 instance cache
- `predicate_access` 是唯一的 `HybridPredicateAccess` 实例
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from griffin.model_ids import FieldId

if TYPE_CHECKING:
    from griffin.dataset.exact_cover import OrdinalIntervalSet
    from griffin.dataset.reader import Dataset
    from griffin.query.truth.predicate import HybridPredicateAccess
    from griffin.query.truth.vector import VectorKnnOracle


@dataclass
class DatasetResources:
    """Dataset identity 绑定的共享 runtime 资源。

    所有 long-lived holders（workload task、coordinator、runtime、
    evaluator、finalizer）通过这个对象访问 dataset-bound exact runtime
    service。

    `DatasetResources` 可被多个 workload task 共享。每个 workload task
    只**引用**它，不**拥有**它。
    """

    dataset: "Dataset"
    vector_oracle: "VectorKnnOracle"
    predicate_access: "HybridPredicateAccess"
    _predicate_interval_covers: dict[str, "OrdinalIntervalSet"] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    @classmethod
    def open(cls, dataset: "Dataset") -> "DatasetResources":
        """从已经打开的 Dataset 构造一份 DatasetResources。

        所有内部 service 在此一次性构造；后续 callsite 不再"现用现造"。
        """
        from griffin.query.truth.predicate import HybridPredicateAccess
        from griffin.query.truth.vector import VectorKnnOracle

        return cls(
            dataset=dataset,
            vector_oracle=VectorKnnOracle(),
            predicate_access=HybridPredicateAccess(),
        )

    def prepare_query_vector_fields(
        self,
        fields: tuple[FieldId, ...],
    ) -> None:
        """Load dataset-level vector truth resources before query coordination."""
        for vector_field in fields:
            self.vector_oracle.vector_source_metadata(
                dataset=self.dataset,
                field_id=vector_field,
            )

    def predicate_interval_cover(
        self,
        key: str,
        producer: Callable[[], "OrdinalIntervalSet"],
    ) -> "OrdinalIntervalSet":
        with self._lock:
            cached = self._predicate_interval_covers.get(key)
            if cached is not None:
                return cached
        cover = producer()
        with self._lock:
            cached = self._predicate_interval_covers.get(key)
            if cached is not None:
                return cached
            self._predicate_interval_covers[key] = cover
            return cover


__all__ = [
    "DatasetResources",
]
