"""显存复用规划的核心数据结构（只依赖标准库）。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Wait:
    """跨流等待：waiter 算子必须等 waitee 算子完成后才能开始。"""

    waiter: str
    waitee: str


@dataclass(frozen=True)
class TensorSpec:
    name: str
    producer: str
    consumers: tuple[str, ...]
    size: int
    alignment: int


@dataclass
class PlanningInput:
    # 流 id -> 该流按提交顺序排列的算子 id
    streams: dict[str, list[str]]
    waits: list[Wait]
    tensors: list[TensorSpec]
    budget_bytes: int

    def all_operators(self) -> list[str]:
        ops: list[str] = []
        for stream_id in sorted(self.streams):
            ops.extend(self.streams[stream_id])
        return ops


@dataclass
class AllocationResult:
    """一次确定依赖图下的最优布局结果。"""

    feasible: bool
    total_bytes: int
    offsets: dict[str, int]
    # 调试/诊断用：每个“生命周期互不相交组”包含的张量
    bins: list[list[str]] = field(default_factory=list)


@dataclass
class PlanOutcome:
    feasible: bool
    total_bytes: int
    budget_bytes: int
    offsets: dict[str, int]
    added_wait: Wait | None
    note: str
    best_effort_total_bytes: int | None = None
    bins: list[list[str]] = field(default_factory=list)


class PlanningError(ValueError):
    """输入本身不合法（与“排不下”的无解不同，这里对应 HTTP 422）。"""
