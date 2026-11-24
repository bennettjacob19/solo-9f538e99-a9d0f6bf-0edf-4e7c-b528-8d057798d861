"""规划编排：原依赖优先，排不下才枚举“加一条不成环等待”的方案。"""

from __future__ import annotations

from .models import PlanOutcome, PlanningInput, Wait
from .solver import (
    build_base_edges,
    build_interference,
    candidate_waits,
    reachability,
    solve_allocation,
    validate_input,
    validate_tensor_lifecycles,
)


def _allocate_with_edges(
    data: PlanningInput, edges: set[tuple[str, str]]
):
    nodes = data.all_operators()
    reach = reachability(edges, nodes)
    validate_tensor_lifecycles(data, reach)
    interfere = build_interference(data, reach)
    return solve_allocation(data, interfere), reach


def plan(data: PlanningInput) -> PlanOutcome:
    validate_input(data)
    base_edges = build_base_edges(data)
    base_result, base_reach = _allocate_with_edges(data, base_edges)

    # 1) 原依赖下能排进预算：保持并行，不加任何等待
    if base_result.feasible:
        return PlanOutcome(
            feasible=True,
            total_bytes=base_result.total_bytes,
            budget_bytes=data.budget_bytes,
            offsets=base_result.offsets,
            added_wait=None,
            note="原依赖下即可在预算内完成显存复用，未新增跨流等待",
            bins=base_result.bins,
        )

    # 2) 排不下：枚举至多一条新增的、不成环的跨流等待。
    # 候选按 (waiter, waitee) 字典序枚举，只在占用严格更小时更新，
    # 因此同占用下自然保留字典序最小的等待，结果确定。
    best = base_result
    best_wait: Wait | None = None
    for wait in candidate_waits(data, base_edges, base_reach):
        edges = set(base_edges)
        # waiter 等 waitee 完成 -> happens-before 边 waitee -> waiter
        edges.add((wait.waitee, wait.waiter))
        result, _ = _allocate_with_edges(data, edges)
        if result.total_bytes < best.total_bytes:
            best = result
            best_wait = wait

    if best.feasible:
        assert best_wait is not None
        return PlanOutcome(
            feasible=True,
            total_bytes=best.total_bytes,
            budget_bytes=data.budget_bytes,
            offsets=best.offsets,
            added_wait=best_wait,
            note=(
                f"原依赖下最小占用 {base_result.total_bytes} 字节超出预算，"
                f"新增一条跨流等待 {best_wait.waiter!r} 等待 "
                f"{best_wait.waitee!r} 完成后，占用降至 {best.total_bytes} 字节"
            ),
            bins=best.bins,
        )

    # 3) 加一条等待仍排不下：无解
    return PlanOutcome(
        feasible=False,
        total_bytes=best.total_bytes,
        budget_bytes=data.budget_bytes,
        offsets={},
        added_wait=best_wait,
        note=(
            "即使新增一条不成环的跨流等待，最小显存占用仍超出预算，无解"
            if best_wait is not None
            else "不存在可新增的合法跨流等待，且原依赖下最小占用超出预算，无解"
        ),
        best_effort_total_bytes=best.total_bytes,
        bins=best.bins,
    )
