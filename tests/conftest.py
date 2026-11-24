"""测试公共工具：把字典形式的用例组装成 PlanningInput，并提供结果校验器。"""

from __future__ import annotations

from app.planner.models import PlanningInput, TensorSpec, Wait


def make_input(
    stream_a: list[str],
    stream_b: list[str],
    tensors: list[dict],
    budget: int,
    waits: list[tuple[str, str]] | None = None,
    stream_names: tuple[str, str] = ("A", "B"),
) -> PlanningInput:
    return PlanningInput(
        streams={stream_names[0]: list(stream_a), stream_names[1]: list(stream_b)},
        waits=[Wait(u, v) for u, v in (waits or [])],
        tensors=[
            TensorSpec(
                name=t["name"],
                producer=t["producer"],
                consumers=tuple(t["consumers"]),
                size=t["size"],
                alignment=t.get("alignment", 1),
            )
            for t in tensors
        ],
        budget_bytes=budget,
    )


def assert_layout_valid(input_data: PlanningInput, outcome, expect_feasible: bool) -> None:
    """与 service 无关地复核一份布局的全部硬性约束。"""
    from app.planner.solver import build_base_edges, reachability

    assert outcome.feasible is expect_feasible
    if not expect_feasible:
        assert outcome.offsets == {}
        return

    edges = build_base_edges(input_data)
    if outcome.added_wait is not None:
        # waiter 等 waitee 完成 -> happens-before 边 waitee -> waiter
        edges.add((outcome.added_wait.waitee, outcome.added_wait.waiter))
    nodes = input_data.all_operators()
    reach = reachability(edges, nodes)  # 不成环，成环会抛异常

    by_name = {t.name: t for t in input_data.tensors}

    # 1. 每个张量都有偏移，且满足自身对齐
    assert set(outcome.offsets) == set(by_name)
    for name, off in outcome.offsets.items():
        t = by_name[name]
        assert off >= 0
        assert off % t.alignment == 0, f"{name} 偏移 {off} 未按 {t.alignment} 对齐"

    # 2. 任何一对可能同时在用的张量，字节区间不得重叠
    def hb(a: str, b: str) -> bool:
        return a == b or b in reach[a]

    def last_consumers(t: TensorSpec) -> list[str]:
        stream_of, pos = {}, {}
        for sid, ops in input_data.streams.items():
            for i, op in enumerate(ops):
                stream_of[op] = sid
                pos[op] = i
        latest: dict[str, str] = {}
        for c in t.consumers:
            s = stream_of[c]
            if s not in latest or pos[c] > pos[latest[s]]:
                latest[s] = c
        return list(latest.values())

    names = list(by_name)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x, y = by_name[names[i]], by_name[names[j]]
            x_dead_first = all(hb(c, y.producer) for c in last_consumers(x))
            y_dead_first = all(hb(c, x.producer) for c in last_consumers(y))
            can_overlap_in_time = not (x_dead_first or y_dead_first)
            if can_overlap_in_time:
                ox, oy = outcome.offsets[x.name], outcome.offsets[y.name]
                intervals = sorted(((ox, ox + x.size), (oy, oy + y.size)))
                assert intervals[0][1] <= intervals[1][0], (
                    f"张量 {x.name} 与 {y.name} 可能同时在用却发生字节重叠"
                )

    # 3. 总占用与预算
    end = max(off + by_name[name].size for name, off in outcome.offsets.items())
    assert end <= outcome.total_bytes
    assert outcome.total_bytes <= input_data.budget_bytes
