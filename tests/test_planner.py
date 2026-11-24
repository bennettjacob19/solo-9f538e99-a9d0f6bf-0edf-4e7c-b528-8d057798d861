"""核心规划逻辑测试：复用、跨流等待、无解、校验与最优性。"""

from __future__ import annotations

import itertools

import pytest

from app.planner.models import PlanningError
from app.planner.service import plan
from app.planner.solver import (
    build_base_edges,
    build_interference,
    candidate_waits,
    reachability,
    solve_allocation,
    validate_input,
)

from conftest import assert_layout_valid, make_input


# --------------------------------------------------------------------------- #
# 1. 基本复用：流内顺序生命周期不相交的张量必须复用同一段显存
# --------------------------------------------------------------------------- #
def test_intra_stream_reuse_same_offset():
    data = make_input(
        ["a1", "a2", "a3"],
        ["b1", "b2"],
        [
            {"name": "T1", "producer": "a1", "consumers": ["a2"], "size": 100, "alignment": 1},
            {"name": "T2", "producer": "a2", "consumers": ["a3"], "size": 100, "alignment": 1},
            {"name": "T3", "producer": "b1", "consumers": ["b2"], "size": 100, "alignment": 1},
        ],
        budget=1000,
    )
    out = plan(data)
    assert out.feasible and out.added_wait is None
    # T1 死在 a2，T2 生于 a2，生命周期严格先后 -> 同偏移
    assert out.offsets["T1"] == out.offsets["T2"]
    assert_layout_valid(data, out, True)


# --------------------------------------------------------------------------- #
# 2. 跨流无依赖：两个流的张量在某些合法时序下同时在用，必须错开
# --------------------------------------------------------------------------- #
def test_cross_stream_no_dependency_must_not_overlap():
    data = make_input(
        ["a1", "a2"],
        ["b1", "b2"],
        [
            {"name": "TA", "producer": "a1", "consumers": ["a2"], "size": 100, "alignment": 1},
            {"name": "TB", "producer": "b1", "consumers": ["b2"], "size": 100, "alignment": 1},
        ],
        budget=1000,
    )
    out = plan(data)
    assert out.feasible
    # 无任何跨流边，存在 TA、TB 同时在用的时序 -> 偏移区间不相交
    assert abs(out.offsets["TA"] - out.offsets["TB"]) >= 100
    assert out.total_bytes == 200
    assert_layout_valid(data, out, True)


# --------------------------------------------------------------------------- #
# 3. 预算充足时保持并行，绝不主动加等待
# --------------------------------------------------------------------------- #
def test_keeps_parallel_when_budget_allows():
    data = make_input(
        ["a1", "a2"],
        ["b1", "b2"],
        [
            {"name": "TA", "producer": "a1", "consumers": ["a2"], "size": 100, "alignment": 1},
            {"name": "TB", "producer": "b1", "consumers": ["b2"], "size": 100, "alignment": 1},
        ],
        budget=200,
    )
    out = plan(data)
    assert out.feasible and out.added_wait is None


# --------------------------------------------------------------------------- #
# 4. 预算不足时新增一条不成环等待，使显存复用成为可能
# --------------------------------------------------------------------------- #
def test_adds_single_wait_when_budget_tight():
    data = make_input(
        ["a1", "a2", "a3"],
        ["b1", "b2", "b3"],
        [
            {"name": "TA", "producer": "a1", "consumers": ["a2"], "size": 100, "alignment": 1},
            {"name": "TB", "producer": "b1", "consumers": ["b2"], "size": 100, "alignment": 1},
        ],
        budget=100,
    )
    out = plan(data)
    assert out.feasible
    assert out.added_wait is not None
    # 等待方向必须让某一张量先死、另一张后出生
    assert out.total_bytes == 100
    assert out.offsets["TA"] == out.offsets["TB"]
    assert_layout_valid(data, out, True)

    # 字典序 tie-break：所有能把占用降到 100 的等待中，
    # 取 (waiter, waitee) 字典序最小者 -> waiter 最小是 a1，即 a1 等 b2 完成
    assert (out.added_wait.waiter, out.added_wait.waitee) == ("a1", "b2")


def test_wait_candidate_never_creates_cycle():
    data = make_input(
        ["a1", "a2"],
        ["b1", "b2"],
        [
            {"name": "TA", "producer": "a1", "consumers": ["a2"], "size": 100, "alignment": 1},
            {"name": "TB", "producer": "b1", "consumers": ["b2"], "size": 100, "alignment": 1},
        ],
        budget=50,
        waits=[("a2", "b1")],  # a2 等 b1（边 b1->a2），那么 b1 等 a2 即成环，候选里不能出现
    )
    edges = build_base_edges(data)
    reach = reachability(edges, data.all_operators())
    cands = candidate_waits(data, edges, reach)
    assert ("b1", "a2") not in {(w.waiter, w.waitee) for w in cands}
    # 预算 50 小于任何单张量，必然无解
    out = plan(data)
    assert not out.feasible
    assert out.offsets == {}
    assert out.best_effort_total_bytes is not None


# --------------------------------------------------------------------------- #
# 5. 加一条等待也排不下 -> 无解
# --------------------------------------------------------------------------- #
def test_infeasible_when_no_wait_helps():
    data = make_input(
        ["a1", "a2"],
        ["b1", "b2"],
        [
            # 两个张量都活到各自流的最后一个算子，一条跨流边最多串起整个流，
            # 但两张量各 300 字节，预算 500：串行后仍需 max(300,300)=300 其实可排，
            # 因此这里用三个互不可复用的张量构造真正无解的预算
            {"name": "TA", "producer": "a1", "consumers": ["a2"], "size": 300, "alignment": 1},
            {"name": "TB", "producer": "b1", "consumers": ["b2"], "size": 300, "alignment": 1},
        ],
        budget=200,
    )
    out = plan(data)
    assert not out.feasible
    # 一条等待可以把两流完全串行，最小占用降到 300，但仍 > 200
    assert out.best_effort_total_bytes == 300


# --------------------------------------------------------------------------- #
# 6. 对齐：偏移必须满足各自对齐，且拼接处做向上取整
# --------------------------------------------------------------------------- #
def test_alignment_is_respected():
    data = make_input(
        ["a1", "a2", "a3", "a4"],
        ["b1"],
        [
            {"name": "T1", "producer": "a1", "consumers": ["a2"], "size": 100, "alignment": 64},
            {"name": "T2", "producer": "a2", "consumers": ["a3"], "size": 100, "alignment": 128},
            {"name": "T3", "producer": "a3", "consumers": ["a4"], "size": 100, "alignment": 32},
        ],
        budget=1000,
    )
    out = plan(data)
    assert out.feasible
    for name, off in out.offsets.items():
        align = {"T1": 64, "T2": 128, "T3": 32}[name]
        assert off % align == 0
    assert_layout_valid(data, out, True)


# --------------------------------------------------------------------------- #
# 7. 输入语义校验
# --------------------------------------------------------------------------- #
def test_validation_errors():
    bad = make_input(
        ["a1"], ["b1", "b2"],
        [{"name": "T", "producer": "a1", "consumers": ["b2"], "size": 10, "alignment": 3}],
        budget=100,
    )
    with pytest.raises(PlanningError, match="2 的幂"):
        validate_input(bad)

    missing = make_input(
        ["a1"], ["b1"],
        [{"name": "T", "producer": "a1", "consumers": ["xx"], "size": 10, "alignment": 1}],
        budget=100,
    )
    with pytest.raises(PlanningError, match="不存在"):
        validate_input(missing)

    race = make_input(
        ["a1", "a2"], ["b1", "b2"],
        [{"name": "T", "producer": "b2", "consumers": ["a1"], "size": 10, "alignment": 1}],
        budget=100,
    )
    with pytest.raises(PlanningError, match="数据竞争"):
        plan(race)

    too_many_ops = make_input(
        [f"a{i}" for i in range(7)], [f"b{i}" for i in range(6)],
        [{"name": "T", "producer": "a0", "consumers": ["a1"], "size": 1, "alignment": 1}],
        budget=100,
    )
    with pytest.raises(PlanningError, match="上限 12"):
        validate_input(too_many_ops)

    too_many_tensors = make_input(
        ["a1", "a2"], ["b1", "b2"],
        [
            {"name": f"T{i}", "producer": "a1", "consumers": ["a2"], "size": 1, "alignment": 1}
            for i in range(9)
        ],
        budget=100,
    )
    with pytest.raises(PlanningError, match="上限 8"):
        validate_input(too_many_tensors)


# --------------------------------------------------------------------------- #
# 8. 已有跨流等待被尊重：b2 等 a2 后，TA 可与 TB 复用
# --------------------------------------------------------------------------- #
def test_existing_wait_enables_reuse_without_adding_new_one():
    data = make_input(
        ["a1", "a2", "a3"],
        ["b1", "b2", "b3"],
        [
            {"name": "TA", "producer": "a1", "consumers": ["a2"], "size": 100, "alignment": 1},
            {"name": "TB", "producer": "b2", "consumers": ["b3"], "size": 100, "alignment": 1},
        ],
        budget=100,
        # b2 等 a2 完成（边 a2->b2）：TA 死于 a2，TB 生于 b2 且 b2 在 a2 之后 ->
        # TA 整个生命周期严格先于 TB，可复用同一段显存，且无需新增等待
        waits=[("b2", "a2")],
    )
    out = plan(data)
    assert out.feasible and out.added_wait is None
    assert out.offsets["TA"] == out.offsets["TB"]
    assert_layout_valid(data, out, True)


# --------------------------------------------------------------------------- #
# 9. 多消费者：张量活到每流最后一个消费者，跨流消费者之间无偏序时不能提前释放
# --------------------------------------------------------------------------- #
def test_multi_consumer_liveness():
    data = make_input(
        ["a1", "a2", "a3", "a4"],
        ["b1", "b2", "b3", "b4"],
        [
            # T 被 a3 和 b3 消费：两个流的最后消费者都完成前 T 都在用。
            # b3 等 a1 保证跨流生产->消费依赖合法。
            {"name": "T", "producer": "a1", "consumers": ["a3", "b3"], "size": 100, "alignment": 1},
            # U 生于 b2：b2 与 a3 之间无偏序，存在 T 仍在用而 U 已出生的时序
            {"name": "U", "producer": "b2", "consumers": ["b4"], "size": 100, "alignment": 1},
        ],
        budget=1000,
        waits=[("b3", "a1")],  # b3 等 a1 完成（边 a1->b3），保证跨流生产->消费合法
    )
    edges = build_base_edges(data)
    reach = reachability(edges, data.all_operators())
    interfere = build_interference(data, reach)
    i_t, i_u = 0, 1
    assert interfere[i_t][i_u]
    out = plan(data)
    assert out.offsets["T"] != out.offsets["U"]
    assert_layout_valid(data, out, True)


# --------------------------------------------------------------------------- #
# 10. 最优性暴力对照：穷举全部有序 bin 划分，DP 结果必须与暴力最优一致
# --------------------------------------------------------------------------- #
def test_dp_matches_brute_force_partitions():
    data = make_input(
        ["a1", "a2", "a3", "a4"],
        ["b1", "b2", "b3", "b4"],
        [
            {"name": "T1", "producer": "a1", "consumers": ["a3"], "size": 130, "alignment": 16},
            {"name": "T2", "producer": "a2", "consumers": ["b2"], "size": 70, "alignment": 32},
            {"name": "T3", "producer": "b1", "consumers": ["b3"], "size": 200, "alignment": 64},
            {"name": "T4", "producer": "a3", "consumers": ["a4"], "size": 90, "alignment": 8},
            {"name": "T5", "producer": "b2", "consumers": ["b4"], "size": 110, "alignment": 128},
        ],
        budget=10_000,
    )
    edges = build_base_edges(data)
    reach = reachability(edges, data.all_operators())
    interfere = build_interference(data, reach)
    result = solve_allocation(data, interfere)

    n = len(data.tensors)

    def align_up(v: int, a: int) -> int:
        return (v + a - 1) & ~(a - 1)

    # 穷举全部集合划分（Bell 数 B5=52，量很小），只保留每个块都是独立集的方案
    def all_partitions(items: tuple[int, ...]) -> list[list[list[int]]]:
        if not items:
            return [[]]
        head, rest = items[0], items[1:]
        out: list[list[list[int]]] = []
        for tail in all_partitions(rest):
            for k in range(len(tail)):
                trial = [list(b) for b in tail]
                trial[k].append(head)
                out.append(trial)
            out.append([[head], *tail])
        return out

    def independent(block: list[int]) -> bool:
        return all(not interfere[i][j] for i, j in itertools.combinations(block, 2))

    brute_best = None
    for partition in all_partitions(tuple(range(n))):
        if not all(independent(block) for block in partition):
            continue
        # bin 顺序也做全排列，取最小
        for ordered in itertools.permutations(partition):
            cursor = 0
            for block in ordered:
                balign = max(data.tensors[i].alignment for i in block)
                bsize = max(data.tensors[i].size for i in block)
                cursor = align_up(cursor, balign) + bsize
            brute_best = cursor if brute_best is None else min(brute_best, cursor)

    assert brute_best is not None
    assert result.total_bytes == brute_best


# --------------------------------------------------------------------------- #
# 11. 随机 fuzz：随机依赖/张量下，布局永远满足硬性约束，且不劣于“全串行”上界
# --------------------------------------------------------------------------- #
def test_fuzz_random_instances():
    import random

    rng = random.Random(20261123)
    for _ in range(300):
        na = rng.randint(1, 5)
        nb = rng.randint(1, 5)
        ops_a = [f"a{i}" for i in range(na)]
        ops_b = [f"b{i}" for i in range(nb)]
        waits: list[tuple[str, str]] = []
        # 随机加若干不成环跨流边：只允许 a_p -> b_q 或 b_q -> a_p 中天然 DAG 的方向组合
        for a in ops_a:
            for b in ops_b:
                r = rng.random()
                if r < 0.2:
                    waits.append((b, a))
                elif r < 0.4:
                    waits.append((a, b))
        # 随机边可能成环，成环用例直接跳过（service 会抛 PlanningError）
        n_tensors = rng.randint(1, min(6, na + nb - 1))
        all_ops = ops_a + ops_b
        tensors = []
        used = set()
        for k in range(n_tensors):
            producer = rng.choice(all_ops[:-1]) if len(all_ops) > 1 else all_ops[0]
            later = [op for op in all_ops if op != producer]
            if not later:
                break
            n_cons = rng.randint(1, min(3, len(later)))
            consumers = tuple(rng.sample(later, n_cons))
            name = f"T{k}"
            if name in used:
                continue
            used.add(name)
            tensors.append(
                {
                    "name": name,
                    "producer": producer,
                    "consumers": list(consumers),
                    "size": rng.choice([1, 7, 16, 100, 255, 1024]),
                    "alignment": rng.choice([1, 2, 4, 8, 16, 64, 256]),
                }
            )
        if not tensors:
            continue
        data = make_input(ops_a, ops_b, tensors, budget=10**9, waits=waits)
        try:
            out = plan(data)
        except PlanningError:
            continue  # 随机数据可能含环或数据竞争，跳过这类非法用例
        assert out.feasible  # 预算给得极大
        assert out.added_wait is None  # 预算极大，绝不应该加等待
        assert_layout_valid(data, out, True)


# --------------------------------------------------------------------------- #
# 12. 单等待最优性：服务选出的等待必须等于暴力枚举全部单条等待的最优结果
# --------------------------------------------------------------------------- #
def test_added_wait_matches_brute_force_single_edge_search():
    import random

    from app.planner.models import Wait as WaitModel
    from app.planner.solver import candidate_waits

    rng = random.Random(99)
    A = [f"a{i}" for i in range(1, 5)]
    B = [f"b{i}" for i in range(1, 5)]

    for _ in range(60):
        tensors = []
        for k in range(rng.randint(2, 5)):
            if rng.random() < 0.5:
                pi = rng.randrange(3)
                tensors.append(
                    {"name": f"T{k}", "producer": A[pi],
                     "consumers": [A[rng.randrange(pi + 1, 4)]],
                     "size": rng.choice([16, 100, 300]),
                     "alignment": rng.choice([1, 16, 64])}
                )
            else:
                pi = rng.randrange(3)
                tensors.append(
                    {"name": f"T{k}", "producer": B[pi],
                     "consumers": [B[rng.randrange(pi + 1, 4)]],
                     "size": rng.choice([16, 100, 300]),
                     "alignment": rng.choice([1, 16, 64])}
                )
        # 紧预算，迫使服务进入“加一条等待”分支
        data = make_input(A, B, tensors, budget=rng.choice([1, 50, 150]))

        base_edges = build_base_edges(data)
        base_reach = reachability(base_edges, data.all_operators())
        base_alloc = solve_allocation(data, build_interference(data, base_reach))

        # 暴力枚举每条合法单等待
        brute: list[tuple[int, str, str]] = []
        for w in candidate_waits(data, base_edges, base_reach):
            edges = set(base_edges)
            edges.add((w.waitee, w.waiter))
            reach = reachability(edges, data.all_operators())
            res = solve_allocation(data, build_interference(data, reach))
            brute.append((res.total_bytes, w.waiter, w.waitee))

        out = plan(data)

        if base_alloc.feasible:
            assert out.added_wait is None
            continue

        if not brute:
            assert not out.feasible
            continue

        best_total, best_waiter, best_waitee = min(brute)
        if best_total <= data.budget_bytes:
            assert out.feasible
            assert out.added_wait == WaitModel(best_waiter, best_waitee)
        else:
            assert not out.feasible
            assert out.best_effort_total_bytes == best_total
            if best_total < base_alloc.total_bytes:
                # 至少有一条等待降低了占用 -> 必须报告字典序最小的最优等待
                assert out.added_wait == WaitModel(best_waiter, best_waitee)
            else:
                # 没有任何等待能改善占用 -> 不报告等待
                assert out.added_wait is None
