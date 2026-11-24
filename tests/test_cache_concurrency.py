"""共享求解缓存的并发自测。

按任务要求，只核对两类场景下的**求解次数**与**返回顺序/结果**：

1. 宽预算并发：同一拓扑、仅算子名不同的多份请求同时到达，原布局只算一次，
   不触发候选求解，各响应与独立求解一致；
2. 宽+紧预算并发：紧预算请求触发候选求解且候选仍在计算时，宽预算请求不等
   候选、按原布局先返回；候选只算一次；占用打平时按本次请求的算子名选等待。
"""

from __future__ import annotations

import threading

from app.planner import caching, solver
from app.planner.caching import PlannerCache
from app.planner.service import plan as independent_plan

from conftest import make_input


def _variant(ops_a: list[str], ops_b: list[str], budget: int):
    """同一拓扑（TA: A0->A1，TB: B0->B1，各 100 字节）的不同算子名变体。"""
    return make_input(
        ops_a,
        ops_b,
        [
            {"name": "TA", "producer": ops_a[0], "consumers": [ops_a[1]], "size": 100, "alignment": 1},
            {"name": "TB", "producer": ops_b[0], "consumers": [ops_b[1]], "size": 100, "alignment": 1},
        ],
        budget=budget,
    )


def _counting_solve(monkeypatch) -> list[int]:
    """把 solver.solve_allocation 包成计数版本，返回 [调用次数]。"""
    calls = [0]
    lock = threading.Lock()
    real_solve = solver.solve_allocation

    def counting(*args, **kwargs):
        with lock:
            calls[0] += 1
        return real_solve(*args, **kwargs)

    monkeypatch.setattr(solver, "solve_allocation", counting)
    return calls


def test_wide_budget_concurrent_requests_share_single_base_solve(monkeypatch):
    cache = PlannerCache()
    variants = [
        (["a1", "a2", "a3"], ["b1", "b2", "b3"]),
        (["x1", "x2", "x3"], ["y1", "y2", "y3"]),
        (["p1", "p2", "p3"], ["q1", "q2", "q3"]),
        (["u1", "u2", "u3"], ["v1", "v2", "v3"]),
    ]
    inputs = [_variant(a, b, budget=200) for a, b in variants]
    # 参照：每份输入独立求解（在统计求解次数之前完成）
    expected = [independent_plan(d) for d in inputs]
    assert all(e.feasible and e.added_wait is None for e in expected)

    solve_calls = _counting_solve(monkeypatch)

    results: list = [None] * len(inputs)
    errors: list = [None] * len(inputs)
    barrier = threading.Barrier(len(inputs))

    def worker(i: int) -> None:
        try:
            barrier.wait(timeout=10)
            results[i] = cache.plan(inputs[i])
        except BaseException as exc:  # pragma: no cover - 失败时下面统一断言
            errors[i] = exc

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(inputs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in threads)
    assert errors == [None] * len(inputs)

    # 求解次数：同一拓扑的并发宽预算请求，原布局只算一次，且不触发候选求解
    assert cache.stats == {"base_computes": 1, "candidate_computes": 0}
    assert solve_calls[0] == 1
    # 返回结果：各请求的响应与独立求解一致（各自使用本次算子名）
    assert results == expected


def test_wide_and_tight_budget_concurrent(monkeypatch):
    cache = PlannerCache()
    # 同一拓扑：宽预算请求用 a/b 命名，紧预算请求换一套算子名（x/a）
    wide = _variant(["a1", "a2", "a3"], ["b1", "b2", "b3"], budget=200)
    tight = _variant(["x1", "x2", "x3"], ["a1", "a2", "a3"], budget=100)

    expected_wide = independent_plan(wide)
    expected_tight = independent_plan(tight)
    assert expected_wide.added_wait is None
    # 紧预算参照结果确实新增等待；该拓扑有 4 条候选占用打平（均为 100），
    # 按本次算子名字典序应选 ("a1", "x2")，而非按位置序的 ("x1", "a2")
    assert (expected_tight.added_wait.waiter, expected_tight.added_wait.waitee) == ("a1", "x2")

    solve_calls = _counting_solve(monkeypatch)

    # 把候选求解挂起，验证宽预算请求不等候选即可返回
    candidates_started = threading.Event()
    release_candidates = threading.Event()
    real_candidates = caching._compute_candidates

    def slow_candidates(cdata, base):
        candidates_started.set()
        assert release_candidates.wait(timeout=10)
        return real_candidates(cdata, base)

    monkeypatch.setattr(caching, "_compute_candidates", slow_candidates)

    completed: list[str] = []
    results: dict = {}
    errors: dict = {}

    def run_tight() -> None:
        try:
            results["tight"] = cache.plan(tight)
        except BaseException as exc:  # pragma: no cover - 失败时下面统一断言
            errors["tight"] = exc
        else:
            completed.append("tight")

    thread = threading.Thread(target=run_tight)
    thread.start()
    assert candidates_started.wait(timeout=10)  # 紧预算请求已阻塞在候选求解中

    results["wide"] = cache.plan(wide)  # 原布局够用，必须不等候选直接返回
    completed.append("wide")

    release_candidates.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert not errors

    # 返回顺序：宽预算先于紧预算完成（未等待同拓扑的候选求解）
    assert completed == ["wide", "tight"]
    # 求解次数：原布局与候选各只算一次，两个请求共享（18 条候选等待）
    assert cache.stats == {"base_computes": 1, "candidate_computes": 1}
    assert solve_calls[0] == 1 + 18
    # 响应与独立求解一致：宽预算不加等待，紧预算按本次算子名选等待
    assert results["wide"] == expected_wide
    assert results["tight"] == expected_tight
