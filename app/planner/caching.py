"""按拓扑共享求解的缓存层（/plan 的并发优化）。

背景：同一模型重新导出后算子名可能变化，发版时还要按各档设备核对不同显存
预算。这两种情况下输入的“拓扑”（流结构、跨流等待、张量生命周期、尺寸与
对齐）其实相同，只是算子改名或预算不同。本模块把算子按流内位置归一化为
位置 id，以拓扑为键缓存两类与预算无关的求解结果：

- 原依赖下的最优布局（预算只影响可行性判定，不影响布局本身）；
- 全部候选新增等待及其布局。

语义约定：

- 同一拓扑、同一阶段（原布局 / 候选等待）的并发请求只计算一次，其余请求
  等待同一个 Future 并复用结果；已算出的结果直接命中缓存。
- 宽预算请求只依赖原布局：原布局够用即返回，绝不等待同拓扑正在进行的候选
  求解。
- 每个请求的响应与独立求解完全一致：候选占用打平时，按本次请求的算子名做
  (waiter, waitee) 字典序选择；校验报错也使用本次请求的算子名。
- 候选求解失败只影响正在等它的请求（异常随 Future 传播），失败结果不入
  缓存，下次紧预算请求会重试；原布局缓存不受影响，可继续沿用。
- 单进程最多缓存 16 个“无请求在途”的拓扑（LRU 淘汰）；有请求在途的拓扑
  不计入该上限。
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass

from . import solver
from .models import PlanOutcome, PlanningError, PlanningInput, TensorSpec, Wait

#: 单进程最多缓存的“无请求在途”拓扑数
MAX_IDLE_TOPOLOGIES = 16

#: 归一化输入的占位预算：共享求解与预算无关，可行性由各请求按自身预算判定
_DUMMY_BUDGET = 1


@dataclass(frozen=True)
class _Canonical:
    """一份输入的归一化视图：拓扑键、位置化输入与算子名双向映射。"""

    key: tuple
    data: PlanningInput
    cid_of: dict[str, str]  # 原始算子名 -> 位置 id
    name_of: dict[str, str]  # 位置 id -> 原始算子名


@dataclass(frozen=True)
class _BaseLayout:
    """原依赖下的共享求解结果（与预算无关）。"""

    total: int
    offsets: dict[str, int]
    bins: list[list[str]]
    edges: frozenset[tuple[str, str]]
    reach: dict[str, frozenset[str]]


@dataclass(frozen=True)
class _CandidateLayout:
    """一条候选新增等待（位置 id 表示）及其布局。"""

    waiter: str
    waitee: str
    total: int
    offsets: dict[str, int]
    bins: list[list[str]]


def _canonicalize(data: PlanningInput) -> _Canonical:
    """把算子按流内位置映射为位置 id，生成拓扑键与归一化输入。

    拓扑键只包含：各流 id 与算子数、位置化后的跨流等待（排序去序）、
    张量（名字、生产/消费算子位置、尺寸、对齐）。算子名与显存预算都不
    参与拓扑键，因此“只差算子名或预算”的两份输入会落到同一条缓存。
    调用前必须先通过 validate_input（保证所有算子引用存在）。
    """
    cid_of: dict[str, str] = {}
    name_of: dict[str, str] = {}
    cstreams: dict[str, list[str]] = {}
    for stream_id, ops in data.streams.items():
        cids = []
        for pos, op in enumerate(ops):
            cid = f"{stream_id}\x1f{pos}"
            cid_of[op] = cid
            name_of[cid] = op
            cids.append(cid)
        cstreams[stream_id] = cids

    # 等待与消费者集合的顺序不影响求解结果，归一化时排序，使拓扑键与
    # 归一化输入都是拓扑的纯函数
    cwaits = sorted((cid_of[w.waiter], cid_of[w.waitee]) for w in data.waits)
    ctensors = [
        TensorSpec(
            name=t.name,
            producer=cid_of[t.producer],
            consumers=tuple(sorted(cid_of[c] for c in t.consumers)),
            size=t.size,
            alignment=t.alignment,
        )
        for t in data.tensors
    ]
    key = (
        tuple(sorted((sid, len(ops)) for sid, ops in data.streams.items())),
        tuple(cwaits),
        tuple(
            (t.name, t.producer, t.consumers, t.size, t.alignment)
            for t in ctensors
        ),
    )
    cdata = PlanningInput(
        streams=cstreams,
        waits=[Wait(waiter=u, waitee=v) for u, v in cwaits],
        tensors=ctensors,
        budget_bytes=_DUMMY_BUDGET,
    )
    return _Canonical(key=key, data=cdata, cid_of=cid_of, name_of=name_of)


def _compute_base(cdata: PlanningInput) -> _BaseLayout:
    """原依赖下的最优布局。依赖图成环时抛出 PlanningError（不含算子名）。"""
    edges = solver.build_base_edges(cdata)
    reach = solver.reachability(edges, cdata.all_operators())
    interfere = solver.build_interference(cdata, reach)
    alloc = solver.solve_allocation(cdata, interfere)
    return _BaseLayout(
        total=alloc.total_bytes,
        offsets=alloc.offsets,
        bins=alloc.bins,
        edges=frozenset(edges),
        reach=reach,
    )


def _compute_candidates(
    cdata: PlanningInput, base: _BaseLayout
) -> list[_CandidateLayout]:
    """逐条求解候选新增等待的布局。

    新增边只会让 happens-before 更强，不会破坏已合法的张量生命周期，
    候选边本身也已按“不成环、非冗余”过滤，因此这里无需重复校验。
    """
    nodes = cdata.all_operators()
    out: list[_CandidateLayout] = []
    for wait in solver.candidate_waits(cdata, base.edges, base.reach):
        edges = set(base.edges)
        # waiter 等 waitee 完成 -> happens-before 边 waitee -> waiter
        edges.add((wait.waitee, wait.waiter))
        reach = solver.reachability(edges, nodes)
        interfere = solver.build_interference(cdata, reach)
        alloc = solver.solve_allocation(cdata, interfere)
        out.append(
            _CandidateLayout(
                waiter=wait.waiter,
                waitee=wait.waitee,
                total=alloc.total_bytes,
                offsets=alloc.offsets,
                bins=alloc.bins,
            )
        )
    return out


def _check_lifecycles(
    data: PlanningInput, canon: _Canonical, base: _BaseLayout
) -> None:
    """用共享可达集复核本次请求的张量生命周期，报错使用本次算子名。"""
    reach = base.reach
    cid_of = canon.cid_of
    for t in data.tensors:
        for consumer in t.consumers:
            born = cid_of[t.producer]
            used = cid_of[consumer]
            if not (born == used or used in reach[born]):
                raise PlanningError(
                    f"张量 {t.name} 的生产算子 {t.producer!r} 不在消费算子 "
                    f"{consumer!r} 之前，缺少依赖（存在非法的数据竞争）"
                )


def _base_outcome(data: PlanningInput, base: _BaseLayout) -> PlanOutcome:
    return PlanOutcome(
        feasible=True,
        total_bytes=base.total,
        budget_bytes=data.budget_bytes,
        offsets=base.offsets,
        added_wait=None,
        note="原依赖下即可在预算内完成显存复用，未新增跨流等待",
        bins=base.bins,
    )


def _best_outcome(
    data: PlanningInput,
    canon: _Canonical,
    base: _BaseLayout,
    candidates: list[_CandidateLayout],
) -> PlanOutcome:
    """与独立求解完全一致的候选选择：只在占用严格更小时更新，占用打平
    时按本次请求的算子名取 (waiter, waitee) 字典序最小者。"""
    budget = data.budget_bytes
    chosen: _CandidateLayout | None = None
    if candidates:
        min_total = min(c.total for c in candidates)
        if min_total < base.total:
            tied = [c for c in candidates if c.total == min_total]
            chosen = min(
                tied,
                key=lambda c: (canon.name_of[c.waiter], canon.name_of[c.waitee]),
            )

    best_total = chosen.total if chosen is not None else base.total
    if best_total <= budget:
        assert chosen is not None
        waiter = canon.name_of[chosen.waiter]
        waitee = canon.name_of[chosen.waitee]
        return PlanOutcome(
            feasible=True,
            total_bytes=chosen.total,
            budget_bytes=budget,
            offsets=chosen.offsets,
            added_wait=Wait(waiter=waiter, waitee=waitee),
            note=(
                f"原依赖下最小占用 {base.total} 字节超出预算，"
                f"新增一条跨流等待 {waiter!r} 等待 "
                f"{waitee!r} 完成后，占用降至 {chosen.total} 字节"
            ),
            bins=chosen.bins,
        )

    if chosen is not None:
        added_wait = Wait(
            waiter=canon.name_of[chosen.waiter],
            waitee=canon.name_of[chosen.waitee],
        )
        bins = chosen.bins
        note = "即使新增一条不成环的跨流等待，最小显存占用仍超出预算，无解"
    else:
        added_wait = None
        bins = base.bins
        note = "不存在可新增的合法跨流等待，且原依赖下最小占用超出预算，无解"
    return PlanOutcome(
        feasible=False,
        total_bytes=best_total,
        budget_bytes=budget,
        offsets={},
        added_wait=added_wait,
        note=note,
        best_effort_total_bytes=best_total,
        bins=bins,
    )


class _Entry:
    """一个拓扑的缓存条目；inflight > 0 表示有请求在途（不计入空闲上限）。"""

    __slots__ = ("cdata", "inflight", "base_future", "candidates_future")

    def __init__(self, cdata: PlanningInput) -> None:
        self.cdata = cdata
        self.inflight = 0
        self.base_future: Future | None = None
        self.candidates_future: Future | None = None


class PlannerCache:
    """按拓扑共享求解结果的规划缓存（线程安全，供单进程内并发请求使用）。"""

    def __init__(self, max_idle: int = MAX_IDLE_TOPOLOGIES) -> None:
        self._max_idle = max_idle
        self._lock = threading.Lock()
        self._entries: dict[tuple, _Entry] = {}
        # 无请求在途的拓扑，按最近使用排序（队尾最新），最多 max_idle 条
        self._idle: OrderedDict[tuple, None] = OrderedDict()
        # 统计：各阶段实际执行求解的次数（缓存命中不计），供自测核对
        self.stats = {"base_computes": 0, "candidate_computes": 0}

    def plan(self, data: PlanningInput) -> PlanOutcome:
        # 语义校验按本次请求的算子名报错（与独立求解一致）
        solver.validate_input(data)
        canon = _canonicalize(data)
        entry = self._acquire(canon)
        try:
            base = self._base_layout(entry)
            # 生命周期校验依赖共享可达集，但按本次请求的算子名报错
            _check_lifecycles(data, canon, base)
            # 原布局够用即返回，即使同拓扑的候选求解仍在进行
            if base.total <= data.budget_bytes:
                return _base_outcome(data, base)
            candidates = self._candidate_layouts(entry, base)
            return _best_outcome(data, canon, base, candidates)
        finally:
            self._release(canon.key, entry)

    # ------------------------------------------------------------------ #
    # 在途 / 空闲管理
    # ------------------------------------------------------------------ #
    def _acquire(self, canon: _Canonical) -> _Entry:
        with self._lock:
            entry = self._entries.get(canon.key)
            if entry is None:
                entry = _Entry(canon.data)
                self._entries[canon.key] = entry
            entry.inflight += 1
            self._idle.pop(canon.key, None)
            return entry

    def _release(self, key: tuple, entry: _Entry) -> None:
        with self._lock:
            entry.inflight -= 1
            if entry.inflight > 0:
                return
            # 变为“无请求在途”：登记为空闲拓扑，超出上限时淘汰最久未用的
            self._idle[key] = None
            while len(self._idle) > self._max_idle:
                evicted, _ = self._idle.popitem(last=False)
                del self._entries[evicted]

    # ------------------------------------------------------------------ #
    # 同阶段并发去重：第一个到达的请求负责计算，其余等待同一个 Future
    # ------------------------------------------------------------------ #
    def _base_layout(self, entry: _Entry) -> _BaseLayout:
        with self._lock:
            fut = entry.base_future
            if fut is None:
                fut = entry.base_future = Future()
                owner = True
                self.stats["base_computes"] += 1
            else:
                owner = False
        if owner:
            try:
                base = _compute_base(entry.cdata)
            except BaseException as exc:
                fut.set_exception(exc)
                with self._lock:
                    if entry.base_future is fut:
                        entry.base_future = None  # 失败不入缓存，下次重试
            else:
                fut.set_result(base)
        return fut.result()

    def _candidate_layouts(
        self, entry: _Entry, base: _BaseLayout
    ) -> list[_CandidateLayout]:
        with self._lock:
            fut = entry.candidates_future
            if fut is None:
                fut = entry.candidates_future = Future()
                owner = True
                self.stats["candidate_computes"] += 1
            else:
                owner = False
        if owner:
            try:
                layouts = _compute_candidates(entry.cdata, base)
            except BaseException as exc:
                # 失败只影响正在等它的请求；清空缓存让下次紧预算请求重试，
                # 原布局（base_future）保留可继续沿用
                fut.set_exception(exc)
                with self._lock:
                    if entry.candidates_future is fut:
                        entry.candidates_future = None
            else:
                fut.set_result(layouts)
        return fut.result()
