"""依赖分析与最优显存布局求解器。

只使用 Python 标准库。张量数上限为 8，采用基于“互不干扰张量组”（bin）划分的
子集 DP 求精确最优解，复杂度 O(3^n)，n=8 时仅 6561 次状态评估。

关键结论（见 README“规划语义”）：两个张量允许复用同一字节区间，当且仅当在
依赖偏序中一个张量的最后消费者 happens-before 另一个张量的生产者。把所有两两
互不干扰的张量放进同一个 bin、共享起始偏移，按 bin 顺序拼接得到的连续布局
与允许碎片化留洞的全局最优 1D 装箱等优，因此 DP 枚举 bin 划分即可得到最优解。
"""

from __future__ import annotations

from functools import lru_cache

from .models import AllocationResult, PlanningError, PlanningInput, TensorSpec, Wait

MAX_OPERATORS = 12
MAX_TENSORS = 8


# --------------------------------------------------------------------------- #
# 输入校验
# --------------------------------------------------------------------------- #
def validate_input(data: PlanningInput) -> None:
    if len(data.streams) != 2:
        raise PlanningError("必须且只能提供两条 CUDA 流")

    all_ops: list[str] = []
    for stream_id, ops in data.streams.items():
        if not ops:
            raise PlanningError(f"流 {stream_id!r} 不能为空")
        if len(set(ops)) != len(ops):
            raise PlanningError(f"流 {stream_id!r} 内算子 id 重复: {ops}")
        all_ops.extend(ops)

    if len(all_ops) > MAX_OPERATORS:
        raise PlanningError(f"算子总数 {len(all_ops)} 超过上限 {MAX_OPERATORS}")
    if len(set(all_ops)) != len(all_ops):
        raise PlanningError("两条流之间算子 id 不允许重复")

    if len(data.tensors) > MAX_TENSORS:
        raise PlanningError(f"张量总数 {len(data.tensors)} 超过上限 {MAX_TENSORS}")
    if not data.tensors:
        raise PlanningError("至少需要规划一个张量")

    op_set = set(all_ops)
    tensor_names: set[str] = set()
    for t in data.tensors:
        if t.name in tensor_names:
            raise PlanningError(f"张量 id 重复: {t.name}")
        tensor_names.add(t.name)
        if t.producer not in op_set:
            raise PlanningError(f"张量 {t.name} 的生产算子 {t.producer!r} 不存在")
        if not t.consumers:
            raise PlanningError(f"张量 {t.name} 必须至少有一个消费算子")
        if len(set(t.consumers)) != len(t.consumers):
            raise PlanningError(f"张量 {t.name} 的消费算子有重复")
        for c in t.consumers:
            if c not in op_set:
                raise PlanningError(f"张量 {t.name} 的消费算子 {c!r} 不存在")
        if t.producer in t.consumers:
            raise PlanningError(f"张量 {t.name} 的生产算子不能同时是消费算子")
        if t.size <= 0:
            raise PlanningError(f"张量 {t.name} 的字节数必须为正整数")
        if t.alignment <= 0 or (t.alignment & (t.alignment - 1)) != 0:
            raise PlanningError(
                f"张量 {t.name} 的对齐 {t.alignment} 不是 2 的幂"
            )

    if data.budget_bytes <= 0:
        raise PlanningError("显存预算必须为正整数")

    wait_pairs: set[tuple[str, str]] = set()
    for w in data.waits:
        if w.waiter not in op_set or w.waitee not in op_set:
            raise PlanningError(
                f"跨流等待引用了不存在的算子: {w.waiter!r} -> {w.waitee!r}"
            )
        if w.waiter == w.waitee:
            raise PlanningError("跨流等待不能是算子等待自身")
        pair = (w.waiter, w.waitee)
        if pair in wait_pairs:
            raise PlanningError(f"重复的跨流等待: {w.waiter!r} -> {w.waitee!r}")
        wait_pairs.add(pair)


# --------------------------------------------------------------------------- #
# 依赖图 / 可达性
# --------------------------------------------------------------------------- #
def build_base_edges(data: PlanningInput) -> set[tuple[str, str]]:
    """流内顺序边 + 已有跨流等待边。

    Wait(waiter=u, waitee=v) 表示 u 必须等 v 完成，即 v happens-before u，
    因此依赖边方向为 v -> u。
    """
    edges: set[tuple[str, str]] = set()
    for ops in data.streams.values():
        for a, b in zip(ops, ops[1:]):
            edges.add((a, b))
    for w in data.waits:
        edges.add((w.waitee, w.waiter))
    return edges


def reachability(
    edges: set[tuple[str, str]], nodes: list[str]
) -> dict[str, frozenset[str]]:
    """返回每个节点经有向边可到达的后继节点集合（不含自身）。

    边 u->v 表示 u happens-before v，因此 v 在 reach[u] 中。
    若图存在环会抛出 PlanningError。
    """
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    indeg: dict[str, int] = {n: 0 for n in nodes}
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1

    # Kahn 拓扑排序；处理 u 时把 u 及其祖先传播给其每个后继 v，
    # 因此 ancestors[v] 最终是“能到达 v 的全部节点”
    ancestors: dict[str, set[str]] = {n: set() for n in nodes}
    queue = [n for n in nodes if indeg[n] == 0]
    visited = 0
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        visited += 1
        for v in adj[u]:
            ancestors[v].update(ancestors[u])
            ancestors[v].add(u)
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)

    if visited != len(nodes):
        raise PlanningError("依赖图存在环，给定的跨流等待不成环约束被破坏")

    # 把“v 的祖先”翻转成“u 的后继可达集”
    reach: dict[str, set[str]] = {n: set() for n in nodes}
    for v, anc in ancestors.items():
        for u in anc:
            reach[u].add(v)
    return {n: frozenset(reach[n]) for n in nodes}


# --------------------------------------------------------------------------- #
# 张量生命周期与干扰图
# --------------------------------------------------------------------------- #
def _happens_before(
    reach: dict[str, frozenset[str]], a: str, b: str
) -> bool:
    """算子 a 是否 happens-before 算子 b。"""
    return a == b or b in reach[a]


def _tensor_born(data: PlanningInput, t: TensorSpec) -> str:
    return t.producer


def _tensor_last_consumers(data: PlanningInput, t: TensorSpec) -> tuple[str, ...]:
    """可能成为“最后完成的消费者”的算子集合。

    同一流内的多个消费者，靠后的一定最后完成，只保留每个流里最靠后的那个；
    不同流的消费者之间无偏序时都要保留。
    """
    stream_of: dict[str, str] = {}
    position: dict[str, int] = {}
    for stream_id, ops in data.streams.items():
        for i, op in enumerate(ops):
            stream_of[op] = stream_id
            position[op] = i

    latest_per_stream: dict[str, str] = {}
    for c in t.consumers:
        s = stream_of[c]
        if s not in latest_per_stream or position[c] > position[latest_per_stream[s]]:
            latest_per_stream[s] = c
    return tuple(latest_per_stream.values())


def _tensor_dead_before_born(
    data: PlanningInput,
    reach: dict[str, frozenset[str]],
    dead: TensorSpec,
    born: TensorSpec,
) -> bool:
    """dead 的整个生命周期是否严格先于 born 的生产（所有合法时序下）。"""
    born_op = _tensor_born(data, born)
    # dead 的每一个“可能最后完成”的消费者都必须 happens-before born 的生产者
    return all(
        _happens_before(reach, c, born_op)
        for c in _tensor_last_consumers(data, dead)
    )


def build_interference(
    data: PlanningInput,
    reach: dict[str, frozenset[str]],
) -> list[list[bool]]:
    """干扰矩阵：interfere[i][j] 为 True 表示存在两张量同时在用的合法时序。"""
    n = len(data.tensors)
    interfere = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = data.tensors[i], data.tensors[j]
            a_before_b = _tensor_dead_before_born(data, reach, a, b)
            b_before_a = _tensor_dead_before_born(data, reach, b, a)
            if not (a_before_b or b_before_a):
                interfere[i][j] = interfere[j][i] = True
    return interfere


def validate_tensor_lifecycles(
    data: PlanningInput,
    reach: dict[str, frozenset[str]],
) -> None:
    """生产必须 happens-before 每一个消费，否则输入本身就是数据竞争。"""
    for t in data.tensors:
        for c in t.consumers:
            if not _happens_before(reach, t.producer, c):
                raise PlanningError(
                    f"张量 {t.name} 的生产算子 {t.producer!r} 不在消费算子 "
                    f"{c!r} 之前，缺少依赖（存在非法的数据竞争）"
                )


# --------------------------------------------------------------------------- #
# 最优布局：独立集 bin 划分 + 子集 DP
# --------------------------------------------------------------------------- #
def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def solve_allocation(
    data: PlanningInput,
    interfere: list[list[bool]],
) -> AllocationResult:
    """在给定干扰图下求满足对齐的最小连续显存布局。

    DP 按 bin 顺序拼接；bin 内张量共享同一个起始偏移，bin 的对齐要求取内部
    最严格者，bin 大小取内部最大张量尺寸。
    """
    tensors = data.tensors
    n = len(tensors)
    size = [t.size for t in tensors]
    align = [t.alignment for t in tensors]

    full = (1 << n) - 1

    # 独立集（两两不干扰）枚举：indep[mask] 表示 mask 是否可放进同一个 bin
    indep: dict[int, bool] = {0: True}
    for mask in range(1, full + 1):
        low = mask & -mask
        i = low.bit_length() - 1
        rest = mask ^ low
        ok = indep[rest]
        if ok:
            bits = rest
            while bits:
                b = bits & -bits
                j = b.bit_length() - 1
                if interfere[i][j]:
                    ok = False
                    break
                bits ^= b
        indep[mask] = ok

    # bin 的尺寸与对齐
    @lru_cache(maxsize=None)
    def bin_metrics(mask: int) -> tuple[int, int]:
        s, a = 0, 1
        bits = mask
        while bits:
            b = bits & -bits
            i = b.bit_length() - 1
            s = max(s, size[i])
            a = max(a, align[i])
            bits ^= b
        return s, a

    # dp[mask] = (最小总字节, 该最优值下字典序最小的 bin 序列)
    # bin 序列用 tuple[int, ...] 表示，每个元素是一个 bin 的张量位掩码。
    @lru_cache(maxsize=None)
    def dp(mask: int) -> tuple[int, tuple[int, ...]]:
        if mask == 0:
            return 0, ()
        best_total: int | None = None
        best_bins: tuple[int, ...] | None = None
        sub = mask
        while sub:
            if indep[sub]:
                bsize, balign = bin_metrics(sub)
                tail_total, tail_bins = dp(mask ^ sub)
                # 前一个 bin 的结束地址需要按本 bin 对齐向上取整
                prev_end = tail_total
                start = _align_up(prev_end, balign)
                total = start + bsize
                bins = tail_bins + (sub,)
                if (
                    best_total is None
                    or total < best_total
                    or (total == best_total and bins < best_bins)
                ):
                    best_total = total
                    best_bins = bins
            sub = (sub - 1) & mask
        assert best_total is not None and best_bins is not None
        return best_total, best_bins

    total, bin_masks = dp(full)

    # 由 bin 序列还原每个张量的偏移
    offsets: dict[str, int] = {}
    cursor = 0
    bins_out: list[list[str]] = []
    for bm in bin_masks:
        bsize, balign = bin_metrics(bm)
        start = _align_up(cursor, balign)
        members: list[str] = []
        bits = bm
        while bits:
            b = bits & -bits
            i = b.bit_length() - 1
            offsets[tensors[i].name] = start
            members.append(tensors[i].name)
            bits ^= b
        bins_out.append(sorted(members))
        cursor = start + bsize

    return AllocationResult(
        feasible=total <= data.budget_bytes,
        total_bytes=total,
        offsets=offsets,
        bins=bins_out,
    )


# --------------------------------------------------------------------------- #
# 候选跨流等待枚举
# --------------------------------------------------------------------------- #
def candidate_waits(
    data: PlanningInput,
    base_edges: set[tuple[str, str]],
    reach: dict[str, frozenset[str]],
) -> list[Wait]:
    """枚举所有可能新增的、不成环且非冗余的跨流等待边。

    规则：
    - 必须跨流（流内顺序边已存在）；
    - 边 u->v 若已存在（u 已可达 v）则冗余，跳过；
    - 若 v 已可达 u，加入 u->v 必然成环，跳过；
    - 自环跳过。
    按 (waiter, waitee) 字典序返回，保证结果确定性。
    """
    stream_of: dict[str, str] = {}
    for stream_id, ops in data.streams.items():
        for op in ops:
            stream_of[op] = stream_id

    out: list[Wait] = []
    nodes = data.all_operators()
    for u in nodes:
        for v in nodes:
            if u == v:
                continue
            if stream_of[u] == stream_of[v]:
                continue
            if v in reach[u]:
                continue  # 已有依赖（边或传递可达），冗余
            if u in reach[v]:
                continue  # 反向已可达，加入即成环
            out.append(Wait(waiter=u, waitee=v))
    out.sort(key=lambda w: (w.waiter, w.waitee))
    return out
