#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用 CPU 模拟多 GPU 的 pipeline stage：
1. 定义一个小网络（MLP）
2. 逐层 profile 前向/反向耗时，得到真实时延 fᵢ / bᵢ
3. 将若干层聚合成 Block（例如 Linear+ReLU），避免粒度过细
4. 基于 Block 的“真实时延均衡 + 通信感知”划分：
   - 目标均衡每段 ∑(fᵢ+bᵢ)
   - 惩罚跨段边界的 activation（减少通信）
5. 对比“按层/按块均分”与“按真实时延均衡+通信感知/DP最优”的瓶颈、均衡度
6. 基于简单调度器（GPipe/1F1B）模拟 makespan 与 bubble，并画图
"""

import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


# =============================
# 1. 定义一个小网络
# =============================

class TinyMLP(nn.Module):
    """
    一个简单的多层感知机：
    输入维度 1024，hidden 尺寸不等，这样每层计算量和 activation 大小都有差异。
    """

    def __init__(self, in_dim: int = 1024):
        super().__init__()
        # 不同隐藏层维度 -> 不同的计算&激活大小
        hidden_dims = [2048, 1536, 4096, 1024, 3072, 512, 2048, 1024]
        layers: List[nn.Module] = []
        dims = [in_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# =============================
# 2. 逐层 profile 前向/反向耗时
# =============================

def _infer_layer_input_shapes(
    model: nn.Sequential,
    input_shape: Tuple[int, ...],
    device: str = "cpu",
) -> List[Tuple[int, ...]]:
    x = torch.randn(*input_shape, device=device)
    in_shapes: List[Tuple[int, ...]] = []
    with torch.no_grad():
        for layer in model:
            in_shapes.append(tuple(x.shape))
            x = layer(x)
    return in_shapes


def profile_model_fwd_bwd(
    model: nn.Sequential,
    input_shape: Tuple[int, ...],
    runs: int = 20,
    device: str = "cpu",
) -> Tuple[List[float], List[float], List[int]]:
    """
    返回：
      - f_times[i]: 第 i 层平均前向耗时 (秒)
      - b_times[i]: 第 i 层平均反向耗时 (秒)
      - act_sizes[i]: 第 i 层输出 activation 的元素个数 (numel)
    """
    model.to(device)
    model.eval()

    num_layers = len(model)
    f_times = [0.0 for _ in range(num_layers)]
    b_times = [0.0 for _ in range(num_layers)]
    act_sizes = [0 for _ in range(num_layers)]

    # forward 统计
    x0 = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        for run in range(runs):
            out = x0
            for idx, layer in enumerate(model):
                start = time.perf_counter()
                out = layer(out)
                end = time.perf_counter()
                f_times[idx] += (end - start)
                if run == 0:
                    act_sizes[idx] = out.numel()
    f_times = [t / runs for t in f_times]

    # backward 统计（逐层独立）
    in_shapes = _infer_layer_input_shapes(model, input_shape, device=device)
    for idx, layer in enumerate(model):
        shape = in_shapes[idx]
        layer.train()
        bt = 0.0
        for _ in range(runs):
            x = torch.randn(*shape, device=device, dtype=torch.float32, requires_grad=True)
            y = layer(x)
            loss = y.sum()
            start = time.perf_counter()
            loss.backward()
            end = time.perf_counter()
            bt += (end - start)
            for p in layer.parameters(recurse=True):
                if p.grad is not None:
                    p.grad = None
        b_times[idx] = bt / runs
        layer.eval()

    return f_times, b_times, act_sizes


# =============================
# 3. 划分算法（均分、贪心、DP）
# =============================

def partition_uniform(num_layers: int, num_stages: int) -> List[Tuple[int, int]]:
    if num_layers < num_stages:
        raise ValueError("层数比 stage 数还少，没法均分")
    base = num_layers // num_stages
    rem = num_layers % num_stages
    boundaries = []
    start = 0
    for s in range(num_stages):
        extra = 1 if s < rem else 0
        end = start + base + extra
        boundaries.append((start, end))
        start = end
    return boundaries


def stage_loads(layer_times: List[float], partitions: List[Tuple[int, int]]) -> List[float]:
    loads = []
    for (s, e) in partitions:
        loads.append(sum(layer_times[s:e]))
    return loads


def partition_profile_balanced(
    layer_times: List[float],
    comm_sizes: List[int],
    num_stages: int,
    alpha: float = 0.05,
) -> List[Tuple[int, int]]:
    """
    贪心：均衡计算 + 通信感知（边界激活越大惩罚越大）
    """
    L = len(layer_times)
    if L < num_stages:
        raise ValueError("层数比 stage 数少，无法划分")

    comm_sizes = np.array(comm_sizes, dtype=np.float64)
    if comm_sizes.max() > 0:
        comm_costs = (comm_sizes / comm_sizes.max()).tolist()
    else:
        comm_costs = comm_sizes.tolist()

    total_time = sum(layer_times)
    target = total_time / num_stages

    partitions: List[Tuple[int, int]] = []
    start = 0
    acc = 0.0
    for i in range(L):
        acc += layer_times[i]
        remaining_layers = L - (i + 1)
        remaining_stages = num_stages - (len(partitions) + 1)
        if acc >= target and remaining_stages > 0 and remaining_layers >= remaining_stages:
            cut_idx = i

            def stage_score(boundary_idx: int) -> float:
                stage_time = sum(layer_times[start:boundary_idx + 1])
                comm_penalty = comm_costs[boundary_idx] if boundary_idx < len(comm_costs) else 0.0
                return abs(stage_time - target) + alpha * comm_penalty

            best_cut = cut_idx
            best_score = stage_score(cut_idx)
            if cut_idx - 1 >= start:
                prev_score = stage_score(cut_idx - 1)
                if prev_score < best_score:
                    best_score = prev_score
                    best_cut = cut_idx - 1

            partitions.append((start, best_cut + 1))
            start = best_cut + 1
            acc = 0.0

    if start < L:
        partitions.append((start, L))

    def _stage_loads(parts: List[Tuple[int, int]]) -> List[float]:
        return stage_loads(layer_times, parts)

    while len(partitions) < num_stages:
        loads = _stage_loads(partitions)
        idx_max = int(np.argmax(loads))
        s, e = partitions[idx_max]
        if e - s <= 1:
            break
        mid = (s + e) // 2
        partitions[idx_max] = (s, mid)
        partitions.insert(idx_max + 1, (mid, e))

    while len(partitions) > num_stages:
        loads = _stage_loads(partitions)
        best_idx = 0
        best_sum = loads[0] + loads[1]
        for i in range(len(partitions) - 1):
            ssum = loads[i] + loads[i + 1]
            if ssum < best_sum:
                best_sum = ssum
                best_idx = i
        s1, e1 = partitions[best_idx]
        s2, e2 = partitions[best_idx + 1]
        partitions[best_idx] = (s1, e2)
        del partitions[best_idx + 1]

    return partitions


# =============================
# 3.1 高级划分：最小化瓶颈，次级最小通信（DP）
# =============================

def partition_min_max_then_min_comm_dp(
    block_times: List[float],
    comm_sizes: List[int],
    num_stages: int,
) -> List[Tuple[int, int]]:
    n = len(block_times)
    if n < num_stages:
        raise ValueError("Block 数少于 stage 数，无法划分")

    prefix = [0.0]
    for t in block_times:
        prefix.append(prefix[-1] + t)

    def seg_sum(l: int, r: int) -> float:
        return prefix[r + 1] - prefix[l]

    lo = max(block_times)
    hi = sum(block_times)

    def dp_min_comm(T: float, need_path: bool = False):
        INF = float("inf")
        p = num_stages
        DP = [[INF] * (n + 1) for _ in range(p + 1)]
        PR = [[-1] * (n + 1) for _ in range(p + 1)]
        DP[0][0] = 0.0
        for k in range(1, p + 1):
            for i in range(1, n + 1):
                best_cost = INF
                best_j = -1
                cur_sum = 0.0
                for j in range(i - 1, -1, -1):
                    cur_sum += block_times[j]
                    if cur_sum > T:
                        break
                    if DP[k - 1][j] == INF:
                        continue
                    comm = comm_sizes[j - 1] if j > 0 else 0
                    cost = DP[k - 1][j] + comm
                    if cost < best_cost:
                        best_cost = cost
                        best_j = j
                DP[k][i] = best_cost
                PR[k][i] = best_j

        feasible = DP[num_stages][n] < INF
        if not feasible:
            return False, None, None
        if not need_path:
            return True, DP[num_stages][n], None

        parts: List[Tuple[int, int]] = []
        k = num_stages
        i = n
        while k > 0:
            j = PR[k][i]
            if j < 0:
                break
            parts.append((j, i))
            i = j
            k -= 1
        parts.reverse()
        return True, DP[num_stages][n], parts

    for _ in range(40):
        mid = (lo + hi) / 2.0
        ok, _, _ = dp_min_comm(mid, need_path=False)
        if ok:
            hi = mid
        else:
            lo = mid

    ok, _, parts = dp_min_comm(hi, need_path=True)
    if not ok or parts is None:
        return partition_uniform(n, num_stages)
    return parts


def partition_min_comm_with_slack_dp(
    block_times: List[float],
    comm_sizes: List[int],
    num_stages: int,
    slack: float = 0.1,
) -> List[Tuple[int, int]]:
    n = len(block_times)
    total = sum(block_times)

    def dp_min_comm(T: float):
        INF = float("inf")
        p = num_stages
        DP = [[INF] * (n + 1) for _ in range(p + 1)]
        PR = [[-1] * (n + 1) for _ in range(p + 1)]
        DP[0][0] = 0.0
        prefix = [0.0]
        for t in block_times:
            prefix.append(prefix[-1] + t)
        def seg_sum(l, r):
            return prefix[r + 1] - prefix[l]
        for k in range(1, p + 1):
            for i in range(1, n + 1):
                best_cost = INF
                best_j = -1
                for j in range(i - 1, -1, -1):
                    if seg_sum(j, i - 1) > T:
                        break
                    if DP[k - 1][j] == INF:
                        continue
                    comm = comm_sizes[j - 1] if j > 0 else 0
                    cost = DP[k - 1][j] + comm
                    if cost < best_cost:
                        best_cost = cost
                        best_j = j
                DP[k][i] = best_cost
                PR[k][i] = best_j
        feasible = DP[num_stages][n] < INF
        if not feasible:
            return False, None
        parts: List[Tuple[int, int]] = []
        k = num_stages
        i = n
        while k > 0:
            j = PR[k][i]
            if j < 0:
                break
            parts.append((j, i))
            i = j
            k -= 1
        parts.reverse()
        return True, parts

    lo = max(block_times)
    hi = total
    for _ in range(40):
        mid = (lo + hi) / 2.0
        ok, _ = dp_min_comm(mid)
        if ok:
            hi = mid
        else:
            lo = mid
    T_min = hi
    T_slack = T_min * (1.0 + max(0.0, slack))
    ok, parts = dp_min_comm(T_slack)
    if ok and parts is not None:
        return parts
    ok2, parts2 = dp_min_comm(T_min)
    if ok2 and parts2 is not None:
        return parts2
    return partition_min_max_then_min_comm_dp(block_times, comm_sizes, num_stages)


# =============================
# 3.2 Block 聚合工具
# =============================

def build_blocks_by_n(num_layers: int, n: int) -> List[Tuple[int, int]]:
    blocks: List[Tuple[int, int]] = []
    start = 0
    while start < num_layers:
        end = min(start + n, num_layers)
        blocks.append((start, end))
        start = end
    return blocks


def aggregate_blocks(
    f_times: List[float],
    b_times: List[float],
    act_sizes: List[int],
    blocks: List[Tuple[int, int]],
) -> Tuple[List[float], List[int]]:
    comp_times: List[float] = []
    comm_per_block: List[int] = []
    for (s, e) in blocks:
        comp = 0.0
        for i in range(s, e):
            comp += (f_times[i] + b_times[i])
        comp_times.append(comp)
        comm_per_block.append(act_sizes[e - 1])
    return comp_times, comm_per_block


def aggregate_blocks_full(
    f_times: List[float],
    b_times: List[float],
    act_sizes: List[int],
    blocks: List[Tuple[int, int]],
) -> Tuple[List[float], List[int], List[float], List[float]]:
    comp_times: List[float] = []
    comm_per_block: List[int] = []
    f_per_block: List[float] = []
    b_per_block: List[float] = []
    for (s, e) in blocks:
        fsum = sum(f_times[i] for i in range(s, e))
        bsum = sum(b_times[i] for i in range(s, e))
        comp_times.append(fsum + bsum)
        f_per_block.append(fsum)
        b_per_block.append(bsum)
        comm_per_block.append(act_sizes[e - 1])
    return comp_times, comm_per_block, f_per_block, b_per_block


def stage_profiles_from_partitions(
    f_per_block: List[float],
    b_per_block: List[float],
    comm_per_block: List[int],
    partitions_blocks: List[Tuple[int, int]],
) -> Tuple[List[float], List[float], List[int]]:
    stage_f: List[float] = []
    stage_b: List[float] = []
    boundary_numel: List[int] = []
    for (s, e) in partitions_blocks:
        stage_f.append(sum(f_per_block[i] for i in range(s, e)))
        stage_b.append(sum(b_per_block[i] for i in range(s, e)))
        boundary_numel.append(comm_per_block[e - 1])
    if boundary_numel:
        boundary_numel = boundary_numel[:-1]
    return stage_f, stage_b, boundary_numel


# =============================
# 3.3 调度模拟（GPipe/1F1B）
# =============================

def simulate_pipeline(
    stage_f: List[float],
    stage_b: List[float],
    boundary_numel: List[int],
    bandwidth_gbps: float,
    microbatches: int,
    schedule: str = "1f1b",
) -> Tuple[float, float]:
    p = len(stage_f)
    m = microbatches
    def comm_time(numel: int) -> float:
        return (numel * 4) / (bandwidth_gbps * 1e9) if numel > 0 else 0.0
    f_dur = [stage_f[s] + (comm_time(boundary_numel[s]) if s < p - 1 else 0.0) for s in range(p)]
    b_dur = [stage_b[s] + (comm_time(boundary_numel[s - 1]) if s > 0 else 0.0) for s in range(p)]

    from collections import deque
    indeg_F = [[0 for _ in range(m)] for _ in range(p)]
    indeg_B = [[0 for _ in range(m)] for _ in range(p)]
    earliest_F = [[0.0 for _ in range(m)] for _ in range(p)]
    earliest_B = [[0.0 for _ in range(m)] for _ in range(p)]
    for s in range(1, p):
        for u in range(m):
            indeg_F[s][u] = 1
    for s in range(p):
        for u in range(m):
            indeg_B[s][u] += 1
    for s in range(p - 1):
        for u in range(m):
            indeg_B[s][u] += 1
    gpipe_barrier = (schedule.lower() == "gpipe")
    stage_free = [0.0 for _ in range(p)]
    busy = [0.0 for _ in range(p)]
    ready: List[deque] = [deque() for _ in range(p)]
    for u in range(m):
        ready[0].append((0, u, 0.0))
    done_F = [[False for _ in range(m)] for _ in range(p)]
    done_B = [[False for _ in range(m)] for _ in range(p)]
    total_tasks = p * m * 2
    finished = 0

    def schedule_one_on_stage(s: int):
        nonlocal finished
        if not ready[s]:
            return False
        idx_best = 0
        best_et = float("inf")
        best_is_b = 1
        for idx, (typ, u, et) in enumerate(ready[s]):
            if et < best_et or (abs(et - best_et) < 1e-12 and typ < best_is_b):
                best_et = et
                best_is_b = typ
                idx_best = idx
        typ, u, et = ready[s][idx_best]
        ready[s].remove((typ, u, et))
        start = max(stage_free[s], et)
        dur = f_dur[s] if typ == 0 else b_dur[s]
        end = start + dur
        stage_free[s] = end
        busy[s] += dur
        finished += 1
        if typ == 0:
            done_F[s][u] = True
            if s + 1 < p:
                indeg_F[s + 1][u] -= 1
                earliest_F[s + 1][u] = max(earliest_F[s + 1][u], end)
                if indeg_F[s + 1][u] == 0:
                    ready[s + 1].append((0, u, earliest_F[s + 1][u]))
            indeg_B[s][u] -= 1
            earliest_B[s][u] = max(earliest_B[s][u], end)
            if indeg_B[s][u] == 0 and not gpipe_barrier:
                ready[s].append((1, u, earliest_B[s][u]))
        else:
            done_B[s][u] = True
            if s - 1 >= 0:
                indeg_B[s - 1][u] -= 1
                earliest_B[s - 1][u] = max(earliest_B[s - 1][u], end)
                if indeg_B[s - 1][u] == 0:
                    ready[s - 1].append((1, u, earliest_B[s - 1][u]))
        return True

    released_barrier = False
    while finished < total_tasks:
        progressed = False
        if gpipe_barrier and not released_barrier:
            if done_F[p - 1][m - 1]:
                barrier_time = stage_free[p - 1]
                for s in range(p):
                    for u in range(m):
                        earliest_B[s][u] = max(earliest_B[s][u], barrier_time)
                        indeg_B[s][u] = max(indeg_B[s][u] - 1, 0)
                        if indeg_B[s][u] == 0:
                            ready[s].append((1, u, earliest_B[s][u]))
                released_barrier = True
        for s in range(p):
            if schedule_one_on_stage(s):
                progressed = True
        if progressed:
            continue
        next_time = float("inf")
        next_stage = -1
        next_item = None
        for s in range(p):
            if not ready[s]:
                continue
            for typ, u, et in ready[s]:
                start = max(stage_free[s], et)
                if start < next_time:
                    next_time = start
                    next_stage = s
                    next_item = (typ, u, et)
        if next_stage >= 0:
            ready[next_stage].remove(next_item)
            ready[next_stage].appendleft(next_item)
            continue
        break

    T = max(stage_free)
    util = (sum(busy)) / (p * T) if T > 0 else 0.0
    bubble = max(0.0, 1.0 - util)
    return T, bubble


# =============================
# 4. 打印与可视化
# =============================

def print_partitions(
    name: str,
    layer_times: List[float],
    partitions: List[Tuple[int, int]],
    comm_sizes: List[int],
):
    print(f"\n===== {name} 划分结果 =====")
    loads = stage_loads(layer_times, partitions)
    total_time = sum(layer_times)
    max_load = max(loads)
    avg_load = total_time / len(partitions)
    balance_ratio = avg_load / max_load
    for stage_id, (s, e) in enumerate(partitions):
        stage_time = loads[stage_id]
        print(f"Stage {stage_id}: 层 {s} ~ {e - 1}, 层数 = {e - s}, 计算时间 = {stage_time * 1000:.3f} ms")
    print(f"总计算时间: {total_time * 1000:.3f} ms")
    print(f"最大 stage 时间: {max_load * 1000:.3f} ms")
    print(f"平均 stage 时间: {avg_load * 1000:.3f} ms")
    print(f"均衡度 (avg / max): {balance_ratio:.4f}")

    comm_total = 0
    for (s, e) in partitions[:-1]:
        boundary_layer = e - 1
        comm_total += comm_sizes[boundary_layer]
    print(f"通信边界数: {len(partitions) - 1}, 简单通信 cost = {comm_total}")

    def comm_time_ms(numel_sum: int, gbps: float, bytes_per_elem: int = 4, directions: int = 2) -> float:
        bytes_total = numel_sum * bytes_per_elem * directions
        return bytes_total / (gbps * 1e9 / 1000.0)
    for gbps in (300.0, 50.0):
        t_comm = comm_time_ms(comm_total, gbps)
        print(f"估算通信时间 @ {gbps:.0f} GB/s: {t_comm:.3f} ms (双向)")
        print(f"粗略总时延(瓶颈+通信) @ {gbps:.0f} GB/s: {(max_load*1000 + t_comm):.3f} ms")


# =============================
# 5. 主程序：对比 + 模拟 + 画图
# =============================

def main():
    torch.manual_seed(0)
    num_stages = 4
    block_n = 2

    net = TinyMLP(in_dim=1024)
    model = net.layers
    batch_size = 256
    input_shape = (batch_size, 1024)

    print("开始逐层 profile 前向/反向时间...")
    f_times, b_times, act_sizes = profile_model_fwd_bwd(
        model=model,
        input_shape=input_shape,
        runs=20,
        device="cpu",
    )

    num_layers = len(f_times)
    print(f"\n共有 {num_layers} 个 layer（包括 Linear 和 ReLU）。")
    print("每层平均前向/反向耗时 & activation 大小：")
    for i, (tf, tb, sz) in enumerate(zip(f_times, b_times, act_sizes)):
        print(f"  Layer {i:2d}: fwd = {tf*1000:.3f} ms, bwd = {tb*1000:.3f} ms, activation numel = {sz}")

    # Blocks
    blocks = build_blocks_by_n(num_layers, n=block_n)
    comp_times_per_block, comm_per_block, f_per_block, b_per_block = aggregate_blocks_full(
        f_times=f_times,
        b_times=b_times,
        act_sizes=act_sizes,
        blocks=blocks,
    )

    print(f"\n聚合为 {len(blocks)} 个 Block（每 {block_n} 层一块，最后一块可能不足 {block_n} 层）")
    for bid, (s, e) in enumerate(blocks):
        print(f"  Block {bid:2d}: 层 {s}~{e-1}, 计算时间(∑f+b) = {comp_times_per_block[bid]*1000:.3f} ms, 出口激活 = {comm_per_block[bid]}")

    # 方案一：按 Block 均分
    uniform_parts_blocks = partition_uniform(len(blocks), num_stages)
    print_partitions("按 Block 均分 (以 Block 计)", comp_times_per_block, uniform_parts_blocks, comm_per_block)

    # 方案二：贪心（∑f+b 均衡 + 通信感知）
    balanced_parts_blocks = partition_profile_balanced(
        layer_times=comp_times_per_block,
        comm_sizes=comm_per_block,
        num_stages=num_stages,
        alpha=0.2,
    )
    print_partitions("按真实时延(∑f+b)均衡 + 通信感知 - 贪心 (Block)", comp_times_per_block, balanced_parts_blocks, comm_per_block)

    # 方案三：DP 最优（最小瓶颈优先 + 通信最小）
    optimal_parts_blocks = partition_min_max_then_min_comm_dp(
        block_times=comp_times_per_block,
        comm_sizes=comm_per_block,
        num_stages=num_stages,
    )
    print_partitions("按最小瓶颈优先 + 通信最小 (DP, Block)", comp_times_per_block, optimal_parts_blocks, comm_per_block)

    # 方案四：DP + 均衡松弛（可选）
    slack = 0.10
    slack_parts_blocks = partition_min_comm_with_slack_dp(
        block_times=comp_times_per_block,
        comm_sizes=comm_per_block,
        num_stages=num_stages,
        slack=slack,
    )
    print_partitions(f"通信最小 + 均衡松弛({slack*100:.0f}%) (DP, Block)", comp_times_per_block, slack_parts_blocks, comm_per_block)

    # ======= 基于调度的效果对比 =======
    def collect_sim(parts: List[Tuple[int, int]]):
        stage_f, stage_b, boundary_numel = stage_profiles_from_partitions(
            f_per_block, b_per_block, comm_per_block, parts
        )
        results = {}
        for sched in ("gpipe", "1f1b"):
            for mbs in (4, 8, 16):
                for bw in (300.0, 50.0):
                    T, bubble = simulate_pipeline(stage_f, stage_b, boundary_numel, bw, mbs, schedule=sched)
                    results[(sched, mbs, bw)] = (T, bubble)
        return results

    def dump_results(tag: str, results):
        for (sched, mbs, bw), (T, bubble) in results.items():
            print(
                f"[Sim] {tag} | {sched.upper()} | m={mbs} | {bw:.0f}GB/s | "
                f"T={T*1000:.3f} ms, bubble={bubble*100:.2f}%"
            )

    print("\n===== 基于真实调度的效果对比 (越小越好) =====")
    res_uni = collect_sim(uniform_parts_blocks)
    res_greedy = collect_sim(balanced_parts_blocks)
    res_dp = collect_sim(optimal_parts_blocks)
    dump_results("均分Block", res_uni)
    dump_results("贪心均衡+通信感知", res_greedy)
    dump_results("DP最小瓶颈+通信最小", res_dp)

    print("\n===== 相对均分Block的改进（>1 表示更快，↓ 表示气泡下降） =====")
    print(f"理想气泡比 (p-1)/m: p={num_stages} -> m=4:{(num_stages-1)/4:.3f}, m=8:{(num_stages-1)/8:.3f}, m=16:{(num_stages-1)/16:.3f}")
    for sched in ("gpipe", "1f1b"):
        for mbs in (4, 8, 16):
            for bw in (300.0, 50.0):
                T0, B0 = res_uni[(sched, mbs, bw)]
                T1, B1 = res_greedy[(sched, mbs, bw)]
                T2, B2 = res_dp[(sched, mbs, bw)]
                sp1 = T0 / T1 if T1 > 0 else float('inf')
                sp2 = T0 / T2 if T2 > 0 else float('inf')
                print(
                    f"[{sched.upper()} m={mbs} {bw:.0f}GB/s] 贪心:×{sp1:.2f} (bubble {B1*100:.1f}%→{B0*100:.1f}%基线), "
                    f"DP:×{sp2:.2f} (bubble {B2*100:.1f}%→{B0*100:.1f}%基线)"
                )

    # ======= 绘图与导出 =======
    def generate_and_save_figures():
        import os
        import csv
        figs_dir = os.path.join(os.getcwd(), "figs_partition")
        os.makedirs(figs_dir, exist_ok=True)

        strategies = {
            "Uniform": res_uni,
            "Greedy": res_greedy,
            "DP": res_dp,
        }
        csv_path = os.path.join(figs_dir, "results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["schedule", "bandwidth_gbps", "microbatches", "strategy", "makespan_ms", "bubble"])
            for strat, res in strategies.items():
                for (sched, mbs, bw), (T, bubble) in res.items():
                    writer.writerow([sched, bw, mbs, strat, T * 1000.0, bubble])

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            print(f"Matplotlib 不可用，已导出 CSV 到 {csv_path}")
            return

        colors = {"Uniform": "#999999", "Greedy": "#1f77b4", "DP": "#d62728"}
        markers = {"Uniform": "o", "Greedy": "s", "DP": "^"}

        for sched in ("gpipe", "1f1b"):
            sched_name = sched.upper()
            for bw in (300.0, 50.0):
                # makespan
                plt.figure(figsize=(6.5, 4))
                for strat, res in strategies.items():
                    xs = []
                    ys = []
                    for mbs in (4, 8, 16):
                        T, _ = res[(sched, mbs, bw)]
                        xs.append(mbs)
                        ys.append(T * 1000.0)
                    plt.plot(xs, ys, marker=markers[strat], color=colors[strat], label=strat)
                plt.title(f"Makespan vs microbatches ({sched_name}, {int(bw)} GB/s)")
                plt.xlabel("microbatches (m)")
                plt.ylabel("makespan (ms)")
                plt.grid(True, alpha=0.3)
                plt.legend()
                out_path = os.path.join(figs_dir, f"makespan_{sched}_{int(bw)}.png")
                plt.tight_layout()
                plt.savefig(out_path, dpi=180)
                plt.close()

                # bubble
                plt.figure(figsize=(6.5, 4))
                for strat, res in strategies.items():
                    xs = []
                    ys = []
                    for mbs in (4, 8, 16):
                        _, bubble = res[(sched, mbs, bw)]
                        xs.append(mbs)
                        ys.append(bubble * 100.0)
                    plt.plot(xs, ys, marker=markers[strat], color=colors[strat], label=strat)
                plt.title(f"Bubble vs microbatches ({sched_name}, {int(bw)} GB/s)")
                plt.xlabel("microbatches (m)")
                plt.ylabel("bubble (%)")
                plt.grid(True, alpha=0.3)
                plt.legend()
                out_path = os.path.join(figs_dir, f"bubble_{sched}_{int(bw)}.png")
                plt.tight_layout()
                plt.savefig(out_path, dpi=180)
                plt.close()

                # speedup vs Uniform
                plt.figure(figsize=(6.5, 4))
                base = strategies["Uniform"]
                for strat in ("Greedy", "DP"):
                    res = strategies[strat]
                    xs = []
                    ys = []
                    for mbs in (4, 8, 16):
                        T0, _ = base[(sched, mbs, bw)]
                        T1, _ = res[(sched, mbs, bw)]
                        xs.append(mbs)
                        ys.append(T0 / T1)
                    plt.plot(xs, ys, marker=markers[strat], color=colors[strat], label=strat)
                plt.title(f"Speedup vs Uniform ({sched_name}, {int(bw)} GB/s)")
                plt.xlabel("microbatches (m)")
                plt.ylabel("speedup (×)")
                plt.grid(True, alpha=0.3)
                plt.legend()
                out_path = os.path.join(figs_dir, f"speedup_{sched}_{int(bw)}.png")
                plt.tight_layout()
                plt.savefig(out_path, dpi=180)
                plt.close()

        print("图像与数据已生成到 figs/ 目录：")
        print(f"  - CSV: {csv_path}")
        for sched in ("gpipe", "1f1b"):
            for bw in (300.0, 50.0):
                print(f"  - figs_partition/makespan_{sched}_{int(bw)}.png")
                print(f"  - figs_partition/bubble_{sched}_{int(bw)}.png")
                print(f"  - figs_partition/speedup_{sched}_{int(bw)}.png")

    generate_and_save_figures()


if __name__ == "__main__":
    main()

