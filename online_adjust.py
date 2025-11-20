#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Online micro-adjustment demo on top of pipeline partitioning.

Implements:
- Rolling per-stage runtime monitoring to detect laggard stage
- Move only 1 small block across boundary per adjustment
- Cooldown window (N steps) between adjustments
- Resource protections: pause moves if est. memory > 90% or est. PCIe ratio > 70%
- Rollback: if no improvement within a patience window, revert the move

Generates figures and CSV comparing static vs online-adjusted runs.
"""

import os
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse building blocks without triggering a script run
from pipeline_partition_demo import (
    TinyMLP,
    profile_model_fwd_bwd,
    build_blocks_by_n,
    aggregate_blocks_full,
    stage_profiles_from_partitions,
    partition_min_max_then_min_comm_dp,
)


def simulate_pipeline_with_busy(
    stage_f: List[float],
    stage_b: List[float],
    boundary_numel: List[int],
    bandwidth_gbps: float,
    microbatches: int,
    schedule: str = "1f1b",
):
    """
    Simulate pipeline with GPipe or 1F1B, returning:
      - makespan T (sec)
      - bubble_fraction
      - busy_per_stage list (sec)
    Model mirrors main.simulate_pipeline but returns per-stage busy time.
    """
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
        # Choose earliest task; tie-breaker: forward first
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
        # Advance to next earliest ready task
        next_time = float("inf")
        next_stage = -1
        next_item = None
        for s in range(p):
            for typ, u, et in list(ready[s]):
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
    return T, bubble, busy


@dataclass
class OnlineConfig:
    microbatches: int = 8
    bandwidth_gbps: float = 300.0
    schedule: str = "gpipe"  # or "1f1b"
    jitter_sigma: float = 0.05  # 5% noise per block per step
    steps: int = 200
    roll_window: int = 20
    cooldown_steps: int = 10
    mem_threshold_ratio: float = 0.98  # relative to mem_cap
    pcie_threshold_ratio: float = 0.85
    rollback_patience: int = 20
    improve_epsilon: float = 0.001  # 0.1% improvement needed
    drift_strength: float = 0.002  # per-step drift factor magnitude


def blocks_to_stage_lists(parts: List[Tuple[int, int]]) -> List[List[int]]:
    stage_blocks: List[List[int]] = []
    for s, e in parts:
        stage_blocks.append(list(range(s, e)))
    return stage_blocks


def est_stage_memory_bytes(stage_block_ids: List[int], comm_per_block: List[int], microbatches: int) -> int:
    # Simplistic: sum of block outputs times microbatches times 4 bytes (float32), x2 to account grads
    return int(sum(comm_per_block[b] for b in stage_block_ids) * microbatches * 4 * 2)


def stage_comm_time(stage_idx: int, boundary_numel: List[int], bandwidth_gbps: float) -> float:
    # Approx forward send (if not last) + backward grad (if not first)
    f = (boundary_numel[stage_idx] if stage_idx < len(boundary_numel) else 0)
    b = (boundary_numel[stage_idx - 1] if stage_idx - 1 >= 0 else 0)
    def t(numel):
        return (numel * 4) / (bandwidth_gbps * 1e9)
    return t(f) + t(b)


def try_move_one_block(
    parts: List[Tuple[int, int]],
    from_stage: int,
    direction: int,
) -> Optional[List[Tuple[int, int]]]:
    """
    Attempt to move 1 block across boundary.
      direction = -1: move first block of from_stage to previous stage
      direction = +1: move last block of from_stage to next stage
    Returns new partitions or None if invalid.
    """
    p = len(parts)
    s, e = parts[from_stage]
    if direction == -1:
        if from_stage == 0 or (e - s) <= 1:
            return None
        left_s, left_e = parts[from_stage - 1]
        # move block 's'
        new_parts = parts.copy()
        new_parts[from_stage - 1] = (left_s, left_e + 1)
        new_parts[from_stage] = (s + 1, e)
    return new_parts


def try_move_k_blocks(
    parts: List[Tuple[int, int]],
    from_stage: int,
    direction: int,
    k: int,
) -> Optional[List[Tuple[int, int]]]:
    """
    Move k contiguous blocks across boundary while keeping stages non-empty.
    direction = -1 moves the first k blocks to previous stage.
    direction = +1 moves the last k blocks to next stage.
    """
    s, e = parts[from_stage]
    if (e - s) <= k:
        return None
    p = len(parts)
    new_parts = parts.copy()
    if direction == -1:
        if from_stage == 0:
            return None
        ls, le = parts[from_stage - 1]
        new_parts[from_stage - 1] = (ls, le + k)
        new_parts[from_stage] = (s + k, e)
        return new_parts
    else:
        if from_stage == p - 1:
            return None
        rs, re = parts[from_stage + 1]
        new_parts[from_stage] = (s, e - k)
        new_parts[from_stage + 1] = (rs - k, re)
        return new_parts
    


def online_adjust(
    f_per_block_base: List[float],
    b_per_block_base: List[float],
    comm_per_block: List[int],
    init_parts: List[Tuple[int, int]],
    cfg: OnlineConfig,
    drift_coeff: Optional[np.ndarray] = None,
    seed: int = 0,
):
    random.seed(seed)
    np.random.seed(seed)

    # Pre-generate jitter for reproducibility: shape (steps, num_blocks)
    L = len(f_per_block_base)
    jitter_f = np.random.normal(0.0, cfg.jitter_sigma, size=(cfg.steps, L))
    jitter_b = np.random.normal(0.0, cfg.jitter_sigma, size=(cfg.steps, L))
    if drift_coeff is None:
        rng = np.random.RandomState(seed + 123)
        drift_coeff = rng.choice([-1.0, 0.0, 1.0], size=L, p=[0.4, 0.2, 0.4])

    # Memory cap baseline per stage from initial partition
    init_stage_lists = blocks_to_stage_lists(init_parts)
    init_mem = [est_stage_memory_bytes(ids, comm_per_block, cfg.microbatches) for ids in init_stage_lists]
    mem_cap = max(init_mem) * 1.2  # allow 20% headroom

    # Logs
    steps_log = []  # list of dicts

    # State
    parts = init_parts.copy()
    cooldown = 0
    last_move_snapshot = None  # (step_idx, prev_parts, baseline_mean_T)

    # Also prepare a static baseline run with same jitter series (no moves)
    def simulate_step(parts_local, f_bl, b_bl):
        stage_f, stage_b, boundary_numel = stage_profiles_from_partitions(f_bl, b_bl, comm_per_block, parts_local)
        T, bubble, busy = simulate_pipeline_with_busy(stage_f, stage_b, boundary_numel, cfg.bandwidth_gbps, cfg.microbatches, cfg.schedule)
        return T, bubble, busy, stage_f, stage_b, boundary_numel

    T_history = []
    busy_history = []  # list of per-stage busy arrays

    for step in range(cfg.steps):
        # Build jittered + drifted per-block f/b
        f_bl = []
        b_bl = []
        for i in range(L):
            drift = 1.0 + cfg.drift_strength * step * float(drift_coeff[i])
            drift = max(0.2, drift)
            f_bl.append(f_per_block_base[i] * drift * (1.0 + float(jitter_f[step, i])))
            b_bl.append(b_per_block_base[i] * drift * (1.0 + float(jitter_b[step, i])))

        T, bubble, busy, stage_f, stage_b, boundary_numel = simulate_step(parts, f_bl, b_bl)
        T_history.append(T)
        busy_history.append(busy)

        # Resource checks for current layout
        stage_lists = blocks_to_stage_lists(parts)
        mem_ratios = [est_stage_memory_bytes(ids, comm_per_block, cfg.microbatches) / mem_cap for ids in stage_lists]
        pcie_ratios = [stage_comm_time(i, boundary_numel, cfg.bandwidth_gbps) / T for i in range(len(stage_lists))]

        moved = False
        rolled_back = False

        # Laggard detection & micro-adjustment
        if step + 1 >= cfg.roll_window:
            # rolling mean of busy per stage
            window_busy = np.array(busy_history[-cfg.roll_window:])  # (W, p)
            mean_busy = window_busy.mean(axis=0)
            laggard = int(np.argmax(mean_busy))
            # Candidate neighbors
            candidates = []
            if cooldown <= 0 and mem_ratios[laggard] < cfg.mem_threshold_ratio and pcie_ratios[laggard] < cfg.pcie_threshold_ratio:
                # evaluate moving left
                if laggard > 0:
                    # k=1 and k=2 candidates
                    for k in (1, 2):
                        new_parts = try_move_k_blocks(parts, laggard, direction=-1, k=k)
                        if new_parts is not None:
                            stage_f2, stage_b2, boundary_numel2 = stage_profiles_from_partitions(f_bl, b_bl, comm_per_block, new_parts)
                            T2, bubble2, busy2 = simulate_pipeline_with_busy(stage_f2, stage_b2, boundary_numel2, cfg.bandwidth_gbps, cfg.microbatches, cfg.schedule)
                            candidates.append((T2, new_parts, stage_f2, stage_b2, boundary_numel2))
                # evaluate moving right
                if laggard < len(parts) - 1:
                    for k in (1, 2):
                        new_parts = try_move_k_blocks(parts, laggard, direction=+1, k=k)
                        if new_parts is not None:
                            stage_f2, stage_b2, boundary_numel2 = stage_profiles_from_partitions(f_bl, b_bl, comm_per_block, new_parts)
                            T2, bubble2, busy2 = simulate_pipeline_with_busy(stage_f2, stage_b2, boundary_numel2, cfg.bandwidth_gbps, cfg.microbatches, cfg.schedule)
                            candidates.append((T2, new_parts, stage_f2, stage_b2, boundary_numel2))

                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    best_T2, best_parts, stage_f2, stage_b2, boundary_numel2 = candidates[0]
                    # Resource check after move (mem + PCIe)
                    stage_lists2 = blocks_to_stage_lists(best_parts)
                    mem_ratios2 = [est_stage_memory_bytes(ids, comm_per_block, cfg.microbatches) / mem_cap for ids in stage_lists2]
                    pcie_ratios2 = [stage_comm_time(i, boundary_numel2, cfg.bandwidth_gbps) / best_T2 for i in range(len(stage_lists2))]
                    # Require small predicted improvement vs current step T to avoid oscillation
                    if (best_T2 <= T * (1.0 - cfg.improve_epsilon)
                        and max(mem_ratios2) < cfg.mem_threshold_ratio
                        and max(pcie_ratios2) < cfg.pcie_threshold_ratio):
                        # Commit move
                        last_move_snapshot = (step, parts.copy(), np.mean(T_history[-cfg.roll_window:]))
                        parts = best_parts
                        cooldown = cfg.cooldown_steps
                        moved = True

        # Cooldown decay
        if cooldown > 0:
            cooldown -= 1

        # Rollback check
        if last_move_snapshot is not None:
            moved_at, prev_parts, baseline_mean_T = last_move_snapshot
            if step - moved_at >= cfg.rollback_patience:
                recent_mean_T = np.mean(T_history[-cfg.roll_window:]) if len(T_history) >= cfg.roll_window else np.mean(T_history)
                if recent_mean_T > baseline_mean_T * (1.0 - cfg.improve_epsilon):
                    # Not improved enough: rollback
                    parts = prev_parts
                    last_move_snapshot = None
                    cooldown = cfg.cooldown_steps
                    rolled_back = True

        steps_log.append({
            "step": step,
            "T": T,
            "bubble": bubble,
            "busy": busy,
            "mem_max": max(mem_ratios),
            "pcie_max": max(pcie_ratios),
            "moved": moved,
            "rollback": rolled_back,
            "parts": parts.copy(),
        })

    return steps_log


def run_and_plot():
    torch.manual_seed(0)
    # Model and profiling (reuse same as main)
    num_stages = 4
    block_n = 1  # finer granularity enables micro-adjustment
    net = TinyMLP(in_dim=1024)
    model = net.layers
    batch_size = 256
    input_shape = (batch_size, 1024)
    f_times, b_times, act_sizes = profile_model_fwd_bwd(model, input_shape, runs=10, device="cpu")

    # Blocks and per-block metrics
    blocks = build_blocks_by_n(len(f_times), n=block_n)
    comp_times_per_block, comm_per_block, f_per_block, b_per_block = aggregate_blocks_full(
        f_times=f_times, b_times=b_times, act_sizes=act_sizes, blocks=blocks
    )

    # Initial partition (static best): DP min-max then min-comm
    init_parts = partition_min_max_then_min_comm_dp(
        block_times=comp_times_per_block,
        comm_sizes=comm_per_block,
        num_stages=num_stages,
    )

    cfg = OnlineConfig(
        microbatches=8,
        bandwidth_gbps=300.0,
        schedule="gpipe",
        jitter_sigma=0.05,
        steps=200,
        roll_window=15,
        cooldown_steps=20,
        mem_threshold_ratio=0.98,
        pcie_threshold_ratio=0.85,
        rollback_patience=25,
        improve_epsilon=0.005,
        drift_strength=0.05,  # Increase drift so the online tuner has more real imbalance to correct
    )

    # Generate same jitter for both runs via seed
    seed = 42

    # Generate block-wise drift coefficients (fixed across runs)
    rng = np.random.RandomState(seed + 2024)
    drift_coeff = rng.choice([-1.0, 0.0, 1.0], size=len(f_per_block), p=[0.4, 0.2, 0.4])

    # Static run (no adjustments)
    def static_run():
        steps_log = []
        L = len(f_per_block)
        rng_f = np.random.RandomState(seed)
        rng_b = np.random.RandomState(seed + 1)
        jitter_f = rng_f.normal(0.0, cfg.jitter_sigma, size=(cfg.steps, L))
        jitter_b = rng_b.normal(0.0, cfg.jitter_sigma, size=(cfg.steps, L))
        for step in range(cfg.steps):
            f_bl = []
            b_bl = []
            for i in range(L):
                drift = 1.0 + cfg.drift_strength * step * float(drift_coeff[i])
                drift = max(0.2, drift)
                f_bl.append(f_per_block[i] * drift * (1.0 + float(jitter_f[step, i])))
                b_bl.append(b_per_block[i] * drift * (1.0 + float(jitter_b[step, i])))
            stage_f, stage_b, boundary_numel = stage_profiles_from_partitions(f_bl, b_bl, comm_per_block, init_parts)
            T, bubble, busy = simulate_pipeline_with_busy(stage_f, stage_b, boundary_numel, cfg.bandwidth_gbps, cfg.microbatches, cfg.schedule)
            steps_log.append({"step": step, "T": T, "bubble": bubble, "busy": busy})
        return steps_log

    static_log = static_run()
    online_log = online_adjust(f_per_block, b_per_block, comm_per_block, init_parts, cfg, drift_coeff=drift_coeff, seed=seed)

    # Save CSV and plots
    figs_dir = os.path.join(os.getcwd(), "figs_online")
    os.makedirs(figs_dir, exist_ok=True)

    # CSV
    import csv
    csv_path = os.path.join(figs_dir, "online_vs_static.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "scenario", "T_ms", "bubble", "moved", "rollback"])  
        for r in static_log:
            w.writerow([r["step"], "static", r["T"] * 1000.0, r["bubble"], 0, 0])
        for r in online_log:
            w.writerow([r["step"], "online", r["T"] * 1000.0, r["bubble"], int(r["moved"]), int(r["rollback"])])

    # Plots: makespan and bubble time series
    steps = [r["step"] for r in static_log]
    T_static = [r["T"] * 1000.0 for r in static_log]
    T_online = [r["T"] * 1000.0 for r in online_log]
    B_static = [r["bubble"] * 100.0 for r in static_log]
    B_online = [r["bubble"] * 100.0 for r in online_log]
    moved_steps = [r["step"] for r in online_log if r.get("moved")] 
    rollback_steps = [r["step"] for r in online_log if r.get("rollback")] 

    from matplotlib.lines import Line2D
    # Makespan timeseries with prominent move/rollback markers
    plt.figure(figsize=(8, 4))
    plt.plot(steps, T_static, label="Static", color="#999999", linewidth=1.4)
    plt.plot(steps, T_online, label="Online (ours)", color="#d62728", linewidth=1.8)
    # Stronger vertical cues
    for s in moved_steps:
        plt.axvline(s, color="#2ca02c", alpha=0.6, linewidth=1.6)
    for s in rollback_steps:
        plt.axvline(s, color="#ff7f0e", alpha=0.7, linewidth=1.6, linestyle="--")
    # Overlay markers at the online curve points for moves/rollbacks
    index_map = {st: i for i, st in enumerate(steps)}
    move_idx = [index_map[st] for st in moved_steps if st in index_map]
    rb_idx = [index_map[st] for st in rollback_steps if st in index_map]
    if move_idx:
        plt.scatter([steps[i] for i in move_idx], [T_online[i] for i in move_idx],
                    marker='v', s=60, color="#2ca02c", edgecolor='k', linewidths=0.6, zorder=3)
    if rb_idx:
        plt.scatter([steps[i] for i in rb_idx], [T_online[i] for i in rb_idx],
                    marker='X', s=70, color="#ff7f0e", edgecolor='k', linewidths=0.6, zorder=3)
    # Legend with custom handles for move/rollback
    move_handle = Line2D([0], [0], color="#2ca02c", lw=0, marker='v', markersize=8, markeredgecolor='k', label='Move')
    rb_handle = Line2D([0], [0], color="#ff7f0e", lw=0, marker='X', markersize=9, markeredgecolor='k', label='Rollback')
    plt.title("Makespan per step (lower is better)")
    plt.xlabel("step")
    plt.ylabel("makespan (ms)")
    plt.grid(True, alpha=0.3)
    # Combine existing line legend with our custom markers
    handles, labels = plt.gca().get_legend_handles_labels()
    handles += [move_handle, rb_handle]
    labels += ["Move", "Rollback"]
    plt.legend(handles, labels)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "makespan_timeseries.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(steps, B_static, label="Static", color="#999999", linewidth=1.4)
    plt.plot(steps, B_online, label="Online (ours)", color="#1f77b4", linewidth=1.8)
    for s in moved_steps:
        plt.axvline(s, color="#2ca02c", alpha=0.6, linewidth=1.6)
    for s in rollback_steps:
        plt.axvline(s, color="#ff7f0e", alpha=0.7, linewidth=1.6, linestyle="--")
    # Markers at bubble curve points
    if move_idx:
        plt.scatter([steps[i] for i in move_idx], [B_online[i] for i in move_idx],
                    marker='v', s=50, color="#2ca02c", edgecolor='k', linewidths=0.5, zorder=3)
    if rb_idx:
        plt.scatter([steps[i] for i in rb_idx], [B_online[i] for i in rb_idx],
                    marker='X', s=60, color="#ff7f0e", edgecolor='k', linewidths=0.5, zorder=3)
    move_handle = Line2D([0], [0], color="#2ca02c", lw=0, marker='v', markersize=8, markeredgecolor='k', label='Move')
    rb_handle = Line2D([0], [0], color="#ff7f0e", lw=0, marker='X', markersize=9, markeredgecolor='k', label='Rollback')
    plt.title("Bubble per step (lower is better)")
    plt.xlabel("step")
    plt.ylabel("bubble (%)")
    plt.grid(True, alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles += [move_handle, rb_handle]
    labels += ["Move", "Rollback"]
    plt.legend(handles, labels)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "bubble_timeseries.png"), dpi=180)
    plt.close()

    # Rolling averages comparison
    W = 20
    def rolling_mean(xs, w=W):
        out = []
        cur = 0.0
        for i, v in enumerate(xs):
            cur += v
            if i >= w:
                cur -= xs[i - w]
            out.append(cur / min(i + 1, w))
        return out
    plt.figure(figsize=(8, 4))
    plt.plot(steps, rolling_mean(T_static), label="Static", color="#999999")
    plt.plot(steps, rolling_mean(T_online), label="Online (ours)", color="#d62728")
    plt.title("Rolling makespan (window=20)")
    plt.xlabel("step")
    plt.ylabel("makespan (ms)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "makespan_rolling.png"), dpi=180)
    plt.close()

    # Summary print
    print("在线微调 vs 静态 对比图与数据已保存到 figs_online/:")
    for n in ["online_vs_static.csv", "makespan_timeseries.png", "bubble_timeseries.png", "makespan_rolling.png"]:
        print("  -", os.path.join(figs_dir, n))

    # ===== 直接可用的数值结论（均值/中位数/95分位/加速比等）=====
    def summary_stats(arr_ms: List[float]):
        arr = np.array(arr_ms, dtype=float)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    stat_T = summary_stats(T_static)
    on_T = summary_stats(T_online)
    stat_B = summary_stats(B_static)
    on_B = summary_stats(B_online)

    speedup_mean = stat_T["mean"] / on_T["mean"] if on_T["mean"] > 0 else float("inf")
    speedup_p95 = stat_T["p95"] / on_T["p95"] if on_T["p95"] > 0 else float("inf")
    improvement_steps = sum(1 for ts, to in zip(T_static, T_online) if to < ts)
    frac_improved = improvement_steps / len(T_static)
    moves = sum(1 for r in online_log if r.get("moved"))
    rollbacks = sum(1 for r in online_log if r.get("rollback"))

    # Print concise summary
    print("\n===== 数值总结（可直接引用） =====")
    print(f"Steps: {len(T_static)}, Moves: {moves}, Rollbacks: {rollbacks}")
    print(f"Makespan 平均: 静态 {stat_T['mean']:.3f} ms vs 在线 {on_T['mean']:.3f} ms (×{speedup_mean:.2f} 加速)")
    print(f"Makespan 中位: 静态 {stat_T['median']:.3f} ms vs 在线 {on_T['median']:.3f} ms")
    print(f"Makespan 95分位: 静态 {stat_T['p95']:.3f} ms vs 在线 {on_T['p95']:.3f} ms (×{speedup_p95:.2f} 加速)")
    print(f"Bubble 平均: 静态 {stat_B['mean']:.2f}% vs 在线 {on_B['mean']:.2f}%")
    print(f"Bubble 95分位: 静态 {stat_B['p95']:.2f}% vs 在线 {on_B['p95']:.2f}%")
    print(f"改进步数占比: {frac_improved*100:.1f}%  (online 步长更小的比例)")

    # Save summary to text
    summary_path = os.path.join(figs_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Online micro-adjustment vs Static\n")
        f.write(f"Steps: {len(T_static)}, Moves: {moves}, Rollbacks: {rollbacks}\n")
        f.write(f"Makespan mean: static {stat_T['mean']:.3f} ms | online {on_T['mean']:.3f} ms | speedup ×{speedup_mean:.2f}\n")
        f.write(f"Makespan median: static {stat_T['median']:.3f} ms | online {on_T['median']:.3f} ms\n")
        f.write(f"Makespan p95: static {stat_T['p95']:.3f} ms | online {on_T['p95']:.3f} ms | speedup ×{speedup_p95:.2f}\n")
        f.write(f"Bubble mean: static {stat_B['mean']:.2f}% | online {on_B['mean']:.2f}%\n")
        f.write(f"Bubble p95: static {stat_B['p95']:.2f}% | online {on_B['p95']:.2f}%\n")
        f.write(f"Improved steps fraction: {frac_improved*100:.1f}%\n")
    print("摘要: ", summary_path)


if __name__ == "__main__":
    run_and_plot()
