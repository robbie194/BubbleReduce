BubbleReduce: Communication-Aware Pipeline Partitioning and Scheduling Demos

Overview
- This repo demonstrates practical techniques to reduce pipeline bubbles when training large models across multiple pipeline stages.
- It focuses on real-world heterogeneity: forward/backward asymmetry, uneven layer runtimes, and communication overhead at stage boundaries.
- You can profile a toy MLP, partition it via several strategies, simulate GPipe/1F1B schedules, and visualize throughput/bubbles. Two additional experiments demonstrate online boundary micro-adjustments and explicit forward/backward imbalance compensation.

Key Components
- `pipeline_partition_demo.py` (formerly `main.py`)
  - Profiles per-layer forward (`fᵢ`) and backward (`bᵢ`) times and activation sizes.
  - Aggregates layers into blocks (e.g., Linear+ReLU per block) to control granularity.
  - Partitions blocks into `p` stages via:
    - Uniform by blocks (baseline)
    - Greedy: balance ∑(f+b) with communication-aware penalty
    - DP optimal: minimize max stage time, then minimize boundary communication
    - DP with slack: allow slight imbalance to reduce communication
  - Simulates GPipe and 1F1B schedules with microbatches and bandwidth; reports makespan and bubble fraction; outputs figures to `figs/`.

- `online_adjust.py`
  - Online micro-adjustment on top of a static split with safeguards:
    - Rolling runtime monitoring to detect laggard stages
    - Move 1–2 small blocks across adjacent boundary at a time
    - Cooldown window and rollback if no improvement
    - Resource guards (VRAM/PCIe proxy) to pause moves under pressure
  - Compares Static vs Online on the same noise/drift series; outputs timeseries plots and summary to `figs_online/`.

- `balance.py`
  - Forward/Backward imbalance compensation: use ∑(f + α·b) (α>1) to avoid piling slow backward blocks into a single stage.
  - Compares baseline (∑(f+b)) vs α-weighted partitions; visualizes per-stage F/B stacks and end-to-end performance; outputs to `figs_fb/`.

Requirements
- Python 3.9+
- PyTorch (CPU is sufficient for demo)
- NumPy
- Matplotlib

Example setup (Conda)
- conda create -n bubble python=3.10 -y
- conda activate bubble
- pip install torch numpy matplotlib

Usage
1) Partitioning + Scheduling Demo
   - Run: `python pipeline_partition_demo.py`
   - Adjust in-file constants in `main()`:
     - `num_stages`: number of pipeline stages
     - `block_n`: block size (2 groups Linear+ReLU; 1 is finer)
     - Greedy comm penalty: `alpha` in `partition_profile_balanced`
     - DP slack: set `slack` in `partition_min_comm_with_slack_dp`
   - Outputs:
     - Console: layer/block times, partition results, simulated makespan/bubble
     - `figs/`: `results.csv`, and per-schedule/bandwidth `makespan_*.png`, `bubble_*.png`, `speedup_*.png`

2) Online Micro-Adjustment
   - Run: `python online_adjust.py`
   - Adjust `OnlineConfig` in `run_and_plot()`:
     - `microbatches`, `bandwidth_gbps`, `schedule` ("gpipe" or "1f1b")
     - `jitter_sigma`, `drift_strength`, `roll_window`, `cooldown_steps`
     - `mem_threshold_ratio`, `pcie_threshold_ratio`, `rollback_patience`, `improve_epsilon`
   - Outputs `figs_online/`:
     - `online_vs_static.csv`
     - `makespan_timeseries.png`, `bubble_timeseries.png`, `makespan_rolling.png`
     - `summary.txt` with directly quotable metrics (mean/median/p95, speedups, move/rollback counts)

3) F/B Imbalance Compensation
   - Run: `python balance.py`
   - Adjust top-of-file config: `alpha` (default 2.0), `num_stages`, `block_n`
   - Outputs `figs_fb/`:
     - `results.csv`
     - Per-schedule/bandwidth: `makespan_*.png`, `bubble_*.png`, `speedup_*.png`
     - `stage_loads_baseline.png` vs `stage_loads_alpha.png` (stacked F/B)
     - `summary.txt` with backward-balance improvements and speedups

Notes and Limitations
- Profiling uses CPU and synthetic data; it captures heterogeneity and F/B asymmetry but is not a substitute for device-level profiling in production.
- Communication is approximated via boundary activation sizes per microbatch; for models with skip connections, extend the proxy to include all tensors crossing boundaries.
- The simulator captures key constraints of GPipe and 1F1B but is simplified for clarity.

Repo Structure
- `pipeline_partition_demo.py`  Partitioning strategies + scheduling simulator + plots
- `online_adjust.py`           Online boundary micro-adjustment with safeguards
- `balance.py`                 F/B imbalance compensation experiment and plots
- `figs/`, `figs_online/`, `figs_fb/`  Generated outputs

License
- Educational/demo code for research and teaching; no license headers added.

