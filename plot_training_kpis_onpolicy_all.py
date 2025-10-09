#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -------------------- style constants (slightly larger fonts) --------------------
TITLE_FS  = 14   # subplot titles
LABEL_FS  = 12   # x/y-axis labels
LEGEND_FS = 13   # legend text
TICK_FS   = 10   # tick labels

# -------------------- helpers --------------------

def load_or_none(path: Path):
    try:
        if path.exists():
            return np.load(path)
    except Exception:
        pass
    return None

def moving_average(y: np.ndarray, k: int) -> np.ndarray:
    """
    Progressive causal moving average (no head padding):
      out[i] = mean(y[max(0, i-k+1) : i+1])
    """
    if y is None:
        return y
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return y
    k_eff = max(1, int(k))
    if k_eff == 1:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))  # c[t] = sum(y[:t])
    idx = np.arange(1, n + 1)            # 1..n
    starts = np.maximum(0, idx - k_eff)  # start index per window
    lens = idx - starts                  # window lengths (grow until k_eff)
    return (c[idx] - c[starts]) / lens

def build_color_map(labels):
    """
    Darker, higher-contrast palette (tab10-like).
    """
    dark_palette = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray (mid-dark)
        "#bcbd22",  # olive
        "#17becf",  # cyan
    ]
    return {lbl: dark_palette[i % len(dark_palette)] for i, lbl in enumerate(labels)}

def aggregate_mean_std(series_list: List[np.ndarray], smooth_k: int, max_pts: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a list of 1D arrays (one per seed), return (mean, std) after:
      1) truncating each to max_pts,
      2) optionally smoothing EACH seed curve (causal MA),
      3) aligning to common length (min over available seeds),
      4) computing mean and std across seeds at each step.
    Returns (mean, std) as 1D arrays; if <2 valid seeds, std will be zeros.
    """
    ys = []
    for y in series_list:
        if y is None or len(y) == 0:
            continue
        y = np.asarray(y, dtype=float)[:max_pts]
        y = moving_average(y, smooth_k) if smooth_k > 1 else y
        ys.append(y)
    if not ys:
        return np.array([]), np.array([])
    L = min(len(y) for y in ys)
    if L == 0:
        return np.array([]), np.array([])
    ys = [y[:L] for y in ys]
    Y = np.stack(ys, axis=0)        # [S, L]
    mean = np.nanmean(Y, axis=0)    # [L]
    std  = np.nanstd(Y, axis=0)     # [L]
    return mean, std

# -------------------- main --------------------

def main():
    parser = argparse.ArgumentParser(
        "Plot KPIs for Grid-3 / Grid-5 / Grid-10 in one figure (rows), mean±std over seeds, with two legend boxes."
    )
    parser.add_argument("--base-dir", type=str, default=".", help="Base directory containing logs_grid_N/")
    parser.add_argument("--seeds", type=str, default="42,238,458", help="Comma-separated seeds, e.g. '42,238,458'")
    parser.add_argument("--smooth", type=int, default=1, help="Moving-average window (eval stages). Use 1 to disable.")
    parser.add_argument("--save-path", type=str, default="", help="Optional path to save the figure")
    parser.add_argument("--eval-every", type=int, default=10, help="Episodes per evaluation (fixed for all runs).")
    parser.add_argument("--max-points", type=int, default=50, help="Max evaluation points per run.")
    parser.add_argument("--grids", type=str, default="3,5,10", help="Comma-separated grid sizes, e.g. '3,5,10'")
    args = parser.parse_args()

    grids = [int(x.strip()) for x in args.grids.split(",") if x.strip()]
    nrows = len(grids)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # Run name prefixes must match your EvalHistory run_name in training scripts
    runs = {
        "IA2C (MLP)"                                : "ia2c_mlp_shared",
        "IA2C (LSTM)"                               : "ia2c_lstm_shared",
        "IA2C (GNN)"                                : "ia2c_gnn_shared",
        "IA2C (GNN+LSTM)"                           : "ia2c_gnn_lstm_shared",
        # "IA2C (LSTM actor, MLP critic)"             : "ia2c_lstm_actor_mlp_critic",
        # "IA2C (MLP actor, LSTM critic)"             : "ia2c_mlp_actor_lstm_critic",
        "MA2C-PA (MLP)"                             : "ma2c_pa_mlp",
        "MA2C-PA (LSTM)"                            : "ma2c_pa_lstm",
        "MA2C-PA (GNN)"                             : "ma2c_pa_gnn_attn",
        "MA2C-PA (GNN+LSTM)"                        : "ma2c_pa_gnn_lstm",
    }

    # Legend grouping
    memoryless_labels = ["IA2C (MLP)", "IA2C (GNN)", "MA2C-PA (MLP)", "MA2C-PA (GNN)"]
    lstm_labels       = ["IA2C (LSTM)", "IA2C (GNN+LSTM)", "MA2C-PA (LSTM)", "MA2C-PA (GNN+LSTM)"]

    labels_order = list(runs.keys())
    color_map = build_color_map(labels_order)

    files = {
        "throughput": "avg_throughput_veh_per_hour.npy",
        "travel"    : "avg_mean_travel_time_s.npy",
        "wait"      : "avg_mean_waiting_time_s.npy",
        "return"    : "avg_return.npy",
    }

    # Put "Episode Return" on the left, others shift right
    metrics_order = ["return", "throughput", "travel", "wait"]

    # Load all series per (grid -> metric -> label -> list of arrays by seed)
    all_series: Dict[int, Dict[str, Dict[str, List[np.ndarray]]]] = {
        g: {m: {lbl: [] for lbl in labels_order} for m in files} for g in grids
    }
    max_pts = int(args.max_points)

    for g in grids:
        for seed in seeds:
            logs_dir = Path(args.base_dir) / f"logs_grid_{g}" / f"seed{seed}"
            for label, prefix in runs.items():
                for metric, fname in files.items():
                    y = load_or_none(logs_dir / f"{prefix}_{fname}")
                    if y is not None and len(y) > 0:
                        all_series[g][metric][label].append(y.astype(float))
                    # if missing, we simply don't append for this seed

    # ---- plotting ----
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(
        nrows=nrows, ncols=4, figsize=(26, 4.8 * nrows),
        constrained_layout=False, sharex=False, sharey=False
    )

    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)  # shape -> [1, 4]

    ylabels = {
        "throughput": "Vehicles per hour (↑ better)",
        "travel"    : "Mean travel time (s) (↓ better)",
        "wait"      : "Average waiting time (s) (↓ better)",
        "return"    : "Average episode return (↑ better)",
    }
    titles = {
        "throughput": "Throughput",
        "travel"    : "Mean Travel Time",
        "wait"      : "Average Waiting Time",
        "return"    : "Episode Return",
    }

    stride = max(1, int(args.eval_every))
    handles_by_label: dict[str, Line2D] = {}

    for r, g in enumerate(grids):
        for c, metric_key in enumerate(metrics_order):
            ax = axes[r, c]
            ax.set_xmargin(0.01)
            ax.set_ymargin(0.02)
            ax.tick_params(axis="both", labelsize=TICK_FS)

            for label in labels_order:
                seed_curves = all_series[g][metric_key][label]
                if not seed_curves:
                    continue

                mean_y, std_y = aggregate_mean_std(seed_curves, args.smooth, max_pts)
                if mean_y.size == 0:
                    continue
                x = stride * np.arange(1, len(mean_y) + 1)

                # main line (mean)
                (line,) = ax.plot(
                    x, mean_y,
                    linewidth=2.0,
                    marker="o",
                    markersize=3.8,
                    markevery=max(1, len(x) // 14),
                    label=label,
                    color=color_map[label],
                    zorder=3,
                )
                # shaded std
                if std_y.size == mean_y.size and np.any(std_y > 0):
                    ax.fill_between(
                        x, mean_y - std_y, mean_y + std_y,
                        color=color_map[label],
                        alpha=0.18, linewidth=0, zorder=2
                    )

                if label not in handles_by_label:
                    handles_by_label[label] = line

            ax.set_title(f"{titles[metric_key]}  (grid {g}x{g})", fontsize=TITLE_FS, pad=3)
            ax.set_ylabel(ylabels[metric_key], fontsize=LABEL_FS)
            if r == nrows - 1:
                ax.set_xlabel("Episodes", fontsize=LABEL_FS)
            ax.grid(True, linestyle="--", alpha=0.6)

    # ---- two legend boxes ----
    mem_handles  = [handles_by_label[l] for l in memoryless_labels if l in handles_by_label]
    mem_labels   = [h.get_label() for h in mem_handles]

    lstm_handles = [handles_by_label[l] for l in lstm_labels if l in handles_by_label]
    lstm_labels_ = [h.get_label() for h in lstm_handles]

    # Adjust layout to leave enough room for two legend boxes at the bottom
    fig.subplots_adjust(
        left=0.045, right=0.992,
        top=0.958,
        bottom=0.15,           # a bit more room for two boxes
        wspace=0.16, hspace=0.22
    )

    # Left legend box (Stateless)
    if mem_handles:
        leg_left = fig.legend(
            handles=mem_handles,
            labels=mem_labels,
            loc="lower center",
            ncol=min(4, len(mem_handles)),
            frameon=True,
            bbox_to_anchor=(0.25, 0.03),
            fontsize=LEGEND_FS,
            columnspacing=1.4,
            handletextpad=0.6,
            borderaxespad=0.6,
            title="Stateless (no LSTM)",
        )
        plt.setp(leg_left.get_title(), fontsize=LEGEND_FS, fontweight="bold")

    # Right legend box (Stateful)
    if lstm_handles:
        leg_right = fig.legend(
            handles=lstm_handles,
            labels=lstm_labels_,
            loc="lower center",
            ncol=min(4, len(lstm_handles)),
            frameon=True,
            bbox_to_anchor=(0.72, 0.03),
            fontsize=LEGEND_FS,
            columnspacing=1.4,
            handletextpad=0.6,
            borderaxespad=0.6,
            title="Stateful (LSTM-augmented)",
        )
        plt.setp(leg_right.get_title(), fontsize=LEGEND_FS, fontweight="bold")

    # Optional save
    if args.save_path:
        out_path = Path(args.save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {out_path}")

    plt.show()

if __name__ == "__main__":
    main()
