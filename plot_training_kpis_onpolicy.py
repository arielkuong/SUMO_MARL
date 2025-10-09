#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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

# -------------------- main --------------------

def main():
    parser = argparse.ArgumentParser(
        "Plot KPIs (IDQN/DRQN variants) with fixed evaluation cadence (e.g., every 5 episodes)."
    )
    parser.add_argument("--grid-n", type=int, default=3, help="Grid size N (folder: logs_grid_N)")
    parser.add_argument("--base-dir", type=str, default=".", help="Base directory containing logs_grid_N/")
    parser.add_argument("--seed", type=int, default=42, help="Seed subfolder to read from (seedN)")
    parser.add_argument("--smooth", type=int, default=1, help="Moving-average window (eval stages). Use 1 to disable.")
    parser.add_argument("--save-path", type=str, default="", help="Optional path to save the figure")
    parser.add_argument("--eval-every", type=int, default=10, help="Episodes per evaluation (fixed for all runs).")
    parser.add_argument("--max-points", type=int, default=150, help="Plot at most this many evaluation points per run.")
    args = parser.parse_args()

    logs_dir = Path(args.base_dir) / f"logs_grid_{args.grid_n}" / f"seed{args.seed}"

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
        # "MA2C (MLP)"                                : "ma2c_mlp",
        # "MA2C-PA (MLP, team R)"                     : "ma2c_pa_mlp_teamreward",
        # "MA2C (Actor LSTM, Critic MLP)"             : "ma2c_actor_lstm",
        # "MA2C (neighbour-critic, MLP)"              : "ma2c_neighbor_critic_mlp",
    }

    files = {
        "throughput": "avg_throughput_veh_per_hour.npy",
        "travel"    : "avg_mean_travel_time_s.npy",
        "wait"      : "avg_mean_waiting_time_s.npy",
        "return"    : "avg_return.npy",
    }

    # Load + truncate
    series = {metric: {} for metric in files}
    max_pts = int(args.max_points)
    for label, prefix in runs.items():
        for metric, fname in files.items():
            y = load_or_none(logs_dir / f"{prefix}_{fname}")
            if y is not None and len(y) > 0:
                series[metric][label] = y.astype(float)[:max_pts]
            else:
                series[metric][label] = None

    # ---- plotting ----
    plt.style.use("seaborn-v0_8-darkgrid")
    # Wider figure but minimal side padding; one row of four plots
    fig, axes = plt.subplots(1, 4, figsize=(26, 5), constrained_layout=False, sharex=False)

    ax_map = {
        "throughput": axes[0],
        "travel"    : axes[1],
        "wait"      : axes[2],
        "return"    : axes[3],
    }
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
    legend_handles, legend_labels = [], []

    for metric_key, ax in ax_map.items():
        ax.set_xmargin(0.01)  # trim empty margins around data
        ax.set_ymargin(0.02)

        for label in runs.keys():
            y = series[metric_key][label]
            if y is None or len(y) == 0:
                continue
            y_plot = moving_average(y, args.smooth) if args.smooth > 1 else y
            x = stride * np.arange(1, len(y_plot) + 1)
            markevery = max(1, len(x) // 14)

            (line,) = ax.plot(
                x, y_plot,
                linewidth=2.0,
                marker="o",
                markersize=4,
                markevery=markevery,
                label=label
            )
            # collect handles only once (first subplot) for shared legend
            if metric_key == "throughput":
                legend_handles.append(line)
                legend_labels.append(label)

        ax.set_title(f"{titles[metric_key]}  (grid_n={args.grid_n})", fontsize=12, pad=6)
        ax.set_ylabel(ylabels[metric_key])
        ax.set_xlabel("Episodes")
        ax.grid(True, linestyle="--", alpha=0.6)

    # Tight outer margins and small inter-axes spacing; room at bottom for legend
    fig.subplots_adjust(left=0.04, right=0.985, top=0.92, bottom=0.23, wspace=0.18)

    # Shared legend below all plots
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="lower center",
            ncol=4,
            frameon=True,
            bbox_to_anchor=(0.5, 0.02),
            fontsize=10,
        )

    if args.save_path:
        out_path = Path(args.save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {out_path}")

    plt.show()

if __name__ == "__main__":
    main()
