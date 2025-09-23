#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_or_none(path: Path):
    try:
        if path.exists():
            return np.load(path)
    except Exception:
        pass
    return None

def moving_average(y: np.ndarray, k: int) -> np.ndarray:
    """
    Causal moving average with head padding to keep length.
    If len(y) < k, it automatically reduces the window to len(y).
    If y is empty or the effective window is 1, it returns y unchanged.
    """
    if y is None:
        return y
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return y

    k_eff = max(1, min(int(k), n))  # effective window in [1, n]
    if k_eff == 1:
        return y

    # cumulative-sum trick
    c = np.cumsum(np.insert(y, 0, 0.0))
    sm = (c[k_eff:] - c[:-k_eff]) / float(k_eff)  # length n - k_eff + 1

    if sm.size == 0:
        # Happens only if something odd slipped through; fall back to raw y
        return y

    # pad the head with the first averaged value to keep original length
    head = np.full(k_eff - 1, sm[0], dtype=np.float64)
    return np.concatenate([head, sm])


def main():
    parser = argparse.ArgumentParser(
        "Plot KPIs (IDQN MLP vs shared-MLP vs shared-GNN vs IDRQN shared-LSTM) in one figure"
    )
    parser.add_argument("--grid-n", type=int, default=3, help="Grid size N (folder: logs_grid_N)")
    parser.add_argument("--base-dir", type=str, default=".", help="Base directory containing logs_grid_N/")
    parser.add_argument("--smooth", type=int, default=5, help="Moving-average window (eval stages). Use 1 to disable.")
    parser.add_argument("--save-path", type=str, default="", help="Optional path to save the figure instead of only showing")
    args = parser.parse_args()

    logs_dir = Path(args.base_dir) / f"logs_grid_{args.grid_n}"

    # Prefixes must match EvalHistory run_name used in your training scripts
    runs = {
        # "DQN (MLP individual)"            : "dqn_mlp",
        "DQN (MLP)"                         : "dqn_mlp_shared",
        "DQN (MLP, with nbobs)"             : "dqn_mlp_shared_neighbor_obs",
        "DQN (GNN)"                         : "dqn_gnn_shared",
        "DQN (CTDE)"                        : "vdn_ctde_shared_mlp",
        "DRQN (LSTM)"                       : "drqn_lstm_shared_seqlen8",
        "DRQN (LSTM, with nbobs)"           : "drqn_lstm_shared_nbobs_seqlen8",
        "DRQN (GNN+LSTM)"                   : "drqn_gnn_lstm_shared_seqlen8",
        "DRQN (CTDE+LSTM)"                  : "vdn_ctde_shared_lstm",
    }

    files = {
        "throughput": "avg_throughput_veh_per_hour.npy",
        "travel"    : "avg_mean_travel_time_s.npy",
        "wait"      : "avg_mean_waiting_time_s.npy",
        "return"    : "avg_return.npy",
    }

    # Load series
    series = {metric: {} for metric in files}
    for label, prefix in runs.items():
        for metric, fname in files.items():
            series[metric][label] = load_or_none(logs_dir / f"{prefix}_{fname}")

    # Styling more suitable for RL curves
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax_map = {
        "throughput": axes[0, 0],
        "travel"    : axes[0, 1],
        "wait"      : axes[1, 0],
        "return"    : axes[1, 1],
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
        "return"    : "Return",
    }

    # Plot each metric
    for metric_key, ax in ax_map.items():
        plotted_any = False
        for label in runs.keys():
            y = series[metric_key][label]
            if y is None or len(y) == 0:
                continue
            y_plot = moving_average(y.astype(float), args.smooth)
            x = np.arange(1, len(y_plot) + 1)
            # Put markers every ~max(1, len/12) points to reduce clutter
            markevery = max(1, len(x) // 12)
            ax.plot(x, y_plot, label=label, linewidth=2.0, marker="o", markersize=4, markevery=markevery)
            plotted_any = True

        ax.set_title(f"{titles[metric_key]}  (grid_n={args.grid_n})", fontsize=12, pad=6)
        ax.set_ylabel(ylabels[metric_key])
        ax.set_xlabel("Evaluation stage")
        if plotted_any:
            ax.legend(fontsize=9, frameon=True)
        else:
            ax.text(0.5, 0.5, "No data found", ha="center", va="center", transform=ax.transAxes)
        # Subtle grid for readability
        ax.grid(True, linestyle="--", alpha=0.6)

    if args.save_path:
        out_path = Path(args.save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Saved figure to: {out_path}")

    plt.show()

if __name__ == "__main__":
    main()
