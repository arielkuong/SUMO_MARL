#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# -------------------- style constants --------------------
TITLE_FS = 14
LABEL_FS = 12
LEGEND_FS = 14
TICK_FS = 10


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
    Progressive causal moving average, without head padding:

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

    c = np.cumsum(np.insert(y, 0, 0.0))
    idx = np.arange(1, n + 1)
    starts = np.maximum(0, idx - k_eff)
    lens = idx - starts

    return (c[idx] - c[starts]) / lens


def build_color_map(labels):
    """
    Darker, higher-contrast palette.
    """
    dark_palette = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # grey
        "#bcbd22",  # olive
        "#17becf",  # cyan
    ]

    return {lbl: dark_palette[i % len(dark_palette)] for i, lbl in enumerate(labels)}


def aggregate_mean_std(
    series_list: List[np.ndarray],
    smooth_k: int,
    max_pts: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a list of 1D arrays, one per seed, return mean and std after:

      1. truncating each to max_pts,
      2. optionally smoothing each seed curve,
      3. aligning to the common minimum length,
      4. computing mean and std across seeds.

    If fewer than two valid seeds are available, std will be zeros.
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
    Y = np.stack(ys, axis=0)

    mean = np.nanmean(Y, axis=0)
    std = np.nanstd(Y, axis=0)

    return mean, std


def parse_grid_values(spec: str, grids: List[int], name: str) -> Dict[int, float]:
    """
    Parse baseline values from command-line input.

    Supported formats:

    1. One scalar used for all grids:

       --fixed-time-throughput 6801.12

    2. Comma-separated values matching --grids order:

       --fixed-time-throughput 6801.12,12000.0,25000.0

    3. Explicit grid:value pairs:

       --fixed-time-throughput 3:6801.12,5:12000.0,10:25000.0

    Empty string disables that baseline metric.
    """
    spec = spec.strip()

    if not spec:
        return {}

    if ":" in spec:
        out: Dict[int, float] = {}

        for item in spec.split(","):
            item = item.strip()

            if not item:
                continue

            if ":" not in item:
                raise ValueError(
                    f"Invalid value for {name}: '{item}'. "
                    "Use either all scalar/list values or all grid:value pairs."
                )

            g_str, v_str = item.split(":", 1)
            out[int(g_str.strip())] = float(v_str.strip())

        return out

    vals = [float(x.strip()) for x in spec.split(",") if x.strip()]

    if len(vals) == 1:
        return {g: vals[0] for g in grids}

    if len(vals) != len(grids):
        raise ValueError(
            f"{name} expects either one value, {len(grids)} values matching --grids, "
            f"or explicit grid:value pairs. Got {len(vals)} values."
        )

    return {g: v for g, v in zip(grids, vals)}


# -------------------- main --------------------

def main():
    parser = argparse.ArgumentParser(
        "Plot KPIs for multiple grid sizes in one figure, mean±std over seeds, "
        "with fixed-time and max-pressure baseline curves."
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base directory containing logs_grid_N/",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="42,137,256,389,451,518,624,733,861,947",
        help="Comma-separated seeds, e.g. '42,238,458'.",
    )

    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Moving-average window over evaluation stages. Use 1 to disable.",
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default="",
        help="Optional path to save the figure.",
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=5,
        help="Episodes per evaluation.",
    )

    parser.add_argument(
        "--max-points",
        type=int,
        default=150,
        help="Maximum evaluation points per run.",
    )

    parser.add_argument(
        "--grids",
        type=str,
        default="3,5,10",
        help="Comma-separated grid sizes, e.g. '3,5,10'.",
    )

    # -------------------- baseline arguments --------------------
    #
    # These defaults are your current baseline values.
    #
    # By default, the same value is used for every grid size.
    # You can override them from the command line if needed.
    #
    # Examples:
    #
    #   Same value for all grids:
    #     --fixed-time-throughput 6801.12
    #
    #   Different values following --grids order:
    #     --fixed-time-throughput 6801.12,12000,25000
    #
    #   Explicit grid:value form:
    #     --fixed-time-throughput 3:6801.12,5:12000,10:25000

    parser.add_argument(
        "--fixed-time-throughput",
        type=str,
        default="3:6574.68,5:10717.20,10:17610.84",
        help="Fixed-time throughput baseline.",
    )

    parser.add_argument(
        "--fixed-time-travel",
        type=str,
        default="3:279.39,5:338.44,10:404.53",
        help="Fixed-time mean travel time baseline.",
    )

    parser.add_argument(
        "--fixed-time-wait",
        type=str,
        default="3:190.19,5:228.83,10:266.69",
        help="Fixed-time average waiting time baseline.",
    )

    parser.add_argument(
        "--max-pressure-throughput",
        type=str,
        default="3:5657.04,5:11027.88,10:25326.72",
        help="Max-pressure throughput baseline.",
    )

    parser.add_argument(
        "--max-pressure-travel",
        type=str,
        default="3:299.45,5:302.92,10:309.52",
        help="Max-pressure mean travel time baseline.",
    )

    parser.add_argument(
        "--max-pressure-wait",
        type=str,
        default="3:190.39,5:163.67,10:103.61",
        help="Max-pressure average waiting time baseline.",
    )

    args = parser.parse_args()

    grids = [int(x.strip()) for x in args.grids.split(",") if x.strip()]
    nrows = len(grids)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    baselines = {
        "Fixed-time": {
            "throughput": parse_grid_values(
                args.fixed_time_throughput,
                grids,
                "--fixed-time-throughput",
            ),
            "travel": parse_grid_values(
                args.fixed_time_travel,
                grids,
                "--fixed-time-travel",
            ),
            "wait": parse_grid_values(
                args.fixed_time_wait,
                grids,
                "--fixed-time-wait",
            ),
        },
        "Max-pressure": {
            "throughput": parse_grid_values(
                args.max_pressure_throughput,
                grids,
                "--max-pressure-throughput",
            ),
            "travel": parse_grid_values(
                args.max_pressure_travel,
                grids,
                "--max-pressure-travel",
            ),
            "wait": parse_grid_values(
                args.max_pressure_wait,
                grids,
                "--max-pressure-wait",
            ),
        },
    }

    # Less visually dominant than the learning curves.
    # The learning curves use linewidth=2.0 and zorder=3.
    # These baselines are thinner, lighter, semi-transparent, and behind the curves.
    baseline_styles = {
        "Fixed-time": {
            "color": "0.20",
            "linestyle": (0, (6, 3)),      # longer dashed line
            "linewidth": 1.35,
            "alpha": 0.50,
        },
        "Max-pressure": {
            "color": "0.50",
            "linestyle": (0, (1, 2)),      # fine dotted line
            "linewidth": 1.70,
            "alpha": 0.55,
        },
    }

    # Run name prefixes must match your EvalHistory run_name in training scripts.
    runs = {
        "IDQN (MLP)": "dqn_mlp_shared",
        "IDRQN (LSTM)": "drqn_lstm_shared_seqlen8",
        "DQN (GNN)": "dqn_gnn_shared",
        "DRQN (GNN+LSTM)": "drqn_gnn_lstm_shared_seqlen8",
        "DQN (CTDE+MLP)": "vdn_ctde_mlp_shared",
        "DRQN (CTDE+LSTM)": "vdn_ctde_lstm_shared_seqlen8",
        "DQN (CTDE+GNN)": "vdn_ctde_gnn_shared",
        "DRQN (CTDE+GNN+LSTM)": "vdn_ctde_gnn_lstm_shared_seqlen8",
    }

    # Legend grouping.
    memoryless_labels = [
        "IDQN (MLP)",
        "DQN (GNN)",
        "DQN (CTDE+MLP)",
        "DQN (CTDE+GNN)",
    ]

    lstm_labels = [
        "IDRQN (LSTM)",
        "DRQN (GNN+LSTM)",
        "DRQN (CTDE+LSTM)",
        "DRQN (CTDE+GNN+LSTM)",
    ]

    labels_order = list(runs.keys())
    color_map = build_color_map(labels_order)

    files = {
        "throughput": "avg_throughput_veh_per_hour.npy",
        "travel": "avg_mean_travel_time_s.npy",
        "wait": "avg_mean_waiting_time_s.npy",
    }

    metrics_order = ["throughput", "travel", "wait"]

    # Load all series per grid, metric, method and seed.
    all_series: Dict[int, Dict[str, Dict[str, List[np.ndarray]]]] = {
        g: {m: {lbl: [] for lbl in labels_order} for m in files}
        for g in grids
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

    # -------------------- plotting --------------------

    plt.style.use("seaborn-v0_8-darkgrid")

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=3,
        figsize=(26, 4.8 * nrows),
        constrained_layout=False,
        sharex=False,
        sharey=False,
    )

    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    ylabels = {
        "throughput": "Vehicles per hour (↑ better)",
        "travel": "Mean travel time (s) (↓ better)",
        "wait": "Average waiting time (s) (↓ better)",
    }

    titles = {
        "throughput": "Throughput",
        "travel": "Mean Travel Time",
        "wait": "Average Waiting Time",
    }

    stride = max(1, int(args.eval_every))

    handles_by_label: Dict[str, Line2D] = {}
    baseline_handles: Dict[str, Line2D] = {}

    for r, g in enumerate(grids):
        for c, metric_key in enumerate(metrics_order):
            ax = axes[r, c]

            ax.set_xmargin(0.01)
            ax.set_ymargin(0.02)
            ax.tick_params(axis="both", labelsize=TICK_FS)

            # RL / learning curves.
            for label in labels_order:
                seed_curves = all_series[g][metric_key][label]

                if not seed_curves:
                    continue

                mean_y, std_y = aggregate_mean_std(
                    seed_curves,
                    args.smooth,
                    max_pts,
                )

                if mean_y.size == 0:
                    continue

                x = stride * np.arange(1, len(mean_y) + 1)

                line, = ax.plot(
                    x,
                    mean_y,
                    linewidth=2.0,
                    marker="o",
                    markersize=3.8,
                    markevery=max(1, len(x) // 14),
                    label=label,
                    color=color_map[label],
                    zorder=3,
                )

                if std_y.size == mean_y.size and np.any(std_y > 0):
                    ax.fill_between(
                        x,
                        mean_y - std_y,
                        mean_y + std_y,
                        color=color_map[label],
                        alpha=0.18,
                        linewidth=0,
                        zorder=2,
                    )

                if label not in handles_by_label:
                    handles_by_label[label] = line

            # Fixed-time and max-pressure baseline curves.
            for baseline_name, baseline_metrics in baselines.items():
                y_value = baseline_metrics[metric_key].get(g, None)

                if y_value is None or not np.isfinite(y_value):
                    continue

                baseline_line = ax.axhline(
                    y=y_value,
                    label=baseline_name,
                    zorder=1,
                    **baseline_styles[baseline_name],
                )

                if baseline_name not in baseline_handles:
                    baseline_handles[baseline_name] = baseline_line

            ax.set_title(
                f"{titles[metric_key]}  (grid {g}x{g})",
                fontsize=TITLE_FS,
                pad=3,
            )

            ax.set_ylabel(ylabels[metric_key], fontsize=LABEL_FS)

            if r == nrows - 1:
                ax.set_xlabel("Episodes", fontsize=LABEL_FS)

            ax.grid(True, linestyle="--", alpha=0.6)

    # -------------------- legends --------------------

    mem_handles = [
        handles_by_label[l]
        for l in memoryless_labels
        if l in handles_by_label
    ]
    mem_labels = [h.get_label() for h in mem_handles]

    lstm_handles = [
        handles_by_label[l]
        for l in lstm_labels
        if l in handles_by_label
    ]
    lstm_labels_ = [h.get_label() for h in lstm_handles]

    base_handles = list(baseline_handles.values())
    base_labels = [h.get_label() for h in base_handles]

    # Legend font sizes.
    LEGEND_BOX_FS = 12
    LEGEND_TITLE_FS = 13
    BASELINE_BOX_FS = 12

    # Leave enough bottom margin for the larger legend row.
    fig.subplots_adjust(
        left=0.045,
        right=0.992,
        top=0.958,
        bottom=0.180,
        wspace=0.16,
        hspace=0.22,
    )

    # Fixed, non-overlapping legend slots in figure coordinates.
    # Increase gap to create more space between the three legend boxes.
    legend_y = 0.026
    legend_h = 0.112   # same height for all three legend boxes

    x0 = 0.006
    total_w = 0.988
    gap = 0.01        # bigger gap between legend boxes

    # Width allocation.
    # Recomputed so the three boxes still fill the row evenly.
    base_w = 0.15
    mem_w = 0.36
    lstm_w = total_w - base_w - mem_w - 2 * gap

    base_box = [
        x0,
        legend_y,
        base_w,
        legend_h,
    ]

    mem_box = [
        x0 + base_w + gap,
        legend_y,
        mem_w,
        legend_h,
    ]

    lstm_box = [
        x0 + base_w + gap + mem_w + gap,
        legend_y,
        lstm_w,
        legend_h,
    ]

    # Internal spacing for the legend entries.
    base_legend_spacing = dict(
        columnspacing=0.34,
        handletextpad=0.26,
        handlelength=1.10,
        borderpad=0.32,
        labelspacing=0.22,
        borderaxespad=0.00,
    )

    method_legend_spacing = dict(
        columnspacing=0.38,
        handletextpad=0.28,
        handlelength=1.14,
        borderpad=0.32,
        labelspacing=0.22,
        borderaxespad=0.00,
    )

    def add_compact_legend_box(
        box,
        handles,
        labels,
        title,
        ncol,
        fontsize,
        spacing,
    ):
        if not handles:
            return None

        # Dedicated invisible axes for this legend.
        # Each legend has its own slot, so the boxes cannot overlap.
        ax_leg = fig.add_axes(box)
        ax_leg.set_axis_off()

        leg = ax_leg.legend(
            handles=handles,
            labels=labels,
            loc="center",
            ncol=ncol,
            mode="expand",
            frameon=True,
            bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
            bbox_transform=ax_leg.transAxes,
            fontsize=fontsize,
            title=title,
            **spacing,
        )

        plt.setp(
            leg.get_title(),
            fontsize=LEGEND_TITLE_FS,
            fontweight="bold",
        )

        return leg

    add_compact_legend_box(
        box=base_box,
        handles=base_handles,
        labels=base_labels,
        title="Baseline rules",
        ncol=max(1, len(base_handles)),
        fontsize=BASELINE_BOX_FS,
        spacing=base_legend_spacing,
    )

    add_compact_legend_box(
        box=mem_box,
        handles=mem_handles,
        labels=mem_labels,
        title="Stateless (no LSTM)",
        ncol=max(1, len(mem_handles)),
        fontsize=LEGEND_BOX_FS,
        spacing=method_legend_spacing,
    )

    add_compact_legend_box(
        box=lstm_box,
        handles=lstm_handles,
        labels=lstm_labels_,
        title="Stateful (LSTM-augmented)",
        ncol=max(1, len(lstm_handles)),
        fontsize=LEGEND_BOX_FS,
        spacing=method_legend_spacing,
    )

    # -------------------- save / show --------------------

    if args.save_path:
        out_path = Path(args.save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
