#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple


# =============================================================================
# Baseline method mapping
# =============================================================================

BASELINE_ORDER: List[Tuple[str, str]] = [
    ("fixed_time", "Fixed-time"),
    ("max_pressure", "Max-pressure"),
]

BASELINE_METHOD_KEYS = {k for k, _ in BASELINE_ORDER}

BASELINE_ALIASES = {
    "fixed_time": {
        "fixed_time",
        "fixed-time",
        "fixedtime",
        "fixed",
        "ft",
        "timed",
        "fixed_time_controller",
    },
    "max_pressure": {
        "max_pressure",
        "max-pressure",
        "maxpressure",
        "mp",
        "max_pressure_controller",
    },
}

GRID_ORDER_DEFAULT = [3, 5, 10]


# =============================================================================
# Demand-type display titles
# =============================================================================

DISPLAY_TITLES = {
    "corridor": "Heavy corridor (E--W arterial)",
    "cross": "Cross (orthogonal OD axes)",
    "platoons": "Platoons (pulse trains)",
    "bursty": "Bursty (on--off)",
    "shifted": "Shifted",
    "incident": "Incident",
    "ampm": "AM--PM",
}


# =============================================================================
# CSV column aliases
# =============================================================================

COLMAP = {
    "throughput": (
        "throughput_veh_per_hour",
        "throughput_vph",
        "throughput_vph_mean",
        "throughput_mean",
        "throughput",
        "tp",
        "tp_mean",
    ),
    "mean_travel": (
        "mean_travel_time_s",
        "mean_travel_time_s_completed",
        "mean_travel_s",
        "mean_travel_s_mean",
        "mean_travel_mean",
        "mean_travel",
        "mtt",
        "mtt_mean",
        "mmt",
        "mmt_mean",
    ),
    "mean_wait": (
        "mean_waiting_time_s",
        "mean_waiting_time_s_completed",
        "mean_wait_s",
        "mean_wait_s_mean",
        "mean_wait_mean",
        "mean_wait",
        "awt",
        "awt_mean",
    ),
}

GRID_ALIASES = ("grid_n", "grid", "grid_size")


# =============================================================================
# Basic helpers
# =============================================================================

def _to_float_or_none(x: object) -> Optional[float]:
    if x is None:
        return None

    s = str(x).strip()
    if s == "":
        return None

    try:
        return float(s)
    except Exception:
        return None


def _get_first_present(row: dict, keys: Tuple[str, ...]) -> Optional[float]:
    for k in keys:
        if k in row:
            v = _to_float_or_none(row.get(k))
            if v is not None:
                return v
    return None


def _parse_grid(row: dict) -> Optional[int]:
    for k in GRID_ALIASES:
        if k not in row:
            continue

        raw = str(row[k]).strip().lower()

        try:
            return int(raw)
        except Exception:
            pass

        raw = raw.replace("gridn", "")
        raw = raw.replace("x", " ")
        parts = raw.split()

        for p in parts:
            try:
                return int(p)
            except Exception:
                pass

    return None


def normalise_method(raw: object) -> str:
    s = str(raw or "").strip().lower()
    s = s.replace("-", "_").replace(" ", "_")

    for canonical, aliases in BASELINE_ALIASES.items():
        normalised_aliases = {
            a.replace("-", "_").replace(" ", "_").lower()
            for a in aliases
        }
        if s in normalised_aliases:
            return canonical

    return s


def mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / len(xs)


def sample_std(xs: List[float]) -> Optional[float]:
    if len(xs) <= 1:
        return None

    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def fmt_num(v: Optional[float], digits: int) -> str:
    if v is None:
        return "\\textemdash{}"
    return f"{v:.{digits}f}"


def fmt_mean_std(
    mean_v: Optional[float],
    std_v: Optional[float],
    digits: int,
) -> str:
    if mean_v is None and std_v is None:
        return "\\textemdash{}"

    m = fmt_num(mean_v, digits) if mean_v is not None else None
    s = fmt_num(std_v, digits) if std_v is not None else None

    if m is not None and s is not None:
        return f"{m}\\,\\(\\pm\\)\\,{s}"

    return m if m is not None else s


# =============================================================================
# CSV discovery and reading
# =============================================================================

def discover_baseline_csv_for_type(
    csv_root: Path,
    demand_type: str,
    verbose: bool = False,
) -> Optional[Path]:
    """
    Discover baseline CSV for one demand type.

    Expected filename:
        all_baseline_results_<demand_type>.csv

    Supports either:
        --csv-dir ./eval_fixed_demand_type

    or:
        --csv-dir ./results
        where ./results/eval_fixed_demand_type/ contains the CSV files.
    """
    filename = f"all_baseline_results_{demand_type}.csv"

    candidates = [
        csv_root / filename,
        csv_root / "eval_fixed_demand_type" / filename,
    ]

    for p in candidates:
        if p.exists():
            if verbose:
                print(f"[INFO] Using baseline CSV for '{demand_type}': {p}")
            return p

    matches = sorted(csv_root.glob(f"**/{filename}"))
    if matches:
        if verbose:
            print(f"[INFO] Using baseline CSV by glob for '{demand_type}': {matches[0]}")
        return matches[0]

    if verbose:
        print(f"[WARN] No baseline CSV found for '{demand_type}' under {csv_root}")

    return None


def read_baseline_csv(
    csv_path: Path,
    verbose: bool = False,
) -> Dict[Tuple[int, str], Dict[str, Optional[float]]]:
    """
    Read one baseline CSV and aggregate repeated rows.

    Returns:
        (grid_n, controller) -> {
            throughput_mean,
            throughput_std,
            mean_travel_mean,
            mean_travel_std,
            mean_wait_mean,
            mean_wait_std,
        }

    Supported CSV columns include:
        controller
        throughput_veh_per_hour
        mean_travel_time_s
        mean_waiting_time_s

    The script ignores longest_queue by default.
    """
    raw_values: DefaultDict[
        Tuple[int, str],
        Dict[str, List[float]],
    ] = defaultdict(lambda: {
        "throughput": [],
        "mean_travel": [],
        "mean_wait": [],
    })

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        n_rows = 0
        n_used = 0
        skipped_methods = set()

        for row in reader:
            n_rows += 1

            raw_method = (
                row.get("controller")
                or row.get("method")
                or row.get("policy")
                or row.get("baseline")
                or ""
            )
            method = normalise_method(raw_method)

            if method not in BASELINE_METHOD_KEYS:
                if method:
                    skipped_methods.add(method)
                continue

            grid_n = _parse_grid(row)
            if grid_n is None:
                continue

            tp = _get_first_present(row, COLMAP["throughput"])
            mtt = _get_first_present(row, COLMAP["mean_travel"])
            awt = _get_first_present(row, COLMAP["mean_wait"])

            key = (grid_n, method)

            if tp is not None:
                raw_values[key]["throughput"].append(tp)
            if mtt is not None:
                raw_values[key]["mean_travel"].append(mtt)
            if awt is not None:
                raw_values[key]["mean_wait"].append(awt)

            n_used += 1

    out: Dict[Tuple[int, str], Dict[str, Optional[float]]] = {}

    for key, vals in raw_values.items():
        out[key] = {
            "throughput_mean": mean(vals["throughput"]),
            "throughput_std": sample_std(vals["throughput"]),
            "mean_travel_mean": mean(vals["mean_travel"]),
            "mean_travel_std": sample_std(vals["mean_travel"]),
            "mean_wait_mean": mean(vals["mean_wait"]),
            "mean_wait_std": sample_std(vals["mean_wait"]),
        }

    if verbose:
        print(f"[INFO] Loaded {n_used}/{n_rows} usable rows from {csv_path.name}")

        if skipped_methods:
            print(f"[INFO] Ignored controllers: {sorted(skipped_methods)}")

        for key in sorted(out):
            print(f"[INFO] Parsed {key}: {out[key]}")

    return out


# =============================================================================
# LaTeX formatting helpers
# =============================================================================

def printable_len_estimate(s: str) -> int:
    if not s:
        return 0

    t = s
    t = t.replace("\\,", "")
    t = t.replace("\\(\\pm\\)", "±")
    t = t.replace("\\textemdash{}", "—")

    for ch in ["{", "}", "\\", "$"]:
        t = t.replace(ch, "")

    return len(t)


def compute_kpi_widths_ex(
    per_type_baseline: Dict[str, Dict[Tuple[int, str], Dict[str, Optional[float]]]],
    demand_types: List[str],
    grids: List[int],
    digits_throughput: int,
    digits_time: int,
) -> Dict[str, int]:
    """
    Compute compact fixed-width KPI boxes.

    In the final compact table, all metric values are placed in grid columns,
    so the largest width among TP, MTT, and AWT is later used.
    """
    max_len = {"tp": 0, "mtt": 0, "awt": 0}

    def update_len(s: str, key: str) -> None:
        max_len[key] = max(max_len[key], printable_len_estimate(s))

    update_len("TP $\\uparrow$", "tp")
    update_len("MTT $\\downarrow$", "mtt")
    update_len("AWT $\\downarrow$", "awt")

    for t in demand_types:
        agg = per_type_baseline.get(t, {})

        for g in grids:
            for method_key, _label in BASELINE_ORDER:
                d = agg.get((g, method_key), {})

                update_len(
                    fmt_mean_std(
                        d.get("throughput_mean"),
                        d.get("throughput_std"),
                        digits_throughput,
                    ),
                    "tp",
                )
                update_len(
                    fmt_mean_std(
                        d.get("mean_travel_mean"),
                        d.get("mean_travel_std"),
                        digits_time,
                    ),
                    "mtt",
                )
                update_len(
                    fmt_mean_std(
                        d.get("mean_wait_mean"),
                        d.get("mean_wait_std"),
                        digits_time,
                    ),
                    "awt",
                )

    return {
        "tp": max(7, int(round(max_len["tp"] * 0.65)) + 1),
        "mtt": max(6, int(round(max_len["mtt"] * 0.55)) + 1),
        "awt": max(6, int(round(max_len["awt"] * 0.55)) + 1),
    }


def fixbox(s: str, ex: int, align: str = "r") -> str:
    return f"\\makebox[{ex}ex][{align}]{{{s}}}"


def maybe_bold(s: str, is_best: bool) -> str:
    if not is_best:
        return s

    if s == "\\textemdash{}":
        return s

    return f"\\textbf{{{s}}}"


def metric_mean(
    d: Dict[str, Optional[float]],
    metric: str,
) -> Optional[float]:
    if metric == "tp":
        return d.get("throughput_mean")
    if metric == "mtt":
        return d.get("mean_travel_mean")
    if metric == "awt":
        return d.get("mean_wait_mean")

    raise ValueError(f"Unknown metric: {metric}")


def metric_string(
    d: Dict[str, Optional[float]],
    metric: str,
    digits_throughput: int,
    digits_time: int,
) -> str:
    if metric == "tp":
        return fmt_mean_std(
            d.get("throughput_mean"),
            d.get("throughput_std"),
            digits_throughput,
        )

    if metric == "mtt":
        return fmt_mean_std(
            d.get("mean_travel_mean"),
            d.get("mean_travel_std"),
            digits_time,
        )

    if metric == "awt":
        return fmt_mean_std(
            d.get("mean_wait_mean"),
            d.get("mean_wait_std"),
            digits_time,
        )

    raise ValueError(f"Unknown metric: {metric}")


def best_flags_for_grid(
    agg: Dict[Tuple[int, str], Dict[str, Optional[float]]],
    grid_n: int,
) -> Dict[Tuple[str, str], bool]:
    """
    Compute best result flags between Fixed-time and Max-pressure.

    TP: higher is better.
    MTT/AWT: lower is better.
    """
    flags: Dict[Tuple[str, str], bool] = {}

    for metric in ("tp", "mtt", "awt"):
        values: List[Tuple[str, float]] = []

        for method_key, _label in BASELINE_ORDER:
            d = agg.get((grid_n, method_key), {})
            v = metric_mean(d, metric)

            if v is not None:
                values.append((method_key, v))

        if not values:
            continue

        if metric == "tp":
            best_value = max(v for _, v in values)
        else:
            best_value = min(v for _, v in values)

        for method_key, v in values:
            flags[(method_key, metric)] = abs(v - best_value) < 1e-12

    return flags


def controller_label(label: str, font_size: float, baseline_skip: float) -> str:
    return (
        "{"
        f"\\fontsize{{{font_size:.1f}pt}}{{{baseline_skip:.1f}pt}}\\selectfont"
        f"\\textbf{{{label}}}"
        "}"
    )


# =============================================================================
# Inner compact subtable builder
# =============================================================================

def inner_baseline_tabular(
    demand_type: str,
    agg: Dict[Tuple[int, str], Dict[str, Optional[float]]],
    grids: List[int],
    digits_throughput: int,
    digits_time: int,
    widths_ex: Dict[str, int],
    bold_best: bool = False,
    controller_font_size: float = 6.4,
    controller_baseline_skip: float = 6.8,
) -> str:
    """
    Build one compact demand-type subtable.

    Layout:

                    3x3       5x5       10x10
    Fixed-time
    TP              ...
    MTT             ...
    AWT             ...

    Max-pressure
    TP              ...
    MTT             ...
    AWT             ...

    This is much narrower than:
        Grid | Fixed-time TP MTT AWT | Max-pressure TP MTT AWT

    Therefore the whole 2x2 table can keep a readable font size.
    """
    title = DISPLAY_TITLES.get(demand_type, demand_type).replace("_", "\\_")

    ncols = 1 + len(grids)
    grid_colspec = "c" * len(grids)

    # Label column + one column per grid size
    colspec = "@{}l@{\\hspace{2pt}}" + grid_colspec + "@{}"

    value_w = max(widths_ex["tp"], widths_ex["mtt"], widths_ex["awt"])

    def val_box(s: str) -> str:
        return fixbox(s, value_w, "r")

    def metric_cell(method_key: str, grid_n: int, metric: str) -> str:
        d = agg.get((grid_n, method_key), {})
        flags = best_flags_for_grid(agg, grid_n) if bold_best else {}

        s = metric_string(
            d=d,
            metric=metric,
            digits_throughput=digits_throughput,
            digits_time=digits_time,
        )

        s = maybe_bold(s, flags.get((method_key, metric), False))
        return val_box(s)

    lines: List[str] = []
    lines.append("{% local spacing")
    lines.append("\\setlength{\\tabcolsep}{0.35pt}%")
    lines.append("\\begin{tabular}{" + colspec + "}")

    lines.append(
        f"\\multicolumn{{{ncols}}}{{c}}{{\\textbf{{{title}}}}}\\\\[0.1ex]"
    )
    lines.append("\\toprule")

    header = "\\textbf{}"
    for g in grids:
        header += f" & \\textbf{{{g}x{g}}}"
    header += "\\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Fixed-time block
    lines.append(
        f"\\multicolumn{{{ncols}}}{{l}}{{"
        f"{controller_label('Fixed-time', controller_font_size, controller_baseline_skip)}"
        f"}}\\\\[-0.2ex]"
    )

    for metric_label, metric_key in [
        ("TP $\\uparrow$", "tp"),
        ("MTT $\\downarrow$", "mtt"),
        ("AWT $\\downarrow$", "awt"),
    ]:
        row = metric_label
        for g in grids:
            row += f" & {metric_cell('fixed_time', g, metric_key)}"
        row += "\\\\"
        lines.append(row)

    lines.append("\\addlinespace[0.15ex]")

    # Max-pressure block
    lines.append(
        f"\\multicolumn{{{ncols}}}{{l}}{{"
        f"{controller_label('Max-pressure', controller_font_size, controller_baseline_skip)}"
        f"}}\\\\[-0.2ex]"
    )

    for metric_label, metric_key in [
        ("TP $\\uparrow$", "tp"),
        ("MTT $\\downarrow$", "mtt"),
        ("AWT $\\downarrow$", "awt"),
    ]:
        row = metric_label
        for g in grids:
            row += f" & {metric_cell('max_pressure', g, metric_key)}"
        row += "\\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")

    return "\n".join(lines)


# =============================================================================
# Full 2x2 one-column table builder
# =============================================================================

def build_2x2_baseline_table(
    demand_types: List[str],
    per_type_baseline: Dict[str, Dict[Tuple[int, str], Dict[str, Optional[float]]]],
    grids: List[int],
    digits_throughput: int,
    digits_time: int,
    scale: float = 0.96,
    bold_best: bool = False,
    body_font_size: float = 7.2,
    body_baseline_skip: float = 7.6,
    controller_font_size: float = 6.2,
    controller_baseline_skip: float = 6.6,
) -> str:
    """
    Build a forced 2x2 baseline table inside one column.

    This version uses an outer tabular to enforce:

        subtable 1 | subtable 2
        subtable 3 | subtable 4

    This avoids accidental 1-2-1 wrapping.
    """
    types = demand_types[:4]

    if len(types) < 4:
        types = types + [""] * (4 - len(types))

    widths_ex = compute_kpi_widths_ex(
        per_type_baseline=per_type_baseline,
        demand_types=[t for t in types if t],
        grids=grids,
        digits_throughput=digits_throughput,
        digits_time=digits_time,
    )

    # Important:
    # Keep each panel safely below half column width.
    # If this is too large, LaTeX may create overfull boxes.
    panel_w = "0.475\\columnwidth"

    body_font_cmd = (
        f"\\fontsize{{{body_font_size:.1f}pt}}"
        f"{{{body_baseline_skip:.1f}pt}}\\selectfont"
    )

    def panel(t: str) -> str:
        if not t:
            return (
                f"\\begin{{minipage}}[t]{{{panel_w}}}\n"
                "\\centering\n"
                "\\end{minipage}"
            )

        agg = per_type_baseline.get(t, {})

        inner = inner_baseline_tabular(
            demand_type=t,
            agg=agg,
            grids=grids,
            digits_throughput=digits_throughput,
            digits_time=digits_time,
            widths_ex=widths_ex,
            bold_best=bold_best,
            controller_font_size=controller_font_size,
            controller_baseline_skip=controller_baseline_skip,
        )

        return (
            f"\\begin{{minipage}}[t]{{{panel_w}}}\n"
            "\\centering\n"
            f"{body_font_cmd}\n"
            f"\\scalebox{{{scale:.2f}}}{{%\n"
            + inner +
            "\n}\n"
            "\\end{minipage}"
        )

    lines: List[str] = []

    lines.append("% =============================================")
    lines.append("% Forced 2x2 one-column table of rule-based baseline results")
    lines.append("% =============================================")

    lines.append("\\begin{table}[!t]")
    lines.append("\\centering")

    caption = (
        "Performance of rule-based traffic signal control baselines across representative "
        "fixed-demand traffic scenarios. Each subtable compares the Fixed-time and "
        "Max-pressure controllers. KPIs are: Throughput (TP, veh/h; higher is better), "
        "Mean Travel Time (MTT, s; lower is better), and Average Waiting Time "
        "(AWT, s; lower is better)."
    )

    if bold_best:
        caption += " Best results within each grid size are shown in bold."

    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\label{tab:baseline_rules}")
    lines.append("\\vspace{-1.0ex}")

    # Outer tabular forces true 2x2 layout.
    # The @{} removes extra left/right padding.
    # The @{\hspace{...}} controls the gap between the two subtables.
    lines.append("\\begin{tabular}{@{}c@{\\hspace{0.015\\columnwidth}}c@{}}")

    lines.append(panel(types[0]) + " & " + panel(types[1]) + "\\\\[0.35ex]")
    lines.append(panel(types[2]) + " & " + panel(types[3]) + "\\\\")

    lines.append("\\end{tabular}")

    lines.append("\\end{table}")

    return "\n".join(lines)

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        "Build one compact one-column 2x2 LaTeX table for Fixed-time and Max-pressure baseline results."
    )

    ap.add_argument(
        "--csv-dir",
        type=str,
        required=True,
        help=(
            "Directory containing all_baseline_results_<type>.csv, "
            "or a parent directory containing eval_fixed_demand_type/."
        ),
    )

    ap.add_argument(
        "--types",
        type=str,
        nargs="*",
        default=["corridor", "cross", "platoons", "bursty"],
        help=(
            "Demand types to include. Four are ideal for the 2x2 layout. "
            "Example: --types corridor cross platoons bursty"
        ),
    )

    ap.add_argument(
        "--grids",
        type=int,
        nargs="*",
        default=GRID_ORDER_DEFAULT,
        help="Grid sizes to include. Default: 3 5 10",
    )

    ap.add_argument(
        "--digits-throughput",
        type=int,
        default=0,
        help="Decimal places for throughput. Default: 0",
    )

    ap.add_argument(
        "--digits-time",
        type=int,
        default=0,
        help="Decimal places for MTT/AWT. Default: 0",
    )

    ap.add_argument(
        "--scale",
        type=float,
        default=1.00,
        help=(
            "Scale factor for each compact subtable body. "
            "Use 1.00 for readable text. Decrease to 0.95 or 0.90 if too wide. "
            "Default: 1.00"
        ),
    )

    ap.add_argument(
        "--body-font-size",
        type=float,
        default=7.4,
        help=(
            "Base font size, in pt, for table values and metric labels. "
            "Increase if the table text is too small. Default: 7.4"
        ),
    )

    ap.add_argument(
        "--body-baseline-skip",
        type=float,
        default=7.8,
        help="Line spacing, in pt, for the table body. Default: 7.8",
    )

    ap.add_argument(
        "--controller-font-size",
        type=float,
        default=7.4,
        help=(
            "Font size, in pt, for the Fixed-time and Max-pressure headers. "
            "Default: 6.4"
        ),
    )

    ap.add_argument(
        "--controller-baseline-skip",
        type=float,
        default=6.8,
        help="Line spacing, in pt, for controller headers. Default: 6.8",
    )

    ap.add_argument(
        "--out",
        type=str,
        default="table_baselines.tex",
        help="Output LaTeX file. Default: table_baselines.tex",
    )

    ap.add_argument(
        "--bold-best",
        action="store_true",
        help="Bold the better baseline result for each grid and KPI.",
    )

    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print which CSV is used and parsing statistics.",
    )

    args = ap.parse_args()

    csv_root = Path(args.csv_dir).resolve()
    out_path = Path(args.out).resolve()

    types: List[str] = []
    for t in args.types:
        for piece in str(t).split(","):
            tt = piece.strip()
            if tt:
                types.append(tt)

    types = types[:4]

    per_type_baseline: Dict[
        str,
        Dict[Tuple[int, str], Dict[str, Optional[float]]],
    ] = {}

    for t in types:
        csv_path = discover_baseline_csv_for_type(
            csv_root=csv_root,
            demand_type=t,
            verbose=args.verbose,
        )

        if csv_path is None:
            per_type_baseline[t] = {}
            continue

        per_type_baseline[t] = read_baseline_csv(
            csv_path=csv_path,
            verbose=args.verbose,
        )

    tex = build_2x2_baseline_table(
        demand_types=types,
        per_type_baseline=per_type_baseline,
        grids=args.grids,
        digits_throughput=args.digits_throughput,
        digits_time=args.digits_time,
        scale=args.scale,
        bold_best=args.bold_best,
        body_font_size=args.body_font_size,
        body_baseline_skip=args.body_baseline_skip,
        controller_font_size=args.controller_font_size,
        controller_baseline_skip=args.controller_baseline_skip,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated LaTeX table for rule-based baseline results\n")
        f.write("% Requires: \\usepackage{booktabs}\n")
        f.write("%           \\usepackage{graphicx}\n\n")
        f.write(tex)
        f.write("\n")

    print("Wrote LaTeX table to:", out_path)


if __name__ == "__main__":
    main()
