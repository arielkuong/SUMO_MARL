#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import csv
from typing import Dict, List, Tuple, Optional

# ---------------- config: method name mapping ----------------
STATELESS_ORDER: List[Tuple[str, str]] = [
    ("dqn_mlp",        "IDQN(MLP)"),
    ("dqn_gnn",        "DQN(GNN)"),
    ("ctde_vdn_mlp",   "CTDE(MLP)"),
    ("ctde_vdn_gnn",   "CTDE(GNN)"),
]
STATEFUL_ORDER: List[Tuple[str, str]] = [
    ("drqn_lstm",            "IDRQN(LSTM)"),
    ("drqn_gnn_lstm",        "DRQN(GNN+LSTM)"),
    ("ctde_vdn_lstm",        "CTDE(LSTM)"),
    ("ctde_vdn_gnn_lstm",    "CTDE(GNN+LSTM)"),
]
ALL_METHOD_KEYS = {k for k, _ in STATELESS_ORDER} | {k for k, _ in STATEFUL_ORDER}
GRID_ORDER_DEFAULT = [3, 5, 10]

# ---------------- demand-type display titles ----------------
DISPLAY_TITLES = {
    "corridor":  "Heavy corridor (E--W arterial)",
    "cross":     "Cross (orthogonal OD axes)",
    "platoons":  "Platoons (pulse trains)",
    "bursty":    "Bursty (on--off)",
    "shifted":   "Shifted",
    "incident":  "Incident",
}

# -------------- helpers --------------
def fmt_num(v: Optional[float], digits: int) -> str:
    if v is None:
        return "\\textemdash{}"
    fmt = "{0:." + str(digits) + "f}"
    return fmt.format(v)

def fmt_mean_std(mean_v: Optional[float], std_v: Optional[float], digits: int) -> str:
    m = fmt_num(mean_v, digits) if mean_v is not None else None
    s = fmt_num(std_v,  digits) if std_v  is not None else None
    if m is None and s is None:
        return "\\textemdash{}"
    if (mean_v is not None) and (std_v is not None):
        return f"{m}\\,\\(\\pm\\)\\,{s}"
    return m if mean_v is not None else s

def _to_float_or_none(s: object) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None

# Flexible column aliases (accept several variants)
COLMAP = {
    "throughput_mean": ("throughput_mean", "throughput_vph_mean", "tp_mean", "throughput_vph"),
    "throughput_std" : ("throughput_std",  "throughput_vph_std",  "tp_std"),
    "mean_travel_mean": ("mean_travel_mean", "mean_travel_s_mean", "mtt_mean", "mean_travel_s"),
    "mean_travel_std" : ("mean_travel_std",  "mean_travel_s_std",  "mtt_std"),
    "mean_wait_mean"  : ("mean_wait_mean",   "mean_wait_s_mean",   "awt_mean", "mean_wait_s"),
    "mean_wait_std"   : ("mean_wait_std",    "mean_wait_s_std",    "awt_std"),
}
GRID_ALIASES = ("grid_n", "grid", "grid_size")

def _get_first_present(row: dict, keys: Tuple[str, ...]) -> Optional[float]:
    for k in keys:
        if k in row:
            val = _to_float_or_none(row.get(k))
            if val is not None:
                return val
    return None

def _parse_grid(row: dict) -> Optional[int]:
    for k in GRID_ALIASES:
        if k in row:
            try:
                return int(str(row[k]).strip())
            except Exception:
                pass
    return None

def read_csv_meanstd(csv_path: Path, verbose: bool=False) -> Dict[Tuple[int, str], Dict[str, float]]:
    """
    Read ONE CSV that already stores mean/std columns (or single-run means).
    Returns (grid_n, method) -> metric dict. Ignores any seed column.
    """
    out: Dict[Tuple[int, str], Dict[str, float]] = {}
    if not csv_path.exists():
        if verbose:
            print(f"[WARN] CSV not found: {csv_path}")
        return out

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        n_rows, n_used = 0, 0
        for row in reader:
            n_rows += 1
            method = str(row.get("method", "")).strip().lower()
            if method not in ALL_METHOD_KEYS:
                continue
            grid_n = _parse_grid(row)
            if grid_n is None:
                continue

            t_mean = _get_first_present(row, COLMAP["throughput_mean"])
            t_std  = _get_first_present(row, COLMAP["throughput_std"])
            tt_mean= _get_first_present(row, COLMAP["mean_travel_mean"])
            tt_std = _get_first_present(row, COLMAP["mean_travel_std"])
            w_mean = _get_first_present(row, COLMAP["mean_wait_mean"])
            w_std  = _get_first_present(row, COLMAP["mean_wait_std"])

            out[(grid_n, method)] = {
                "throughput_mean": t_mean,
                "throughput_std":  t_std,
                "mean_travel_mean": tt_mean,
                "mean_travel_std":  tt_std,
                "mean_wait_mean":   w_mean,
                "mean_wait_std":    w_std,
            }
            n_used += 1

    if verbose:
        print(f"[INFO] Loaded {n_used}/{n_rows} usable rows from {csv_path.name}")
    return out

def discover_csv_for_type(csv_root: Path, demand_type: str, seed: int, verbose: bool=False) -> Optional[Path]:
    """
    Prefer eval_fixed_demand_type/; accept both with and without _seed<seed>.
    """
    candidates = [
        csv_root / "eval_fixed_demand_type" / f"eval_results_fixed_trip_{demand_type}_seed{seed}.csv",
        csv_root / f"eval_results_fixed_trip_{demand_type}_seed{seed}.csv",
        csv_root / "eval_fixed_demand_type" / f"eval_results_fixed_trip_{demand_type}.csv",
        csv_root / f"eval_results_fixed_trip_{demand_type}.csv",
    ]
    for p in candidates:
        if p.exists():
            if verbose:
                print(f"[INFO] Using CSV for '{demand_type}': {p}")
            return p

    for pattern in [
        f"**/eval_results_fixed_trip_{demand_type}_seed{seed}.csv",
        f"**/eval_results_fixed_trip_{demand_type}.csv",
    ]:
        matches = list(csv_root.glob(pattern))
        if matches:
            p = sorted(matches)[0]
            if verbose:
                print(f"[INFO] Using CSV (glob) for '{demand_type}': {p}")
            return p

    if verbose:
        print(f"[WARN] No CSV found for type='{demand_type}' under {csv_root}")
    return None

# ---------- width estimation for KPI columns ----------
def printable_len_estimate(s: str) -> int:
    """
    Rough glyph count closer to the printed width:
    - Remove thin spaces '\\,', braces, slashes, backslashes.
    - Replace '\\(\\pm\\)' with a single '±'.
    """
    if not s:
        return 0
    t = s
    t = t.replace("\\,", "")
    t = t.replace("\\(\\pm\\)", "±")
    for ch in ["{", "}", "\\", "$"]:
        t = t.replace(ch, "")
    return len(t)

def compute_kpi_widths_ex(
    per_type_agg: Dict[str, Dict[Tuple[int, str], Dict[str, float]]],
    demand_types: List[str],
    grids: List[int],
    digits_throughput: int,
    digits_time: int,
) -> Dict[str, int]:
    """
    Compute fixed widths (in ex units) for TP / MTT / AWT based on the longest
    formatted mean±std string across all selected types + grids.

    Widen TP a bit; make MTT slightly narrower; AWT even narrower.
    """
    max_len = {"tp": 0, "mtt": 0, "awt": 0}

    def update_len(s: str, key: str):
        max_len[key] = max(max_len[key], printable_len_estimate(s))

    # include headers so they also fit
    update_len("TP $\\uparrow$", "tp")
    update_len("MTT $\\downarrow$", "mtt")
    update_len("AWT $\\downarrow$", "awt")

    for t in demand_types:
        agg = per_type_agg.get(t, {})
        for g in grids:
            for k, _label in STATELESS_ORDER + STATEFUL_ORDER:
                d = agg.get((g, k), {})
                s_tp  = fmt_mean_std(d.get("throughput_mean"), d.get("throughput_std"), digits_throughput)
                s_mtt = fmt_mean_std(d.get("mean_travel_mean"), d.get("mean_travel_std"), digits_time)
                s_awt = fmt_mean_std(d.get("mean_wait_mean"),   d.get("mean_wait_std"),   digits_time)
                update_len(s_tp,  "tp")
                update_len(s_mtt, "mtt")
                update_len(s_awt, "awt")

    # Heuristics: TP wider; MTT narrower; AWT narrowest
    tp_scale, tp_pad, tp_min = 0.75, 2, 9
    mtt_scale, mtt_pad, mtt_min = 0.55, 1, 6
    awt_scale, awt_pad, awt_min = 0.50, 1, 5

    widths_ex = {
        "tp":  max(tp_min,  int(round(max_len["tp"]  * tp_scale))  + tp_pad),
        "mtt": max(mtt_min, int(round(max_len["mtt"] * mtt_scale)) + mtt_pad),
        "awt": max(awt_min, int(round(max_len["awt"] * awt_scale)) + awt_pad),
    }
    return widths_ex

def fixbox(s: str, ex: int, align: str = "r") -> str:
    """Wrap cell in a fixed-width box (in ex units)."""
    return f"\\makebox[{ex}ex][{align}]{{{s}}}"

# ---------------- inner tables (with / without grid) ----------------
def inner_tabular_with_grid(
    demand_type: str,
    agg: Dict[Tuple[int, str], Dict[str, float]],
    grids: List[int],
    digits_throughput: int,
    digits_time: int,
    widths_ex: Dict[str, int],
) -> str:
    title = DISPLAY_TITLES.get(demand_type, demand_type).replace("_", "\\_")
    # Slightly narrower inter-block gap (2pt) and smaller colsep
    colspec = "l lccc@{\\hspace{2pt}}lccc"
    lines: List[str] = []
    lines.append("{% local spacing (with grid)")
    lines.append("\\setlength{\\tabcolsep}{2.0pt}%")
    lines.append("\\begin{tabular}{" + colspec + "}")
    # Bigger title and a bit more vertical space
    lines.append(f"\\multicolumn{{9}}{{c}}{{\\textbf{{\\small {title}}}}}\\\\[0.6ex]")
    lines.append("\\toprule")
    lines.append(
        "\\multicolumn{1}{c}{\\textbf{Grid}} & "
        "\\multicolumn{4}{c}{\\textbf{Stateless (no LSTM)}}"
        " & "
        "\\multicolumn{4}{c}{\\textbf{LSTM-augmented}}\\\\"
    )
    lines.append("\\cmidrule(lr){2-5} \\cmidrule(lr){6-9}")
    lines.append(
        "\\textbf{} & \\textbf{Method} & "
        f"{fixbox('TP $\\uparrow$', widths_ex['tp'], 'c')} & "
        f"{fixbox('MTT $\\downarrow$', widths_ex['mtt'], 'c')} & "
        f"{fixbox('AWT $\\downarrow$', widths_ex['awt'], 'c')}"
        " & \\textbf{Method} & "
        f"{fixbox('TP $\\uparrow$', widths_ex['tp'], 'c')} & "
        f"{fixbox('MTT $\\downarrow$', widths_ex['mtt'], 'c')} & "
        f"{fixbox('AWT $\\downarrow$', widths_ex['awt'], 'c')}\\\\"
    )
    lines.append("\\midrule")

    rows_per_grid = max(len(STATELESS_ORDER), len(STATEFUL_ORDER))
    # Narrower grid column
    grid_box_w_ex = 4

    for g in grids:
        for i in range(rows_per_grid):
            # left block
            if i < len(STATELESS_ORDER):
                kL, labelL = STATELESS_ORDER[i]
                tL = agg.get((g, kL), {})
                thL = fixbox(fmt_mean_std(tL.get("throughput_mean"), tL.get("throughput_std"), digits_throughput), widths_ex["tp"])
                ttL = fixbox(fmt_mean_std(tL.get("mean_travel_mean"), tL.get("mean_travel_std"), digits_time), widths_ex["mtt"])
                wtL = fixbox(fmt_mean_std(tL.get("mean_wait_mean"),   tL.get("mean_wait_std"),   digits_time), widths_ex["awt"])
            else:
                labelL, thL, ttL, wtL = "", fixbox("\\textemdash{}", widths_ex["tp"]), fixbox("\\textemdash{}", widths_ex["mtt"]), fixbox("\\textemdash{}", widths_ex["awt"])
            # right block
            if i < len(STATEFUL_ORDER):
                kR, labelR = STATEFUL_ORDER[i]
                tR = agg.get((g, kR), {})
                thR = fixbox(fmt_mean_std(tR.get("throughput_mean"), tR.get("throughput_std"), digits_throughput), widths_ex["tp"])
                ttR = fixbox(fmt_mean_std(tR.get("mean_travel_mean"), tR.get("mean_travel_std"), digits_time), widths_ex["mtt"])
                wtR = fixbox(fmt_mean_std(tR.get("mean_wait_mean"),   tR.get("mean_wait_std"),   digits_time), widths_ex["awt"])
            else:
                labelR, thR, ttR, wtR = "", fixbox("\\textemdash{}", widths_ex["tp"]), fixbox("\\textemdash{}", widths_ex["mtt"]), fixbox("\\textemdash{}", widths_ex["awt"])

            if i == 0:
                grid_text = f"\\textbf{{{g}x{g}}}"
                grid_cell = f"\\multirow{{{rows_per_grid}}}{{*}}{{\\makebox[{grid_box_w_ex}ex][c]{{{grid_text}}}}}"
            else:
                grid_cell = ""

            lines.append(
                f"{grid_cell} & {labelL} & {thL} & {ttL} & {wtL}  &  "
                f"{labelR} & {thR} & {ttR} & {wtR}\\\\"
            )
        lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    return "\n".join(lines)

def inner_tabular_no_grid(
    demand_type: str,
    agg: Dict[Tuple[int, str], Dict[str, float]],
    grids: List[int],
    digits_throughput: int,
    digits_time: int,
    widths_ex: Dict[str, int],
) -> str:
    title = DISPLAY_TITLES.get(demand_type, demand_type).replace("_", "\\_")
    colspec = "lccc@{\\hspace{2pt}}lccc"
    lines: List[str] = []
    lines.append("{% local spacing (no grid)")
    lines.append("\\setlength{\\tabcolsep}{2.0pt}%")
    lines.append("\\begin{tabular}{" + colspec + "}")
    lines.append(f"\\multicolumn{{8}}{{c}}{{\\textbf{{\\small {title}}}}}\\\\[0.6ex]")
    lines.append("\\toprule")
    lines.append(
        "\\multicolumn{4}{c}{\\textbf{Stateless (no LSTM)}}"
        " & "
        "\\multicolumn{4}{c}{\\textbf{LSTM-augmented}}\\\\"
    )
    lines.append("\\cmidrule(lr){1-4} \\cmidrule(lr){5-8}")
    lines.append(
        "\\textbf{Method} & "
        f"{fixbox('TP $\\uparrow$', widths_ex['tp'], 'c')} & "
        f"{fixbox('MTT $\\downarrow$', widths_ex['mtt'], 'c')} & "
        f"{fixbox('AWT $\\downarrow$', widths_ex['awt'], 'c')}"
        " & \\textbf{Method} & "
        f"{fixbox('TP $\\uparrow$', widths_ex['tp'], 'c')} & "
        f"{fixbox('MTT $\\downarrow$', widths_ex['mtt'], 'c')} & "
        f"{fixbox('AWT $\\downarrow$', widths_ex['awt'], 'c')}\\\\"
    )
    lines.append("\\midrule")

    rows_per_grid = max(len(STATELESS_ORDER), len(STATEFUL_ORDER))
    for g in grids:
        for i in range(rows_per_grid):
            if i < len(STATELESS_ORDER):
                kL, labelL = STATELESS_ORDER[i]
                tL = agg.get((g, kL), {})
                thL = fixbox(fmt_mean_std(tL.get("throughput_mean"), tL.get("throughput_std"), digits_throughput), widths_ex["tp"])
                ttL = fixbox(fmt_mean_std(tL.get("mean_travel_mean"), tL.get("mean_travel_std"), digits_time), widths_ex["mtt"])
                wtL = fixbox(fmt_mean_std(tL.get("mean_wait_mean"),   tL.get("mean_wait_std"),   digits_time), widths_ex["awt"])
            else:
                labelL, thL, ttL, wtL = "", fixbox("\\textemdash{}", widths_ex["tp"]), fixbox("\\textemdash{}", widths_ex["mtt"]), fixbox("\\textemdash{}", widths_ex["awt"])
            if i < len(STATEFUL_ORDER):
                kR, labelR = STATEFUL_ORDER[i]
                tR = agg.get((g, kR), {})
                thR = fixbox(fmt_mean_std(tR.get("throughput_mean"), tR.get("throughput_std"), digits_throughput), widths_ex["tp"])
                ttR = fixbox(fmt_mean_std(tR.get("mean_travel_mean"), tR.get("mean_travel_std"), digits_time), widths_ex["mtt"])
                wtR = fixbox(fmt_mean_std(tR.get("mean_wait_mean"),   tR.get("mean_wait_std"),   digits_time), widths_ex["awt"])
            else:
                labelR, thR, ttR, wtR = "", fixbox("\\textemdash{}", widths_ex["tp"]), fixbox("\\textemdash{}", widths_ex["mtt"]), fixbox("\\textemdash{}", widths_ex["awt"])
            lines.append(
                f"{labelL} & {thL} & {ttL} & {wtL}  &  "
                f"{labelR} & {thR} & {ttR} & {wtR}\\\\"
            )
        lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    return "\n".join(lines)

# ---------------- big table builder ----------------
def build_2x2_single_table(
    demand_types: List[str],
    per_type_agg: Dict[str, Dict[Tuple[int, str], Dict[str, float]]],
    grids: List[int],
    digits_throughput: int,
    digits_time: int,
    scale: float = 0.93,
) -> str:
    types = demand_types[:4]
    if len(types) < 4:
        types = types + [""] * (4 - len(types))

    # Compute fixed KPI widths (ex units) over all selected types/grids
    widths_ex = compute_kpi_widths_ex(
        per_type_agg=per_type_agg,
        demand_types=[t for t in types if t],
        grids=grids,
        digits_throughput=digits_throughput,
        digits_time=digits_time,
    )

    def panel(idx: int) -> str:
        t = types[idx]
        if not t:
            return (
                "\\begin{minipage}[t]{0.49\\linewidth}\n"
                "\\centering\\scriptsize\n"
                "{\\bfseries (empty)}\\\\[0.5ex]\n"
                "\\end{minipage}"
            )
        agg = per_type_agg.get(t, {})
        inner = (inner_tabular_with_grid if idx in (0, 2) else inner_tabular_no_grid)(
            demand_type=t,
            agg=agg,
            grids=grids,
            digits_throughput=digits_throughput,
            digits_time=digits_time,
            widths_ex=widths_ex,
        )
        return (
            "\\begin{minipage}[t]{0.49\\linewidth}\n"
            "\\centering\\scriptsize\n" +
            inner + "\n"
            "\\end{minipage}"
        )

    row1 = panel(0) + "\n\\hfill\n" + panel(1)
    row2 = panel(2) + "\n\\hfill\n" + panel(3)

    lines: List[str] = []
    lines.append("% =============================================")
    lines.append("% One table: 2×2 grid of off-policy results (mean ± std) with fixed KPI widths")
    lines.append("% =============================================")
    lines.append("\\begin{table*}[p]")
    lines.append("\\centering")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.03}")
    lines.append("\\scriptsize")
    # Caption ABOVE the content, then a small negative vspace to tighten gap
    lines.append("\\caption{Performance of off-policy (DQN-based) MARL policies across representative traffic demand scenarios. "
                 "Each mini-table shows \\emph{Stateless} (no LSTM) and \\emph{LSTM-augmented} methods side-by-side for comparison. "
                 "KPI metrics are reported as mean~$\\pm$~std: Throughput (TP, veh/h; higher is better), "
                 "Mean Travel Time (MTT, s; lower is better), and Average Waiting Time (AWT, s; lower is better).}")
    lines.append("\\label{tab:offpolicy_all_2x2}")
    lines.append("\\vspace{-1.0ex}")
    lines.append(f"\\scalebox{{{scale:.2f}}}{{%")
    lines.append("\\begin{minipage}{\\linewidth}")
    lines.append(row1)
    lines.append("\\vspace{0.8ex}")
    lines.append(row2)
    lines.append("\\end{minipage}")
    lines.append("}")
    lines.append("\\end{table*}")
    return "\n".join(lines)

# -------------- main --------------
def main():
    ap = argparse.ArgumentParser(
        "Build ONE LaTeX table (2×2 grid) for off-policy evaluations from CSVs (mean ± std inside each CSV)."
    )
    ap.add_argument("--csv-dir", type=str, required=True,
                    help="Directory that contains eval_fixed_demand_type/ or the CSVs themselves.")
    ap.add_argument("--seed", type=int, required=True,
                    help="Seed used to pick CSV filenames (we ignore any seed column inside CSVs).")
    ap.add_argument("--types", type=str, nargs="*",
                    default=["corridor", "cross", "platoons", "bursty"],
                    help=("Exactly four demand types are ideal for a 2×2 layout "
                          "[corridor, cross, platoons, bursty, shifted, incident]."))
    ap.add_argument("--grids", type=int, nargs="*", default=GRID_ORDER_DEFAULT,
                    help="Grid sizes to include (default: 3 5 10)")
    ap.add_argument("--digits-throughput", type=int, default=0,
                    help="Decimal places for throughput (default: 0)")
    ap.add_argument("--digits-time", type=int, default=0,
                    help="Decimal places for times (default: 0)")
    ap.add_argument("--verbose", action="store_true", help="Print which CSV is used and parsing stats")
    args = ap.parse_args()

    csv_root = Path(args.csv_dir).resolve()
    out_path = Path("table_offpolicy.tex").resolve()

    # Normalize types list (up to 4)
    types: List[str] = []
    for t in args.types:
        for piece in str(t).split(","):
            tt = piece.strip()
            if tt:
                types.append(tt)
    types = types[:4]

    # Load per-type data (no averaging; use CSV means/stds)
    per_type_agg: Dict[str, Dict[Tuple[int, str], Dict[str, float]]] = {}
    for t in types:
        csv_path = discover_csv_for_type(csv_root, t, args.seed, verbose=args.verbose)
        if csv_path is None:
            per_type_agg[t] = {}
            continue
        per_type_agg[t] = read_csv_meanstd(csv_path, args.verbose)

    tex = build_2x2_single_table(
        demand_types=types,
        per_type_agg=per_type_agg,
        grids=args.grids,
        digits_throughput=args.digits_throughput,
        digits_time=args.digits_time,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated LaTeX table (off-policy MARL results, mean ± std)\n")
        f.write("% Requires: \\usepackage{booktabs}\n")
        f.write("%           \\usepackage{multirow}\n\n")
        f.write(tex)
    print("Wrote LaTeX table to:", out_path)

if __name__ == "__main__":
    main()
