#!/usr/bin/env python3
# Generate K SUMO trips/flows XML files for an NxN grid with fringe stubs.
# Each file: time horizon split into 10 sections; each section chooses R random
# fringe->fringe routes (R in [0.05*(4N)^2 , 0.30*(4N)^2]) with random vph in [100, 2500].

from __future__ import annotations
import os
import re
import argparse
import xml.etree.ElementTree as ET
import numpy as np
from typing import List, Tuple

# ---------- Edge helpers that MATCH your env's naming ----------
def W_in(j: int) -> str:   return f"e_fw{j}_to_n0_{j}"
def W_out(j: int) -> str:  return f"e_n0_{j}_to_fw{j}"

def E_in(j: int, N: int) -> str:   return f"e_fe{j}_to_n{N-1}_{j}"
def E_out(j: int, N: int) -> str:  return f"e_n{N-1}_{j}_to_fe{j}"

def S_in(i: int) -> str:   return f"e_fs{i}_to_n{i}_0"
def S_out(i: int) -> str:  return f"e_n{i}_0_to_fs{i}"

def N_in(i: int, N: int) -> str:   return f"e_fn{i}_to_n{i}_{N-1}"
def N_out(i: int, N: int) -> str:  return f"e_n{i}_{N-1}_to_fn{i}"

# ---------- Robust node mappers (fixes your error) ----------
def in_to_node(edge_id: str, N: int) -> str:
    """
    Map an 'incoming' fringe->grid edge id to its grid node id n_i_j.
    Patterns:
      e_fw{j}_to_n0_{j}           -> n_0_{j}
      e_fe{j}_to_n{N-1}_{j}       -> n_{N-1}_{j}
      e_fs{i}_to_n{i}_0           -> n_{i}_0
      e_fn{i}_to_n{i}_{N-1}       -> n_{i}_{N-1}
    """
    m = re.match(r"^e_fw(\d+)_to_n0_(\d+)$", edge_id)
    if m: return f"n_0_{int(m.group(2))}"

    m = re.match(rf"^e_fe(\d+)_to_n{N-1}_(\d+)$", edge_id)
    if m: return f"n_{N-1}_{int(m.group(2))}"

    m = re.match(r"^e_fs(\d+)_to_n(\d+)_0$", edge_id)
    if m: return f"n_{int(m.group(2))}_0"

    m = re.match(rf"^e_fn(\d+)_to_n(\d+)_{N-1}$", edge_id)
    if m: return f"n_{int(m.group(2))}_{N-1}"

    raise ValueError(f"in_to_node: unrecognized incoming edge id: {edge_id}")

def out_from_node(edge_id: str, N: int) -> str:
    """
    Map an 'outgoing' grid->fringe edge id to its source grid node id n_i_j.
    Patterns:
      e_n{i}_{j}_to_fw{j}
      e_n{N-1}_{j}_to_fe{j}
      e_n{i}_0_to_fs{i}
      e_n{i}_{N-1}_to_fn{i}
    We only need the 'e_n{i}_{j}_to_' prefix to recover n_i_j.
    """
    m = re.match(r"^e_n(\d+)_(\d+)_to_", edge_id)
    if m:
        i = int(m.group(1)); j = int(m.group(2))
        return f"n_{i}_{j}"
    raise ValueError(f"out_from_node: unrecognized outgoing edge id: {edge_id}")

# ---------- XML helpers ----------
def new_root():
    root = ET.Element("routes")
    ET.SubElement(root, "vType",
        id="car", accel="2.0", decel="4.5", sigma="0.5",
        length="3.5", width="1.0", maxSpeed="16.7",
        vClass="passenger", guiShape="passenger")
    return root

def add_flow(root, fid, begin, end, vph, src, dst):
    vph_i = int(max(1, round(vph)))
    ET.SubElement(root, "flow",
        id=fid, type="car",
        begin=f"{begin:.1f}", end=f"{end:.1f}",
        vehsPerHour=str(vph_i),
        **{"from": src, "to": dst}
    )

def write_case(path: str, flows: List[dict]):
    root = new_root()
    for f in flows:
        add_flow(root, **f)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

# ---------- Build full fringe->fringe route pool ----------
def all_fringe_routes(N: int) -> List[Tuple[str, str]]:
    """
    Return list of (src_in_edge, dst_out_edge) pairs for all fringe->fringe ODs,
    excluding trivial 'same-node' pairs (in_node == out_node).
    """
    in_edges = (
        [W_in(j) for j in range(N)] +
        [E_in(j, N) for j in range(N)] +
        [S_in(i) for i in range(N)] +
        [N_in(i, N) for i in range(N)]
    )
    out_edges = (
        [W_out(j) for j in range(N)] +
        [E_out(j, N) for j in range(N)] +
        [S_out(i) for i in range(N)] +
        [N_out(i, N) for i in range(N)]
    )

    pairs: List[Tuple[str, str]] = []
    for src in in_edges:
        src_node = in_to_node(src, N)
        for dst in out_edges:
            dst_node = out_from_node(dst, N)
            if src_node == dst_node:
                continue  # avoid immediate in->out at the same junction
            pairs.append((src, dst))
    return pairs

# ---------- Random per-section flow drawing ----------
def draw_random_case(
    rng: np.random.Generator,
    N: int,
    T: float,
    sections: int,
    routes_all: List[Tuple[str, str]],
    r_min_frac: float,
    r_max_frac: float,
    vph_low: int,
    vph_high: int,
) -> List[dict]:
    """
    Build a single case (list of flows) with time-varying random routes.
    Each of `sections` time windows picks R routes uniformly without replacement,
    where R is drawn uniformly from [r_min_frac*M, r_max_frac*M], M=(4N)^2.
    """
    flows: List[dict] = []
    M = (4 * N) ** 2  # matches your spec for the R-range baseline
    r_min = max(1, int(np.floor(r_min_frac * M)))
    r_max = max(r_min, int(np.floor(r_max_frac * M)))

    # But we also cannot exceed the actual number of available routes
    M_actual = len(routes_all)

    section_len = T / float(sections)
    for s in range(sections):
        begin = s * section_len
        end = (s + 1) * section_len

        R = int(rng.integers(r_min, r_max + 1))
        R = min(R, M_actual)  # cap by available OD pairs

        # Choose R distinct routes
        idxs = rng.choice(M_actual, size=R, replace=False)
        vphs = rng.integers(vph_low, vph_high + 1, size=R)

        for k, (idx, v) in enumerate(zip(idxs, vphs)):
            src, dst = routes_all[idx]
            flows.append(dict(
                fid=f"sec{s:02d}_r{idx:04d}",
                begin=begin,
                end=end,
                vph=int(v),
                src=src,
                dst=dst,
            ))
    return flows

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="Output directory for trips files")
    ap.add_argument("--sim-end", type=float, default=500.0, help="End time (s) for flows")
    ap.add_argument("--grid-n", type=int, default=3, help="Grid size N used by your env")
    ap.add_argument("--seed", type=int, default=2025, help="RNG seed")
    ap.add_argument("--count", type=int, default=10, help="How many trip files to generate")
    ap.add_argument("--sections", type=int, default=10, help="Number of time sections per file")
    # R ~ UniformInt( r_min_frac*(4N)^2 , r_max_frac*(4N)^2 )
    ap.add_argument("--r-min-frac", type=float, default=0.05, help="Min fraction for R range baseline (of (4N)^2)")
    ap.add_argument("--r-max-frac", type=float, default=0.20, help="Max fraction for R range baseline (of (4N)^2)")
    ap.add_argument("--vph-low", type=int, default=100, help="Minimum vehsPerHour for any chosen route")
    ap.add_argument("--vph-high", type=int, default=2500, help="Maximum vehsPerHour for any chosen route")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    N = args.grid_n
    if N < 2:
        raise ValueError("grid-n must be >= 2")
    T = float(args.sim_end)
    rng = np.random.default_rng(args.seed)

    # Build the full OD pool (fringe->fringe)
    routes_all = all_fringe_routes(N)
    print(f"[info] grid N={N}: total fringe->fringe OD routes available = {len(routes_all)}")

    for k in range(args.count):
        flows = draw_random_case(
            rng=rng,
            N=N,
            T=T,
            sections=args.sections,
            routes_all=routes_all,
            r_min_frac=args.r_min_frac,
            r_max_frac=args.r_max_frac,
            vph_low=args.vph_low,
            vph_high=args.vph_high,
        )
        fname = f"eval_trips_random_case_{k+1:02d}_gridN{N}.xml"
        path = os.path.join(args.out, fname)
        write_case(path, flows)

        print(f"Wrote {path} ({len(flows)} flows)")
        # brief preview (first few flows)
        for f in flows[:min(8, len(flows))]:
            print(f"  - {f['fid']}: {f['vph']} vph  {f['src']} -> {f['dst']}  [{f['begin']:.0f}-{f['end']:.0f}]")

if __name__ == "__main__":
    main()
