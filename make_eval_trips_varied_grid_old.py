# make_eval_trips_varied_grid.py
# Generate 10 SUMO trips/flows XML files for an NxN grid with fringe stubs.
# Each case contains route-specific (non-uniform) vehsPerHour across flows.

from __future__ import annotations
import os
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

# ---------- XML helpers ----------
def new_root():
    root = ET.Element("routes")
    ET.SubElement(root, "vType",
        id="car", accel="2.0", decel="4.5", sigma="0.5",
        length="3.5", width="1.0", maxSpeed="16.7",
        vClass="passenger", guiShape="passenger")
    return root

def add_flow(root, fid, begin, end, vph, src, dst):
    vph = int(max(1, vph))
    ET.SubElement(root, "flow",
        id=fid, type="car",
        begin=f"{begin:.1f}", end=f"{end:.1f}",
        vehsPerHour=str(vph),
        **{"from": src, "to": dst}
    )

def write_case(path, flows):
    root = new_root()
    for f in flows:
        add_flow(root, **f)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

# ---------- drawing non-uniform VPH per route ----------
def draw_vphs(rng: np.random.Generator, n: int, low: int, high: int, lognormal_skew: float = 0.0) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    if lognormal_skew > 0:
        sample = rng.lognormal(mean=0.0, sigma=lognormal_skew, size=n)
        sample = (sample - sample.min()) / max(sample.ptp(), 1e-9)
    else:
        sample = rng.random(n)
    vals = low + sample * (high - low)
    return np.maximum(1, vals.astype(int))

# ---------- case builders ----------
def case_light_straight(N: int, T: float, rng) -> List[dict]:
    flows = []
    routes = [(f"w{j}_e{j}", W_in(j), E_out(j, N)) for j in range(N)]
    routes += [(f"s{i}_n{i}", S_in(i), N_out(i, N)) for i in range(N)]
    vphs = draw_vphs(rng, len(routes), 150, 500, lognormal_skew=0.8)
    for (rid, src, dst), v in zip(routes, vphs):
        flows.append(dict(fid=f"c01_{rid}", begin=0, end=T, vph=v, src=src, dst=dst))
    return flows

def case_light_shifted(N: int, T: float, rng) -> List[dict]:
    flows = []
    routes = [(f"w{j}_e{(j+1)%N}", W_in(j), E_out((j+1)%N, N)) for j in range(N)]
    routes += [(f"s{i}_n{(i+2)%N}", S_in(i), N_out((i+2)%N, N)) for i in range(N)]
    vphs = draw_vphs(rng, len(routes), 250, 700, lognormal_skew=0.6)
    for (rid, src, dst), v in zip(routes, vphs):
        flows.append(dict(fid=f"c02_{rid}", begin=0, end=T, vph=v, src=src, dst=dst))
    return flows

def case_medium_cross(N: int, T: float, rng) -> List[dict]:
    flows = []
    routes = [(f"w{j}_n{j}", W_in(j), N_out(j, N)) for j in range(N)]
    routes += [(f"s{i}_e{i}", S_in(i), E_out(i, N)) for i in range(N)]
    vphs = draw_vphs(rng, len(routes), 700, 1200, lognormal_skew=0.7)
    for (rid, src, dst), v in zip(routes, vphs):
        flows.append(dict(fid=f"c03_{rid}", begin=0, end=T, vph=v, src=src, dst=dst))
    return flows

def case_medium_mixed(N: int, T: float, rng) -> List[dict]:
    flows = []
    routes = [(f"e{j}_w{(j+2)%N}", E_in(j, N), W_out((j+2)%N)) for j in range(N)]
    routes += [(f"n{i}_s{(i+2)%N}", N_in(i, N), S_out((i+2)%N)) for i in range(N)]
    for j in range(0, N, max(1, N // 3)):
        routes.append((f"w{j}_e{(j+2)%N}", W_in(j), E_out((j+2) % N, N)))
    for i in range(0, N, max(1, N // 3)):
        routes.append((f"s{i}_n{(i+1)%N}", S_in(i), N_out((i+1) % N, N)))
    vphs = draw_vphs(rng, len(routes), 800, 1400, lognormal_skew=0.8)
    for (rid, src, dst), v in zip(routes, vphs):
        flows.append(dict(fid=f"c04_{rid}", begin=0, end=T, vph=v, src=src, dst=dst))
    return flows

def case_peak_am_pm(N: int, T: float, rng) -> List[dict]:
    flows = []
    am_end = 0.4 * T
    routes_am = [(f"s{i}_n{i}", S_in(i), N_out(i, N)) for i in range(N)]
    routes_pm = [(f"w{j}_e{j}", W_in(j), E_out(j, N)) for j in range(N)]
    vphs_am = draw_vphs(rng, len(routes_am), 1100, 1700, lognormal_skew=0.7)
    vphs_pm = draw_vphs(rng, len(routes_pm), 1100, 1700, lognormal_skew=0.7)
    for (rid, src, dst), v in zip(routes_am, vphs_am):
        flows.append(dict(fid=f"c05_am_{rid}", begin=0, end=am_end, vph=v, src=src, dst=dst))
    for (rid, src, dst), v in zip(routes_pm, vphs_pm):
        flows.append(dict(fid=f"c05_pm_{rid}", begin=am_end, end=T, vph=v, src=src, dst=dst))
    return flows

def case_heavy_corridor_ew(N: int, T: float, rng) -> List[dict]:
    flows = []
    routes = []
    for j in range(N):
        routes.append((f"w{j}_e{j}", W_in(j), E_out(j, N)))
        routes.append((f"e{j}_w{j}", E_in(j, N), W_out(j)))
    vphs = draw_vphs(rng, len(routes), 1400, 2200, lognormal_skew=0.9)
    for (rid, src, dst), v in zip(routes, vphs):
        flows.append(dict(fid=f"c06_{rid}", begin=0, end=T, vph=v, src=src, dst=dst))
    return flows

def case_heavy_shifted(N: int, T: float, rng) -> List[dict]:
    flows = []
    routes = []
    for j in range(N):
        routes.append((f"w{j}_e{(j+1)%N}", W_in(j), E_out((j+1) % N, N)))
        routes.append((f"e{j}_w{(j+2)%N}", E_in(j, N), W_out((j+2) % N)))
    for i in range(N):
        routes.append((f"s{i}_n{(i+1)%N}", S_in(i), N_out((i+1) % N, N)))
        routes.append((f"n{i}_s{(i+2)%N}", N_in(i, N), S_out((i+2) % N)))
    vphs = draw_vphs(rng, len(routes), 1200, 2200, lognormal_skew=1.0)
    for (rid, src, dst), v in zip(routes, vphs):
        flows.append(dict(fid=f"c07_{rid}", begin=0, end=T, vph=v, src=src, dst=dst))
    return flows

def case_asym_diagonal(N: int, T: float, rng) -> List[dict]:
    flows = []
    routes = []
    step = max(1, N // 2)
    for i in range(0, N, step):
        routes.append((f"n{i}_e{(i+step)%N}", N_in(i, N), E_out((i+step) % N, N)))
    for j in range(0, N, step):
        routes.append((f"w{j}_s{(j+1)%N}", W_in(j), S_out((j+1) % N)))
    if N >= 3:
        routes.append((f"w0_e{(2%N)}", W_in(0), E_out(2 % N, N)))
        routes.append((f"n{(N-1)}_s0", N_in(N-1, N), S_out(0)))
    vphs = draw_vphs(rng, len(routes), 500, 1600, lognormal_skew=0.9)
    for (rid, src, dst), v in zip(routes, vphs):
        flows.append(dict(fid=f"c08_{rid}", begin=0, end=T, vph=v, src=src, dst=dst))
    return flows

def case_sparse_bursty(N: int, T: float, rng) -> List[dict]:
    flows = []
    windows = [(0.00*T, 0.20*T), (0.20*T, 0.40*T), (0.40*T, 0.60*T), (0.60*T, 0.80*T)]
    candidates: List[Tuple[str, str, str]] = []
    for j in range(N):
        candidates.append((f"w{j}_n{j}", W_in(j), N_out(j, N)))
        candidates.append((f"w{j}_e{(j+1)%N}", W_in(j), E_out((j+1)%N, N)))
    for i in range(N):
        candidates.append((f"s{i}_e{i}", S_in(i), E_out(i, N)))
        candidates.append((f"n{i}_w{i}", N_in(i, N), W_out(i)))
    rng.shuffle(candidates)
    chosen = candidates[:4] if len(candidates) >= 4 else candidates
    vphs = draw_vphs(rng, len(chosen), 120, 450, lognormal_skew=0.7)
    for (rid, src, dst), (beg, end), v in zip(chosen, windows[:len(chosen)], vphs):
        flows.append(dict(fid=f"c09_{rid}", begin=beg, end=end, vph=v, src=src, dst=dst))
    return flows

def case_balanced_moderate(N: int, T: float, rng) -> List[dict]:
    flows = []
    routes = []
    for j in range(N):
        if j % 2 == 0:
            routes.append((f"w{j}_e{(j+1)%N}", W_in(j), E_out((j+1) % N, N)))
    for i in range(N):
        if i % 2 == 1:
            routes.append((f"s{i}_n{(i+N-1)%N}", S_in(i), N_out((i+N-1) % N, N)))
    if N % 2 == 1:
        c = N // 2
        routes.append((f"e{c}_w{c}", E_in(c, N), W_out(c)))
        routes.append((f"n{c}_s{c}", N_in(c, N), S_out(c)))
    vphs = draw_vphs(rng, len(routes), 400, 1000, lognormal_skew=0.8)
    for (rid, src, dst), v in zip(routes, vphs):
        flows.append(dict(fid=f"c10_{rid}", begin=0, end=T, vph=v, src=src, dst=dst))
    return flows

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="Output directory for trips files")
    ap.add_argument("--sim-end", type=float, default=1000.0, help="End time (s) for flows")
    ap.add_argument("--grid-n", type=int, default=3, help="Grid size N used by your env")
    ap.add_argument("--seed", type=int, default=2025, help="RNG seed")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    N = args.grid_n
    if N < 2:
        raise ValueError("grid-n must be >= 2")
    T = float(args.sim_end)
    rng = np.random.default_rng(args.seed)

    builders = [
        ("01_light_straight",  case_light_straight),
        ("02_light_shifted",   case_light_shifted),
        ("03_medium_cross",    case_medium_cross),
        ("04_medium_mixed",    case_medium_mixed),
        ("05_peak_am_pm",      case_peak_am_pm),
        ("06_heavy_corridor",  case_heavy_corridor_ew),
        ("07_heavy_shifted",   case_heavy_shifted),
        ("08_asym_diag",       case_asym_diagonal),
        ("09_sparse_bursty",   case_sparse_bursty),
        ("10_balanced_mod",    case_balanced_moderate),
    ]

    for name, fn in builders:
        flows = fn(N, T, rng)
        fname = f"eval_trips_case_{name}_gridN{N}.xml"
        path = os.path.join(args.out, fname)
        write_case(path, flows)
        print(f"Wrote {path} ({len(flows)} flows)")
        if not flows:
            print("  (no flows)")
        else:
            for f in flows:
                # FIXED: use keys actually present in dict (fid, src, dst, vph, begin, end)
                print(f"  - {f['fid']}: {f['vph']} vph  {f['src']} -> {f['dst']}  [{f['begin']:.0f}-{f['end']:.0f}]")

if __name__ == "__main__":
    main()
