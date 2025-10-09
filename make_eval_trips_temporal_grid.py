# make_eval_trips_temporal_grid.py
# Generate SUMO trips/flows XML files (temporal stress tests) for an NxN grid with fringe stubs.
# Cases:
#   t01_platoons_pulse_trains
#   t02_bursty_on_off
#   t03_shifted_peaks_dir_staggered
#   t04_ampm_global
#   t05_incident_west_boundary
#
# Notes:
# - Uses vehsPerHour on <flow> (duarouter-friendly).
# - All edge IDs match your environment’s naming.
# - “Sections” = 10 equal time windows across [0, T] for shifted/AMPM/incident.

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
    begin = float(begin); end = float(end)
    if end <= begin:  # skip degenerate windows
        return
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

# ---------- utils ----------
def draw_ints(rng: np.random.Generator, n: int, low: int, high: int) -> np.ndarray:
    """Uniform integers in [low, high]."""
    if n <= 0: return np.array([], dtype=int)
    return rng.integers(low, high + 1, size=n, dtype=int)

def ten_sections(T: float) -> List[Tuple[float, float]]:
    """Split [0,T] into 10 equal sections."""
    edges = np.linspace(0.0, T, 11)
    return [(float(edges[i]), float(edges[i+1])) for i in range(10)]

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

# ======================================================================
# Case builders
# ======================================================================

def case_platoons_pulse_trains(N: int, T: float, rng,
                               delta_min: float = 25.0, delta_max: float = 40.0,
                               width_min: float = 5.0, width_max: float = 8.0,
                               vph_min: int = 900, vph_max: int = 1200) -> List[dict]:
    """
    Pulsed platoons along a single E-W corridor (both directions).
    Mean flow ~ (vph * width / delta) -> around flat-moderate by chosen ranges.
    """
    flows: List[dict] = []
    r = N // 2  # middle row corridor
    routes = []
    # EB/WB along row r
    routes += [(f"w{r}_e{r}", W_in(r), E_out(r, N))]
    routes += [(f"e{r}_w{r}", E_in(r, N), W_out(r))]
    # Optional: add a light NB/SB corridor for distraction (commented out)
    # c = N // 2
    # routes += [(f"s{c}_n{c}", S_in(c), N_out(c, N))]
    # routes += [(f"n{c}_s{c}", N_in(c, N), S_out(c))]

    for rid, src, dst in routes:
        delta = rng.uniform(delta_min, delta_max)
        width = rng.uniform(width_min, width_max)
        # randomize initial phase so pulses from opposite fringes don't align trivially
        t = rng.uniform(0.0, delta)
        k = 0
        while t < T:
            vph = int(rng.integers(vph_min, vph_max + 1))
            b = t; e = min(T, t + width)
            flows.append(dict(fid=f"t01_{rid}_pulse{k}", begin=b, end=e, vph=vph, src=src, dst=dst))
            t += delta; k += 1
    return flows

def case_bursty_on_off(N: int, T: float, rng,
                       window_min: float = 10.0, window_max: float = 20.0,
                       vph_on_min: int = 600, vph_on_max: int = 900,
                       vph_off_min: int = 0, vph_off_max: int = 60,
                       route_multiplier: float = 1.0) -> List[dict]:
    """
    ON/OFF windows per route with short micro-windows. We choose ~2N routes to cap flow count.
    """
    flows: List[dict] = []
    candidates: List[Tuple[str, str, str]] = []
    for j in range(N):
        candidates.append((f"w{j}_e{j}", W_in(j), E_out(j, N)))
        candidates.append((f"e{j}_w{j}", E_in(j, N), W_out(j)))
    for i in range(N):
        candidates.append((f"s{i}_n{i}", S_in(i), N_out(i, N)))
        candidates.append((f"n{i}_s{i}", N_in(i, N), S_out(i)))

    rng.shuffle(candidates)
    K = max(1, int(route_multiplier * 2 * N))  # default ~2N routes
    routes = candidates[:K]

    # Per-route window length; randomized to de-synchronize
    for rid, src, dst in routes:
        win = rng.uniform(window_min, window_max)
        # Random phase
        t = rng.uniform(0.0, win)
        k = 0
        while t < T:
            on = rng.random() < 0.5
            if on:
                vph = int(rng.integers(vph_on_min, vph_on_max + 1))
            else:
                vph = int(rng.integers(vph_off_min, vph_off_max + 1))
            b = t; e = min(T, t + win)
            flows.append(dict(fid=f"t02_{rid}_w{k}", begin=b, end=e, vph=vph, src=src, dst=dst))
            t += win; k += 1
    return flows

def case_shifted_peaks_dir_staggered(N: int, T: float, rng,
                                     vph_lo_min: int = 100, vph_lo_max: int = 150,
                                     vph_hi_min: int = 350, vph_hi_max: int = 600) -> List[dict]:
    """
    Direction-staggered peaks over 10 equal sections:
      EB: sections 2–5, WB: 4–7, NB: 1–3, SB: 6–8.
    """
    flows: List[dict] = []
    secs = ten_sections(T)

    # Build directional route lists
    EB = [(f"w{j}_e{j}", W_in(j), E_out(j, N)) for j in range(N)]
    WB = [(f"e{j}_w{j}", E_in(j, N), W_out(j)) for j in range(N)]
    NB = [(f"s{i}_n{i}", S_in(i), N_out(i, N)) for i in range(N)]
    SB = [(f"n{i}_s{i}", N_in(i, N), S_out(i)) for i in range(N)]

    def add_dir(name_prefix: str, routes, s_idx: int, hi: bool):
        v = int(rng.integers(vph_hi_min, vph_hi_max + 1) if hi
                else rng.integers(vph_lo_min, vph_lo_max + 1))
        b, e = secs[s_idx]
        for rid, src, dst in routes:
            flows.append(dict(fid=f"t03_s{s_idx:02d}_{name_prefix}_{rid}", begin=b, end=e, vph=v, src=src, dst=dst))

    for s in range(10):
        add_dir("EB", EB, s, hi=(2 <= s <= 5))
        add_dir("WB", WB, s, hi=(4 <= s <= 7))
        add_dir("NB", NB, s, hi=(1 <= s <= 3))
        add_dir("SB", SB, s, hi=(6 <= s <= 8))
    return flows

def case_ampm_global(N: int, T: float, rng,
                     vph_lo_min: int = 120, vph_lo_max: int = 150,
                     vph_hi_min: int = 380, vph_hi_max: int = 420) -> List[dict]:
    """
    Global time-varying: sections 0–2 low, 3–6 high, 7–9 low across ALL directions.
    """
    flows: List[dict] = []
    secs = ten_sections(T)
    routes = []
    for j in range(N):
        routes.append((f"w{j}_e{j}", W_in(j), E_out(j, N)))
        routes.append((f"e{j}_w{j}", E_in(j, N), W_out(j)))
    for i in range(N):
        routes.append((f"s{i}_n{i}", S_in(i), N_out(i, N)))
        routes.append((f"n{i}_s{i}", N_in(i, N), S_out(i)))

    for s in range(10):
        lo = (s <= 2) or (s >= 7)
        v = int(rng.integers(vph_lo_min, vph_lo_max + 1) if lo
                else rng.integers(vph_hi_min, vph_hi_max + 1))
        b, e = secs[s]
        for rid, src, dst in routes:
            flows.append(dict(fid=f"t04_s{s:02d}_{rid}", begin=b, end=e, vph=v, src=src, dst=dst))
    return flows

def case_incident_west_boundary(N: int, T: float, rng,
                                base_min: int = 200, base_max: int = 350,
                                incident_cut_frac_min: float = 0.7,
                                incident_cut_frac_max: float = 1.0,
                                incident_sections: Tuple[int, int] = (5, 7)) -> List[dict]:
    """
    Baseline moderate across directions, but at sections ~5–7, drastically cut
    ALL ODs whose source is a WEST fringe (W_in(*)) to emulate a closure/diversion.
    """
    flows: List[dict] = []
    secs = ten_sections(T)

    routes = []
    # EB/WB/NB/SB balanced set
    for j in range(N):
        routes.append((f"w{j}_e{j}", W_in(j), E_out(j, N)))  # affected by incident
        routes.append((f"e{j}_w{j}", E_in(j, N), W_out(j)))
    for i in range(N):
        routes.append((f"s{i}_n{i}", S_in(i), N_out(i, N)))
        routes.append((f"n{i}_s{i}", N_in(i, N), S_out(i)))

    i_lo, i_hi = incident_sections
    cut_frac = rng.uniform(incident_cut_frac_min, incident_cut_frac_max)  # 0.7..1.0

    for s in range(10):
        b, e = secs[s]
        for rid, src, dst in routes:
            is_west_src = src.startswith("e_fw")  # all WEST entries start with e_fw*
            v_base = int(rng.integers(base_min, base_max + 1))
            if i_lo <= s <= i_hi and is_west_src:
                v = max(1, int(v_base * (1.0 - cut_frac)))  # 70–100% cut
                flows.append(dict(fid=f"t05_s{s:02d}_INC_{rid}", begin=b, end=e, vph=v, src=src, dst=dst))
            else:
                flows.append(dict(fid=f"t05_s{s:02d}_BASE_{rid}", begin=b, end=e, vph=v_base, src=src, dst=dst))
    return flows

def case_heavy_corridor_ew_arterial(N: int, T: float, rng) -> list[dict]:
    """
    Heavy Corridor (E↔W arterial):
      - Concentrate high bidirectional demand along a single east–west corridor (middle row).
      - Stresses spatial coordination/offsets along an arterial.
    """
    flows = []
    j = N // 2  # middle row as the arterial
    routes = [
        (f"w{j}_e{j}", W_in(j), E_out(j, N)),  # W -> E
        (f"e{j}_w{j}", E_in(j, N), W_out(j)),  # E -> W
    ]
    # Strong rates to saturate the corridor; slightly skewed distribution
    vphs = draw_vphs(rng, len(routes), low=1500, high=2400, lognormal_skew=0.9)
    for (rid, src, dst), v in zip(routes, vphs):
        flows.append(dict(fid=f"t06_{rid}", begin=0, end=T, vph=int(v), src=src, dst=dst))
    return flows


def case_cross_orthogonal_axes(N: int, T: float, rng) -> list[dict]:
    """
    Cross (Orthogonal OD axes):
      - Two perpendicular arterials (E↔W along middle row and S↔N along middle column) peak simultaneously.
      - Creates conflicting green-wave demands at the hub so agents must trade off fairly and time offsets.
    """
    flows = []
    j = N // 2  # middle row (east–west axis)
    i = N // 2  # middle column (south–north axis)

    routes = [
        (f"w{j}_e{j}", W_in(j), E_out(j, N)),  # W -> E on the row
        (f"e{j}_w{j}", E_in(j, N), W_out(j)),  # E -> W on the row
        (f"s{i}_n{i}", S_in(i), N_out(i, N)),  # S -> N on the column
        (f"n{i}_s{i}", N_in(i, N), S_out(i)),  # N -> S on the column
    ]
    # High-moderate demand to press both axes without total gridlock
    vphs = draw_vphs(rng, len(routes), low=1000, high=1800, lognormal_skew=0.8)
    for (rid, src, dst), v in zip(routes, vphs):
        flows.append(dict(fid=f"t07_{rid}", begin=0, end=T, vph=int(v), src=src, dst=dst))
    return flows

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Temporal-eval trips generator for NxN grid with fringe stubs")
    ap.add_argument("--out", type=str, required=True, help="Output directory for trips files")
    ap.add_argument("--sim-end", type=float, default=500.0, help="End time (s) for flows")
    ap.add_argument("--grid-n", type=int, default=3, help="Grid size N used by your env")
    ap.add_argument("--seed", type=int, default=2025, help="RNG seed")

    # Optional knobs (keep defaults aligned with your guidance)
    ap.add_argument("--pulse-delta-min", type=float, default=25.0)
    ap.add_argument("--pulse-delta-max", type=float, default=40.0)
    ap.add_argument("--pulse-width-min", type=float, default=5.0)
    ap.add_argument("--pulse-width-max", type=float, default=8.0)
    ap.add_argument("--pulse-vph-min", type=int, default=900)
    ap.add_argument("--pulse-vph-max", type=int, default=2000)

    ap.add_argument("--bursty-win-min", type=float, default=10.0)
    ap.add_argument("--bursty-win-max", type=float, default=20.0)
    ap.add_argument("--bursty-on-min", type=int, default=600)
    ap.add_argument("--bursty-on-max", type=int, default=900)
    ap.add_argument("--bursty-off-min", type=int, default=0)
    ap.add_argument("--bursty-off-max", type=int, default=60)
    ap.add_argument("--bursty-route-mult", type=float, default=1.0, help="~2N*mult routes used")

    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    N = args.grid_n
    if N < 2:
        raise ValueError("grid-n must be >= 2")
    T = float(args.sim_end)
    rng = np.random.default_rng(args.seed)

    builders = [
        ("t01_platoons_pulse_trains", lambda N,T,rng: case_platoons_pulse_trains(
            N, T, rng,
            delta_min=args.pulse_delta_min, delta_max=args.pulse_delta_max,
            width_min=args.pulse_width_min, width_max=args.pulse_width_max,
            vph_min=args.pulse_vph_min, vph_max=args.pulse_vph_max
        )),
        ("t02_bursty_on_off", lambda N,T,rng: case_bursty_on_off(
            N, T, rng,
            window_min=args.bursty_win_min, window_max=args.bursty_win_max,
            vph_on_min=args.bursty_on_min, vph_on_max=args.bursty_on_max,
            vph_off_min=args.bursty_off_min, vph_off_max=args.bursty_off_max,
            route_multiplier=args.bursty_route_mult
        )),
        ("t03_shifted_peaks_dir_staggered", case_shifted_peaks_dir_staggered),
        ("t04_ampm_global", case_ampm_global),
        ("t05_incident_west_boundary", case_incident_west_boundary),
        ("t06_heavy_corridor_ew_arterial",  case_heavy_corridor_ew_arterial),
        ("t07_cross_orthogonal_axes",       case_cross_orthogonal_axes),
    ]

    for name, builder in builders:
        flows = builder(N, T, rng)
        fname = f"eval_trips_{name}_gridN{N}.xml"
        path = os.path.join(args.out, fname)
        write_case(path, flows)
        print(f"Wrote {path} ({len(flows)} flows)")
        if not flows:
            print("  (no flows)")
        else:
            for f in flows[:min(len(flows), 40)]:  # keep printout readable
                print(f"  - {f['fid']}: {f['vph']} vph  {f['src']} -> {f['dst']}  [{f['begin']:.0f}-{f['end']:.0f}]")
            if len(flows) > 40:
                print(f"  ... ({len(flows)-40} more)")

if __name__ == "__main__":
    main()
