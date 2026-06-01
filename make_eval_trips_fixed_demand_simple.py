#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np


# ============================================================================
# Edge helpers that match the grid-with-fringe-stubs naming convention
# ============================================================================

def W_in(j: int) -> str:
    return f"e_fw{j}_to_n0_{j}"


def W_out(j: int) -> str:
    return f"e_n0_{j}_to_fw{j}"


def E_in(j: int, N: int) -> str:
    return f"e_fe{j}_to_n{N - 1}_{j}"


def E_out(j: int, N: int) -> str:
    return f"e_n{N - 1}_{j}_to_fe{j}"


def S_in(i: int) -> str:
    return f"e_fs{i}_to_n{i}_0"


def S_out(i: int) -> str:
    return f"e_n{i}_0_to_fs{i}"


def N_in(i: int, N: int) -> str:
    return f"e_fn{i}_to_n{i}_{N - 1}"


def N_out(i: int, N: int) -> str:
    return f"e_n{i}_{N - 1}_to_fn{i}"


# ============================================================================
# XML helpers
# ============================================================================

def new_root(vehicle_sigma: float = 0.5) -> ET.Element:
    root = ET.Element("routes")
    ET.SubElement(
        root,
        "vType",
        id="car",
        accel="2.0",
        decel="4.5",
        sigma=f"{vehicle_sigma:g}",
        length="3.5",
        width="1.0",
        maxSpeed="16.7",
        vClass="passenger",
        guiShape="passenger",
    )
    return root


def add_flow(
    root: ET.Element,
    fid: str,
    begin: float,
    end: float,
    vph: int,
    src: str,
    dst: str,
) -> None:
    begin = float(begin)
    end = float(end)
    vph = int(max(1, vph))

    if end <= begin:
        return

    ET.SubElement(
        root,
        "flow",
        id=fid,
        type="car",
        begin=f"{begin:.1f}",
        end=f"{end:.1f}",
        vehsPerHour=str(vph),
        **{"from": src, "to": dst},
    )


def write_case(path: str, flows: List[dict], vehicle_sigma: float = 0.5) -> None:
    root = new_root(vehicle_sigma=vehicle_sigma)
    for flow in flows:
        add_flow(root, **flow)

    ET.ElementTree(root).write(
        path,
        encoding="utf-8",
        xml_declaration=True,
    )


# ============================================================================
# Utility
# ============================================================================

def ten_sections(T: float) -> List[Tuple[float, float]]:
    edges = np.linspace(0.0, T, 11)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(10)]


# ============================================================================
# Simple case builders
# ============================================================================

def case_platoons_pulse_trains(
    N: int,
    T: float,
    rng: np.random.Generator,
    delta_min: float = 25.0,
    delta_max: float = 40.0,
    width_min: float = 5.0,
    width_max: float = 8.0,
    vph_min: int = 900,
    vph_max: int = 1200,
) -> List[dict]:
    """
    Simple platoon pulse trains on the middle E-W corridor.

    Both E-W directions receive repeated short pulses. This is the simplest
    temporal platoon case: no precursor pulse, no train blocks, and no
    grid-specific presets.
    """
    flows: List[dict] = []
    r = N // 2

    routes = [
        (f"w{r}_e{r}", W_in(r), E_out(r, N)),
        (f"e{r}_w{r}", E_in(r, N), W_out(r)),
    ]

    for rid, src, dst in routes:
        delta = rng.uniform(delta_min, delta_max)
        width = rng.uniform(width_min, width_max)
        t = rng.uniform(0.0, delta)
        k = 0

        while t < T:
            vph = int(rng.integers(vph_min, vph_max + 1))
            flows.append(
                dict(
                    fid=f"t01_{rid}_pulse{k}",
                    begin=t,
                    end=min(T, t + width),
                    vph=vph,
                    src=src,
                    dst=dst,
                )
            )
            t += delta
            k += 1

    return flows


def case_bursty_on_off(
    N: int,
    T: float,
    rng: np.random.Generator,
    window_min: float = 10.0,
    window_max: float = 20.0,
    vph_on_min: int = 600,
    vph_on_max: int = 900,
    vph_off_min: int = 0,
    vph_off_max: int = 60,
    route_multiplier: float = 1.0,
) -> List[dict]:
    """
    Simple bursty ON/OFF case.

    A random subset of routes is selected. Each route is split into short
    windows, and every window is independently assigned ON or OFF demand.
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
    num_routes = max(1, int(route_multiplier * 2 * N))
    routes = candidates[:num_routes]

    for rid, src, dst in routes:
        win = rng.uniform(window_min, window_max)
        t = rng.uniform(0.0, win)
        k = 0

        while t < T:
            is_on = rng.random() < 0.5
            if is_on:
                vph = int(rng.integers(vph_on_min, vph_on_max + 1))
            else:
                vph = int(rng.integers(vph_off_min, vph_off_max + 1))

            flows.append(
                dict(
                    fid=f"t02_{rid}_w{k}",
                    begin=t,
                    end=min(T, t + win),
                    vph=vph,
                    src=src,
                    dst=dst,
                )
            )
            t += win
            k += 1

    return flows


def case_shifted_peaks_dir_staggered(
    N: int,
    T: float,
    rng: np.random.Generator,
    vph_lo_min: int = 100,
    vph_lo_max: int = 150,
    vph_hi_min: int = 350,
    vph_hi_max: int = 600,
) -> List[dict]:
    """
    Simple direction-staggered peaks over 10 equal sections.

    EB: sections 2-5
    WB: sections 4-7
    NB: sections 1-3
    SB: sections 6-8
    """
    flows: List[dict] = []
    secs = ten_sections(T)

    EB = [(f"w{j}_e{j}", W_in(j), E_out(j, N)) for j in range(N)]
    WB = [(f"e{j}_w{j}", E_in(j, N), W_out(j)) for j in range(N)]
    NB = [(f"s{i}_n{i}", S_in(i), N_out(i, N)) for i in range(N)]
    SB = [(f"n{i}_s{i}", N_in(i, N), S_out(i)) for i in range(N)]

    def add_dir(name: str, routes: List[Tuple[str, str, str]], section_idx: int, high: bool) -> None:
        vph = int(
            rng.integers(vph_hi_min, vph_hi_max + 1)
            if high
            else rng.integers(vph_lo_min, vph_lo_max + 1)
        )
        begin, end = secs[section_idx]

        for rid, src, dst in routes:
            flows.append(
                dict(
                    fid=f"t03_s{section_idx:02d}_{name}_{rid}",
                    begin=begin,
                    end=end,
                    vph=vph,
                    src=src,
                    dst=dst,
                )
            )

    for s in range(10):
        add_dir("EB", EB, s, high=(2 <= s <= 5))
        add_dir("WB", WB, s, high=(4 <= s <= 7))
        add_dir("NB", NB, s, high=(1 <= s <= 3))
        add_dir("SB", SB, s, high=(6 <= s <= 8))

    return flows


def case_ampm_global(
    N: int,
    T: float,
    rng: np.random.Generator,
    vph_lo_min: int = 120,
    vph_lo_max: int = 150,
    vph_hi_min: int = 380,
    vph_hi_max: int = 420,
) -> List[dict]:
    """
    Simple global low-high-low demand pattern.

    Sections 0-2: low
    Sections 3-6: high
    Sections 7-9: low
    """
    flows: List[dict] = []
    secs = ten_sections(T)
    routes: List[Tuple[str, str, str]] = []

    for j in range(N):
        routes.append((f"w{j}_e{j}", W_in(j), E_out(j, N)))
        routes.append((f"e{j}_w{j}", E_in(j, N), W_out(j)))

    for i in range(N):
        routes.append((f"s{i}_n{i}", S_in(i), N_out(i, N)))
        routes.append((f"n{i}_s{i}", N_in(i, N), S_out(i)))

    for s in range(10):
        is_low = (s <= 2) or (s >= 7)
        vph = int(
            rng.integers(vph_lo_min, vph_lo_max + 1)
            if is_low
            else rng.integers(vph_hi_min, vph_hi_max + 1)
        )
        begin, end = secs[s]

        for rid, src, dst in routes:
            flows.append(
                dict(
                    fid=f"t04_s{s:02d}_{rid}",
                    begin=begin,
                    end=end,
                    vph=vph,
                    src=src,
                    dst=dst,
                )
            )

    return flows


def case_incident_west_boundary(
    N: int,
    T: float,
    rng: np.random.Generator,
    base_min: int = 200,
    base_max: int = 350,
    incident_cut_frac_min: float = 0.7,
    incident_cut_frac_max: float = 1.0,
    incident_sections: Tuple[int, int] = (5, 7),
) -> List[dict]:
    """
    Simple incident case.

    Demand is moderate and balanced, but during selected sections all routes
    starting from the west fringe are reduced by 70-100%.
    """
    flows: List[dict] = []
    secs = ten_sections(T)
    routes: List[Tuple[str, str, str]] = []

    for j in range(N):
        routes.append((f"w{j}_e{j}", W_in(j), E_out(j, N)))
        routes.append((f"e{j}_w{j}", E_in(j, N), W_out(j)))

    for i in range(N):
        routes.append((f"s{i}_n{i}", S_in(i), N_out(i, N)))
        routes.append((f"n{i}_s{i}", N_in(i, N), S_out(i)))

    incident_start, incident_end = incident_sections
    cut_frac = rng.uniform(incident_cut_frac_min, incident_cut_frac_max)

    for s in range(10):
        begin, end = secs[s]

        for rid, src, dst in routes:
            is_west_src = src.startswith("e_fw")
            base_vph = int(rng.integers(base_min, base_max + 1))

            if incident_start <= s <= incident_end and is_west_src:
                vph = max(1, int(base_vph * (1.0 - cut_frac)))
                prefix = "INC"
            else:
                vph = base_vph
                prefix = "BASE"

            flows.append(
                dict(
                    fid=f"t05_s{s:02d}_{prefix}_{rid}",
                    begin=begin,
                    end=end,
                    vph=vph,
                    src=src,
                    dst=dst,
                )
            )

    return flows


def case_heavy_corridor_ew_arterial(
    N: int,
    T: float,
    rng: np.random.Generator,
    corridor_vph: int = 1000,
) -> List[dict]:
    """
    Simple heavy E-W corridor.

    Constant bidirectional demand is placed on the middle row only.
    """
    flows: List[dict] = []
    j = N // 2

    routes = [
        (f"w{j}_e{j}", W_in(j), E_out(j, N)),
        (f"e{j}_w{j}", E_in(j, N), W_out(j)),
    ]

    for rid, src, dst in routes:
        flows.append(
            dict(
                fid=f"t06_{rid}",
                begin=0.0,
                end=T,
                vph=int(corridor_vph),
                src=src,
                dst=dst,
            )
        )

    return flows


def case_cross_orthogonal_axes(
    N: int,
    T: float,
    rng: np.random.Generator,
    ew_vph: int = 600,
    ns_vph: int = 450,
) -> List[dict]:
    """
    Simple central cross case.

    The middle E-W row and middle N-S column are both active. E-W is slightly
    stronger than N-S to avoid perfect symmetry.
    """
    flows: List[dict] = []
    j = N // 2
    i = N // 2

    routes = [
        (f"w{j}_e{j}", W_in(j), E_out(j, N), ew_vph),
        (f"e{j}_w{j}", E_in(j, N), W_out(j), ew_vph),
        (f"s{i}_n{i}", S_in(i), N_out(i, N), ns_vph),
        (f"n{i}_s{i}", N_in(i, N), S_out(i), ns_vph),
    ]

    for rid, src, dst, vph in routes:
        flows.append(
            dict(
                fid=f"t07_{rid}",
                begin=0.0,
                end=T,
                vph=int(vph),
                src=src,
                dst=dst,
            )
        )

    return flows


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        "Simple evaluation trip generator for NxN grid with fringe stubs"
    )

    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--sim-end", type=float, default=500.0, help="Simulation end time in seconds")
    parser.add_argument("--grid-n", type=int, default=3, help="Grid size N")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument(
        "--vehicle-sigma",
        type=float,
        default=0.5,
        help="SUMO vType sigma. Use 0.0 for deterministic driving, 0.5 for standard stochastic driving.",
    )

    # t01 simple platoon pulse trains
    parser.add_argument("--pulse-delta-min", type=float, default=25.0)
    parser.add_argument("--pulse-delta-max", type=float, default=40.0)
    parser.add_argument("--pulse-width-min", type=float, default=5.0)
    parser.add_argument("--pulse-width-max", type=float, default=8.0)
    parser.add_argument("--pulse-vph-min", type=int, default=900)
    parser.add_argument("--pulse-vph-max", type=int, default=1200)

    # t02 simple bursty ON/OFF
    parser.add_argument("--bursty-win-min", type=float, default=10.0)
    parser.add_argument("--bursty-win-max", type=float, default=20.0)
    parser.add_argument("--bursty-on-min", type=int, default=600)
    parser.add_argument("--bursty-on-max", type=int, default=900)
    parser.add_argument("--bursty-off-min", type=int, default=0)
    parser.add_argument("--bursty-off-max", type=int, default=60)
    parser.add_argument(
        "--bursty-route-mult",
        type=float,
        default=1.0,
        help="Number of bursty routes is approximately 2N * multiplier.",
    )

    # t06 simple heavy corridor
    parser.add_argument(
        "--corridor-vph",
        type=int,
        default=1000,
        help="Fixed veh/h for both directions of the middle E-W corridor.",
    )

    # t07 simple central cross
    parser.add_argument("--cross-ew-vph", type=int, default=600)
    parser.add_argument("--cross-ns-vph", type=int, default=450)

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    N = int(args.grid_n)
    if N < 2:
        raise ValueError("grid-n must be >= 2")

    T = float(args.sim_end)
    rng = np.random.default_rng(args.seed)

    builders = [
        (
            "t01_platoons_pulse_trains",
            lambda n, t, r: case_platoons_pulse_trains(
                n,
                t,
                r,
                delta_min=args.pulse_delta_min,
                delta_max=args.pulse_delta_max,
                width_min=args.pulse_width_min,
                width_max=args.pulse_width_max,
                vph_min=args.pulse_vph_min,
                vph_max=args.pulse_vph_max,
            ),
        ),
        (
            "t02_bursty_on_off",
            lambda n, t, r: case_bursty_on_off(
                n,
                t,
                r,
                window_min=args.bursty_win_min,
                window_max=args.bursty_win_max,
                vph_on_min=args.bursty_on_min,
                vph_on_max=args.bursty_on_max,
                vph_off_min=args.bursty_off_min,
                vph_off_max=args.bursty_off_max,
                route_multiplier=args.bursty_route_mult,
            ),
        ),
        ("t03_shifted_peaks_dir_staggered", case_shifted_peaks_dir_staggered),
        ("t04_ampm_global", case_ampm_global),
        ("t05_incident_west_boundary", case_incident_west_boundary),
        (
            "t06_heavy_corridor_ew_arterial",
            lambda n, t, r: case_heavy_corridor_ew_arterial(
                n,
                t,
                r,
                corridor_vph=args.corridor_vph,
            ),
        ),
        (
            "t07_cross_orthogonal_axes",
            lambda n, t, r: case_cross_orthogonal_axes(
                n,
                t,
                r,
                ew_vph=args.cross_ew_vph,
                ns_vph=args.cross_ns_vph,
            ),
        ),
    ]

    print(f"[INFO] grid_n={N}, sim_end={T}, seed={args.seed}, vehicle_sigma={args.vehicle_sigma}")

    for name, builder in builders:
        flows = builder(N, T, rng)
        filename = f"eval_trips_{name}_gridN{N}.xml"
        path = os.path.join(args.out, filename)

        write_case(path, flows, vehicle_sigma=args.vehicle_sigma)

        print(f"Wrote {path} ({len(flows)} flows)")
        if not flows:
            print("  (no flows)")
            continue

        for flow in flows[: min(len(flows), 40)]:
            print(
                f"  - {flow['fid']}: "
                f"{flow['vph']} vph  "
                f"{flow['src']} -> {flow['dst']}  "
                f"[{flow['begin']:.0f}-{flow['end']:.0f}]"
            )

        if len(flows) > 40:
            print(f"  ... ({len(flows) - 40} more)")


if __name__ == "__main__":
    main()
