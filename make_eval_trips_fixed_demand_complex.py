#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np


# ============================================================================
# Edge helpers
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

def new_root(vehicle_sigma: float = 0.0) -> ET.Element:
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


def write_case(path: str, flows: List[dict], vehicle_sigma: float = 0.0) -> None:
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
# Case builders
# ============================================================================

def case_platoons_pulse_trains(
    N: int,
    T: float,
    rng: np.random.Generator,
    heavy_pulses_per_train: int = 3,
    light_pulses_per_train: int = 2,
    precursor_width_min: float = 3.0,
    precursor_width_max: float = 5.0,
    precursor_vph_min: int = 500,
    precursor_vph_max: int = 900,
    precursor_to_main_gap_min: float = 10.0,
    precursor_to_main_gap_max: float = 18.0,
    heavy_pulse_gap_min: float = 12.0,
    heavy_pulse_gap_max: float = 20.0,
    light_pulse_gap_min: float = 20.0,
    light_pulse_gap_max: float = 32.0,
    heavy_width_min: float = 4.0,
    heavy_width_max: float = 6.0,
    light_width_min: float = 3.0,
    light_width_max: float = 5.0,
    heavy_vph_min: int = 1800,
    heavy_vph_max: int = 2600,
    light_vph_min: int = 600,
    light_vph_max: int = 1200,
    train_gap_min: float = 35.0,
    train_gap_max: float = 70.0,
    jitter_frac: float = 0.08,
) -> List[dict]:
    """
    LSTM-oriented platoon case.

    Each train has:
      1. a small precursor pulse;
      2. a short delay;
      3. several main pulses.

    The precursor gives a recurrent policy useful short-horizon information:
    after seeing the precursor, a larger pulse train is likely to arrive soon.
    """
    flows: List[dict] = []
    r = N // 2

    route_cfgs = [
        {
            "rid": f"w{r}_e{r}",
            "src": W_in(r),
            "dst": E_out(r, N),
            "pulses_per_train": heavy_pulses_per_train,
            "gap_min": heavy_pulse_gap_min,
            "gap_max": heavy_pulse_gap_max,
            "width_min": heavy_width_min,
            "width_max": heavy_width_max,
            "vph_min": heavy_vph_min,
            "vph_max": heavy_vph_max,
        },
        {
            "rid": f"e{r}_w{r}",
            "src": E_in(r, N),
            "dst": W_out(r),
            "pulses_per_train": light_pulses_per_train,
            "gap_min": light_pulse_gap_min,
            "gap_max": light_pulse_gap_max,
            "width_min": light_width_min,
            "width_max": light_width_max,
            "vph_min": light_vph_min,
            "vph_max": light_vph_max,
        },
    ]

    for cfg in route_cfgs:
        t = rng.uniform(0.0, 20.0)
        pulse_id = 0

        while t < T:
            # Small precursor pulse.
            precursor_width = rng.uniform(precursor_width_min, precursor_width_max)
            precursor_vph = int(rng.integers(precursor_vph_min, precursor_vph_max + 1))

            flows.append(
                dict(
                    fid=f"t01_{cfg['rid']}_pre{pulse_id}",
                    begin=t,
                    end=min(T, t + precursor_width),
                    vph=precursor_vph,
                    src=cfg["src"],
                    dst=cfg["dst"],
                )
            )

            t += rng.uniform(precursor_to_main_gap_min, precursor_to_main_gap_max)

            # Main pulse train.
            base_gap = rng.uniform(cfg["gap_min"], cfg["gap_max"])

            for _ in range(cfg["pulses_per_train"]):
                if t >= T:
                    break

                width = rng.uniform(cfg["width_min"], cfg["width_max"])
                vph = int(rng.integers(cfg["vph_min"], cfg["vph_max"] + 1))

                flows.append(
                    dict(
                        fid=f"t01_{cfg['rid']}_main{pulse_id}",
                        begin=t,
                        end=min(T, t + width),
                        vph=vph,
                        src=cfg["src"],
                        dst=cfg["dst"],
                    )
                )

                jitter = rng.uniform(1.0 - jitter_frac, 1.0 + jitter_frac)
                t += base_gap * jitter
                pulse_id += 1

            # Quiet gap between trains.
            t += rng.uniform(train_gap_min, train_gap_max)

    return flows


def case_bursty_on_off(
    N: int,
    T: float,
    rng: np.random.Generator,
    window_min: float = 5.0,
    window_max: float = 8.0,
    vph_on_min: int = 950,
    vph_on_max: int = 1400,
    vph_off_min: int = 0,
    vph_off_max: int = 40,
    route_multiplier: float = 1.0,
    on_block_min: float = 25.0,
    on_block_max: float = 45.0,
    off_block_min: float = 25.0,
    off_block_max: float = 45.0,
    p_on_dropout: float = 0.20,
    p_off_false_burst: float = 0.10,
) -> List[dict]:
    """
    LSTM-oriented bursty case.

    ON/OFF blocks are kept short enough to fit approximately within a seqlen8
    recurrent horizon. Short dropouts and false bursts make the current window
    ambiguous, so recent history becomes useful.
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
        on_regime = rng.random() < 0.5

        while t < T:
            if on_regime:
                block_len = rng.uniform(on_block_min, on_block_max)
            else:
                block_len = rng.uniform(off_block_min, off_block_max)

            block_end = min(T, t + block_len)

            while t < block_end:
                if on_regime:
                    if rng.random() < p_on_dropout:
                        vph = int(rng.integers(vph_off_min, vph_off_max + 1))
                    else:
                        vph = int(rng.integers(vph_on_min, vph_on_max + 1))
                else:
                    if rng.random() < p_off_false_burst:
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

            on_regime = not on_regime

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
    flows: List[dict] = []
    secs = ten_sections(T)

    EB = [(f"w{j}_e{j}", W_in(j), E_out(j, N)) for j in range(N)]
    WB = [(f"e{j}_w{j}", E_in(j, N), W_out(j)) for j in range(N)]
    NB = [(f"s{i}_n{i}", S_in(i), N_out(i, N)) for i in range(N)]
    SB = [(f"n{i}_s{i}", N_in(i, N), S_out(i)) for i in range(N)]

    def add_dir(name_prefix: str, routes, section_idx: int, high: bool) -> None:
        vph = int(
            rng.integers(vph_hi_min, vph_hi_max + 1)
            if high
            else rng.integers(vph_lo_min, vph_lo_max + 1)
        )
        begin, end = secs[section_idx]

        for rid, src, dst in routes:
            flows.append(
                dict(
                    fid=f"t03_s{section_idx:02d}_{name_prefix}_{rid}",
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
            west_source = src.startswith("e_fw")
            base_vph = int(rng.integers(base_min, base_max + 1))

            if incident_start <= s <= incident_end and west_source:
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
    corridor_vph: int = 1200,
    corridor_pulse_width_min: float = 8.0,
    corridor_pulse_width_max: float = 12.0,
    corridor_pulse_gap_min: float = 22.0,
    corridor_pulse_gap_max: float = 34.0,
    corridor_jitter_frac: float = 0.10,
    side_vph: int = 150,
) -> List[dict]:
    """
    GNN-oriented corridor-wave case.

    The main E-W corridor receives pulse waves. Side-street demand is kept
    continuous and light. Downstream junctions benefit from upstream queue
    information, making this more GNN-friendly than a steady corridor.
    """
    flows: List[dict] = []
    r = N // 2

    corridor_routes = [
        (f"w{r}_e{r}", W_in(r), E_out(r, N)),
        (f"e{r}_w{r}", E_in(r, N), W_out(r)),
    ]

    for rid, src, dst in corridor_routes:
        t = rng.uniform(0.0, 20.0)
        pulse_id = 0
        base_gap = rng.uniform(corridor_pulse_gap_min, corridor_pulse_gap_max)

        while t < T:
            width = rng.uniform(corridor_pulse_width_min, corridor_pulse_width_max)
            jitter = rng.uniform(1.0 - corridor_jitter_frac, 1.0 + corridor_jitter_frac)

            flows.append(
                dict(
                    fid=f"t06_MAIN_{rid}_pulse{pulse_id}",
                    begin=t,
                    end=min(T, t + width),
                    vph=int(corridor_vph),
                    src=src,
                    dst=dst,
                )
            )

            t += base_gap * jitter
            pulse_id += 1

    if side_vph > 0:
        for i in range(N):
            side_routes = [
                (f"s{i}_n{i}", S_in(i), N_out(i, N)),
                (f"n{i}_s{i}", N_in(i, N), S_out(i)),
            ]

            for rid, src, dst in side_routes:
                flows.append(
                    dict(
                        fid=f"t06_SIDE_{rid}",
                        begin=0.0,
                        end=T,
                        vph=int(side_vph),
                        src=src,
                        dst=dst,
                    )
                )

    return flows


def case_cross_orthogonal_axes(
    N: int,
    T: float,
    rng: np.random.Generator,
    cross_centre_ew_vph: int = 650,
    cross_centre_ns_vph: int = 550,
    cross_adjacent_ew_vph: int = 350,
    cross_adjacent_ns_vph: int = 300,
    cross_width: int = 1,
) -> List[dict]:
    flows: List[dict] = []
    centre = N // 2

    idxs = [
        k
        for k in range(centre - cross_width, centre + cross_width + 1)
        if 0 <= k < N
    ]

    for j in idxs:
        if j == centre:
            vph = cross_centre_ew_vph
            band = "CENTRE"
        else:
            vph = cross_adjacent_ew_vph
            band = "ADJ"

        routes = [
            (f"w{j}_e{j}", W_in(j), E_out(j, N)),
            (f"e{j}_w{j}", E_in(j, N), W_out(j)),
        ]

        for rid, src, dst in routes:
            flows.append(
                dict(
                    fid=f"t07_EW_{band}_{rid}",
                    begin=0.0,
                    end=T,
                    vph=int(vph),
                    src=src,
                    dst=dst,
                )
            )

    for i in idxs:
        if i == centre:
            vph = cross_centre_ns_vph
            band = "CENTRE"
        else:
            vph = cross_adjacent_ns_vph
            band = "ADJ"

        routes = [
            (f"s{i}_n{i}", S_in(i), N_out(i, N)),
            (f"n{i}_s{i}", N_in(i, N), S_out(i)),
        ]

        for rid, src, dst in routes:
            flows.append(
                dict(
                    fid=f"t07_NS_{band}_{rid}",
                    begin=0.0,
                    end=T,
                    vph=int(vph),
                    src=src,
                    dst=dst,
                )
            )

    return flows


# ============================================================================
# Grid-specific presets
# ============================================================================

GRID_TEMPORAL_PRESETS = {
    3: {
        "platoon_heavy_pulses_per_train": 3,
        "platoon_light_pulses_per_train": 2,
        "platoon_precursor_width_min": 3.0,
        "platoon_precursor_width_max": 5.0,
        "platoon_precursor_vph_min": 500,
        "platoon_precursor_vph_max": 900,
        "platoon_precursor_to_main_gap_min": 10.0,
        "platoon_precursor_to_main_gap_max": 18.0,
        "platoon_heavy_pulse_gap_min": 12.0,
        "platoon_heavy_pulse_gap_max": 20.0,
        "platoon_light_pulse_gap_min": 20.0,
        "platoon_light_pulse_gap_max": 32.0,
        "platoon_heavy_width_min": 4.0,
        "platoon_heavy_width_max": 6.0,
        "platoon_light_width_min": 3.0,
        "platoon_light_width_max": 5.0,
        "platoon_heavy_vph_min": 1800,
        "platoon_heavy_vph_max": 2600,
        "platoon_light_vph_min": 600,
        "platoon_light_vph_max": 1200,
        "platoon_train_gap_min": 35.0,
        "platoon_train_gap_max": 70.0,
        "platoon_jitter_frac": 0.08,
        "bursty_win_min": 5.0,
        "bursty_win_max": 8.0,
        "bursty_on_min": 950,
        "bursty_on_max": 1400,
        "bursty_off_min": 0,
        "bursty_off_max": 40,
        "bursty_route_mult": 1.0,
        "bursty_on_block_min": 25.0,
        "bursty_on_block_max": 45.0,
        "bursty_off_block_min": 25.0,
        "bursty_off_block_max": 45.0,
        "bursty_p_on_dropout": 0.20,
        "bursty_p_off_false_burst": 0.10,
    },
    5: {
        "platoon_heavy_pulses_per_train": 3,
        "platoon_light_pulses_per_train": 2,
        "platoon_precursor_width_min": 3.0,
        "platoon_precursor_width_max": 5.0,
        "platoon_precursor_vph_min": 450,
        "platoon_precursor_vph_max": 800,
        "platoon_precursor_to_main_gap_min": 12.0,
        "platoon_precursor_to_main_gap_max": 20.0,
        "platoon_heavy_pulse_gap_min": 14.0,
        "platoon_heavy_pulse_gap_max": 22.0,
        "platoon_light_pulse_gap_min": 24.0,
        "platoon_light_pulse_gap_max": 36.0,
        "platoon_heavy_width_min": 4.0,
        "platoon_heavy_width_max": 6.0,
        "platoon_light_width_min": 3.0,
        "platoon_light_width_max": 5.0,
        "platoon_heavy_vph_min": 1400,
        "platoon_heavy_vph_max": 2200,
        "platoon_light_vph_min": 500,
        "platoon_light_vph_max": 1000,
        "platoon_train_gap_min": 40.0,
        "platoon_train_gap_max": 75.0,
        "platoon_jitter_frac": 0.08,
        "bursty_win_min": 6.0,
        "bursty_win_max": 10.0,
        "bursty_on_min": 800,
        "bursty_on_max": 1250,
        "bursty_off_min": 0,
        "bursty_off_max": 40,
        "bursty_route_mult": 0.8,
        "bursty_on_block_min": 30.0,
        "bursty_on_block_max": 50.0,
        "bursty_off_block_min": 30.0,
        "bursty_off_block_max": 50.0,
        "bursty_p_on_dropout": 0.20,
        "bursty_p_off_false_burst": 0.10,
    },
    10: {
        "platoon_heavy_pulses_per_train": 2,
        "platoon_light_pulses_per_train": 1,
        "platoon_precursor_width_min": 3.0,
        "platoon_precursor_width_max": 5.0,
        "platoon_precursor_vph_min": 350,
        "platoon_precursor_vph_max": 700,
        "platoon_precursor_to_main_gap_min": 14.0,
        "platoon_precursor_to_main_gap_max": 24.0,
        "platoon_heavy_pulse_gap_min": 18.0,
        "platoon_heavy_pulse_gap_max": 30.0,
        "platoon_light_pulse_gap_min": 30.0,
        "platoon_light_pulse_gap_max": 45.0,
        "platoon_heavy_width_min": 3.5,
        "platoon_heavy_width_max": 5.5,
        "platoon_light_width_min": 3.0,
        "platoon_light_width_max": 4.5,
        "platoon_heavy_vph_min": 1000,
        "platoon_heavy_vph_max": 1700,
        "platoon_light_vph_min": 350,
        "platoon_light_vph_max": 800,
        "platoon_train_gap_min": 50.0,
        "platoon_train_gap_max": 90.0,
        "platoon_jitter_frac": 0.08,
        "bursty_win_min": 8.0,
        "bursty_win_max": 12.0,
        "bursty_on_min": 550,
        "bursty_on_max": 950,
        "bursty_off_min": 0,
        "bursty_off_max": 30,
        "bursty_route_mult": 0.5,
        "bursty_on_block_min": 35.0,
        "bursty_on_block_max": 60.0,
        "bursty_off_block_min": 35.0,
        "bursty_off_block_max": 60.0,
        "bursty_p_on_dropout": 0.20,
        "bursty_p_off_false_burst": 0.10,
    },
}


GRID_SPATIAL_PRESETS = {
    3: {
        "corridor_vph": 1300,
        "corridor_pulse_width_min": 8.0,
        "corridor_pulse_width_max": 12.0,
        "corridor_pulse_gap_min": 22.0,
        "corridor_pulse_gap_max": 34.0,
        "corridor_jitter_frac": 0.10,
        "corridor_side_vph": 180,
        "cross_centre_ew_vph": 600,
        "cross_centre_ns_vph": 500,
        "cross_adjacent_ew_vph": 200,
        "cross_adjacent_ns_vph": 160,
        "cross_width": 0,
    },
    5: {
        "corridor_vph": 1150,
        "corridor_pulse_width_min": 8.0,
        "corridor_pulse_width_max": 12.0,
        "corridor_pulse_gap_min": 24.0,
        "corridor_pulse_gap_max": 38.0,
        "corridor_jitter_frac": 0.10,
        "corridor_side_vph": 150,
        "cross_centre_ew_vph": 650,
        "cross_centre_ns_vph": 550,
        "cross_adjacent_ew_vph": 320,
        "cross_adjacent_ns_vph": 280,
        "cross_width": 1,
    },
    10: {
        "corridor_vph": 850,
        "corridor_pulse_width_min": 7.0,
        "corridor_pulse_width_max": 11.0,
        "corridor_pulse_gap_min": 28.0,
        "corridor_pulse_gap_max": 45.0,
        "corridor_jitter_frac": 0.10,
        "corridor_side_vph": 100,
        "cross_centre_ew_vph": 600,
        "cross_centre_ns_vph": 500,
        "cross_adjacent_ew_vph": 180,
        "cross_adjacent_ns_vph": 160,
        "cross_width": 2,
    },
}


def get_nearest_preset(grid_n: int, presets: dict, name: str) -> dict:
    if grid_n in presets:
        return presets[grid_n]

    available = sorted(presets.keys())
    nearest = min(available, key=lambda x: abs(x - grid_n))

    print(
        f"[WARN] No {name} preset defined for grid_n={grid_n}; "
        f"using nearest preset grid_n={nearest}."
    )

    return presets[nearest]


def apply_defaults_from_preset(args, preset: dict) -> None:
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        "Temporal and spatial evaluation trip generator for NxN grid with fringe stubs"
    )

    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--sim-end", type=float, default=500.0)
    parser.add_argument("--grid-n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--vehicle-sigma", type=float, default=0.0)

    # Platoon arguments
    parser.add_argument("--platoon-heavy-pulses-per-train", type=int, default=None)
    parser.add_argument("--platoon-light-pulses-per-train", type=int, default=None)
    parser.add_argument("--platoon-precursor-width-min", type=float, default=None)
    parser.add_argument("--platoon-precursor-width-max", type=float, default=None)
    parser.add_argument("--platoon-precursor-vph-min", type=int, default=None)
    parser.add_argument("--platoon-precursor-vph-max", type=int, default=None)
    parser.add_argument("--platoon-precursor-to-main-gap-min", type=float, default=None)
    parser.add_argument("--platoon-precursor-to-main-gap-max", type=float, default=None)
    parser.add_argument("--platoon-heavy-pulse-gap-min", type=float, default=None)
    parser.add_argument("--platoon-heavy-pulse-gap-max", type=float, default=None)
    parser.add_argument("--platoon-light-pulse-gap-min", type=float, default=None)
    parser.add_argument("--platoon-light-pulse-gap-max", type=float, default=None)
    parser.add_argument("--platoon-heavy-width-min", type=float, default=None)
    parser.add_argument("--platoon-heavy-width-max", type=float, default=None)
    parser.add_argument("--platoon-light-width-min", type=float, default=None)
    parser.add_argument("--platoon-light-width-max", type=float, default=None)
    parser.add_argument("--platoon-heavy-vph-min", type=int, default=None)
    parser.add_argument("--platoon-heavy-vph-max", type=int, default=None)
    parser.add_argument("--platoon-light-vph-min", type=int, default=None)
    parser.add_argument("--platoon-light-vph-max", type=int, default=None)
    parser.add_argument("--platoon-train-gap-min", type=float, default=None)
    parser.add_argument("--platoon-train-gap-max", type=float, default=None)
    parser.add_argument("--platoon-jitter-frac", type=float, default=None)

    # Bursty arguments
    parser.add_argument("--bursty-win-min", type=float, default=None)
    parser.add_argument("--bursty-win-max", type=float, default=None)
    parser.add_argument("--bursty-on-min", type=int, default=None)
    parser.add_argument("--bursty-on-max", type=int, default=None)
    parser.add_argument("--bursty-off-min", type=int, default=None)
    parser.add_argument("--bursty-off-max", type=int, default=None)
    parser.add_argument("--bursty-route-mult", type=float, default=None)
    parser.add_argument("--bursty-on-block-min", type=float, default=None)
    parser.add_argument("--bursty-on-block-max", type=float, default=None)
    parser.add_argument("--bursty-off-block-min", type=float, default=None)
    parser.add_argument("--bursty-off-block-max", type=float, default=None)
    parser.add_argument("--bursty-p-on-dropout", type=float, default=None)
    parser.add_argument("--bursty-p-off-false-burst", type=float, default=None)

    # Spatial arguments
    parser.add_argument("--corridor-vph", type=int, default=None)
    parser.add_argument("--corridor-pulse-width-min", type=float, default=None)
    parser.add_argument("--corridor-pulse-width-max", type=float, default=None)
    parser.add_argument("--corridor-pulse-gap-min", type=float, default=None)
    parser.add_argument("--corridor-pulse-gap-max", type=float, default=None)
    parser.add_argument("--corridor-jitter-frac", type=float, default=None)
    parser.add_argument("--corridor-side-vph", type=int, default=None)

    parser.add_argument("--cross-centre-ew-vph", type=int, default=None)
    parser.add_argument("--cross-centre-ns-vph", type=int, default=None)
    parser.add_argument("--cross-adjacent-ew-vph", type=int, default=None)
    parser.add_argument("--cross-adjacent-ns-vph", type=int, default=None)
    parser.add_argument("--cross-width", type=int, default=None)

    args = parser.parse_args()

    temporal_preset = get_nearest_preset(
        args.grid_n,
        GRID_TEMPORAL_PRESETS,
        "temporal",
    )
    spatial_preset = get_nearest_preset(
        args.grid_n,
        GRID_SPATIAL_PRESETS,
        "spatial",
    )

    apply_defaults_from_preset(args, temporal_preset)
    apply_defaults_from_preset(args, spatial_preset)

    os.makedirs(args.out, exist_ok=True)

    N = args.grid_n

    if N < 2:
        raise ValueError("grid-n must be >= 2")

    T = float(args.sim_end)
    rng = np.random.default_rng(args.seed)

    print(f"[INFO] grid_n={N}, sim_end={T}, seed={args.seed}")
    print(f"[INFO] vehicle_sigma={args.vehicle_sigma}")
    print(
        f"[INFO] Platoons precursor={args.platoon_precursor_vph_min}-"
        f"{args.platoon_precursor_vph_max} vph, "
        f"heavy={args.platoon_heavy_vph_min}-{args.platoon_heavy_vph_max} vph"
    )
    print(
        f"[INFO] Bursty ON={args.bursty_on_min}-{args.bursty_on_max} vph, "
        f"blocks={args.bursty_on_block_min}-{args.bursty_on_block_max}s"
    )
    print(
        f"[INFO] Corridor wave vph={args.corridor_vph}, "
        f"side_vph={args.corridor_side_vph}"
    )
    print(
        f"[INFO] Cross centre EW/NS={args.cross_centre_ew_vph}/"
        f"{args.cross_centre_ns_vph}, adjacent EW/NS="
        f"{args.cross_adjacent_ew_vph}/{args.cross_adjacent_ns_vph}, "
        f"width={args.cross_width}"
    )

    builders = [
        (
            "t01_platoons_pulse_trains",
            lambda n, t, r: case_platoons_pulse_trains(
                n,
                t,
                r,
                heavy_pulses_per_train=args.platoon_heavy_pulses_per_train,
                light_pulses_per_train=args.platoon_light_pulses_per_train,
                precursor_width_min=args.platoon_precursor_width_min,
                precursor_width_max=args.platoon_precursor_width_max,
                precursor_vph_min=args.platoon_precursor_vph_min,
                precursor_vph_max=args.platoon_precursor_vph_max,
                precursor_to_main_gap_min=args.platoon_precursor_to_main_gap_min,
                precursor_to_main_gap_max=args.platoon_precursor_to_main_gap_max,
                heavy_pulse_gap_min=args.platoon_heavy_pulse_gap_min,
                heavy_pulse_gap_max=args.platoon_heavy_pulse_gap_max,
                light_pulse_gap_min=args.platoon_light_pulse_gap_min,
                light_pulse_gap_max=args.platoon_light_pulse_gap_max,
                heavy_width_min=args.platoon_heavy_width_min,
                heavy_width_max=args.platoon_heavy_width_max,
                light_width_min=args.platoon_light_width_min,
                light_width_max=args.platoon_light_width_max,
                heavy_vph_min=args.platoon_heavy_vph_min,
                heavy_vph_max=args.platoon_heavy_vph_max,
                light_vph_min=args.platoon_light_vph_min,
                light_vph_max=args.platoon_light_vph_max,
                train_gap_min=args.platoon_train_gap_min,
                train_gap_max=args.platoon_train_gap_max,
                jitter_frac=args.platoon_jitter_frac,
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
                on_block_min=args.bursty_on_block_min,
                on_block_max=args.bursty_on_block_max,
                off_block_min=args.bursty_off_block_min,
                off_block_max=args.bursty_off_block_max,
                p_on_dropout=args.bursty_p_on_dropout,
                p_off_false_burst=args.bursty_p_off_false_burst,
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
                corridor_pulse_width_min=args.corridor_pulse_width_min,
                corridor_pulse_width_max=args.corridor_pulse_width_max,
                corridor_pulse_gap_min=args.corridor_pulse_gap_min,
                corridor_pulse_gap_max=args.corridor_pulse_gap_max,
                corridor_jitter_frac=args.corridor_jitter_frac,
                side_vph=args.corridor_side_vph,
            ),
        ),
        (
            "t07_cross_orthogonal_axes",
            lambda n, t, r: case_cross_orthogonal_axes(
                n,
                t,
                r,
                cross_centre_ew_vph=args.cross_centre_ew_vph,
                cross_centre_ns_vph=args.cross_centre_ns_vph,
                cross_adjacent_ew_vph=args.cross_adjacent_ew_vph,
                cross_adjacent_ns_vph=args.cross_adjacent_ns_vph,
                cross_width=args.cross_width,
            ),
        ),
    ]

    for name, builder in builders:
        flows = builder(N, T, rng)

        filename = f"eval_trips_{name}_gridN{N}.xml"
        path = os.path.join(args.out, filename)

        write_case(
            path,
            flows,
            vehicle_sigma=args.vehicle_sigma,
        )

        print(f"Wrote {path} ({len(flows)} flows)")

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
