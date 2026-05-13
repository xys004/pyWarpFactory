import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory.analyzer.pipeline import analyze_metric, summarize_energy_conditions
from warpfactory.constants import C, G
from warpfactory.generator.warp_shell import create_warp_shell_metric
from warpfactory.recipes.fuchs_warp_shell import fuchs_constant_warp_shell_parameters


def _parse_float_list(value):
    return [float(item) for item in value.split(",") if item.strip()]


def _parse_int_list(value):
    return [int(float(item)) for item in value.split(",") if item.strip()]


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the Fuchs et al. matter shell / Warp Shell recipe by "
            "sweeping mass, smoothing, and shift velocity."
        )
    )
    parser.add_argument("--profile", choices=("quick", "original"), default="quick")
    parser.add_argument("--mass-factors", default="0.1,0.2,0.3333333333333333,0.45")
    parser.add_argument("--smooth-factors", default=None)
    parser.add_argument("--r-buffs", default=None)
    parser.add_argument("--density-smooth-ratios", default="1.79")
    parser.add_argument("--velocities", default="0")
    parser.add_argument("--num-vecs", type=int, default=40)
    parser.add_argument("--num-time-vecs", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--solver-method", choices=("christoffel", "warpfactory_direct"), default="christoffel")
    parser.add_argument("--energy-condition-method", choices=("standard", "warpfactory"), default="warpfactory")
    parser.add_argument("--output", default=None, help="Optional CSV output path.")
    parser.add_argument("--json", action="store_true")
    return parser


def run_case(base_params, mass_factor, smooth_factor, r_buff, density_smooth_ratio, velocity, args):
    mass = base_params["R2"] / (2.0 * G) * C**2 * mass_factor
    metric = create_warp_shell_metric(
        base_params["grid_size"],
        base_params["grid_scaling"],
        base_params["world_center"],
        mass=mass,
        r_inner=base_params["R1"],
        r_outer=base_params["R2"],
        r_buff=r_buff,
        sigma=base_params["sigma"],
        smooth_factor=smooth_factor,
        v_warp=velocity,
        do_warp=velocity != 0.0,
        density_smooth_ratio=density_smooth_ratio,
    )
    result = analyze_metric(
        metric,
        num_vecs=args.num_vecs,
        num_time_vecs=args.num_time_vecs,
        solver_method=args.solver_method,
        energy_condition_method=args.energy_condition_method,
    )
    summary = result.summary
    radius = (
        metric.coords["x"] * metric.coords["x"]
        + metric.coords["y"] * metric.coords["y"]
        + metric.coords["z"] * metric.coords["z"]
    ) ** 0.5
    shell_mask = (radius >= base_params["R1"]) & (radius <= base_params["R2"])
    exterior_mask = radius > base_params["R2"]
    shell_summary = summarize_energy_conditions(
        result.energy_conditions,
        mask=shell_mask,
        tolerance=args.tolerance,
    )
    exterior_summary = summarize_energy_conditions(
        result.energy_conditions,
        mask=exterior_mask,
        tolerance=args.tolerance,
    )
    return {
        "profile": args.profile,
        "mass_factor": mass_factor,
        "smooth_factor": smooth_factor,
        "r_buff": r_buff,
        "density_smooth_ratio": density_smooth_ratio,
        "v_warp": velocity,
        "mass_kg": mass,
        "M_end": float(np.nanmax(metric.params["M"])),
        "A_min": float(np.nanmin(metric.params["A"])),
        "A_max": float(np.nanmax(metric.params["A"])),
        "B_max": float(np.nanmax(metric.params["B"])),
        "Null_min": summary["Null_min"],
        "Weak_min": summary["Weak_min"],
        "Strong_min": summary["Strong_min"],
        "Dominant_min": summary["Dominant_min"],
        "Null_violating_fraction": summary["Null_violating_fraction"],
        "Weak_violating_fraction": summary["Weak_violating_fraction"],
        "Strong_violating_fraction": summary["Strong_violating_fraction"],
        "Dominant_violating_fraction": summary["Dominant_violating_fraction"],
        "Null_shell_min": shell_summary["Null_min"],
        "Null_shell_violating_fraction": shell_summary["Null_violating_fraction"],
        "Weak_shell_min": shell_summary["Weak_min"],
        "Weak_shell_violating_fraction": shell_summary["Weak_violating_fraction"],
        "Dominant_shell_min": shell_summary["Dominant_min"],
        "Dominant_shell_violating_fraction": shell_summary["Dominant_violating_fraction"],
        "Null_exterior_min": exterior_summary["Null_min"],
        "Null_exterior_violating_fraction": exterior_summary["Null_violating_fraction"],
    }


def run(args):
    base_params = fuchs_constant_warp_shell_parameters(args.profile)
    smooth_factors = (
        _parse_int_list(args.smooth_factors)
        if args.smooth_factors is not None
        else [base_params["smooth_factor"]]
    )
    r_buffs = (
        _parse_float_list(args.r_buffs)
        if args.r_buffs is not None
        else [base_params["Rbuff"]]
    )
    rows = []
    for smooth_factor in smooth_factors:
        for r_buff in r_buffs:
            for density_smooth_ratio in _parse_float_list(args.density_smooth_ratios):
                for mass_factor in _parse_float_list(args.mass_factors):
                    for velocity in _parse_float_list(args.velocities):
                        rows.append(
                            run_case(
                                base_params,
                                mass_factor,
                                smooth_factor,
                                r_buff,
                                density_smooth_ratio,
                                velocity,
                                args,
                            )
                        )
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_rows(rows):
    print(
        "{:>7s} {:>7s} {:>8s} {:>11s} {:>11s} {:>11s} {:>11s} {:>9s}".format(
            "mfac", "smooth", "v", "NEC_min", "WEC_min", "SEC_min", "DEC_min", "NEC_frac"
        )
    )
    for row in rows:
        print(
            "{mass_factor:7.3f} {smooth_factor:7d} {v_warp:8.3g} "
            "{Null_min:11.3e} {Weak_min:11.3e} {Strong_min:11.3e} "
            "{Dominant_min:11.3e} {Null_violating_fraction:9.3f} "
            "shell_NEC={Null_shell_violating_fraction:.3f} "
            "Rb={r_buff:g} ratio={density_smooth_ratio:g}".format(**row)
        )


def main():
    args = build_parser().parse_args()
    rows = run(args)
    if args.output:
        write_csv(args.output, rows)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print_rows(rows)


if __name__ == "__main__":
    main()
