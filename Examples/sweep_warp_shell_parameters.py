import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory import analyze_metric
from warpfactory.generator.warp_shell import create_warp_shell_metric


def parse_float_list(value):
    return [float(item) for item in value.split(",") if item.strip()]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Small transparent parameter sweep for WarpShell energy-condition calibration."
    )
    parser.add_argument("--masses", default="0,1e20,1e24,1e26,1e27", help="Comma-separated mass values.")
    parser.add_argument("--velocities", default="0,0.01,0.05", help="Comma-separated v_warp values.")
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--spacing", type=float, default=0.5)
    parser.add_argument("--num-vecs", type=int, default=4)
    parser.add_argument("--r-inner", type=float, default=1.0)
    parser.add_argument("--r-outer", type=float, default=2.0)
    parser.add_argument("--r-buff", type=float, default=0.25)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--smooth-factor", type=int, default=2)
    parser.add_argument("--flat-tolerance", type=float, default=None)
    parser.add_argument("--energy-condition-method", choices=("standard", "warpfactory"), default="standard")
    parser.add_argument("--json", action="store_true")
    return parser


def analyze_case(args, mass, velocity):
    grid_size = (5, args.grid, args.grid, args.grid)
    grid_scale = (1.0, args.spacing, args.spacing, args.spacing)
    center = (
        0.0,
        args.spacing * args.grid / 2.0,
        args.spacing * args.grid / 2.0,
        args.spacing * args.grid / 2.0,
    )

    metric = create_warp_shell_metric(
        grid_size,
        grid_scale,
        center,
        mass=mass,
        r_inner=args.r_inner,
        r_outer=args.r_outer,
        r_buff=args.r_buff,
        sigma=args.sigma,
        smooth_factor=args.smooth_factor,
        v_warp=velocity,
        do_warp=velocity != 0.0,
    )

    eta = np.diag([-1.0, 1.0, 1.0, 1.0]).reshape(4, 4, 1, 1, 1, 1)
    metric_deviation = float(np.nanmax(np.abs(metric.tensor - eta)))
    result = analyze_metric(
        metric,
        num_vecs=args.num_vecs,
        flat_tolerance=args.flat_tolerance,
        energy_condition_method=args.energy_condition_method,
    )

    return {
        "mass": mass,
        "v_warp": velocity,
        "metric_deviation": metric_deviation,
        "A_min": float(np.nanmin(metric.params["A"])),
        "A_max": float(np.nanmax(metric.params["A"])),
        "B_min": float(np.nanmin(metric.params["B"])),
        "B_max": float(np.nanmax(metric.params["B"])),
        "Null_min": result.summary["Null_min"],
        "Weak_min": result.summary["Weak_min"],
        "Strong_min": result.summary["Strong_min"],
        "Dominant_min": result.summary["Dominant_min"],
        "Null_violating_fraction": result.summary["Null_violating_fraction"],
        "Weak_violating_fraction": result.summary["Weak_violating_fraction"],
        "Strong_violating_fraction": result.summary["Strong_violating_fraction"],
        "Dominant_violating_fraction": result.summary["Dominant_violating_fraction"],
        "flat_floor_applied": result.methodology["solver_diagnostics"].get("flat_floor_applied", False),
    }


def run(args):
    rows = []
    for mass in parse_float_list(args.masses):
        for velocity in parse_float_list(args.velocities):
            rows.append(analyze_case(args, mass, velocity))
    return rows


def print_rows(rows):
    header = (
        "mass",
        "v",
        "metric_dev",
        "A_min",
        "B_max",
        "NEC_min",
        "WEC_min",
        "SEC_min",
        "DEC_min",
        "NEC_frac",
    )
    print("{:>12s} {:>8s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>10s}".format(*header))
    for row in rows:
        print(
            "{mass:12.3e} {v_warp:8.3g} {metric_deviation:12.3e} {A_min:12.3e} {B_max:12.3e} "
            "{Null_min:12.3e} {Weak_min:12.3e} {Strong_min:12.3e} {Dominant_min:12.3e} "
            "{Null_violating_fraction:10.3f}".format(**row)
        )


def main():
    parser = build_parser()
    args = parser.parse_args()
    rows = run(args)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print_rows(rows)


if __name__ == "__main__":
    main()
