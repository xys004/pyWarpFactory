import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory import analyze_metric
from warpfactory.generator.warp_shell import create_warp_shell_metric


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Verify a constant-velocity Warp Shell candidate with sampled "
            "energy-condition maps and optional optimized observer audit."
        )
    )
    parser.add_argument("--grid", type=int, default=12, help="Spatial grid size Nx=Ny=Nz.")
    parser.add_argument("--nt", type=int, default=5, help="Temporal grid size. Must be >=5.")
    parser.add_argument("--spacing", type=float, default=0.5, help="Spatial grid spacing.")
    parser.add_argument("--mass", type=float, default=1000.0, help="Total shell mass parameter.")
    parser.add_argument("--r-inner", type=float, default=1.0, help="Inner shell radius.")
    parser.add_argument("--r-outer", type=float, default=2.0, help="Outer shell radius.")
    parser.add_argument("--r-buff", type=float, default=0.25, help="Shift buffer distance.")
    parser.add_argument("--sigma", type=float, default=4.0, help="Shift sigmoid sharpness.")
    parser.add_argument("--smooth-factor", type=int, default=2, help="Shell smoothing factor.")
    parser.add_argument("--v-warp", type=float, default=0.1, help="Subluminal warp speed in factors of c.")
    parser.add_argument("--static", action="store_true", help="Disable the warp shift.")
    parser.add_argument("--num-vecs", type=int, default=12, help="Sampled observer directions.")
    parser.add_argument("--audit-points", type=int, default=2, help="Optimized audit points per condition.")
    parser.add_argument("--optimized-max-speed", type=float, default=0.95, help="Max observer speed for optimized audit.")
    parser.add_argument(
        "--flat-tolerance",
        type=float,
        default=None,
        help="Optional near-Minkowski numerical floor passed to solve_energy_tensor.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--energy-condition-method",
        choices=("standard", "warpfactory"),
        default="standard",
        help="Energy-condition convention to use.",
    )
    return parser


def run(args):
    grid_size = (args.nt, args.grid, args.grid, args.grid)
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
        mass=args.mass,
        r_inner=args.r_inner,
        r_outer=args.r_outer,
        r_buff=args.r_buff,
        sigma=args.sigma,
        smooth_factor=args.smooth_factor,
        v_warp=args.v_warp,
        do_warp=not args.static,
    )

    minkowski = np.diag([-1.0, 1.0, 1.0, 1.0]).reshape(4, 4, 1, 1, 1, 1)
    metric_deviation = float(np.nanmax(np.abs(metric.tensor - minkowski)))
    finite_metric = bool(np.all(np.isfinite(metric.tensor)))

    result = analyze_metric(
        metric,
        num_vecs=args.num_vecs,
        observer_mode="optimized",
        audit_points=args.audit_points,
        optimized_max_speed=args.optimized_max_speed,
        flat_tolerance=args.flat_tolerance,
        energy_condition_method=args.energy_condition_method,
    )

    pass_fail = {}
    for condition in ("Null", "Weak", "Strong", "Dominant"):
        pass_fail[condition] = {
            "sampled_pass": not result.has_violation(condition),
            "sampled_min": result.summary[f"{condition}_min"],
            "sampled_violating_fraction": result.summary[f"{condition}_violating_fraction"],
        }

        audit_entries = result.observer_audit.get("conditions", {}).get(condition, [])
        if audit_entries:
            optimized_min = min(entry["optimized_value"] for entry in audit_entries)
            pass_fail[condition]["optimized_audit_pass"] = optimized_min >= 0
            pass_fail[condition]["optimized_audit_min"] = optimized_min

    report = {
        "metric": "Comoving Warp Shell",
        "purpose": "Constant-velocity subluminal warp-shell candidate verification",
        "parameters": {
            "grid_size": grid_size,
            "grid_scale": grid_scale,
            "center": center,
            "mass": args.mass,
            "r_inner": args.r_inner,
            "r_outer": args.r_outer,
            "r_buff": args.r_buff,
            "sigma": args.sigma,
            "smooth_factor": args.smooth_factor,
            "v_warp": args.v_warp,
            "do_warp": not args.static,
            "flat_tolerance": args.flat_tolerance,
            "energy_condition_method": args.energy_condition_method,
        },
        "methodology": result.methodology,
        "numerics": {
            "finite_metric": finite_metric,
            "max_abs_metric_deviation_from_minkowski": metric_deviation,
            "note": (
                "For near-flat toy runs, roundoff-level metric deviations can be "
                "amplified by c^4/G in the stress-energy solve. Interpret very "
                "small-grid SI-unit runs as numerical smoke tests, not publication "
                "reproductions."
            ),
        },
        "energy_conditions": pass_fail,
        "observer_audit": result.observer_audit,
    }

    return report


def print_report(report):
    print("pyWarpFactory physical Warp Shell verification")
    print("=" * 52)
    print(f"Metric     : {report['metric']}")
    print(f"Purpose    : {report['purpose']}")
    print(f"Parameters : {report['parameters']}")
    print(f"Numerics   : {report['numerics']}")
    print()
    print("Energy-condition status")
    for condition, data in report["energy_conditions"].items():
        sampled = "PASS" if data["sampled_pass"] else "FAIL"
        print(
            f"- {condition:8s} sampled={sampled:4s} "
            f"min={data['sampled_min']:.4e} "
            f"violating_fraction={data['sampled_violating_fraction']:.4e}"
        )
        if "optimized_audit_min" in data:
            optimized = "PASS" if data["optimized_audit_pass"] else "FAIL"
            print(
                f"           optimized_audit={optimized:4s} "
                f"min={data['optimized_audit_min']:.4e}"
            )

    print()
    print("Methodology")
    for key, value in report["methodology"].items():
        print(f"- {key}: {value}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    report = run(args)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
