import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory import analyze_metric
from warpfactory.recipes import create_fuchs_constant_warp_shell, fuchs_constant_warp_shell_parameters


def build_parser():
    parser = argparse.ArgumentParser(
        description="Reproduce the WarpFactory W1 constant-velocity Warp Shell workflow in Python."
    )
    parser.add_argument(
        "--profile",
        choices=("quick", "original"),
        default="quick",
        help="quick is for smoke tests; original follows W1_Warp_Shell.mlx dimensions.",
    )
    parser.add_argument(
        "--grid-scale-factor",
        type=float,
        default=None,
        help="Override the MATLAB spaceScale parameter.",
    )
    parser.add_argument("--num-vecs", type=int, default=8)
    parser.add_argument("--num-time-vecs", type=int, default=10)
    parser.add_argument(
        "--energy-condition-method",
        choices=("standard", "warpfactory"),
        default="warpfactory",
        help="Use warpfactory for direct convention comparison with MATLAB.",
    )
    parser.add_argument("--solver-method", choices=("christoffel", "warpfactory_direct"), default="christoffel")
    parser.add_argument("--observer-mode", choices=("sampled", "optimized"), default="sampled")
    parser.add_argument("--audit-points", type=int, default=2)
    parser.add_argument("--flat-tolerance", type=float, default=None)
    parser.add_argument("--json", action="store_true")
    return parser


def run(args):
    params = fuchs_constant_warp_shell_parameters(
        profile=args.profile,
        grid_scale_factor=args.grid_scale_factor,
    )
    metric = create_fuchs_constant_warp_shell(
        profile=args.profile,
        grid_scale_factor=args.grid_scale_factor,
    )
    result = analyze_metric(
        metric,
        num_vecs=args.num_vecs,
        num_time_vecs=args.num_time_vecs,
        observer_mode=args.observer_mode,
        audit_points=args.audit_points,
        flat_tolerance=args.flat_tolerance,
        energy_condition_method=args.energy_condition_method,
        solver_method=args.solver_method,
    )

    eta = np.diag([-1.0, 1.0, 1.0, 1.0]).reshape(4, 4, 1, 1, 1, 1)
    metric_deviation = float(np.nanmax(np.abs(metric.tensor - eta)))
    report = {
        "recipe": "Fuchs et al. / WarpFactory W1_Warp_Shell.mlx",
        "profile": args.profile,
        "parameters": params,
        "methodology": result.methodology,
        "metric_diagnostics": {
            "shape": metric.tensor.shape,
            "finite": bool(np.all(np.isfinite(metric.tensor))),
            "max_abs_deviation_from_minkowski": metric_deviation,
            "g_tx_min": float(np.nanmin(metric.tensor[0, 1])),
            "g_tx_max": float(np.nanmax(metric.tensor[0, 1])),
            "A_min": float(np.nanmin(metric.params["A"])),
            "A_max": float(np.nanmax(metric.params["A"])),
            "B_min": float(np.nanmin(metric.params["B"])),
            "B_max": float(np.nanmax(metric.params["B"])),
        },
        "energy_conditions": {},
        "observer_audit": result.observer_audit,
    }

    for condition in ("Null", "Weak", "Strong", "Dominant"):
        report["energy_conditions"][condition] = {
            "min": result.summary[f"{condition}_min"],
            "max": result.summary[f"{condition}_max"],
            "violating_points": result.summary[f"{condition}_violating_points"],
            "violating_fraction": result.summary[f"{condition}_violating_fraction"],
            "pass_sampled": not result.has_violation(condition),
        }

    return report


def print_report(report):
    print("pyWarpFactory reproduction: constant-velocity Warp Shell")
    print("=" * 62)
    print(f"Recipe  : {report['recipe']}")
    print(f"Profile : {report['profile']}")
    print(f"Grid    : {report['parameters']['grid_size']}")
    print(f"Scale   : {report['parameters']['grid_scaling']}")
    print(f"Mass    : {report['parameters']['mass']:.6e}")
    print(f"vWarp   : {report['parameters']['v_warp']}")
    print(f"Method  : {report['methodology']['energy_condition_mode']}")
    print()
    print("Metric diagnostics")
    for key, value in report["metric_diagnostics"].items():
        print(f"- {key}: {value}")
    print()
    print("Energy conditions")
    for condition, data in report["energy_conditions"].items():
        status = "PASS" if data["pass_sampled"] else "FAIL"
        print(
            f"- {condition:8s} {status:4s} "
            f"min={data['min']:.4e} max={data['max']:.4e} "
            f"violating_fraction={data['violating_fraction']:.4e}"
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
