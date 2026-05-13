import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory import analyze_metric
from warpfactory.analyzer.spatial_geometry import spatial_geometry_summary
from warpfactory.generator.warp_shell import create_warp_shell_metric


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose Warp Shell energy-condition failures by separating "
            "static mass, pure shift, and full shell cases."
        )
    )
    parser.add_argument("--grid", type=int, default=12, help="Spatial grid size Nx=Ny=Nz.")
    parser.add_argument("--nt", type=int, default=5, help="Temporal grid size.")
    parser.add_argument("--spacing", type=float, default=0.5, help="Spatial grid spacing.")
    parser.add_argument("--dt", type=float, default=1.0, help="Temporal grid spacing.")
    parser.add_argument("--mass", type=float, default=1.0e26, help="Mass for static/full shell cases.")
    parser.add_argument("--velocity", type=float, default=0.1, help="v_warp for shift/full cases.")
    parser.add_argument("--r-inner", type=float, default=1.0)
    parser.add_argument("--r-outer", type=float, default=2.0)
    parser.add_argument("--r-buff", type=float, default=0.25)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--smooth-factor", type=int, default=2)
    parser.add_argument("--num-vecs", type=int, default=6)
    parser.add_argument("--radial-bins", type=int, default=10)
    parser.add_argument(
        "--skip-spatial-curvature",
        action="store_true",
        help="Skip numerical 3D Ricci scalar calculation for faster large sweeps.",
    )
    parser.add_argument("--flat-tolerance", type=float, default=None)
    parser.add_argument("--observer-mode", choices=("sampled", "optimized"), default="sampled")
    parser.add_argument("--audit-points", type=int, default=1)
    parser.add_argument("--optimized-max-speed", type=float, default=0.95)
    parser.add_argument("--energy-condition-method", choices=("warpfactory", "standard"), default="warpfactory")
    parser.add_argument(
        "--adm-shift-g00",
        action="store_true",
        help=(
            "Experimental: include beta_i beta^i in g00 after adding shift, "
            "so the static alpha remains the ADM lapse. Default preserves "
            "MATLAB WarpFactory parity."
        ),
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser


def _grid(args):
    grid_size = (args.nt, args.grid, args.grid, args.grid)
    grid_scale = (args.dt, args.spacing, args.spacing, args.spacing)
    center = (
        0.0,
        args.spacing * args.grid / 2.0,
        args.spacing * args.grid / 2.0,
        args.spacing * args.grid / 2.0,
    )
    return grid_size, grid_scale, center


def _radius(metric):
    x = metric.coords["x"]
    y = metric.coords["y"]
    z = metric.coords["z"]
    return np.sqrt(x**2 + y**2 + z**2)


def _radial_profile(values, radius, bins):
    finite = np.isfinite(values) & np.isfinite(radius)
    if not np.any(finite):
        return []

    r_values = radius[finite]
    v_values = values[finite]
    edges = np.linspace(float(np.nanmin(r_values)), float(np.nanmax(r_values)), bins + 1)
    rows = []
    for start, stop in zip(edges[:-1], edges[1:]):
        mask = (r_values >= start) & (r_values < stop)
        if not np.any(mask):
            continue
        rows.append(
            {
                "r_min": float(start),
                "r_max": float(stop),
                "points": int(np.count_nonzero(mask)),
                "min": float(np.nanmin(v_values[mask])),
                "median": float(np.nanmedian(v_values[mask])),
                "violating_fraction": float(np.mean(v_values[mask] < 0.0)),
            }
        )
    return rows


def _condition_report(values, radius, shell_mask, bins):
    min_index = tuple(int(i) for i in np.unravel_index(np.nanargmin(values), values.shape))
    violations = values < 0.0
    violation_count = int(np.count_nonzero(violations))
    shell_violation_count = int(np.count_nonzero(violations & shell_mask))
    return {
        "min": float(values[min_index]),
        "max": float(np.nanmax(values)),
        "violating_points": violation_count,
        "violating_fraction": float(np.mean(violations)),
        "min_index": min_index,
        "r_at_min": float(radius[min_index]),
        "shell_fraction_of_violations": (
            float(shell_violation_count / violation_count) if violation_count else 0.0
        ),
        "radial_profile": _radial_profile(values, radius, bins),
    }


def _build_metric(args, case):
    grid_size, grid_scale, center = _grid(args)
    if case == "static_mass":
        mass = args.mass
        velocity = 0.0
        do_warp = False
    elif case == "pure_shift":
        mass = 0.0
        velocity = args.velocity
        do_warp = True
    elif case == "full_shell":
        mass = args.mass
        velocity = args.velocity
        do_warp = True
    else:
        raise ValueError(f"Unknown case {case!r}.")

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
        do_warp=do_warp,
        adm_shift_g00=args.adm_shift_g00,
    )
    return metric, {"mass": mass, "v_warp": velocity, "do_warp": do_warp}


def _adm_lapse_shift(metric):
    g = metric.tensor
    gamma = g[1:, 1:]
    beta_down = g[0, 1:]
    gamma_T = np.moveaxis(gamma, [0, 1], [-2, -1])
    gamma_up_T = np.linalg.inv(gamma_T)
    gamma_up = np.moveaxis(gamma_up_T, [-2, -1], [0, 1])
    beta_up = np.einsum("ij...,j...->i...", gamma_up, beta_down)
    beta_sq = np.sum(beta_down * beta_up, axis=0)
    alpha_sq = beta_sq - g[0, 0]
    return np.sqrt(np.maximum(alpha_sq, 0.0)), beta_sq


def analyze_case(args, case):
    metric, case_params = _build_metric(args, case)
    result = analyze_metric(
        metric,
        num_vecs=args.num_vecs,
        observer_mode=args.observer_mode,
        audit_points=args.audit_points,
        optimized_max_speed=args.optimized_max_speed,
        flat_tolerance=args.flat_tolerance,
        energy_condition_method=args.energy_condition_method,
    )

    radius = _radius(metric)
    shell_mask = (radius >= args.r_inner) & (radius <= args.r_outer)
    eta = np.diag([-1.0, 1.0, 1.0, 1.0]).reshape(4, 4, 1, 1, 1, 1)
    adm_alpha, beta_sq = _adm_lapse_shift(metric)
    intended_alpha = np.sqrt(np.maximum(-metric.tensor[0, 0], 0.0))
    spatial_summary = spatial_geometry_summary(
        metric,
        compute_curvature=not args.skip_spatial_curvature,
    )

    return {
        "case": case,
        "parameters": case_params,
        "metric_diagnostics": {
            "max_abs_deviation_from_minkowski": float(np.nanmax(np.abs(metric.tensor - eta))),
            "g00_min": float(np.nanmin(metric.tensor[0, 0])),
            "g00_max": float(np.nanmax(metric.tensor[0, 0])),
            "g01_min": float(np.nanmin(metric.tensor[0, 1])),
            "g01_max": float(np.nanmax(metric.tensor[0, 1])),
            "A_min": float(np.nanmin(metric.params["A"])),
            "A_max": float(np.nanmax(metric.params["A"])),
            "B_min": float(np.nanmin(metric.params["B"])),
            "B_max": float(np.nanmax(metric.params["B"])),
            "adm_alpha_min": float(np.nanmin(adm_alpha)),
            "adm_alpha_max": float(np.nanmax(adm_alpha)),
            "sqrt_minus_g00_min": float(np.nanmin(intended_alpha)),
            "sqrt_minus_g00_max": float(np.nanmax(intended_alpha)),
            "beta_sq_max": float(np.nanmax(beta_sq)),
        },
        "spatial_hypersurface": spatial_summary,
        "solver_diagnostics": result.methodology["solver_diagnostics"],
        "energy_conditions": {
            name: _condition_report(values, radius, shell_mask, args.radial_bins)
            for name, values in result.energy_conditions.items()
        },
        "observer_audit": result.observer_audit,
    }


def run(args):
    report = {
        "purpose": "Warp Shell component energy-condition diagnosis",
        "common_parameters": {
            "grid": args.grid,
            "nt": args.nt,
            "spacing": args.spacing,
            "dt": args.dt,
            "r_inner": args.r_inner,
            "r_outer": args.r_outer,
            "r_buff": args.r_buff,
            "sigma": args.sigma,
            "smooth_factor": args.smooth_factor,
            "num_vecs": args.num_vecs,
            "flat_tolerance": args.flat_tolerance,
            "energy_condition_method": args.energy_condition_method,
            "observer_mode": args.observer_mode,
            "adm_shift_g00": args.adm_shift_g00,
        },
        "cases": [
            analyze_case(args, "static_mass"),
            analyze_case(args, "pure_shift"),
            analyze_case(args, "full_shell"),
        ],
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    return report


def print_report(report):
    print("pyWarpFactory Warp Shell component diagnosis")
    print("=" * 54)
    print(f"Common parameters: {report['common_parameters']}")
    print()
    for case in report["cases"]:
        print(f"Case: {case['case']}")
        print(f"  parameters      : {case['parameters']}")
        print(f"  metric deviation: {case['metric_diagnostics']['max_abs_deviation_from_minkowski']:.4e}")
        print(
            "  g00 range       : "
            f"{case['metric_diagnostics']['g00_min']:.4e}, "
            f"{case['metric_diagnostics']['g00_max']:.4e}"
        )
        print(
            "  g01 range       : "
            f"{case['metric_diagnostics']['g01_min']:.4e}, "
            f"{case['metric_diagnostics']['g01_max']:.4e}"
        )
        print(
            "  ADM alpha range : "
            f"{case['metric_diagnostics']['adm_alpha_min']:.4e}, "
            f"{case['metric_diagnostics']['adm_alpha_max']:.4e}"
        )
        print(
            "  sqrt(-g00) range: "
            f"{case['metric_diagnostics']['sqrt_minus_g00_min']:.4e}, "
            f"{case['metric_diagnostics']['sqrt_minus_g00_max']:.4e}"
        )
        print(f"  beta^2 max      : {case['metric_diagnostics']['beta_sq_max']:.4e}")
        hypersurface = case["spatial_hypersurface"]
        print(
            "  gamma-I max     : "
            f"{hypersurface['max_abs_gamma_minus_identity']:.4e}"
        )
        print(
            "  gamma eig range : "
            f"{hypersurface['gamma_eigen_min']:.4e}, "
            f"{hypersurface['gamma_eigen_max']:.4e}"
        )
        if "spatial_ricci_scalar_abs_max" in hypersurface:
            print(
                "  3-Ricci range   : "
                f"{hypersurface['spatial_ricci_scalar_min']:.4e}, "
                f"{hypersurface['spatial_ricci_scalar_max']:.4e}"
            )
        for condition, data in case["energy_conditions"].items():
            status = "FAIL" if data["min"] < 0.0 else "PASS"
            print(
                f"  - {condition:8s} {status:4s} min={data['min']:.4e} "
                f"frac={data['violating_fraction']:.4e} "
                f"r_min={data['r_at_min']:.4e} "
                f"shell_share={data['shell_fraction_of_violations']:.4e}"
            )
        print()


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
