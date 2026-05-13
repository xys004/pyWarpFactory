import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory import analyze_metric
from warpfactory.analyzer.stress_energy_diagnostics import eulerian_stress_decomposition
from warpfactory.generator.warp_shell import create_warp_shell_metric


def build_parser():
    parser = argparse.ArgumentParser(
        description="Diagnose Eulerian stress-energy components for Warp Shell cases."
    )
    parser.add_argument("--grid", type=int, default=12)
    parser.add_argument("--nt", type=int, default=5)
    parser.add_argument("--spacing", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--mass", type=float, default=1e26)
    parser.add_argument("--velocity", type=float, default=0.1)
    parser.add_argument("--r-inner", type=float, default=1.0)
    parser.add_argument("--r-outer", type=float, default=2.0)
    parser.add_argument("--r-buff", type=float, default=0.25)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--smooth-factor", type=int, default=2)
    parser.add_argument("--num-vecs", type=int, default=6)
    parser.add_argument("--radial-bins", type=int, default=8)
    parser.add_argument("--flat-tolerance", type=float, default=None)
    parser.add_argument("--energy-condition-method", choices=("warpfactory", "standard"), default="warpfactory")
    parser.add_argument("--adm-shift-g00", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", default=None)
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


def _case_params(args, case):
    if case == "static_mass":
        return args.mass, 0.0, False
    if case == "pure_shift":
        return 0.0, args.velocity, True
    if case == "full_shell":
        return args.mass, args.velocity, True
    raise ValueError(f"Unknown case {case!r}.")


def _build_metric(args, case):
    grid_size, grid_scale, center = _grid(args)
    mass, velocity, do_warp = _case_params(args, case)
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


def _summary(values):
    values = np.real_if_close(values)
    return {
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "median": float(np.nanmedian(values)),
        "abs_max": float(np.nanmax(np.abs(values))),
    }


def _radial_rows(radius, maps, bins):
    finite = np.isfinite(radius)
    r_values = radius[finite]
    edges = np.linspace(float(np.nanmin(r_values)), float(np.nanmax(r_values)), bins + 1)
    rows = []
    for start, stop in zip(edges[:-1], edges[1:]):
        mask = finite & (radius >= start) & (radius < stop)
        if not np.any(mask):
            continue
        row = {
            "r_min": float(start),
            "r_max": float(stop),
            "r_mid": float(0.5 * (start + stop)),
            "points": int(np.count_nonzero(mask)),
        }
        for name, values in maps.items():
            row[f"{name}_median"] = float(np.nanmedian(values[mask]))
            row[f"{name}_min"] = float(np.nanmin(values[mask]))
            row[f"{name}_max"] = float(np.nanmax(values[mask]))
            if name in {"Null", "Weak", "Strong", "Dominant", "rho_plus_p_min", "rho_plus_radial_pressure", "rho_plus_tangential_pressure"}:
                row[f"{name}_negative_fraction"] = float(np.mean(values[mask] < 0.0))
        rows.append(row)
    return rows


def analyze_case(args, case):
    metric, case_params = _build_metric(args, case)
    result = analyze_metric(
        metric,
        num_vecs=args.num_vecs,
        flat_tolerance=args.flat_tolerance,
        energy_condition_method=args.energy_condition_method,
    )
    stress = eulerian_stress_decomposition(result.eulerian_energy_tensor, coords=metric.coords)

    maps = {
        "rho": stress["rho"],
        "flux_norm": stress["flux_norm"],
        "flux_over_abs_rho": stress["flux_over_abs_rho"],
        "pressure_min": stress["pressure_min"],
        "pressure_mid": stress["pressure_mid"],
        "pressure_max": stress["pressure_max"],
        "pressure_iso": stress["pressure_iso"],
        "anisotropy": stress["anisotropy"],
        "radial_pressure": stress["radial_pressure"],
        "tangential_pressure": stress["tangential_pressure"],
        "radial_flux": stress["radial_flux"],
        "rho_plus_p_min": stress["rho_plus_p_min"],
        "rho_plus_radial_pressure": stress["rho_plus_radial_pressure"],
        "rho_plus_tangential_pressure": stress["rho_plus_tangential_pressure"],
        "Null": result.energy_conditions["Null"],
        "Weak": result.energy_conditions["Weak"],
        "Strong": result.energy_conditions["Strong"],
        "Dominant": result.energy_conditions["Dominant"],
    }

    return {
        "case": case,
        "parameters": case_params,
        "global": {name: _summary(values) for name, values in maps.items()},
        "radial_profile": _radial_rows(stress["radius"], maps, args.radial_bins),
    }


def run(args):
    report = {
        "purpose": "Eulerian stress-energy diagnostic for Warp Shell metrics",
        "common_parameters": {
            "grid": args.grid,
            "nt": args.nt,
            "spacing": args.spacing,
            "dt": args.dt,
            "mass": args.mass,
            "velocity": args.velocity,
            "r_inner": args.r_inner,
            "r_outer": args.r_outer,
            "smooth_factor": args.smooth_factor,
            "energy_condition_method": args.energy_condition_method,
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
    print("pyWarpFactory Warp Shell stress-energy diagnosis")
    print("=" * 56)
    print(f"Common parameters: {report['common_parameters']}")
    print()
    for case in report["cases"]:
        print(f"Case: {case['case']}")
        for name in (
            "rho",
            "flux_norm",
            "pressure_min",
            "pressure_max",
            "radial_pressure",
            "tangential_pressure",
            "rho_plus_p_min",
            "rho_plus_radial_pressure",
            "rho_plus_tangential_pressure",
            "Null",
        ):
            values = case["global"][name]
            print(
                f"  {name:28s} min={values['min']:.4e} "
                f"max={values['max']:.4e} median={values['median']:.4e}"
            )
        print("  radial bins:")
        for row in case["radial_profile"]:
            print(
                f"    r=[{row['r_min']:.3g},{row['r_max']:.3g}] "
                f"rho_med={row['rho_median']:.3e} "
                f"p_r_med={row['radial_pressure_median']:.3e} "
                f"p_t_med={row['tangential_pressure_median']:.3e} "
                f"rho+pmin_neg={row['rho_plus_p_min_negative_fraction']:.3f} "
                f"NEC_frac={row['Null_negative_fraction']:.3f}"
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
