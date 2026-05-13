import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory import analyze_metric, summarize_energy_conditions
from warpfactory.analyzer.stress_energy_diagnostics import eulerian_stress_decomposition
from warpfactory.generator.warp_shell import create_warp_shell_metric
from warpfactory.recipes.fuchs_warp_shell import fuchs_constant_warp_shell_parameters


CONDITIONS = ("Null", "Weak", "Strong", "Dominant")


def parse_float_list(value):
    return [float(item) for item in value.split(",") if item.strip()]


def build_metric(profile, grid_scale_factor, v_warp, static=False, adm_shift_g00=False):
    params = fuchs_constant_warp_shell_parameters(
        profile=profile,
        grid_scale_factor=grid_scale_factor,
    )
    params["v_warp"] = 0.0 if static else float(v_warp)
    params["do_warp"] = not static and float(v_warp) != 0.0

    metric = create_warp_shell_metric(
        params["grid_size"],
        params["grid_scaling"],
        params["world_center"],
        mass=params["mass"],
        r_inner=params["R1"],
        r_outer=params["R2"],
        r_buff=params["Rbuff"],
        sigma=params["sigma"],
        smooth_factor=params["smooth_factor"],
        v_warp=params["v_warp"],
        do_warp=params["do_warp"],
        adm_shift_g00=adm_shift_g00,
    )
    metric.params["fuchs_recipe"] = params
    return metric, params


def radius(metric):
    return np.sqrt(metric.coords["x"] ** 2 + metric.coords["y"] ** 2 + metric.coords["z"] ** 2)


def mask_summary(conditions, mask):
    if not np.any(mask):
        return {f"{name}_{field}": None for name in CONDITIONS for field in ("min", "violating_fraction")}
    return summarize_energy_conditions(conditions, mask=mask)


def slice_summaries(metric, conditions, params):
    r = radius(metric)
    shell = (r >= params["R1"]) & (r <= params["R2"])
    z_coords = metric.coords["z"]
    x_coords = metric.coords["x"]
    y_coords = metric.coords["y"]

    summaries = {
        "all": mask_summary(conditions, np.ones_like(shell, dtype=bool)),
        "material_shell": mask_summary(conditions, shell),
        "z_slices_material_shell": [],
    }

    for z_index in range(shell.shape[3]):
        z_mask = np.zeros_like(shell, dtype=bool)
        z_mask[:, :, :, z_index] = True
        summaries["z_slices_material_shell"].append(
            {
                "z_index": z_index,
                "z_value": float(np.nanmedian(z_coords[z_mask])),
                "summary": mask_summary(conditions, shell & z_mask),
            }
        )

    central_z_index = int(np.argmin(np.abs(np.nanmedian(z_coords, axis=(0, 1, 2)))))
    central_x_mask = np.abs(x_coords) == np.nanmin(np.abs(x_coords))
    central_y_mask = np.abs(y_coords) == np.nanmin(np.abs(y_coords))
    central_z_mask = np.zeros_like(shell, dtype=bool)
    central_z_mask[:, :, :, central_z_index] = True

    summaries["central_z_material_shell"] = mask_summary(conditions, shell & central_z_mask)
    summaries["central_xz_line_material_shell"] = mask_summary(
        conditions,
        shell & central_z_mask & central_y_mask,
    )
    summaries["central_yz_line_material_shell"] = mask_summary(
        conditions,
        shell & central_z_mask & central_x_mask,
    )
    return summaries


def flux_summary(eulerian_tensor, metric, params):
    r = radius(metric)
    shell = (r >= params["R1"]) & (r <= params["R2"])
    stress = eulerian_stress_decomposition(eulerian_tensor, coords=metric.coords)

    def summarize(values):
        selected = values[shell]
        return {
            "min": float(np.nanmin(selected)),
            "max": float(np.nanmax(selected)),
            "median": float(np.nanmedian(selected)),
            "abs_max": float(np.nanmax(np.abs(selected))),
        }

    return {
        "rho": summarize(stress["rho"]),
        "flux_norm": summarize(stress["flux_norm"]),
        "rho_plus_p_min": summarize(stress["rho_plus_p_min"]),
        "T01": summarize(eulerian_tensor[0, 1]),
        "T02": summarize(eulerian_tensor[0, 2]),
        "T03": summarize(eulerian_tensor[0, 3]),
    }


def analyze_case(args, v_warp, solver_method):
    metric, params = build_metric(
        args.profile,
        args.grid_scale_factor,
        v_warp,
        static=(v_warp == 0.0),
        adm_shift_g00=args.adm_shift_g00,
    )
    result = analyze_metric(
        metric,
        num_vecs=args.num_vecs,
        num_time_vecs=args.num_time_vecs,
        energy_condition_method=args.energy_condition_method,
        solver_method=solver_method,
    )
    return {
        "v_warp": v_warp,
        "solver_method": solver_method,
        "params": params,
        "global_summary": result.summary,
        "slice_summary": slice_summaries(metric, result.energy_conditions, params),
        "flux_summary": flux_summary(result.eulerian_energy_tensor, metric, params),
    }


def run(args):
    velocities = parse_float_list(args.velocities)
    rows = []
    for solver_method in args.solver_methods.split(","):
        solver_method = solver_method.strip()
        for v_warp in velocities:
            rows.append(analyze_case(args, v_warp, solver_method))
    report = {
        "purpose": "Fuchs W1 velocity and slice audit",
        "settings": {
            "profile": args.profile,
            "grid_scale_factor": args.grid_scale_factor,
            "velocities": velocities,
            "solver_methods": args.solver_methods,
            "num_vecs": args.num_vecs,
            "num_time_vecs": args.num_time_vecs,
            "energy_condition_method": args.energy_condition_method,
            "adm_shift_g00": args.adm_shift_g00,
        },
        "cases": rows,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def print_report(report):
    print("Fuchs W1 velocity/slice audit")
    print("=" * 34)
    for case in report["cases"]:
        shell = case["slice_summary"]["material_shell"]
        central = case["slice_summary"]["central_z_material_shell"]
        flux = case["flux_summary"]
        print(
            f"{case['solver_method']:18s} v={case['v_warp']:<8g} "
            f"shell NEC={shell['Null_min']:.4e} frac={shell['Null_violating_fraction']:.4f} "
            f"central-z NEC={central['Null_min']:.4e} frac={central['Null_violating_fraction']:.4f} "
            f"T01_absmax={flux['T01']['abs_max']:.4e} rho_min={flux['rho']['min']:.4e}"
        )


def build_parser():
    parser = argparse.ArgumentParser(description="Sweep Fuchs W1 vWarp and summarize slices.")
    parser.add_argument("--profile", choices=("quick", "original"), default="quick")
    parser.add_argument("--grid-scale-factor", type=float, default=None)
    parser.add_argument("--velocities", default="0,0.001,0.002,0.005,0.01,0.015,0.02")
    parser.add_argument("--solver-methods", default="christoffel,warpfactory_direct")
    parser.add_argument("--num-vecs", type=int, default=4)
    parser.add_argument("--num-time-vecs", type=int, default=10)
    parser.add_argument("--energy-condition-method", choices=("standard", "warpfactory"), default="warpfactory")
    parser.add_argument("--adm-shift-g00", action="store_true")
    parser.add_argument("--output", default="outputs/fuchs_w1_velocity_slices.json")
    parser.add_argument("--json", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    report = run(args)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
