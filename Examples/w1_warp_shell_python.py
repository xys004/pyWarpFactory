import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory import analyze_metric
from warpfactory.recipes import create_fuchs_constant_warp_shell, fuchs_constant_warp_shell_parameters


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Pure-Python reproduction scaffold for WarpFactory's "
            "Examples/4 Warp Shell/W1_Warp_Shell.mlx workflow."
        )
    )
    parser.add_argument("--profile", choices=("quick", "original"), default="quick")
    parser.add_argument("--grid-scale-factor", type=float, default=None)
    parser.add_argument("--num-vecs", type=int, default=8)
    parser.add_argument("--num-time-vecs", type=int, default=10)
    parser.add_argument("--energy-condition-method", choices=("standard", "warpfactory"), default="warpfactory")
    parser.add_argument("--solver-method", choices=("christoffel", "warpfactory_direct"), default="christoffel")
    parser.add_argument("--observer-mode", choices=("sampled", "optimized"), default="sampled")
    parser.add_argument("--audit-points", type=int, default=2)
    parser.add_argument("--flat-tolerance", type=float, default=None)
    parser.add_argument("--output-dir", default="outputs/w1_warp_shell_python")
    parser.add_argument("--plots", action="store_true", help="Save simple PNG slices if matplotlib is available.")
    return parser


def center_slice(array):
    if array.ndim != 4:
        raise ValueError(f"Expected a 4D grid array, got shape {array.shape}.")
    return array[0, :, :, array.shape[3] // 2]


def save_optional_plots(output_dir, metric, result):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping plots.")
        return

    plots = {
        "metric_g00.png": center_slice(metric.tensor[0, 0]),
        "metric_g01.png": center_slice(metric.tensor[0, 1]),
        "energy_density.png": center_slice(result.eulerian_energy_tensor[0, 0]),
        "null_condition.png": center_slice(result.energy_conditions["Null"]),
        "weak_condition.png": center_slice(result.energy_conditions["Weak"]),
        "strong_condition.png": center_slice(result.energy_conditions["Strong"]),
        "dominant_condition.png": center_slice(result.energy_conditions["Dominant"]),
    }

    for filename, values in plots.items():
        plt.figure(figsize=(6, 5))
        plt.imshow(values.T, origin="lower")
        plt.colorbar()
        plt.title(filename.replace("_", " ").replace(".png", ""))
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=140)
        plt.close()


def summarize_result(metric, result, recipe_params):
    eta = np.diag([-1.0, 1.0, 1.0, 1.0]).reshape(4, 4, 1, 1, 1, 1)
    summary = {
        "source": "WarpFactory Examples/4 Warp Shell/W1_Warp_Shell.mlx",
        "status": (
            "Python scaffold for reproduction. Passing all energy conditions "
            "is not yet expected until MATLAB parity checks are completed."
        ),
        "recipe_parameters": recipe_params,
        "methodology": result.methodology,
        "metric_diagnostics": {
            "tensor_shape": metric.tensor.shape,
            "finite_metric": bool(np.all(np.isfinite(metric.tensor))),
            "max_abs_deviation_from_minkowski": float(np.nanmax(np.abs(metric.tensor - eta))),
            "g00_min": float(np.nanmin(metric.tensor[0, 0])),
            "g00_max": float(np.nanmax(metric.tensor[0, 0])),
            "g01_min": float(np.nanmin(metric.tensor[0, 1])),
            "g01_max": float(np.nanmax(metric.tensor[0, 1])),
            "A_min": float(np.nanmin(metric.params["A"])),
            "A_max": float(np.nanmax(metric.params["A"])),
            "B_min": float(np.nanmin(metric.params["B"])),
            "B_max": float(np.nanmax(metric.params["B"])),
        },
        "energy_conditions": {},
        "observer_audit": result.observer_audit,
    }

    for condition in ("Null", "Weak", "Strong", "Dominant"):
        summary["energy_conditions"][condition] = {
            "min": result.summary[f"{condition}_min"],
            "max": result.summary[f"{condition}_max"],
            "violating_points": result.summary[f"{condition}_violating_points"],
            "violating_fraction": result.summary[f"{condition}_violating_fraction"],
            "pass": not result.has_violation(condition),
        }

    return summary


def save_arrays(output_dir, metric, result):
    np.savez_compressed(
        output_dir / "w1_python_arrays.npz",
        metric_tensor=metric.tensor,
        eulerian_energy_tensor=result.eulerian_energy_tensor,
        null=result.energy_conditions["Null"],
        weak=result.energy_conditions["Weak"],
        strong=result.energy_conditions["Strong"],
        dominant=result.energy_conditions["Dominant"],
        r_sample=metric.params["r_sample"],
        rho=metric.params["rho"],
        P=metric.params["P"],
        M=metric.params["M"],
        A=metric.params["A"],
        B=metric.params["B"],
    )


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recipe_params = fuchs_constant_warp_shell_parameters(
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

    summary = summarize_result(metric, result, recipe_params)
    with open(output_dir / "w1_python_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    save_arrays(output_dir, metric, result)

    if args.plots:
        save_optional_plots(output_dir, metric, result)

    return output_dir, summary


def print_summary(output_dir, summary):
    print("W1 Warp Shell Python workflow")
    print("=" * 42)
    print(f"Output directory : {output_dir}")
    print(f"Grid             : {summary['recipe_parameters']['grid_size']}")
    print(f"Method           : {summary['methodology']['energy_condition_mode']}")
    print(f"Observer mode    : {summary['methodology']['observer_mode']}")
    print()
    print("Energy conditions")
    for condition, data in summary["energy_conditions"].items():
        status = "PASS" if data["pass"] else "FAIL"
        print(
            f"- {condition:8s} {status:4s} "
            f"min={data['min']:.4e} "
            f"violating_fraction={data['violating_fraction']:.4e}"
        )
    print()
    print("Saved:")
    print(f"- {output_dir / 'w1_python_summary.json'}")
    print(f"- {output_dir / 'w1_python_arrays.npz'}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    output_dir, summary = run(args)
    print_summary(output_dir, summary)


if __name__ == "__main__":
    main()
