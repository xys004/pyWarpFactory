import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory import analyze_metric
from warpfactory.constants import C
from warpfactory.generator.alcubierre import create_alcubierre_metric


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Verify the ordinary Alcubierre metric as a WarpFactory-style "
            "negative control for energy-condition violations."
        )
    )
    parser.add_argument("--grid", type=int, default=20, help="Spatial grid size Nx=Ny=Nz.")
    parser.add_argument("--nt", type=int, default=5, help="Temporal grid size.")
    parser.add_argument("--spacing", type=float, default=0.2, help="Spatial grid spacing.")
    parser.add_argument("--dt", type=float, default=1.0, help="Temporal grid spacing.")
    parser.add_argument("--v", type=float, default=0.5, help="Bubble speed as a fraction of c.")
    parser.add_argument("--radius", type=float, default=1.4, help="Alcubierre bubble radius R.")
    parser.add_argument("--sigma", type=float, default=5.0, help="Alcubierre wall sharpness.")
    parser.add_argument("--num-vecs", type=int, default=12, help="Sampled null/timelike directions.")
    parser.add_argument(
        "--observer-mode",
        choices=("sampled", "optimized"),
        default="sampled",
        help="Use sampled maps only, or add optimized local observer audit.",
    )
    parser.add_argument("--audit-points", type=int, default=2, help="Optimized audit points per condition.")
    parser.add_argument("--optimized-max-speed", type=float, default=0.95, help="Max speed for observer audit.")
    parser.add_argument(
        "--energy-condition-method",
        choices=("warpfactory", "standard"),
        default="warpfactory",
        help="Use MATLAB-compatible WarpFactory convention or stricter pyWarpFactory convention.",
    )
    parser.add_argument("--output-dir", default=None, help="Optional directory for JSON and NPZ outputs.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def _grid_parameters(args):
    grid_size = (args.nt, args.grid, args.grid, args.grid)
    grid_scale = (args.dt, args.spacing, args.spacing, args.spacing)
    world_center = (
        args.dt,
        args.spacing * args.grid / 2.0,
        args.spacing * args.grid / 2.0,
        args.spacing * args.grid / 2.0,
    )
    return grid_size, grid_scale, world_center


def _bubble_radius(metric, v):
    t = metric.coords["t"]
    x = metric.coords["x"]
    y = metric.coords["y"]
    z = metric.coords["z"]
    xs = t * v * C
    return np.sqrt((x - xs) ** 2 + y**2 + z**2)


def _condition_location_report(values, radius_from_bubble, coords, bubble_radius):
    min_index = tuple(int(i) for i in np.unravel_index(np.nanargmin(values), values.shape))
    min_value = float(values[min_index])
    coord = {axis: float(coords[axis][min_index]) for axis in ("t", "x", "y", "z")}
    r_at_min = float(radius_from_bubble[min_index])
    return {
        "min": min_value,
        "max": float(np.nanmax(values)),
        "violating_points": int(np.count_nonzero(values < 0.0)),
        "violating_fraction": float(np.mean(values < 0.0)),
        "min_index": min_index,
        "min_coordinate": coord,
        "r_from_bubble_at_min": r_at_min,
        "distance_from_expected_wall": float(abs(r_at_min - bubble_radius)),
    }


def run(args):
    grid_size, grid_scale, world_center = _grid_parameters(args)
    metric = create_alcubierre_metric(
        grid_size=grid_size,
        grid_scale=grid_scale,
        world_center=world_center,
        v=args.v,
        R=args.radius,
        sigma=args.sigma,
    )

    result = analyze_metric(
        metric,
        num_vecs=args.num_vecs,
        observer_mode=args.observer_mode,
        audit_points=args.audit_points,
        optimized_max_speed=args.optimized_max_speed,
        energy_condition_method=args.energy_condition_method,
    )

    radius_from_bubble = _bubble_radius(metric, args.v)
    condition_reports = {
        name: _condition_location_report(values, radius_from_bubble, metric.coords, args.radius)
        for name, values in result.energy_conditions.items()
    }

    wall_band = np.abs(radius_from_bubble - args.radius) <= max(args.spacing, 1.0 / max(args.sigma, 1.0))
    wall_points = int(np.count_nonzero(wall_band))
    wall_report = {}
    for name, values in result.energy_conditions.items():
        violations = values < 0.0
        violations_in_wall = int(np.count_nonzero(violations & wall_band))
        wall_report[name] = {
            "wall_band_points": wall_points,
            "violating_points_in_wall_band": violations_in_wall,
            "fraction_of_violations_in_wall_band": (
                float(violations_in_wall / np.count_nonzero(violations))
                if np.count_nonzero(violations)
                else 0.0
            ),
        }

    report = {
        "metric": "Alcubierre",
        "purpose": (
            "Negative-control verification. Ordinary Alcubierre is expected "
            "to violate energy conditions, with strongest violations near "
            "the bubble wall."
        ),
        "parameters": {
            "grid_size": grid_size,
            "grid_scale": grid_scale,
            "world_center": world_center,
            "velocity_fraction_c": args.v,
            "radius": args.radius,
            "sigma": args.sigma,
            "energy_condition_method": args.energy_condition_method,
        },
        "methodology": result.methodology,
        "energy_conditions": condition_reports,
        "wall_alignment": wall_report,
        "observer_audit": result.observer_audit,
    }

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        json_path = os.path.join(args.output_dir, "alcubierre_verification_summary.json")
        npz_path = os.path.join(args.output_dir, "alcubierre_verification_arrays.npz")
        report["outputs"] = {"json": json_path, "npz": npz_path}
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        np.savez_compressed(
            npz_path,
            metric_tensor=metric.tensor,
            energy_tensor=result.energy_tensor.tensor,
            eulerian_energy_tensor=result.eulerian_energy_tensor,
            radius_from_bubble=radius_from_bubble,
            t=metric.coords["t"],
            x=metric.coords["x"],
            y=metric.coords["y"],
            z=metric.coords["z"],
            **{f"{name}_map": values for name, values in result.energy_conditions.items()},
        )

    return report


def print_report(report):
    print("pyWarpFactory Alcubierre verification")
    print("=" * 46)
    print(f"Metric     : {report['metric']}")
    print(f"Purpose    : {report['purpose']}")
    print(f"Parameters : {report['parameters']}")
    print()
    print("Energy-condition violations")
    for condition, data in report["energy_conditions"].items():
        status = "FAIL" if data["min"] < 0.0 else "PASS"
        coord = data["min_coordinate"]
        print(
            f"- {condition:8s} {status:4s} min={data['min']:.4e} "
            f"violating_fraction={data['violating_fraction']:.4e}"
        )
        print(
            "           "
            f"min_index={data['min_index']} "
            f"coord=(t={coord['t']:.3e}, x={coord['x']:.3e}, "
            f"y={coord['y']:.3e}, z={coord['z']:.3e})"
        )
        print(
            "           "
            f"r_from_bubble={data['r_from_bubble_at_min']:.4e} "
            f"|r-R|={data['distance_from_expected_wall']:.4e}"
        )

    print()
    print("Wall alignment")
    for condition, data in report["wall_alignment"].items():
        print(
            f"- {condition:8s} "
            f"fraction_of_violations_in_wall_band="
            f"{data['fraction_of_violations_in_wall_band']:.4e}"
        )

    if "outputs" in report:
        print()
        print(f"Saved JSON : {report['outputs']['json']}")
        print(f"Saved NPZ  : {report['outputs']['npz']}")


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
