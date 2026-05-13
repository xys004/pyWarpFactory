import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory.generator.warp_shell import create_warp_shell_metric
from warpfactory.recipes.fuchs_warp_shell import fuchs_constant_warp_shell_parameters
from warpfactory.utils.helpers import compact_sigmoid


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _parse_float_list(value):
    return np.array([float(item) for item in value.split(",") if item.strip()], dtype=float)


def _build_metric(profile, v_warp, density_smooth_ratio=1.79):
    params = fuchs_constant_warp_shell_parameters(profile=profile)
    params["v_warp"] = float(v_warp)
    params["do_warp"] = abs(float(v_warp)) > 0.0
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
        density_smooth_ratio=density_smooth_ratio,
    )
    params["density_smooth_ratio"] = density_smooth_ratio
    metric.params["fuchs_recipe"] = params
    return metric, params


def _interp_profile(r_sample, values, radii):
    return np.interp(radii, r_sample, values)


def _finite_derivatives(r, values):
    first = np.gradient(values, r, edge_order=2)
    second = np.gradient(first, r, edge_order=2)
    return first, second


def _nearest_grid_point(metric, radius):
    grid_radius = np.sqrt(metric.coords["x"] ** 2 + metric.coords["y"] ** 2 + metric.coords["z"] ** 2)
    index = tuple(int(i) for i in np.unravel_index(int(np.argmin(np.abs(grid_radius - radius))), grid_radius.shape))
    return index, float(grid_radius[index])


def audit(args):
    metric, params = _build_metric(args.profile, args.v_warp, args.density_smooth_ratio)
    radii = _parse_float_list(args.radii)

    r_sample = metric.params["r_sample"]
    shift_smoothed = metric.params["shift_radial"]
    shift_raw = compact_sigmoid(r_sample, params["R1"], params["R2"], params["sigma"], params["Rbuff"])
    d_shift, d2_shift = _finite_derivatives(r_sample, shift_smoothed)
    d_raw, d2_raw = _finite_derivatives(r_sample, shift_raw)

    S = _interp_profile(r_sample, shift_smoothed, radii)
    S_raw = _interp_profile(r_sample, shift_raw, radii)
    dS = _interp_profile(r_sample, d_shift, radii)
    d2S = _interp_profile(r_sample, d2_shift, radii)
    dS_raw = _interp_profile(r_sample, d_raw, radii)
    d2S_raw = _interp_profile(r_sample, d2_raw, radii)

    rows = []
    for radius, raw, smooth, ds_raw, d2s_raw, ds, d2s in zip(radii, S_raw, S, dS_raw, d2S_raw, dS, d2S):
        index, grid_r = _nearest_grid_point(metric, radius)
        g01 = float(metric.tensor[0, 1][index])
        coords = {name: float(metric.coords[name][index]) for name in ("x", "y", "z")}
        rows.append(
            {
                "r": float(radius),
                "nearest_grid_r": grid_r,
                "nearest_grid_index": list(index),
                "nearest_grid_coords": coords,
                "S_raw": float(raw),
                "S_smoothed": float(smooth),
                "S_eff_nearest_grid": float(-g01 / args.v_warp) if args.v_warp != 0.0 else None,
                "g01_nearest_grid": g01,
                "g01_expected_from_smoothed": float(-smooth * args.v_warp),
                "dS_raw_dr": float(ds_raw),
                "d2S_raw_dr2": float(d2s_raw),
                "dS_smoothed_dr": float(ds),
                "d2S_smoothed_dr2": float(d2s),
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "w1_shift_profile_audit.json"
    csv_path = out_dir / "w1_shift_profile_audit.csv"
    report = {
        "profile": args.profile,
        "v_warp": args.v_warp,
        "R1": params["R1"],
        "R2": params["R2"],
        "Rbuff": params["Rbuff"],
        "sigma": params["sigma"],
        "smooth_factor": params["smooth_factor"],
        "factor": params["factor"],
        "factor_meaning": "mass factor: mass = R2/(2G)*c^2*factor; it is not Rbuff in the current recipe",
        "rows": rows,
    }
    json_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    print("r          S_raw       S_smooth    S_eff_grid  g01_grid       dS/dr        d2S/dr2")
    for row in rows:
        print(
            f"{row['r']:9.6f} "
            f"{row['S_raw']:11.6e} "
            f"{row['S_smoothed']:11.6e} "
            f"{row['S_eff_nearest_grid']:11.6e} "
            f"{row['g01_nearest_grid']:12.6e} "
            f"{row['dS_smoothed_dr']:11.6e} "
            f"{row['d2S_smoothed_dr2']:11.6e}"
        )


def main():
    parser = argparse.ArgumentParser(description="Audit the radial Swarp/g01 profile for Fuchs W1.")
    parser.add_argument("--profile", choices=("quick", "original"), default="original")
    parser.add_argument("--v-warp", type=float, default=0.02)
    parser.add_argument(
        "--radii",
        default="0,5,9.9,10,10.277159,11.700427,13.333333,15,19.663672,20",
    )
    parser.add_argument("--density-smooth-ratio", type=float, default=1.79)
    parser.add_argument("--output-dir", default="outputs/w1_shift_profile_audit")
    args = parser.parse_args()
    audit(args)


if __name__ == "__main__":
    main()
