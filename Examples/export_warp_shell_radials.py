import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory.generator.warp_shell import create_warp_shell_metric


def build_parser():
    parser = argparse.ArgumentParser(description="Export WarpShell radial profiles for MATLAB/Python parity checks.")
    parser.add_argument("--output", default="warp_shell_radials.npz")
    parser.add_argument("--grid", type=int, default=10)
    parser.add_argument("--spacing", type=float, default=0.5)
    parser.add_argument("--mass", type=float, default=1e26)
    parser.add_argument("--r-inner", type=float, default=1.0)
    parser.add_argument("--r-outer", type=float, default=2.0)
    parser.add_argument("--r-buff", type=float, default=0.25)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--smooth-factor", type=int, default=2)
    parser.add_argument("--v-warp", type=float, default=0.0)
    parser.add_argument("--do-warp", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
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
        mass=args.mass,
        r_inner=args.r_inner,
        r_outer=args.r_outer,
        r_buff=args.r_buff,
        sigma=args.sigma,
        smooth_factor=args.smooth_factor,
        v_warp=args.v_warp,
        do_warp=args.do_warp,
    )

    params = metric.params
    output = os.path.abspath(args.output)
    np.savez(
        output,
        r_sample=params["r_sample"],
        rho=params["rho"],
        P=params["P"],
        M=params["M"],
        A=params["A"],
        B=params["B"],
        grid_size=np.asarray(grid_size),
        grid_scale=np.asarray(grid_scale),
        center=np.asarray(center),
    )

    summary = {
        "output": output,
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
        "do_warp": args.do_warp,
        "A_min": float(np.nanmin(params["A"])),
        "A_max": float(np.nanmax(params["A"])),
        "B_min": float(np.nanmin(params["B"])),
        "B_max": float(np.nanmax(params["B"])),
        "rho_max": float(np.nanmax(params["rho"])),
        "P_min": float(np.nanmin(params["P"])),
        "P_max": float(np.nanmax(params["P"])),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
