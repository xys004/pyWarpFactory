import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from warpfactory.analyzer.pipeline import analyze_metric, summarize_energy_conditions
from warpfactory.generator.warp_shell import create_warp_shell_metric
from warpfactory.recipes.fuchs_warp_shell import (
    create_fuchs_constant_warp_shell,
    fuchs_constant_warp_shell_parameters,
)


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=_json_default)


def _maybe_upload_to_gcs(local_path, output_uri):
    if not output_uri or not output_uri.startswith("gs://"):
        return None

    try:
        from google.cloud import storage
    except ImportError as exc:
        raise RuntimeError(
            "google-cloud-storage is required for gs:// uploads. "
            "Install requirements-cloud.txt in the Vertex image."
        ) from exc

    bucket_name, _, prefix = output_uri[5:].partition("/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    destination = "/".join(part.strip("/") for part in (prefix, local_path.name) if part)
    blob = bucket.blob(destination)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{destination}"


def _build_metric(args):
    if args.recipe != "fuchs-w1":
        raise ValueError("Only recipe='fuchs-w1' is currently supported.")

    params = fuchs_constant_warp_shell_parameters(args.profile)
    if args.v_warp is None and args.r_buff is None and args.density_smooth_ratio == 1.79:
        metric = create_fuchs_constant_warp_shell(profile=args.profile)
        if args.static:
            metric.tensor[0, 1] = 0.0
            metric.tensor[1, 0] = 0.0
        return metric, params

    v_warp = params["v_warp"] if args.v_warp is None else args.v_warp
    r_buff = params["Rbuff"] if args.r_buff is None else args.r_buff
    metric = create_warp_shell_metric(
        params["grid_size"],
        params["grid_scaling"],
        params["world_center"],
        mass=params["mass"],
        r_inner=params["R1"],
        r_outer=params["R2"],
        r_buff=r_buff,
        sigma=params["sigma"],
        smooth_factor=params["smooth_factor"],
        v_warp=0.0 if args.static else v_warp,
        do_warp=not args.static and v_warp != 0.0,
        density_smooth_ratio=args.density_smooth_ratio,
    )
    metric.params["fuchs_recipe"] = {
        **params,
        "Rbuff": r_buff,
        "v_warp": 0.0 if args.static else v_warp,
        "density_smooth_ratio": args.density_smooth_ratio,
    }
    return metric, metric.params["fuchs_recipe"]


def _region_summaries(metric, conditions, params, tolerance):
    radius = (
        metric.coords["x"] * metric.coords["x"]
        + metric.coords["y"] * metric.coords["y"]
        + metric.coords["z"] * metric.coords["z"]
    ) ** 0.5
    masks = {
        "interior": radius < params["R1"],
        "material_shell": (radius >= params["R1"]) & (radius <= params["R2"]),
        "exterior": radius > params["R2"],
    }
    return {
        name: summarize_energy_conditions(conditions, mask=mask, tolerance=tolerance)
        for name, mask in masks.items()
    }


def run(args):
    metric, params = _build_metric(args)
    result = analyze_metric(
        metric,
        num_vecs=args.num_vecs,
        num_time_vecs=args.num_time_vecs,
        observer_mode=args.observer_mode,
        audit_points=args.audit_points,
        optimized_max_speed=args.optimized_max_speed,
        energy_condition_method=args.energy_condition_method,
        solver_method=args.solver_method,
        flat_tolerance=args.flat_tolerance,
    )

    payload = {
        "job": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "recipe": args.recipe,
            "profile": args.profile,
            "static": args.static,
            "execution_target": args.execution_target,
        },
        "params": params,
        "summary": result.summary,
        "region_summary": _region_summaries(
            metric,
            result.energy_conditions,
            params,
            tolerance=args.tolerance,
        ),
        "methodology": result.methodology,
        "observer_audit": result.observer_audit,
    }

    local_dir = Path(args.local_output_dir or tempfile.mkdtemp(prefix="pywarpfactory-run-"))
    local_dir.mkdir(parents=True, exist_ok=True)
    summary_path = local_dir / "summary.json"
    _write_json(summary_path, payload)

    if args.save_arrays:
        arrays_path = local_dir / "arrays.npz"
        np.savez_compressed(
            arrays_path,
            metric_tensor=metric.tensor,
            energy_tensor=result.energy_tensor.tensor,
            eulerian_energy_tensor=result.eulerian_energy_tensor,
            **{name.lower(): values for name, values in result.energy_conditions.items()},
        )
    else:
        arrays_path = None

    uploaded = []
    if args.output_uri:
        uploaded.append(_maybe_upload_to_gcs(summary_path, args.output_uri))
        if arrays_path is not None:
            uploaded.append(_maybe_upload_to_gcs(arrays_path, args.output_uri))

    payload["artifacts"] = {
        "local_summary": str(summary_path),
        "local_arrays": str(arrays_path) if arrays_path is not None else None,
        "uploaded": [item for item in uploaded if item],
    }
    print(json.dumps(payload, indent=2, default=_json_default))
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description="Portable pyWarpFactory job runner.")
    parser.add_argument("--config", default=None, help="Optional JSON job config. CLI flags override config values.")
    parser.add_argument("--execution-target", choices=("local", "vertex"), default=os.environ.get("PYWARPFACTORY_EXECUTION_TARGET", "local"))
    parser.add_argument("--recipe", default="fuchs-w1", choices=("fuchs-w1",))
    parser.add_argument("--profile", default="quick", choices=("quick", "original"))
    parser.add_argument("--static", action="store_true", help="Disable the warp shift and analyze the static shell.")
    parser.add_argument("--v-warp", type=float, default=None)
    parser.add_argument("--r-buff", type=float, default=None)
    parser.add_argument("--density-smooth-ratio", type=float, default=1.79)
    parser.add_argument("--num-vecs", type=int, default=40)
    parser.add_argument("--num-time-vecs", type=int, default=10)
    parser.add_argument("--observer-mode", choices=("sampled", "optimized"), default="sampled")
    parser.add_argument("--audit-points", type=int, default=3)
    parser.add_argument("--optimized-max-speed", type=float, default=0.999)
    parser.add_argument("--energy-condition-method", choices=("standard", "warpfactory"), default="warpfactory")
    parser.add_argument("--solver-method", choices=("christoffel", "warpfactory_direct"), default="christoffel")
    parser.add_argument("--flat-tolerance", type=float, default=None)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument("--local-output-dir", default=None)
    parser.add_argument("--output-uri", default=os.environ.get("PYWARPFACTORY_OUTPUT_URI"))
    return parser


def _load_config_defaults(config_path):
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Job config must be a JSON object.")
    return data


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args(argv)

    parser = build_parser()
    defaults = _load_config_defaults(config_args.config)
    if defaults:
        unknown_keys = sorted(set(defaults) - {action.dest for action in parser._actions})
        if unknown_keys:
            raise ValueError(f"Unknown job config keys: {', '.join(unknown_keys)}")
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
