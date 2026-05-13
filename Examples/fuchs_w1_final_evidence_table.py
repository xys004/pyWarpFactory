import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory import analyze_metric, summarize_energy_conditions
from warpfactory.analyzer.energy_conditions import _lower_indices_minkowski, generate_warpfactory_field
from warpfactory.generator.warp_shell import create_warp_shell_metric
from warpfactory.recipes.fuchs_warp_shell import fuchs_constant_warp_shell_parameters


CONDITIONS = ("Null", "Weak", "Strong", "Dominant")


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _parse_float_list(value):
    return [float(item) for item in value.split(",") if item.strip()]


def _radius(metric):
    return np.sqrt(metric.coords["x"] ** 2 + metric.coords["y"] ** 2 + metric.coords["z"] ** 2)


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


def _quadratic_decomposition(T_lower, vector):
    contribution = T_lower * np.outer(vector, vector)
    diag = float(np.trace(contribution))
    offdiag = float(np.sum(contribution) - diag)
    return {
        "diag": diag,
        "offdiag": offdiag,
        "total": diag + offdiag,
    }


def _critical_nec_certificate(metric, eulerian_tensor, null_map, shell_mask, num_vecs):
    masked = np.where(shell_mask, null_map, np.inf)
    index = tuple(int(i) for i in np.unravel_index(int(np.argmin(masked)), masked.shape))
    local_upper = eulerian_tensor[(slice(None), slice(None)) + index]
    local_lower = _lower_indices_minkowski(local_upper[..., np.newaxis])[..., 0]
    null_vectors = generate_warpfactory_field("nulllike", num_vecs)

    best_value = np.inf
    best_vector = None
    best_vector_index = None
    for vector_index in range(null_vectors.shape[1]):
        vector = null_vectors[:, vector_index]
        value = float(vector @ local_lower @ vector)
        if value < best_value:
            best_value = value
            best_vector = vector
            best_vector_index = vector_index

    coords = {name: float(metric.coords[name][index]) for name in ("t", "x", "y", "z")}
    decomposition = _quadratic_decomposition(local_lower, best_vector)
    return {
        "index": list(index),
        "coords": coords,
        "radius": float(np.sqrt(coords["x"] ** 2 + coords["y"] ** 2 + coords["z"] ** 2)),
        "map_value": float(null_map[index]),
        "Tkk": best_value,
        "diag": decomposition["diag"],
        "offdiag": decomposition["offdiag"],
        "vector": [float(item) for item in best_vector],
        "vector_index": int(best_vector_index),
        "eta_k_k": float(-best_vector[0] ** 2 + np.sum(best_vector[1:] ** 2)),
        "T00": float(local_upper[0, 0]),
        "T01": float(local_upper[0, 1]),
        "T11": float(local_upper[1, 1]),
        "T22": float(local_upper[2, 2]),
        "T33": float(local_upper[3, 3]),
    }


def _analyze_velocity(args, v_warp):
    metric, params = _build_metric(args.profile, v_warp, args.density_smooth_ratio)
    result = analyze_metric(
        metric,
        num_vecs=args.num_vecs,
        num_time_vecs=args.num_time_vecs,
        energy_condition_method="warpfactory",
        solver_method="christoffel",
    )
    radius = _radius(metric)
    shell = (radius >= params["R1"]) & (radius <= params["R2"])
    shell_summary = summarize_energy_conditions(result.energy_conditions, mask=shell)
    euler = result.eulerian_energy_tensor
    cert = _critical_nec_certificate(metric, euler, result.energy_conditions["Null"], shell, args.num_vecs)

    row = {
        "v_warp": float(v_warp),
        "num_vecs": args.num_vecs,
        "num_time_vecs": args.num_time_vecs,
        "T01_absmax": float(np.nanmax(np.abs(euler[0, 1][shell]))),
        "T00_min": float(np.nanmin(euler[0, 0][shell])),
        "critical": cert,
    }
    for condition in CONDITIONS:
        row[f"{condition}_min"] = float(shell_summary[f"{condition}_min"])
        row[f"{condition}_violating_fraction"] = float(shell_summary[f"{condition}_violating_fraction"])
    return row


def _write_csv(rows, path):
    fields = [
        "v_warp",
        "num_vecs",
        "Null_min",
        "Null_violating_fraction",
        "Weak_min",
        "Weak_violating_fraction",
        "Strong_min",
        "Strong_violating_fraction",
        "Dominant_min",
        "Dominant_violating_fraction",
        "T01_absmax",
        "T00_min",
        "critical_x",
        "critical_y",
        "critical_z",
        "critical_r",
        "critical_Tkk",
        "critical_diag",
        "critical_offdiag",
        "critical_T00",
        "critical_T01",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            critical = row["critical"]
            writer.writerow(
                {
                    **{field: row[field] for field in fields if field in row},
                    "critical_x": critical["coords"]["x"],
                    "critical_y": critical["coords"]["y"],
                    "critical_z": critical["coords"]["z"],
                    "critical_r": critical["radius"],
                    "critical_Tkk": critical["Tkk"],
                    "critical_diag": critical["diag"],
                    "critical_offdiag": critical["offdiag"],
                    "critical_T00": critical["T00"],
                    "critical_T01": critical["T01"],
                }
            )


def _write_markdown(rows, path):
    headers = [
        "vWarp",
        "NEC min",
        "NEC frac",
        "WEC min",
        "SEC min",
        "DEC min",
        "max \\|T01\\|",
        "min T00",
        "critical r",
        "Tkk",
        "diag",
        "offdiag",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        critical = row["critical"]
        values = [
            f"{row['v_warp']:.6g}",
            f"{row['Null_min']:.6e}",
            f"{row['Null_violating_fraction']:.6f}",
            f"{row['Weak_min']:.6e}",
            f"{row['Strong_min']:.6e}",
            f"{row['Dominant_min']:.6e}",
            f"{row['T01_absmax']:.6e}",
            f"{row['T00_min']:.6e}",
            f"{critical['radius']:.6f}",
            f"{critical['Tkk']:.6e}",
            f"{critical['diag']:.6e}",
            f"{critical['offdiag']:.6e}",
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for v_warp in _parse_float_list(args.velocities):
        print(f"Analyzing vWarp={v_warp:g} with num_vecs={args.num_vecs}...")
        rows.append(_analyze_velocity(args, v_warp))

    report = {
        "purpose": "Final paper-only evidence table for Fuchs W1",
        "profile": args.profile,
        "solver_method": "christoffel",
        "energy_condition_method": "warpfactory",
        "num_vecs": args.num_vecs,
        "num_time_vecs": args.num_time_vecs,
        "velocities": _parse_float_list(args.velocities),
        "rows": rows,
    }
    json_path = out_dir / "fuchs_w1_final_evidence_table.json"
    csv_path = out_dir / "fuchs_w1_final_evidence_table.csv"
    md_path = out_dir / "fuchs_w1_final_evidence_table.md"
    json_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    _write_csv(rows, csv_path)
    _write_markdown(rows, md_path)
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {md_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Build a W1 vWarp evidence table with NEC certificates.")
    parser.add_argument("--profile", choices=("quick", "original"), default="original")
    parser.add_argument("--velocities", default="0,0.005,0.006,0.007,0.01,0.02")
    parser.add_argument("--num-vecs", type=int, default=4)
    parser.add_argument("--num-time-vecs", type=int, default=5)
    parser.add_argument("--density-smooth-ratio", type=float, default=1.79)
    parser.add_argument("--output-dir", default="outputs/fuchs_w1_final_evidence")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
