import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory.analyzer.pipeline import summarize_energy_conditions
from warpfactory.analyzer.transform import do_frame_transfer
from warpfactory.generator.base import Metric
from warpfactory.recipes.fuchs_warp_shell import create_fuchs_constant_warp_shell
from warpfactory.solver.tensor_utils import (
    get_c4_inv,
    get_einstein_tensor,
    get_energy_tensor,
    get_ricci_scalar,
    get_ricci_tensor,
    get_ricci_tensor_warpfactory_direct,
)
from warpfactory.analyzer.energy_conditions import _evaluate_warpfactory_compatible_maps_from_eulerian


CONDITIONS = ("Null", "Weak", "Strong", "Dominant")


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _radius(metric):
    return np.sqrt(metric.coords["x"] ** 2 + metric.coords["y"] ** 2 + metric.coords["z"] ** 2)


def _region_masks(metric):
    r = _radius(metric)
    return {
        "passenger_core": r < 8.0,
        "inner_transition": (r > 9.5) & (r < 12.0),
        "middle_shell": (r >= 12.0) & (r < 18.0),
        "outer_transition": (r >= 18.0) & (r < 20.5),
        "exterior_vacuum": r > 22.0,
        "material_shell": (r >= 10.0) & (r <= 20.0),
        "all": np.ones_like(r, dtype=bool),
    }


def _build_metric(profile, static):
    metric = create_fuchs_constant_warp_shell(profile=profile)
    if static:
        metric.tensor[0, 1] = 0.0
        metric.tensor[1, 0] = 0.0
        metric.params["fuchs_recipe"]["v_warp"] = 0.0
        metric.params["fuchs_recipe"]["do_warp"] = False
    return metric


def _wrap_energy(metric, tensor, label):
    energy = Metric(
        tensor=tensor,
        coords=metric.coords,
        scaling=metric.scaling,
        name=f"{label} energy tensor",
        index="contravariant",
        params=dict(metric.params),
    )
    energy.type = "Stress-Energy"
    return energy


def _curvature_chain(metric, ricci):
    g_inv = get_c4_inv(metric.tensor)
    ricci_scalar = get_ricci_scalar(ricci, g_inv)
    einstein = get_einstein_tensor(ricci, ricci_scalar, metric.tensor)
    energy = get_energy_tensor(einstein, g_inv)
    energy_metric = _wrap_energy(metric, energy, "chain")
    eulerian = do_frame_transfer(metric, energy_metric, "Eulerian").tensor
    maps = _evaluate_warpfactory_compatible_maps_from_eulerian(
        eulerian,
        num_vecs=4,
        num_time_vecs=5,
    )
    return {
        "RicciTensor": ricci,
        "RicciScalar": ricci_scalar,
        "EinsteinTensor": einstein,
        "EnergyTensor": energy,
        "EnergyEulerianTensor": eulerian,
        "Null": maps["Null"],
        "Weak": maps["Weak"],
        "Strong": maps["Strong"],
        "Dominant": maps["Dominant"],
    }


def _component_iter(values):
    if values.ndim >= 6 and values.shape[:2] == (4, 4):
        for mu in range(4):
            for nu in range(4):
                yield f"{mu}{nu}", values[mu, nu]
    else:
        yield "scalar", values


def _worst_point(diff, mask):
    selected = np.where(mask, np.abs(diff), -np.inf)
    flat_index = int(np.argmax(selected))
    return tuple(int(i) for i in np.unravel_index(flat_index, diff.shape))


def _compare_field(name, direct, reference, metric, masks):
    report = {"components": {}, "worst": None}
    radius = _radius(metric)
    for component, direct_values in _component_iter(direct):
        if component == "scalar":
            ref_values = reference
        else:
            mu = int(component[0])
            nu = int(component[1])
            ref_values = reference[mu, nu]
        diff = direct_values - ref_values
        component_report = {}
        for region, mask in masks.items():
            if not np.any(mask):
                continue
            local_diff = diff[mask]
            local_direct = direct_values[mask]
            local_ref = ref_values[mask]
            abs_max = float(np.nanmax(np.abs(local_diff)))
            scale = float(max(np.nanmax(np.abs(local_ref)), np.nanmax(np.abs(local_direct)), 1e-300))
            index = _worst_point(diff, mask)
            entry = {
                "abs_max": abs_max,
                "rel_max": abs_max / scale,
                "direct_at_max": float(direct_values[index]),
                "christoffel_at_max": float(ref_values[index]),
                "diff_at_max": float(diff[index]),
                "index": list(index),
                "coords": {
                    "x": float(metric.coords["x"][index]),
                    "y": float(metric.coords["y"][index]),
                    "z": float(metric.coords["z"][index]),
                },
                "radius": float(radius[index]),
            }
            component_report[region] = entry
            worst = report["worst"]
            if worst is None or abs_max > worst["abs_max"]:
                report["worst"] = {
                    "field": name,
                    "component": component,
                    "region": region,
                    **entry,
                }
        report["components"][component] = component_report
    return report


def _region_condition_summary(chain, masks):
    return {
        region: summarize_energy_conditions(
            {condition: chain[condition] for condition in CONDITIONS},
            mask=mask,
        )
        for region, mask in masks.items()
        if np.any(mask)
    }


def _analyze_case(profile, static):
    metric = _build_metric(profile, static=static)
    masks = _region_masks(metric)
    print(f"Computing Ricci tensors for case={'static' if static else 'shifted'}...")
    direct_ricci = get_ricci_tensor_warpfactory_direct(metric.tensor, metric.scaling)
    christoffel_ricci = get_ricci_tensor(metric.tensor, metric.scaling)
    print("Building curvature chains...")
    direct = _curvature_chain(metric, direct_ricci)
    christoffel = _curvature_chain(metric, christoffel_ricci)
    fields = ("RicciTensor", "RicciScalar", "EinsteinTensor", "EnergyTensor", "EnergyEulerianTensor")
    comparisons = {
        field: _compare_field(field, direct[field], christoffel[field], metric, masks)
        for field in fields
    }
    return {
        "case": "static" if static else "shifted",
        "params": metric.params["fuchs_recipe"],
        "comparisons": comparisons,
        "condition_summary": {
            "warpfactory_direct": _region_condition_summary(direct, masks),
            "christoffel": _region_condition_summary(christoffel, masks),
        },
        "worst_by_field": {
            field: comparisons[field]["worst"]
            for field in fields
        },
    }


def _write_markdown(report, path):
    lines = [
        "# W1 warpfactory_direct vs christoffel",
        "",
        f"Profile: `{report['profile']}`",
        "",
    ]
    for case in report["cases"]:
        lines.extend([f"## {case['case']}", ""])
        lines.append("| field | component | region | abs_max | rel_max | r | direct | christoffel |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for field, worst in case["worst_by_field"].items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        field,
                        worst["component"],
                        worst["region"],
                        f"{worst['abs_max']:.6e}",
                        f"{worst['rel_max']:.6e}",
                        f"{worst['radius']:.6f}",
                        f"{worst['direct_at_max']:.6e}",
                        f"{worst['christoffel_at_max']:.6e}",
                    ]
                )
                + " |"
            )
        lines.extend(["", "### Material-shell conditions", ""])
        lines.append("| engine | NEC min | NEC frac | WEC min | SEC min | DEC min |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for engine in ("christoffel", "warpfactory_direct"):
            shell = case["condition_summary"][engine]["material_shell"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        engine,
                        f"{shell['Null_min']:.6e}",
                        f"{shell['Null_violating_fraction']:.6f}",
                        f"{shell['Weak_min']:.6e}",
                        f"{shell['Strong_min']:.6e}",
                        f"{shell['Dominant_min']:.6e}",
                    ]
                )
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args):
    cases = []
    if args.case in {"static", "both"}:
        cases.append(_analyze_case(args.profile, static=True))
    if args.case in {"shifted", "both"}:
        cases.append(_analyze_case(args.profile, static=False))
    report = {
        "purpose": "Legacy audit: compare warpfactory_direct against christoffel on post-Swarp-fix W1.",
        "profile": args.profile,
        "cases": cases,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "wfdirect_vs_christoffel_w1.json"
    md_path = out_dir / "wfdirect_vs_christoffel_w1.md"
    json_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    _write_markdown(report, md_path)
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")
    for case in cases:
        print(f"Case {case['case']}")
        for field, worst in case["worst_by_field"].items():
            print(
                f"  {field:22s} worst {worst['component']} {worst['region']} "
                f"abs={worst['abs_max']:.4e} rel={worst['rel_max']:.4e} r={worst['radius']:.4f}"
            )
    return report


def main():
    parser = argparse.ArgumentParser(description="Compare W1 warpfactory_direct and christoffel curvature chains.")
    parser.add_argument("--profile", choices=("quick", "original"), default="quick")
    parser.add_argument("--case", choices=("static", "shifted", "both"), default="both")
    parser.add_argument("--output-dir", default="outputs/wfdirect_vs_christoffel_w1")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
