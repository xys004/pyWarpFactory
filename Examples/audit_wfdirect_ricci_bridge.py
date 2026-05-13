import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory.recipes.fuchs_warp_shell import create_fuchs_constant_warp_shell
from warpfactory.solver.tensor_utils import (
    get_ricci_tensor,
    get_ricci_tensor_christoffel_from_warpfactory_derivatives,
    get_ricci_tensor_warpfactory_direct,
)


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


def _worst_point(diff, mask):
    selected = np.where(mask, np.abs(diff), -np.inf)
    flat_index = int(np.argmax(selected))
    return tuple(int(i) for i in np.unravel_index(flat_index, diff.shape))


def _compare(label, candidate, reference, metric, masks):
    radius = _radius(metric)
    report = {"label": label, "worst": None, "components": {}}
    for mu in range(4):
        for nu in range(4):
            component = f"{mu}{nu}"
            diff = candidate[mu, nu] - reference[mu, nu]
            component_report = {}
            for region, mask in masks.items():
                if not np.any(mask):
                    continue
                abs_max = float(np.nanmax(np.abs(diff[mask])))
                scale = float(
                    max(
                        np.nanmax(np.abs(candidate[mu, nu][mask])),
                        np.nanmax(np.abs(reference[mu, nu][mask])),
                        1e-300,
                    )
                )
                index = _worst_point(diff, mask)
                entry = {
                    "abs_max": abs_max,
                    "rel_max": abs_max / scale,
                    "index": list(index),
                    "coords": {
                        "x": float(metric.coords["x"][index]),
                        "y": float(metric.coords["y"][index]),
                        "z": float(metric.coords["z"][index]),
                    },
                    "radius": float(radius[index]),
                    "candidate_at_max": float(candidate[mu, nu][index]),
                    "reference_at_max": float(reference[mu, nu][index]),
                    "diff_at_max": float(diff[index]),
                }
                component_report[region] = entry
                if report["worst"] is None or abs_max > report["worst"]["abs_max"]:
                    report["worst"] = {
                        "label": label,
                        "component": component,
                        "region": region,
                        **entry,
                    }
            report["components"][component] = component_report
    return report


def _point_table(metric, index, current, bridge, direct):
    return {
        "index": list(index),
        "coords": {
            "x": float(metric.coords["x"][index]),
            "y": float(metric.coords["y"][index]),
            "z": float(metric.coords["z"][index]),
        },
        "radius": float(_radius(metric)[index]),
        "components": {
            f"{mu}{nu}": {
                "christoffel_current": float(current[mu, nu][index]),
                "christoffel_from_wfdirect_derivatives": float(bridge[mu, nu][index]),
                "warpfactory_direct": float(direct[mu, nu][index]),
                "bridge_minus_current": float(bridge[mu, nu][index] - current[mu, nu][index]),
                "direct_minus_current": float(direct[mu, nu][index] - current[mu, nu][index]),
                "direct_minus_bridge": float(direct[mu, nu][index] - bridge[mu, nu][index]),
            }
            for mu in range(4)
            for nu in range(4)
        },
    }


def _analyze_case(profile, static):
    metric = _build_metric(profile, static=static)
    masks = _region_masks(metric)
    case_name = "static" if static else "shifted"
    print(f"Computing Ricci bridge for {case_name}...")
    current = get_ricci_tensor(metric.tensor, metric.scaling)
    bridge = get_ricci_tensor_christoffel_from_warpfactory_derivatives(metric.tensor, metric.scaling)
    direct = get_ricci_tensor_warpfactory_direct(metric.tensor, metric.scaling)

    bridge_vs_current = _compare("bridge_vs_current", bridge, current, metric, masks)
    direct_vs_current = _compare("direct_vs_current", direct, current, metric, masks)
    direct_vs_bridge = _compare("direct_vs_bridge", direct, bridge, metric, masks)

    worst_index = tuple(direct_vs_current["worst"]["index"])
    return {
        "case": case_name,
        "params": metric.params["fuchs_recipe"],
        "comparisons": {
            "bridge_vs_current": bridge_vs_current,
            "direct_vs_current": direct_vs_current,
            "direct_vs_bridge": direct_vs_bridge,
        },
        "point_at_direct_worst": _point_table(metric, worst_index, current, bridge, direct),
    }


def _write_markdown(report, path):
    lines = [
        "# W1 Ricci Bridge Audit",
        "",
        f"Profile: `{report['profile']}`",
        "",
        "A = current christoffel",
        "B = christoffel reconstructed from the derivative arrays used by `warpfactory_direct`",
        "C = `warpfactory_direct` / direct Ricci expansion",
        "",
    ]
    for case in report["cases"]:
        lines.extend([f"## {case['case']}", ""])
        lines.append("| comparison | component | region | abs_max | rel_max | r | candidate | reference |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for key in ("bridge_vs_current", "direct_vs_current", "direct_vs_bridge"):
            worst = case["comparisons"][key]["worst"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        key,
                        worst["component"],
                        worst["region"],
                        f"{worst['abs_max']:.6e}",
                        f"{worst['rel_max']:.6e}",
                        f"{worst['radius']:.6f}",
                        f"{worst['candidate_at_max']:.6e}",
                        f"{worst['reference_at_max']:.6e}",
                    ]
                )
                + " |"
            )

        point = case["point_at_direct_worst"]
        lines.extend(
            [
                "",
                "### Direct-Worst Point",
                "",
                f"index: `{point['index']}`",
                f"coords: x={point['coords']['x']:.6f}, y={point['coords']['y']:.6f}, z={point['coords']['z']:.6f}",
                f"r: {point['radius']:.6f}",
                "",
                "| component | A current | B same-deriv Christoffel | C direct | B-A | C-A | C-B |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for component in ("00", "11", "22", "33"):
            row = point["components"][component]
            lines.append(
                "| "
                + " | ".join(
                    [
                        component,
                        f"{row['christoffel_current']:.6e}",
                        f"{row['christoffel_from_wfdirect_derivatives']:.6e}",
                        f"{row['warpfactory_direct']:.6e}",
                        f"{row['bridge_minus_current']:.6e}",
                        f"{row['direct_minus_current']:.6e}",
                        f"{row['direct_minus_bridge']:.6e}",
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
        "purpose": "Separate derivative-layout errors from the algebraic direct Ricci expansion.",
        "profile": args.profile,
        "cases": cases,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "wfdirect_ricci_bridge.json"
    md_path = out_dir / "wfdirect_ricci_bridge.md"
    json_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    _write_markdown(report, md_path)
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")
    for case in cases:
        print(f"Case {case['case']}")
        for key in ("bridge_vs_current", "direct_vs_current", "direct_vs_bridge"):
            worst = case["comparisons"][key]["worst"]
            print(
                f"  {key:20s} {worst['component']} {worst['region']} "
                f"abs={worst['abs_max']:.4e} rel={worst['rel_max']:.4e} r={worst['radius']:.4f}"
            )
    return report


def main():
    parser = argparse.ArgumentParser(description="Audit warpfactory_direct Ricci against a same-derivative Christoffel bridge.")
    parser.add_argument("--profile", choices=("quick", "original"), default="quick")
    parser.add_argument("--case", choices=("static", "shifted", "both"), default="static")
    parser.add_argument("--output-dir", default="outputs/wfdirect_ricci_bridge")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
