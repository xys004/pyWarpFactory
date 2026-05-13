import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory import analyze_metric, summarize_energy_conditions
from warpfactory.analyzer.energy_conditions import (
    _evaluate_energy_condition_maps_from_eulerian,
    _evaluate_warpfactory_compatible_maps_from_eulerian,
    _lower_indices_minkowski,
    _mixed_up_down_minkowski,
    _trace_minkowski,
    generate_timelike_observers,
    generate_uniform_field,
    generate_warpfactory_field,
)
from warpfactory.analyzer.matlab_compat import matlab_eval_metric
from warpfactory.analyzer.stress_energy_diagnostics import eulerian_stress_decomposition
from warpfactory.recipes.fuchs_warp_shell import create_fuchs_constant_warp_shell
from warpfactory.solver.tensor_utils import (
    get_c4_inv,
    get_einstein_tensor,
    get_ricci_scalar,
    get_ricci_tensor_warpfactory_direct,
)


CONDITION_NAMES = ("Null", "Weak", "Strong", "Dominant")
COMPONENT_NAMES = {
    "T00": (0, 0),
    "T01": (0, 1),
    "T11": (1, 1),
    "T22": (2, 2),
    "T33": (3, 3),
    "T12": (1, 2),
    "T13": (1, 3),
    "T23": (2, 3),
}


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _summary(values, mask=None):
    data = values if mask is None else values[mask]
    data = np.asarray(np.real_if_close(data))
    return {
        "min": float(np.nanmin(data)),
        "max": float(np.nanmax(data)),
        "median": float(np.nanmedian(data)),
        "mean": float(np.nanmean(data)),
        "abs_max": float(np.nanmax(np.abs(data))),
        "negative_fraction": float(np.mean(data < 0.0)),
        "points": int(data.size),
    }


def _radius(metric):
    return np.sqrt(metric.coords["x"] ** 2 + metric.coords["y"] ** 2 + metric.coords["z"] ** 2)


def _region_masks(radius, params):
    return {
        "interior": radius < params["R1"],
        "material_shell": (radius >= params["R1"]) & (radius <= params["R2"]),
        "exterior": radius > params["R2"],
    }


def _critical_points(values, metric, mask, count):
    masked = np.where(mask, values, np.inf)
    flat = masked.reshape(-1)
    finite = np.isfinite(flat)
    if not np.any(finite):
        return []
    candidate_indices = np.argpartition(flat, min(count, flat.size - 1))[:count]
    candidate_indices = candidate_indices[np.argsort(flat[candidate_indices])]

    points = []
    for flat_index in candidate_indices:
        if not np.isfinite(flat[flat_index]):
            continue
        index = np.unravel_index(int(flat_index), values.shape)
        points.append(
            {
                "index": [int(item) for item in index],
                "value": float(values[index]),
                "coords": {
                    "t": float(metric.coords["t"][index]),
                    "x": float(metric.coords["x"][index]),
                    "y": float(metric.coords["y"][index]),
                    "z": float(metric.coords["z"][index]),
                },
                "radius": float(
                    np.sqrt(
                        metric.coords["x"][index] ** 2
                        + metric.coords["y"][index] ** 2
                        + metric.coords["z"][index] ** 2
                    )
                ),
            }
        )
    return points


def _point_tensor_report(T_euler, index):
    tensor = T_euler[(slice(None), slice(None)) + tuple(index)]
    return {
        name: float(tensor[mu, nu])
        for name, (mu, nu) in COMPONENT_NAMES.items()
    }


def _condition_point_report(result, metric, shell_mask, count):
    report = {}
    for name in CONDITION_NAMES:
        points = _critical_points(result.energy_conditions[name], metric, shell_mask, count)
        for point in points:
            point["T_euler_components"] = _point_tensor_report(
                result.eulerian_energy_tensor,
                point["index"],
            )
        report[name] = points
    return report


def _contract(T_lower, vector):
    return float(vector @ T_lower @ vector)


def _quadratic_decomposition(T_lower, vector):
    contribution = T_lower * np.outer(vector, vector)
    diag = float(np.trace(contribution))
    offdiag = float(np.sum(contribution) - diag)
    return {
        "diag": diag,
        "offdiag": offdiag,
        "total": diag + offdiag,
        "component_matrix": contribution.tolist(),
    }


def _flux_report(T_mixed, vector):
    flux = -(T_mixed @ vector)
    flux_norm = _minkowski_norm(flux)
    return {
        "flux": [float(item) for item in flux],
        "flux_time": float(flux[0]),
        "flux_norm": float(flux_norm),
        "minus_flux_norm": float(-flux_norm),
        "signed_norm": float(np.sign(flux_norm) * np.sqrt(abs(flux_norm))),
    }


def _minkowski_norm(vector):
    return float(-vector[0] ** 2 + np.sum(vector[1:] ** 2))


def _critical_observer_for_point(T_euler, condition, method, num_vecs, num_time_vecs):
    T_upper = np.asarray(T_euler, dtype=float)
    T_lower = _lower_indices_minkowski(T_upper[..., np.newaxis])[..., 0]
    trace = float(_trace_minkowski(T_lower[..., np.newaxis])[0])

    best = {
        "value": np.inf,
        "vector": None,
        "vector_kind": None,
        "vector_index": None,
        "time_index": None,
        "subcriterion": None,
        "quadratic_decomposition": None,
        "flux_report": None,
    }

    def consider(
        value,
        vector,
        kind,
        vector_index,
        time_index=None,
        subcriterion=None,
        tensor_for_decomposition=None,
        flux_tensor=None,
    ):
        if value < best["value"]:
            quadratic = None
            if tensor_for_decomposition is not None:
                quadratic = _quadratic_decomposition(tensor_for_decomposition, vector)
            flux = None
            if flux_tensor is not None:
                flux = _flux_report(flux_tensor, vector)
            best.update(
                {
                    "value": float(value),
                    "vector": [float(item) for item in vector],
                    "vector_kind": kind,
                    "vector_index": int(vector_index),
                    "time_index": None if time_index is None else int(time_index),
                    "subcriterion": subcriterion,
                    "quadratic_decomposition": quadratic,
                    "flux_report": flux,
                }
            )

    if method == "standard":
        if condition in {"Null", "Weak"}:
            vectors = generate_uniform_field("nulllike", num_vecs)
            for i in range(vectors.shape[1]):
                k = vectors[:, i]
                consider(
                    _contract(T_lower, k),
                    k,
                    "null",
                    i,
                    subcriterion="T_ab k^a k^b",
                    tensor_for_decomposition=T_lower,
                )

        if condition in {"Weak", "Strong", "Dominant"}:
            vectors = generate_timelike_observers(num_vecs)
            trace_reversed = T_lower - 0.5 * trace * np.diag([-1.0, 1.0, 1.0, 1.0])
            T_mixed = _mixed_up_down_minkowski(T_lower[..., np.newaxis])[..., 0]
            for i in range(vectors.shape[1]):
                u = vectors[:, i]
                if condition == "Weak":
                    consider(
                        _contract(T_lower, u),
                        u,
                        "timelike",
                        i,
                        subcriterion="T_ab u^a u^b",
                        tensor_for_decomposition=T_lower,
                    )
                elif condition == "Strong":
                    consider(
                        _contract(trace_reversed, u),
                        u,
                        "timelike",
                        i,
                        subcriterion="(T_ab - 1/2 T eta_ab) u^a u^b",
                        tensor_for_decomposition=trace_reversed,
                    )
                else:
                    rho_u = _contract(T_lower, u)
                    flux = -(T_mixed @ u)
                    flux_norm = _minkowski_norm(flux)
                    candidates = {
                        "rho_u": rho_u,
                        "flux_time": float(flux[0]),
                        "-flux_norm": -flux_norm,
                    }
                    subcriterion, value = min(candidates.items(), key=lambda item: item[1])
                    consider(value, u, "timelike", i, subcriterion=subcriterion, flux_tensor=T_mixed)

    elif method == "warpfactory":
        if condition in {"Null", "Dominant"}:
            vectors = generate_warpfactory_field("nulllike", num_vecs)
            T_mixed = _mixed_up_down_minkowski(T_lower[..., np.newaxis])[..., 0]
            for i in range(vectors.shape[1]):
                k = vectors[:, i]
                if condition == "Null":
                    consider(
                        _contract(T_lower, k),
                        k,
                        "nulllike",
                        i,
                        subcriterion="T_ab k^a k^b",
                        tensor_for_decomposition=T_lower,
                    )
                else:
                    flux = -(T_mixed @ k)
                    flux_norm = _minkowski_norm(flux)
                    signed_norm = np.sign(flux_norm) * np.sqrt(abs(flux_norm))
                    consider(
                        -signed_norm,
                        k,
                        "nulllike",
                        i,
                        subcriterion="-sign(J.J)*sqrt(abs(J.J))",
                        flux_tensor=T_mixed,
                    )

        if condition in {"Weak", "Strong"}:
            vectors = generate_warpfactory_field("timelike", num_vecs, num_time=num_time_vecs)
            trace_reversed = T_lower - 0.5 * trace * np.diag([-1.0, 1.0, 1.0, 1.0])
            for j in range(vectors.shape[2]):
                for i in range(vectors.shape[1]):
                    u = vectors[:, i, j]
                    if condition == "Weak":
                        consider(
                            _contract(T_lower, u),
                            u,
                            "timelike",
                            i,
                            j,
                            "T_ab u^a u^b",
                            tensor_for_decomposition=T_lower,
                        )
                    else:
                        consider(
                            _contract(trace_reversed, u),
                            u,
                            "timelike",
                            i,
                            j,
                            "(T_ab - 1/2 T eta_ab) u^a u^b",
                            tensor_for_decomposition=trace_reversed,
                        )
    else:
        raise ValueError("method must be 'standard' or 'warpfactory'.")

    return best


def _critical_observer_report(T_euler, maps_by_method, metric, shell_mask, count, num_vecs, num_time_vecs):
    report = {}
    for method, maps in maps_by_method.items():
        method_report = {}
        for condition in CONDITION_NAMES:
            points = _critical_points(maps[condition], metric, shell_mask, count)
            for point in points:
                index = tuple(point["index"])
                point["T_euler_components"] = _point_tensor_report(T_euler, index)
                point["critical_observer"] = _critical_observer_for_point(
                    T_euler[(slice(None), slice(None)) + index],
                    condition,
                    method,
                    num_vecs,
                    num_time_vecs,
                )
            method_report[condition] = points
        report[method] = method_report
    return report


def _component_summaries(T_euler, stress, masks):
    component_maps = {
        name: T_euler[mu, nu]
        for name, (mu, nu) in COMPONENT_NAMES.items()
    }
    component_maps.update(
        {
            "rho": stress["rho"],
            "flux_norm": stress["flux_norm"],
            "pressure_min": stress["pressure_min"],
            "pressure_mid": stress["pressure_mid"],
            "pressure_max": stress["pressure_max"],
            "pressure_trace": stress["pressure_trace"],
            "radial_pressure": stress["radial_pressure"],
            "tangential_pressure": stress["tangential_pressure"],
            "rho_plus_p_min": stress["rho_plus_p_min"],
            "rho_plus_radial_pressure": stress["rho_plus_radial_pressure"],
            "rho_plus_tangential_pressure": stress["rho_plus_tangential_pressure"],
            "radial_flux": stress["radial_flux"],
        }
    )
    return {
        region: {name: _summary(values, mask=mask) for name, values in component_maps.items()}
        for region, mask in masks.items()
    }


def _build_case(profile, static):
    metric = create_fuchs_constant_warp_shell(profile=profile)
    if static:
        metric.tensor[0, 1] = 0.0
        metric.tensor[1, 0] = 0.0
        metric.params["fuchs_recipe"]["v_warp"] = 0.0
    return metric


def analyze_case(args, static):
    metric = _build_case(args.profile, static=static)
    params = dict(metric.params["fuchs_recipe"])
    radius = _radius(metric)
    masks = _region_masks(radius, params)
    if args.analysis_mode == "matlab_compat":
        compat = matlab_eval_metric(
            metric,
            num_angular_vec=args.num_vecs,
            num_time_vec=args.num_time_vecs,
            solver_method=args.solver_method,
        )
        energy_conditions = compat.energy_conditions
        energy_tensor = compat.energy_tensor.tensor
        eulerian_energy_tensor = compat.energy_tensor_eulerian.tensor
        summary = summarize_energy_conditions(energy_conditions)
        methodology = compat.methodology
        observer_audit = {}
    else:
        result = analyze_metric(
            metric,
            num_vecs=args.num_vecs,
            num_time_vecs=args.num_time_vecs,
            observer_mode=args.observer_mode,
            audit_points=args.audit_points,
            solver_method=args.solver_method,
            energy_condition_method=args.energy_condition_method,
        )
        energy_conditions = result.energy_conditions
        energy_tensor = result.energy_tensor.tensor
        eulerian_energy_tensor = result.eulerian_energy_tensor
        summary = result.summary
        methodology = result.methodology
        observer_audit = result.observer_audit

    stress = eulerian_stress_decomposition(eulerian_energy_tensor, coords=metric.coords)
    ricci_tensor = get_ricci_tensor_warpfactory_direct(metric.tensor, metric.scaling)
    metric_inverse = get_c4_inv(metric.tensor)
    ricci_scalar = get_ricci_scalar(ricci_tensor, metric_inverse)
    einstein_tensor = get_einstein_tensor(ricci_tensor, ricci_scalar, metric.tensor)
    standard_conditions = _evaluate_energy_condition_maps_from_eulerian(
        eulerian_energy_tensor,
        num_vecs=args.num_vecs,
    )
    warpfactory_conditions = _evaluate_warpfactory_compatible_maps_from_eulerian(
        eulerian_energy_tensor,
        num_vecs=args.num_vecs,
        num_time_vecs=args.num_time_vecs,
    )
    maps_by_method = {
        "standard": standard_conditions,
        "warpfactory": warpfactory_conditions,
    }

    case_name = "static" if static else "shifted"
    arrays = {
        f"{case_name}_metric_tensor": metric.tensor,
        f"{case_name}_ricci_tensor": ricci_tensor,
        f"{case_name}_ricci_scalar": ricci_scalar,
        f"{case_name}_einstein_tensor": einstein_tensor,
        f"{case_name}_energy_tensor": energy_tensor,
        f"{case_name}_eulerian_energy_tensor": eulerian_energy_tensor,
        f"{case_name}_radius": radius,
        **{
            f"{case_name}_{name.lower()}": values
            for name, values in energy_conditions.items()
        },
        **{
            f"{case_name}_standard_{name.lower()}": values
            for name, values in standard_conditions.items()
        },
        **{
            f"{case_name}_warpfactory_{name.lower()}": values
            for name, values in warpfactory_conditions.items()
        },
    }

    return (
        {
            "case": case_name,
            "params": params,
            "summary": summary,
            "region_summary": {
                region: summarize_energy_conditions(energy_conditions, mask=mask)
                for region, mask in masks.items()
            },
            "component_summary": _component_summaries(eulerian_energy_tensor, stress, masks),
            "method_region_summary": {
                method: {
                    region: summarize_energy_conditions(method_maps, mask=mask)
                    for region, mask in masks.items()
                }
                for method, method_maps in maps_by_method.items()
            },
            "critical_points": _condition_point_report(
                type(
                    "AuditResult",
                    (),
                    {
                        "energy_conditions": energy_conditions,
                        "eulerian_energy_tensor": eulerian_energy_tensor,
                    },
                )(),
                metric,
                masks["material_shell"],
                args.critical_points,
            ),
            "critical_observers": _critical_observer_report(
                eulerian_energy_tensor,
                maps_by_method,
                metric,
                masks["material_shell"],
                args.critical_points,
                args.num_vecs,
                args.num_time_vecs,
            ),
            "methodology": methodology,
            "observer_audit": observer_audit,
        },
        arrays,
    )


def run(args):
    report = {
        "purpose": "Fuchs/WarpFactory W1 component-level reproduction audit",
        "settings": {
            "profile": args.profile,
            "num_vecs": args.num_vecs,
            "num_time_vecs": args.num_time_vecs,
            "solver_method": args.solver_method,
            "energy_condition_method": args.energy_condition_method,
            "observer_mode": args.observer_mode,
            "analysis_mode": args.analysis_mode,
        },
        "cases": [],
    }
    all_arrays = {}

    if args.case in {"static", "both"}:
        case_report, arrays = analyze_case(args, static=True)
        report["cases"].append(case_report)
        all_arrays.update(arrays)
    if args.case in {"shifted", "both"}:
        case_report, arrays = analyze_case(args, static=False)
        report["cases"].append(case_report)
        all_arrays.update(arrays)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "fuchs_w1_component_audit.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=_json_default)

    arrays_path = None
    if args.save_arrays:
        arrays_path = output_dir / "fuchs_w1_component_audit_arrays.npz"
        np.savez_compressed(arrays_path, **all_arrays)

    return report, report_path, arrays_path


def print_report(report, report_path, arrays_path):
    print("Fuchs/WarpFactory W1 component audit")
    print("=" * 42)
    print(f"Report: {report_path}")
    if arrays_path is not None:
        print(f"Arrays: {arrays_path}")
    print()
    for case in report["cases"]:
        shell = case["region_summary"]["material_shell"]
        print(f"Case: {case['case']}")
        for name in CONDITION_NAMES:
            print(
                f"  {name:8s} shell_min={shell[f'{name}_min']:.4e} "
                f"shell_frac={shell[f'{name}_violating_fraction']:.4f}"
            )
        print("  Critical material-shell NEC points:")
        for point in case["critical_points"]["Null"]:
            coords = point["coords"]
            t_components = point["T_euler_components"]
            print(
                "    "
                f"idx={point['index']} r={point['radius']:.4g} "
                f"value={point['value']:.4e} "
                f"x={coords['x']:.4g} y={coords['y']:.4g} z={coords['z']:.4g} "
                f"T00={t_components['T00']:.4e} T01={t_components['T01']:.4e}"
            )
        if "critical_observers" in case:
            print("  Critical observer comparison (material-shell Null):")
            for method in ("standard", "warpfactory"):
                points = case["critical_observers"][method]["Null"]
                if not points:
                    continue
                point = points[0]
                observer = point["critical_observer"]
                decomposition = observer.get("quadratic_decomposition") or {}
                print(
                    "    "
                    f"{method:11s} value={point['value']:.4e} "
                    f"idx={point['index']} vector={observer['vector']} "
                    f"diag={decomposition.get('diag', float('nan')):.4e} "
                    f"offdiag={decomposition.get('offdiag', float('nan')):.4e}"
                )
        print()


def build_parser():
    parser = argparse.ArgumentParser(description="Audit Fuchs/WarpFactory W1 intermediate components.")
    parser.add_argument("--profile", choices=("quick", "original"), default="quick")
    parser.add_argument("--case", choices=("static", "shifted", "both"), default="both")
    parser.add_argument("--num-vecs", type=int, default=4)
    parser.add_argument("--num-time-vecs", type=int, default=10)
    parser.add_argument("--solver-method", choices=("christoffel", "warpfactory_direct"), default="christoffel")
    parser.add_argument("--energy-condition-method", choices=("standard", "warpfactory"), default="warpfactory")
    parser.add_argument("--analysis-mode", choices=("pipeline", "matlab_compat"), default="pipeline")
    parser.add_argument("--observer-mode", choices=("sampled", "optimized"), default="sampled")
    parser.add_argument("--audit-points", type=int, default=3)
    parser.add_argument("--critical-points", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs/fuchs_w1_component_audit")
    parser.add_argument("--save-arrays", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    report, report_path, arrays_path = run(args)
    print_report(report, report_path, arrays_path)


if __name__ == "__main__":
    main()
