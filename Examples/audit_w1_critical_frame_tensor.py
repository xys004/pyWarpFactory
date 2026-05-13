import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory.analyzer.energy_conditions import _lower_indices_minkowski, generate_warpfactory_field
from warpfactory.analyzer.transform import (
    change_tensor_index,
    do_frame_transfer,
    get_eulerian_transformation_matrix,
)
from warpfactory.constants import C, G
from warpfactory.generator.warp_shell import create_warp_shell_metric
from warpfactory.recipes.fuchs_warp_shell import fuchs_constant_warp_shell_parameters
from warpfactory.solver.solvers import solve_energy_tensor
from warpfactory.solver.tensor_utils import (
    get_c4_inv,
    get_einstein_tensor,
    get_ricci_scalar,
    get_ricci_tensor,
)


ETA = np.diag([-1.0, 1.0, 1.0, 1.0])


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _build_metric(profile, v_warp, density_smooth_ratio=1.79, adm_shift_g00=False):
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
        adm_shift_g00=adm_shift_g00,
        density_smooth_ratio=density_smooth_ratio,
    )
    params["density_smooth_ratio"] = density_smooth_ratio
    params["adm_shift_g00"] = adm_shift_g00
    metric.params["fuchs_recipe"] = params
    return metric


def _nearest_index(metric, target):
    distance2 = (
        (metric.coords["x"] - target[0]) ** 2
        + (metric.coords["y"] - target[1]) ** 2
        + (metric.coords["z"] - target[2]) ** 2
    )
    return tuple(int(i) for i in np.unravel_index(int(np.argmin(distance2)), distance2.shape))


def _matrix_at(field, index):
    return np.asarray(field[(slice(None), slice(None)) + index], dtype=float)


def _scalar_at(field, index):
    return float(np.asarray(field[index]))


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


def _best_null_observer(T_hat_upper, num_vecs):
    T_hat_lower = _lower_indices_minkowski(T_hat_upper[..., np.newaxis])[..., 0]
    vectors = generate_warpfactory_field("nulllike", num_vecs)
    best = {
        "value": np.inf,
        "vector": None,
        "vector_index": None,
    }
    for i in range(vectors.shape[1]):
        vector = vectors[:, i]
        value = float(vector @ T_hat_lower @ vector)
        if value < best["value"]:
            best = {
                "value": value,
                "vector": vector,
                "vector_index": i,
            }
    return best, T_hat_lower


def _shift_report(g_cov):
    gamma = g_cov[1:, 1:]
    gamma_inv = np.linalg.inv(gamma)
    beta_cov = g_cov[0, 1:].copy()
    beta_contra_spatial = gamma_inv @ beta_cov
    beta_sq = float(beta_cov @ beta_contra_spatial)
    return {
        "beta_cov": beta_cov,
        "beta_contra_spatial_gamma_inv_beta_cov": beta_contra_spatial,
        "beta_cov_beta_contra": beta_sq,
        "g01": float(g_cov[0, 1]),
        "gamma": gamma,
        "gamma_inv": gamma_inv,
    }


def audit(args):
    metric = _build_metric(
        args.profile,
        args.v_warp,
        density_smooth_ratio=args.density_smooth_ratio,
        adm_shift_g00=args.adm_shift_g00,
    )
    target = tuple(float(item) for item in args.target.split(","))
    if len(target) != 3:
        raise ValueError("--target must be x,y,z")
    index = _nearest_index(metric, target)
    coords = {name: float(metric.coords[name][index]) for name in ("t", "x", "y", "z")}
    radius = float(np.sqrt(coords["x"] ** 2 + coords["y"] ** 2 + coords["z"] ** 2))

    g_inv = get_c4_inv(metric.tensor)
    ricci_cov = get_ricci_tensor(metric.tensor, metric.scaling)
    ricci_scalar = get_ricci_scalar(ricci_cov, g_inv)
    einstein_cov = get_einstein_tensor(ricci_cov, ricci_scalar, metric.tensor)
    factor = (C**4) / (8.0 * np.pi * G)
    T_cov_direct = factor * einstein_cov

    energy_contra = solve_energy_tensor(metric, solver_method="christoffel")
    energy_cov_from_pipeline = change_tensor_index(energy_contra, "covariant", metric)
    euler_pipeline = do_frame_transfer(metric, energy_contra, "Eulerian")

    g_local = _matrix_at(metric.tensor, index)
    g_inv_local = _matrix_at(g_inv, index)
    ricci_local = _matrix_at(ricci_cov, index)
    einstein_local = _matrix_at(einstein_cov, index)
    T_cov_direct_local = _matrix_at(T_cov_direct, index)
    T_contra_pipeline_local = _matrix_at(energy_contra.tensor, index)
    T_cov_pipeline_local = _matrix_at(energy_cov_from_pipeline.tensor, index)
    T_hat_pipeline_upper = _matrix_at(euler_pipeline.tensor, index)

    M_full = get_eulerian_transformation_matrix(metric.tensor)
    M_local = _matrix_at(M_full, index)
    orthonormality = M_local.T @ g_local @ M_local

    T_hat_manual_lower = M_local.T @ T_cov_direct_local @ M_local
    T_hat_manual_upper = T_hat_manual_lower.copy()
    for i in range(1, 4):
        T_hat_manual_upper[0, i] = -T_hat_manual_upper[0, i]
        T_hat_manual_upper[i, 0] = -T_hat_manual_upper[i, 0]

    best, T_hat_pipeline_lower = _best_null_observer(T_hat_pipeline_upper, args.num_vecs)
    k = best["vector"]
    Tkk_pipeline = float(k @ T_hat_pipeline_lower @ k)
    Tkk_manual_lower = float(k @ T_hat_manual_lower @ k)
    Tkk_manual_upper_lowered = float(
        k @ _lower_indices_minkowski(T_hat_manual_upper[..., np.newaxis])[..., 0] @ k
    )

    report = {
        "purpose": "Local W1 critical point frame/tensor audit",
        "profile": args.profile,
        "v_warp": args.v_warp,
        "target": {"x": target[0], "y": target[1], "z": target[2]},
        "index": list(index),
        "coords": coords,
        "radius": radius,
        "num_vecs": args.num_vecs,
        "scalars": {
            "ricci_scalar": _scalar_at(ricci_scalar, index),
            "factor_c4_over_8piG": factor,
            "eta_k_k": float(k @ ETA @ k),
            "MtgM_minus_eta_absmax": float(np.max(np.abs(orthonormality - ETA))),
            "T_cov_direct_minus_pipeline_absmax": float(
                np.max(np.abs(T_cov_direct_local - T_cov_pipeline_local))
            ),
            "T_cov_direct_minus_pipeline_relmax": float(
                np.max(np.abs(T_cov_direct_local - T_cov_pipeline_local))
                / max(np.max(np.abs(T_cov_direct_local)), 1e-300)
            ),
            "T_hat_manual_upper_minus_pipeline_absmax": float(
                np.max(np.abs(T_hat_manual_upper - T_hat_pipeline_upper))
            ),
            "T_hat_manual_upper_minus_pipeline_relmax": float(
                np.max(np.abs(T_hat_manual_upper - T_hat_pipeline_upper))
                / max(np.max(np.abs(T_hat_manual_upper)), 1e-300)
            ),
            "Tkk_pipeline": Tkk_pipeline,
            "Tkk_manual_lower": Tkk_manual_lower,
            "Tkk_manual_upper_lowered": Tkk_manual_upper_lowered,
        },
        "critical_observer": {
            "vector": k,
            "vector_index": best["vector_index"],
            "quadratic_pipeline": _quadratic_decomposition(T_hat_pipeline_lower, k),
            "quadratic_manual": _quadratic_decomposition(T_hat_manual_lower, k),
        },
        "shift_report": _shift_report(g_local),
        "g_cov": g_local,
        "g_inv": g_inv_local,
        "ricci_cov": ricci_local,
        "einstein_cov": einstein_local,
        "T_cov_direct": T_cov_direct_local,
        "T_contra_pipeline": T_contra_pipeline_local,
        "T_cov_from_pipeline_lowering": T_cov_pipeline_local,
        "tetrad_M": M_local,
        "M_transpose_g_M": orthonormality,
        "T_hat_manual_lower": T_hat_manual_lower,
        "T_hat_manual_upper": T_hat_manual_upper,
        "T_hat_pipeline_upper": T_hat_pipeline_upper,
        "T_hat_pipeline_lower": T_hat_pipeline_lower,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "w1_critical_frame_tensor_audit.json"
    npz_path = out_dir / "w1_critical_frame_tensor_audit_arrays.npz"
    json_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    np.savez_compressed(
        npz_path,
        **{
            key: value
            for key, value in report.items()
            if isinstance(value, np.ndarray)
        },
        observer=k,
    )

    print(f"Saved {json_path}")
    print(f"Saved {npz_path}")
    print(f"coords=({coords['x']:.6g}, {coords['y']:.6g}, {coords['z']:.6g}), r={radius:.6g}")
    print(f"M^T g M - eta absmax = {report['scalars']['MtgM_minus_eta_absmax']:.4e}")
    print(
        "T_cov direct-vs-pipeline "
        f"absmax = {report['scalars']['T_cov_direct_minus_pipeline_absmax']:.4e}, "
        f"relmax = {report['scalars']['T_cov_direct_minus_pipeline_relmax']:.4e}"
    )
    print(
        "T_hat manual-vs-pipeline "
        f"absmax = {report['scalars']['T_hat_manual_upper_minus_pipeline_absmax']:.4e}, "
        f"relmax = {report['scalars']['T_hat_manual_upper_minus_pipeline_relmax']:.4e}"
    )
    print(f"eta(k,k) = {report['scalars']['eta_k_k']:.4e}")
    print(f"Tkk pipeline = {Tkk_pipeline:.4e}")
    print(f"Tkk manual   = {Tkk_manual_lower:.4e}")
    q = report["critical_observer"]["quadratic_manual"]
    print(f"manual diag={q['diag']:.4e} offdiag={q['offdiag']:.4e}")


def main():
    parser = argparse.ArgumentParser(description="Audit W1 critical point tensor/frame conventions.")
    parser.add_argument("--profile", choices=("quick", "original"), default="original")
    parser.add_argument("--v-warp", type=float, default=0.02)
    parser.add_argument("--target", default="0.1,-11.7,0.0", help="Target x,y,z coordinates.")
    parser.add_argument("--num-vecs", type=int, default=40)
    parser.add_argument("--density-smooth-ratio", type=float, default=1.79)
    parser.add_argument("--adm-shift-g00", action="store_true")
    parser.add_argument("--output-dir", default="outputs/w1_critical_frame_tensor_audit")
    args = parser.parse_args()
    audit(args)


if __name__ == "__main__":
    main()
