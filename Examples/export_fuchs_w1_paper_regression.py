import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory import analyze_metric
from warpfactory.analyzer.energy_conditions import (
    _lower_indices_minkowski,
    generate_warpfactory_field,
)
from warpfactory.analyzer.stress_energy_diagnostics import eulerian_stress_decomposition
from warpfactory.generator.warp_shell import create_warp_shell_metric
from warpfactory.recipes.fuchs_warp_shell import fuchs_constant_warp_shell_parameters


CONDITIONS = ("Null", "Weak", "Strong", "Dominant")


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _radius(metric):
    return np.sqrt(metric.coords["x"] ** 2 + metric.coords["y"] ** 2 + metric.coords["z"] ** 2)


def _build_metric(profile, v_warp, adm_shift_g00=False, density_smooth_ratio=1.79):
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
    params["adm_shift_g00"] = adm_shift_g00
    params["density_smooth_ratio"] = density_smooth_ratio
    metric.params["fuchs_recipe"] = params
    return metric


def _nearest_axis_indices(metric):
    x0 = int(np.argmin(np.abs(metric.coords["x"][0, :, 0, 0])))
    y0 = int(np.argmin(np.abs(metric.coords["y"][0, 0, :, 0])))
    z0 = int(np.argmin(np.abs(metric.coords["z"][0, 0, 0, :])))
    return x0, y0, z0


def _line(values, x0, z0):
    return np.asarray(values[0, x0, :, z0])


def _z_centered_slice(values, z0):
    return np.asarray(values[0, :, :, z0])


def _summary(values, mask=None):
    data = values if mask is None else values[mask]
    data = np.real_if_close(data)
    return {
        "min": float(np.nanmin(data)),
        "max": float(np.nanmax(data)),
        "median": float(np.nanmedian(data)),
        "mean": float(np.nanmean(data)),
        "negative_fraction": float(np.mean(data < 0.0)),
        "points": int(data.size),
    }


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


def _null_certificate(metric, T_euler, null_map, shell_mask, num_vecs, count):
    candidates = np.where(shell_mask, null_map, np.inf).reshape(-1)
    finite = np.isfinite(candidates)
    if not np.any(finite):
        return []

    take = min(count, int(np.count_nonzero(finite)))
    flat_indices = np.argpartition(candidates, take - 1)[:take]
    flat_indices = flat_indices[np.argsort(candidates[flat_indices])]
    null_vectors = generate_warpfactory_field("nulllike", num_vecs)

    certificates = []
    for flat_index in flat_indices:
        index = tuple(int(i) for i in np.unravel_index(int(flat_index), null_map.shape))
        local_upper = T_euler[(slice(None), slice(None)) + index]
        local_lower = _lower_indices_minkowski(local_upper[..., np.newaxis])[..., 0]

        best_value = np.inf
        best_vector = None
        best_vector_index = None
        for vector_index in range(null_vectors.shape[1]):
            k = null_vectors[:, vector_index]
            value = float(k @ local_lower @ k)
            if value < best_value:
                best_value = value
                best_vector = k
                best_vector_index = vector_index

        g_local = metric.tensor[(slice(None), slice(None)) + index]
        eta_norm = float(-best_vector[0] ** 2 + np.sum(best_vector[1:] ** 2))
        g_norm = float(best_vector @ g_local @ best_vector)
        coords = {
            name: float(metric.coords[name][index])
            for name in ("t", "x", "y", "z")
        }
        certificates.append(
            {
                "index": list(index),
                "coords": coords,
                "radius": float(
                    np.sqrt(coords["x"] ** 2 + coords["y"] ** 2 + coords["z"] ** 2)
                ),
                "map_value": float(null_map[index]),
                "Tkk": best_value,
                "vector": [float(item) for item in best_vector],
                "vector_index": int(best_vector_index),
                "eta_k_k": eta_norm,
                "g_k_k_coordinate_basis_check": g_norm,
                "metric": g_local.tolist(),
                "T_hat": local_upper.tolist(),
                "quadratic_decomposition": _quadratic_decomposition(local_lower, best_vector),
            }
        )

    return certificates


def _plot_signed_line(ax, x, series, title, ylabel):
    ax.axhline(0.0, color="0.3", linewidth=0.8)
    for label, values in series.items():
        ax.plot(x, values, label=label, linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("y")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def _set_symmetric_symlog(ax, values):
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return
    abs_max = float(np.nanmax(np.abs(finite)))
    if abs_max == 0.0:
        return
    ax.set_yscale("symlog", linthresh=max(abs_max * 1e-6, 1e-12))


def _save_plots(out_dir, metric_profiles, stress_profiles, condition_profiles, condition_slices, metric):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = metric_profiles["y"]

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    _plot_signed_line(
        ax,
        y,
        {
            "g00": metric_profiles["g00"],
            "g01": metric_profiles["g01"],
            "g11": metric_profiles["g11"],
            "g22": metric_profiles["g22"],
            "g33": metric_profiles["g33"],
        },
        "Fig. 8 style: metric components on y-axis",
        "metric component",
    )
    fig.savefig(out_dir / "fig8_metric_y_axis.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    stress_line_values = {
        "rho": stress_profiles["rho"],
        "flux_norm": stress_profiles["flux_norm"],
        "p_min": stress_profiles["pressure_min"],
        "p_mid": stress_profiles["pressure_mid"],
        "p_max": stress_profiles["pressure_max"],
        "radial_p": stress_profiles["radial_pressure"],
        "tangential_p": stress_profiles["tangential_pressure"],
        "shear": stress_profiles["anisotropy"],
    }
    _plot_signed_line(
        ax,
        y,
        stress_line_values,
        "Fig. 9 style: Eulerian stress diagnostics on y-axis",
        "stress-energy scale",
    )
    _set_symmetric_symlog(ax, np.stack(list(stress_line_values.values())))
    fig.savefig(out_dir / "fig9_stress_y_axis.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    _plot_signed_line(
        ax,
        y,
        condition_profiles,
        "Fig. 10 style: energy conditions on y-axis",
        "condition value",
    )
    _set_symmetric_symlog(ax, np.stack(list(condition_profiles.values())))
    fig.savefig(out_dir / "fig10_energy_conditions_y_axis.png", dpi=180)
    plt.close(fig)

    x = metric.coords["x"][0, :, 0, 0]
    yy = metric.coords["y"][0, 0, :, 0]
    extent = [float(yy.min()), float(yy.max()), float(x.min()), float(x.max())]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    for ax, condition in zip(axes.ravel(), CONDITIONS):
        values = condition_slices[condition]
        abs_max = float(np.nanmax(np.abs(values)))
        if abs_max == 0.0:
            abs_max = 1.0
        image = ax.imshow(
            values,
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            vmin=-abs_max,
            vmax=abs_max,
            aspect="equal",
        )
        ax.set_title(condition)
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle("Fig. 14 style: central-z maps, observer-minimized samples")
    fig.savefig(out_dir / "fig14_energy_conditions_central_z.png", dpi=180)
    plt.close(fig)


def export_regression(args):
    metric = _build_metric(
        profile=args.profile,
        v_warp=args.v_warp,
        adm_shift_g00=args.adm_shift_g00,
        density_smooth_ratio=args.density_smooth_ratio,
    )
    result = analyze_metric(
        metric,
        num_vecs=args.num_vecs,
        num_time_vecs=args.num_time_vecs,
        energy_condition_method="warpfactory",
        solver_method="christoffel",
    )
    stress = eulerian_stress_decomposition(result.eulerian_energy_tensor, coords=metric.coords)
    radius = _radius(metric)
    params = metric.params["fuchs_recipe"]
    shell = (radius >= params["R1"]) & (radius <= params["R2"])
    interior = radius < params["R1"]
    exterior = radius > params["R2"]
    x0, y0, z0 = _nearest_axis_indices(metric)

    metric_profiles = {
        "y": _line(metric.coords["y"], x0, z0),
        "radius": _line(radius, x0, z0),
        "g00": _line(metric.tensor[0, 0], x0, z0),
        "g01": _line(metric.tensor[0, 1], x0, z0),
        "g11": _line(metric.tensor[1, 1], x0, z0),
        "g22": _line(metric.tensor[2, 2], x0, z0),
        "g33": _line(metric.tensor[3, 3], x0, z0),
    }
    stress_profiles = {
        "rho": _line(stress["rho"], x0, z0),
        "flux_norm": _line(stress["flux_norm"], x0, z0),
        "radial_flux": _line(stress["radial_flux"], x0, z0),
        "pressure_min": _line(stress["pressure_min"], x0, z0),
        "pressure_mid": _line(stress["pressure_mid"], x0, z0),
        "pressure_max": _line(stress["pressure_max"], x0, z0),
        "radial_pressure": _line(stress["radial_pressure"], x0, z0),
        "tangential_pressure": _line(stress["tangential_pressure"], x0, z0),
        "anisotropy": _line(stress["anisotropy"], x0, z0),
        "rho_plus_p_min": _line(stress["rho_plus_p_min"], x0, z0),
    }
    condition_profiles = {
        condition: _line(result.energy_conditions[condition], x0, z0)
        for condition in CONDITIONS
    }
    condition_slices = {
        condition: _z_centered_slice(result.energy_conditions[condition], z0)
        for condition in CONDITIONS
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = out_dir / "fuchs_w1_paper_regression_arrays.npz"
    json_path = out_dir / "fuchs_w1_paper_regression_summary.json"

    np.savez_compressed(
        arrays_path,
        radius=radius,
        shell_mask=shell,
        interior_mask=interior,
        exterior_mask=exterior,
        y_axis_x_index=x0,
        y_axis_y_index=y0,
        y_axis_z_index=z0,
        metric_tensor=metric.tensor,
        energy_tensor=result.energy_tensor.tensor,
        eulerian_energy_tensor=result.eulerian_energy_tensor,
        **{f"profile_metric_{key}": value for key, value in metric_profiles.items()},
        **{f"profile_stress_{key}": value for key, value in stress_profiles.items()},
        **{f"profile_condition_{key.lower()}": value for key, value in condition_profiles.items()},
        **{f"slice_z_condition_{key.lower()}": value for key, value in condition_slices.items()},
    )

    summary = {
        "profile": args.profile,
        "v_warp": args.v_warp,
        "solver_method": "christoffel",
        "energy_condition_method": "warpfactory",
        "num_vecs": args.num_vecs,
        "num_time_vecs": args.num_time_vecs,
        "axis_indices": {"x0": x0, "y0": y0, "z0": z0},
        "axis_coordinates": {
            "x": float(metric.coords["x"][0, x0, y0, z0]),
            "y": float(metric.coords["y"][0, x0, y0, z0]),
            "z": float(metric.coords["z"][0, x0, y0, z0]),
        },
        "recipe": params,
        "summaries": {
            "global": {
                condition: _summary(result.energy_conditions[condition])
                for condition in CONDITIONS
            },
            "material_shell": {
                condition: _summary(result.energy_conditions[condition], mask=shell)
                for condition in CONDITIONS
            },
            "interior": {
                condition: _summary(result.energy_conditions[condition], mask=interior)
                for condition in CONDITIONS
            },
            "exterior": {
                condition: _summary(result.energy_conditions[condition], mask=exterior)
                for condition in CONDITIONS
            },
        },
        "stress_shell_summary": {
            key: _summary(np.real_if_close(value), mask=shell)
            for key, value in stress.items()
            if isinstance(value, np.ndarray) and value.shape == radius.shape
        },
        "nec_certificates": _null_certificate(
            metric,
            result.eulerian_energy_tensor,
            result.energy_conditions["Null"],
            shell,
            args.num_vecs,
            args.critical_points,
        ),
        "arrays_file": str(arrays_path),
    }

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)

    if args.save_plots:
        _save_plots(
            out_dir,
            metric_profiles,
            stress_profiles,
            condition_profiles,
            condition_slices,
            metric,
        )

    print(f"Saved arrays: {arrays_path}")
    print(f"Saved summary: {json_path}")
    if args.save_plots:
        print(f"Saved plots: {out_dir}")
    print(
        "Material shell NEC: "
        f"min={summary['summaries']['material_shell']['Null']['min']:.4e}, "
        f"negative_fraction={summary['summaries']['material_shell']['Null']['negative_fraction']:.4e}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export Fuchs W1 paper-style regression profiles, slices, and NEC certificates."
    )
    parser.add_argument("--profile", choices=("quick", "original"), default="quick")
    parser.add_argument("--v-warp", type=float, default=0.02)
    parser.add_argument("--num-vecs", type=int, default=8)
    parser.add_argument("--num-time-vecs", type=int, default=5)
    parser.add_argument("--critical-points", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs/fuchs_w1_paper_regression")
    parser.add_argument("--adm-shift-g00", action="store_true")
    parser.add_argument("--density-smooth-ratio", type=float, default=1.79)
    parser.add_argument("--save-plots", action="store_true")
    args = parser.parse_args()
    export_regression(args)


if __name__ == "__main__":
    main()
