import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory import analyze_metric, summarize_energy_conditions
from warpfactory.analyzer.transform import do_frame_transfer
from warpfactory.generator.base import Metric
from warpfactory.generator.commons import create_grid, get_minkowski_metric
from warpfactory.generator.schwarzschild import create_schwarzschild_metric
from warpfactory.recipes.fuchs_warp_shell import create_fuchs_constant_warp_shell
from warpfactory.solver.finite_difference import derive_1st_4th_order
from warpfactory.solver.solvers import solve_energy_tensor
from warpfactory.solver.tensor_utils import get_c4_inv, get_ricci_scalar, get_ricci_tensor


def _metric_from_tensor(tensor, grid_scale, name):
    coords = create_grid(tensor.shape[2:], grid_scale)
    return Metric(
        tensor=tensor,
        coords=coords,
        scaling=np.array(grid_scale, dtype=float),
        name=name,
        index="covariant",
    )


def _minkowski_metric(grid_size):
    return get_minkowski_metric(grid_size)


def _constant_shift_metric(grid_size, beta=0.2):
    tensor = get_minkowski_metric(grid_size)
    tensor[0, 0] = -1.0 + beta**2
    tensor[0, 1] = beta
    tensor[1, 0] = beta
    return tensor


def _constant_boost_metric(grid_size, beta=0.2):
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    lam = np.array(
        [
            [gamma, -gamma * beta, 0.0, 0.0],
            [-gamma * beta, gamma, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    eta = np.diag([-1.0, 1.0, 1.0, 1.0])
    boosted = lam.T @ eta @ lam
    return np.broadcast_to(
        boosted.reshape((4, 4) + (1,) * len(grid_size)),
        (4, 4) + tuple(grid_size),
    ).copy()


def _rindler_metric(grid_size, grid_scale, acceleration=0.015):
    tensor = get_minkowski_metric(grid_size)
    x = np.arange(grid_size[1], dtype=float) * grid_scale[1]
    x = x - x.mean()
    lapse = 1.0 + acceleration * x
    tensor[0, 0] = -(lapse.reshape(1, grid_size[1], 1, 1) ** 2)
    return tensor


def _interior4(field, width=2):
    slices = [slice(None)] * field.ndim
    for axis in range(field.ndim - 4, field.ndim):
        slices[axis] = slice(width, -width)
    return tuple(slices)


def _max_ricci(metric, width=2):
    ricci = get_ricci_tensor(metric.tensor, metric.scaling)
    scalar = get_ricci_scalar(ricci, get_c4_inv(metric.tensor))
    ricci_abs = float(np.nanmax(np.abs(ricci[_interior4(ricci, width=width)])))
    scalar_abs = float(np.nanmax(np.abs(scalar[_interior4(scalar, width=width)])))
    return ricci_abs, scalar_abs


def _max_energy(metric, width=2):
    energy = solve_energy_tensor(metric, solver_method="christoffel")
    euler = do_frame_transfer(metric, energy, "Eulerian").tensor
    return (
        float(np.nanmax(np.abs(energy.tensor[_interior4(energy.tensor, width=width)]))),
        float(np.nanmax(np.abs(euler[_interior4(euler, width=width)]))),
    )


def _radius(metric):
    return np.sqrt(metric.coords["x"] ** 2 + metric.coords["y"] ** 2 + metric.coords["z"] ** 2)


def _validate_flat_cases():
    grid_size = (7, 9, 9, 9)
    grid_scale = (0.5, 0.25, 0.25, 0.25)
    cases = [
        ("Minkowski", _minkowski_metric(grid_size), 1e-12, 1e-12, 1e-12),
        ("constant ADM shift", _constant_shift_metric(grid_size), 1e-12, 1e-12, 1e-12),
        ("constant Lorentz boost", _constant_boost_metric(grid_size), 1e-12, 1e-12, 1e-12),
        ("Rindler", _rindler_metric(grid_size, grid_scale), 2e-4, 3e-4, None),
    ]

    for label, tensor, ricci_tol, scalar_tol, energy_tol in cases:
        metric = _metric_from_tensor(tensor, grid_scale, label)
        ricci_abs, scalar_abs = _max_ricci(metric)
        print(f"{label}: max |Ricci|={ricci_abs:.4e}, max |R|={scalar_abs:.4e}")
        assert ricci_abs < ricci_tol, f"{label}: Ricci max {ricci_abs}"
        assert scalar_abs < scalar_tol, f"{label}: Ricci scalar max {scalar_abs}"

        if energy_tol is not None:
            energy_abs, euler_abs = _max_energy(metric)
            print(f"{label}: max |T|={energy_abs:.4e}, max |T_hat|={euler_abs:.4e}")
            assert energy_abs < energy_tol, f"{label}: stress-energy max {energy_abs}"
            assert euler_abs < energy_tol, f"{label}: Eulerian stress-energy max {euler_abs}"
        else:
            print(f"{label}: skipped stress-energy assert; Ricci floor is finite-difference dominated.")


def _validate_schwarzschild_exterior():
    grid_size = (1, 28, 28, 28)
    grid_scale = (1.0, 0.5, 0.5, 0.5)
    center = (0.0, -8.0, -8.0, -8.0)
    rs = 1.0
    metric = create_schwarzschild_metric(grid_size, grid_scale, center, rs)

    ricci = get_ricci_tensor(metric.tensor, metric.scaling)
    scalar = get_ricci_scalar(ricci, get_c4_inv(metric.tensor))
    radius = _radius(metric)
    exterior = (radius > 3.0 * rs)
    interior_grid = np.zeros(radius.shape, dtype=bool)
    interior_grid[(slice(None), slice(3, -3), slice(3, -3), slice(3, -3))] = True
    mask = exterior & interior_grid

    ricci_abs = float(np.nanmax(np.abs(ricci[:, :, mask])))
    scalar_abs = float(np.nanmax(np.abs(scalar[mask])))
    print(f"Schwarzschild exterior: max |Ricci|={ricci_abs:.4e}, max |R|={scalar_abs:.4e}")
    assert scalar_abs < 2e-2, f"Schwarzschild exterior Ricci scalar max {scalar_abs}"
    assert ricci_abs < 2e-2, f"Schwarzschild exterior Ricci max {ricci_abs}"


def _validate_static_w1_shell():
    metric = create_fuchs_constant_warp_shell(profile="original")
    metric.tensor[0, 1] = 0.0
    metric.tensor[1, 0] = 0.0
    metric.params["fuchs_recipe"]["v_warp"] = 0.0
    metric.params["fuchs_recipe"]["do_warp"] = False

    result = analyze_metric(
        metric,
        num_vecs=4,
        num_time_vecs=5,
        energy_condition_method="warpfactory",
        solver_method="christoffel",
    )
    params = metric.params["fuchs_recipe"]
    radius = _radius(metric)
    shell = (radius >= params["R1"]) & (radius <= params["R2"])
    shell_summary = summarize_energy_conditions(result.energy_conditions, mask=shell)
    print("Static W1 original shell summary:")
    for condition in ("Null", "Weak", "Strong", "Dominant"):
        print(
            f"  {condition}: min={shell_summary[f'{condition}_min']:.4e}, "
            f"violating_fraction={shell_summary[f'{condition}_violating_fraction']:.4e}"
        )
        assert shell_summary[f"{condition}_min"] >= -1e32, condition
        assert shell_summary[f"{condition}_violating_fraction"] == 0.0, condition

    passenger = radius < min(2.0, params["R1"] - 8.0)
    energy_abs = float(np.nanmax(np.abs(result.energy_tensor.tensor[:, :, passenger])))
    euler_abs = float(np.nanmax(np.abs(result.eulerian_energy_tensor[:, :, passenger])))
    spatial_derivative_abs = 0.0
    for axis, spacing in zip((3, 4, 5), metric.scaling[1:]):
        derivative = derive_1st_4th_order(metric.tensor, axis, spacing)
        spatial_derivative_abs = max(spatial_derivative_abs, float(np.nanmax(np.abs(derivative[:, :, passenger]))))
    print(
        "Static W1 passenger interior: "
        f"max |spatial dg|={spatial_derivative_abs:.4e}, "
        f"max |T|={energy_abs:.4e}, max |T_hat|={euler_abs:.4e}"
    )
    assert spatial_derivative_abs < 1e-10
    assert energy_abs < 1e32
    assert euler_abs < 1e32


def main():
    _validate_flat_cases()
    _validate_schwarzschild_exterior()
    _validate_static_w1_shell()
    print("Christoffel geometry-only validation passed.")


if __name__ == "__main__":
    main()
