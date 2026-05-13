from warpfactory.solver.tensor_utils import (
    get_c4_inv, 
    get_ricci_tensor, 
    get_ricci_tensor_warpfactory_direct,
    get_ricci_scalar, 
    get_einstein_tensor, 
    get_energy_tensor,
    verify_tensor
)
from warpfactory.generator.base import Metric
import numpy as np
import warnings


def _minkowski_tensor(shape):
    eta = np.diag([-1.0, 1.0, 1.0, 1.0])
    return eta.reshape((4, 4) + (1,) * len(shape))


def _max_minkowski_deviation(metric_tensor):
    eta = _minkowski_tensor(metric_tensor.shape[2:])
    return float(np.nanmax(np.abs(metric_tensor - eta)))


def _wrap_energy_tensor(metric, tensor, diagnostics=None):
    params = dict(metric.params)
    if diagnostics is not None:
        params["solver_diagnostics"] = diagnostics

    energy_tensor = Metric(
        tensor=tensor,
        coords=metric.coords,
        scaling=metric.scaling,
        name=f"Energy Tensor of {metric.name}",
        index="contravariant",
        params=params
    )
    energy_tensor.type = "Stress-Energy"
    return energy_tensor


def solve_energy_tensor(metric, flat_tolerance=None, solver_method="christoffel"):
    """
    Computes the Stress-Energy Tensor for a given Metric using Einstein Field Equations.
    Replicates met2den.m functionality.
    
    Args:
        metric (Metric): A valid Metric object.
        flat_tolerance (float, optional): If provided and the full metric is
            within this absolute tolerance of Cartesian Minkowski, return an
            exact zero stress-energy tensor. This is an explicit numerical-floor
            control for near-flat smoke tests, not a physics approximation.
        solver_method (str): 'christoffel' uses the existing Python Ricci path.
            'warpfactory_direct' ports MATLAB WarpFactory's ricciT.m direct
            first/second-derivative formula for numerical parity.
        
    Returns:
        Metric: A new 'Metric' object representing the Stress-Energy tensor (T^uv).
                Type will be 'Stress-Energy'.
    """
    # 1. Verify input
    verify_tensor(metric)
    if solver_method not in {"christoffel", "warpfactory_direct"}:
        raise ValueError("solver_method must be 'christoffel' or 'warpfactory_direct'.")
    
    grid_scale = metric.scaling
    diagnostics = {
        "flat_tolerance": flat_tolerance,
        "solver_method": solver_method,
        "max_abs_metric_deviation_from_minkowski": _max_minkowski_deviation(metric.tensor),
        "flat_floor_applied": False,
    }

    if flat_tolerance is not None and diagnostics["max_abs_metric_deviation_from_minkowski"] <= flat_tolerance:
        diagnostics["flat_floor_applied"] = True
        return _wrap_energy_tensor(metric, np.zeros_like(metric.tensor), diagnostics)
    
    # 2. Compute Inverse Metric
    g_inv = get_c4_inv(metric.tensor)
    
    # 3. Compute Ricci Tensor R_mn
    if solver_method == "warpfactory_direct":
        warnings.warn(
            "solver_method='warpfactory_direct' is experimental/legacy-audit only. "
            "It is known to diverge from the christoffel solver on W1 in the "
            "RicciTensor/middle_shell region; do not use it for physical "
            "energy-condition conclusions.",
            RuntimeWarning,
            stacklevel=2,
        )
        R_mn = get_ricci_tensor_warpfactory_direct(metric.tensor, grid_scale)
    else:
        R_mn = get_ricci_tensor(metric.tensor, grid_scale)
    
    # 4. Compute Ricci Scalar R
    R = get_ricci_scalar(R_mn, g_inv)
    
    # 5. Compute Einstein Tensor G_mn
    G_mn = get_einstein_tensor(R_mn, R, metric.tensor)
    
    # 6. Compute Energy Tensor T^uv (Contravariant)
    # Note: get_energy_tensor returns T^uv
    T_uv = get_energy_tensor(G_mn, g_inv)
    
    # 7. Wrap in Metric/Tensor object
    # We reuse the Metric class container for now, but change type.
    # ideally we should have a Tensor class, but Metric is generic enough.
    
    return _wrap_energy_tensor(metric, T_uv, diagnostics)
