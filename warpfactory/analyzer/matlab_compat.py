"""
MATLAB WarpFactory-compatible analysis helpers.

This module intentionally mirrors the public MATLAB flow:

    evalMetric
      -> getEnergyTensor / met2den
      -> doFrameTransfer(..., "Eulerian")
      -> getEnergyConditions

The goal is not to replace the higher-level Python pipeline. It is a parity
lane for reproducing and debugging WarpFactory examples component by component.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from warpfactory.analyzer.energy_conditions import (
    _evaluate_warpfactory_compatible_maps_from_eulerian,
)
from warpfactory.analyzer.transform import do_frame_transfer
from warpfactory.generator.base import Metric
from warpfactory.solver.solvers import solve_energy_tensor
from warpfactory.solver.tensor_utils import get_c4_inv


@dataclass
class MatlabCompatResult:
    """Container matching the main products of MATLAB evalMetric."""

    metric: Metric
    energy_tensor: Metric
    energy_tensor_eulerian: Metric
    energy_conditions: Dict[str, np.ndarray]
    methodology: Dict[str, object]


def matlab_even_points_on_sphere(radius: float, number_of_points: int) -> np.ndarray:
    """Port of Analyzer/utils/getEvenPointsOnSphere.m."""
    golden_ratio = (1.0 + 5.0**0.5) / 2.0
    points = np.zeros((3, number_of_points), dtype=float)
    for i in range(number_of_points):
        theta = 2.0 * np.pi * i / golden_ratio
        phi = np.arccos(1.0 - 2.0 * (i + 0.5) / number_of_points)
        points[0, i] = radius * np.cos(theta) * np.sin(phi)
        points[1, i] = radius * np.sin(theta) * np.sin(phi)
        points[2, i] = radius * np.cos(phi)
    return np.real(points)


def matlab_generate_uniform_field(
    field_type: str,
    num_angular_vec: int,
    num_time_vec: int = 10,
) -> np.ndarray:
    """Port of Analyzer/utils/generateUniformField.m."""
    field_type = field_type.lower()
    if field_type not in {"nulllike", "timelike"}:
        raise ValueError("field_type must be 'nulllike' or 'timelike'.")

    if field_type == "nulllike":
        vec_field = np.ones((4, num_angular_vec), dtype=float)
        vec_field[1:] = matlab_even_points_on_sphere(1.0, num_angular_vec)
        norm = np.sqrt(np.sum(vec_field**2, axis=0))
        return vec_field / norm

    bb = np.linspace(0.0, 1.0, num_time_vec)
    vec_field = np.ones((4, num_angular_vec, num_time_vec), dtype=float)
    for jj, b in enumerate(bb):
        vec_field[1:, :, jj] = matlab_even_points_on_sphere(1.0 - b, num_angular_vec)
        norm = np.sqrt(np.sum(vec_field[:, :, jj] ** 2, axis=0))
        vec_field[:, :, jj] = vec_field[:, :, jj] / norm
    return vec_field


def matlab_change_tensor_index(input_tensor: Metric, index: str, metric_tensor: Optional[Metric] = None) -> Metric:
    """
    Array port of Analyzer/changeTensorIndex.m for rank-2 tensors.

    Supported target indices: covariant, contravariant, mixedupdown,
    mixeddownup. For metric tensors, only covariant <-> contravariant is
    supported, matching MATLAB.
    """
    index = index.lower()
    if index not in {"covariant", "contravariant", "mixedupdown", "mixeddownup"}:
        raise ValueError("index must be covariant, contravariant, mixedupdown, or mixeddownup.")

    current = input_tensor.index.lower()
    if current == index:
        return input_tensor

    tensor_type = getattr(input_tensor, "type", None)
    is_metric = tensor_type == "metric" or input_tensor.name.lower().startswith("metric")

    if is_metric:
        if {current, index} == {"covariant", "contravariant"}:
            return _wrap_like(input_tensor, get_c4_inv(input_tensor.tensor), index)
        raise ValueError("MATLAB does not convert metric tensors to mixed index.")

    if metric_tensor is None:
        raise ValueError("metric_tensor is required for non-metric tensor index changes.")

    metric_index = metric_tensor.index.lower()
    if metric_index in {"mixedupdown", "mixeddownup"}:
        raise ValueError("Metric tensor cannot be used in mixed index.")

    if index in {"contravariant", "mixedupdown", "mixeddownup"} and metric_index == "covariant":
        g_raise = get_c4_inv(metric_tensor.tensor)
        g_lower = metric_tensor.tensor
    elif index in {"covariant", "mixedupdown", "mixeddownup"} and metric_index == "contravariant":
        g_raise = metric_tensor.tensor
        g_lower = get_c4_inv(metric_tensor.tensor)
    else:
        g_lower = metric_tensor.tensor
        g_raise = get_c4_inv(metric_tensor.tensor)

    if current == "covariant" and index == "contravariant":
        tensor = _flip_index(input_tensor.tensor, g_raise)
    elif current == "contravariant" and index == "covariant":
        tensor = _flip_index(input_tensor.tensor, g_lower)
    elif current == "contravariant" and index == "mixedupdown":
        tensor = _mix_index_2(input_tensor.tensor, g_lower)
    elif current == "contravariant" and index == "mixeddownup":
        tensor = _mix_index_1(input_tensor.tensor, g_lower)
    elif current == "covariant" and index == "mixedupdown":
        tensor = _mix_index_1(input_tensor.tensor, g_raise)
    elif current == "covariant" and index == "mixeddownup":
        tensor = _mix_index_2(input_tensor.tensor, g_raise)
    elif current == "mixedupdown" and index == "contravariant":
        tensor = _mix_index_2(input_tensor.tensor, g_raise)
    elif current == "mixedupdown" and index == "covariant":
        tensor = _mix_index_1(input_tensor.tensor, g_lower)
    elif current == "mixeddownup" and index == "covariant":
        tensor = _mix_index_2(input_tensor.tensor, g_lower)
    elif current == "mixeddownup" and index == "contravariant":
        tensor = _mix_index_1(input_tensor.tensor, g_raise)
    else:
        raise ValueError(f"Unsupported index transformation: {current} to {index}.")

    return _wrap_like(input_tensor, tensor, index)


def matlab_do_frame_transfer(metric: Metric, energy_tensor: Metric) -> Metric:
    """Port target for doFrameTransfer.m using the existing explicit Cholesky implementation."""
    return do_frame_transfer(metric, energy_tensor, "Eulerian")


def matlab_energy_conditions(
    energy_tensor: Metric,
    metric: Metric,
    num_angular_vec: int = 100,
    num_time_vec: int = 10,
) -> Dict[str, np.ndarray]:
    """Port target for getEnergyConditions.m in WarpFactory-compatible mode."""
    energy_eulerian = matlab_do_frame_transfer(metric, energy_tensor)
    return _evaluate_warpfactory_compatible_maps_from_eulerian(
        energy_eulerian.tensor,
        num_vecs=num_angular_vec,
        num_time_vecs=num_time_vec,
    )


def matlab_eval_metric(
    metric: Metric,
    num_angular_vec: int = 100,
    num_time_vec: int = 10,
    solver_method: str = "warpfactory_direct",
) -> MatlabCompatResult:
    """
    Python port of evalMetric.m's core numerical products.

    `solver_method="warpfactory_direct"` mirrors MATLAB met2den.m most closely.
    Passing "christoffel" is useful for comparing pyWarpFactory's independent
    Ricci path against the direct MATLAB formula.
    """
    energy_tensor = solve_energy_tensor(metric, solver_method=solver_method)
    energy_eulerian = matlab_do_frame_transfer(metric, energy_tensor)
    energy_conditions = _evaluate_warpfactory_compatible_maps_from_eulerian(
        energy_eulerian.tensor,
        num_vecs=num_angular_vec,
        num_time_vecs=num_time_vec,
    )
    return MatlabCompatResult(
        metric=metric,
        energy_tensor=energy_tensor,
        energy_tensor_eulerian=energy_eulerian,
        energy_conditions=energy_conditions,
        methodology={
            "mode": "matlab_compat",
            "solver_method": solver_method,
            "energy_condition_method": "warpfactory",
            "num_angular_vec": num_angular_vec,
            "num_time_vec": num_time_vec,
        },
    )


def _flip_index(tensor: np.ndarray, metric: np.ndarray) -> np.ndarray:
    return np.einsum("ab...,ai...,bj...->ij...", tensor, metric, metric)


def _mix_index_1(tensor: np.ndarray, metric: np.ndarray) -> np.ndarray:
    return np.einsum("aj...,ai...->ij...", tensor, metric)


def _mix_index_2(tensor: np.ndarray, metric: np.ndarray) -> np.ndarray:
    return np.einsum("ia...,aj...->ij...", tensor, metric)


def _wrap_like(source: Metric, tensor: np.ndarray, index: str) -> Metric:
    wrapped = Metric(
        tensor=tensor,
        coords=source.coords,
        scaling=source.scaling,
        name=source.name,
        index=index,
        params=source.params,
    )
    if hasattr(source, "type"):
        wrapped.type = source.type
    if hasattr(source, "frame"):
        wrapped.frame = source.frame
    return wrapped
