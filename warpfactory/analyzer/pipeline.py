from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np

from warpfactory.analyzer.energy_conditions import _condition_alias, calculate_energy_conditions
from warpfactory.analyzer.energy_conditions import _lower_indices_minkowski
from warpfactory.analyzer.observer_search import optimize_timelike_conditions
from warpfactory.analyzer.transform import do_frame_transfer
from warpfactory.generator.base import Metric
from warpfactory.solver.solvers import solve_energy_tensor


@dataclass
class AnalysisResult:
    """Container for a complete metric analysis."""

    metric: Metric
    energy_tensor: Metric
    energy_conditions: Dict[str, np.ndarray]
    eulerian_energy_tensor: Optional[np.ndarray]
    summary: Dict[str, float]
    methodology: Dict[str, object] = field(default_factory=dict)
    observer_audit: Dict[str, object] = field(default_factory=dict)

    def min_condition(self, name: str) -> float:
        """Return the minimum value of an energy-condition map."""
        return float(np.nanmin(self.energy_conditions[name]))

    def has_violation(self, name: str, tolerance: float = 0.0) -> bool:
        """Return True when a condition map has values below -tolerance."""
        return self.min_condition(name) < -abs(tolerance)


def _normalize_conditions(conditions: Optional[Iterable[str]]) -> Optional[list]:
    if conditions is None:
        return None
    return [_condition_alias(condition) for condition in conditions]


def summarize_energy_conditions(
    conditions: Dict[str, np.ndarray],
    mask: Optional[np.ndarray] = None,
    tolerance: float = 0.0,
) -> Dict[str, float]:
    """Create scalar summary statistics for energy-condition maps."""
    summary = {}
    threshold = -abs(tolerance)
    for name, values in conditions.items():
        selected = values[mask] if mask is not None else values
        summary[f"{name}_min"] = float(np.nanmin(selected))
        summary[f"{name}_max"] = float(np.nanmax(selected))
        summary[f"{name}_evaluated_points"] = int(np.size(selected))
        summary[f"{name}_violating_points"] = int(np.count_nonzero(selected < threshold))
        summary[f"{name}_violating_fraction"] = float(np.mean(selected < threshold))
    return summary


def _candidate_points(values: np.ndarray, count: int) -> list:
    flat = values.reshape(-1)
    count = min(count, flat.size)
    if count <= 0:
        return []
    indices = np.argpartition(flat, count - 1)[:count]
    indices = indices[np.argsort(flat[indices])]
    return [tuple(int(i) for i in np.unravel_index(index, values.shape)) for index in indices]


def _optimized_observer_audit(
    eulerian_tensor: np.ndarray,
    sampled_conditions: Dict[str, np.ndarray],
    audit_points: int,
    max_speed: float,
) -> Dict[str, object]:
    audit = {
        "mode": "optimized",
        "scope": "candidate_points",
        "max_speed": max_speed,
        "points_per_condition": audit_points,
        "conditions": {},
    }

    for condition in ("Weak", "Strong", "Dominant"):
        if condition not in sampled_conditions:
            continue

        entries = []
        for point in _candidate_points(sampled_conditions[condition], audit_points):
            local_upper = eulerian_tensor[(slice(None), slice(None)) + point]
            local_lower = _lower_indices_minkowski(local_upper)
            optimized = optimize_timelike_conditions(
                local_lower,
                conditions=(condition,),
                max_speed=max_speed,
            )[condition]
            entries.append(
                {
                    "point": point,
                    "sampled_value": float(sampled_conditions[condition][point]),
                    "optimized_value": optimized.value,
                    "observer": optimized.observer.tolist(),
                    "iterations": optimized.iterations,
                    "converged": optimized.converged,
                    "violates": optimized.violates,
                }
            )

        audit["conditions"][condition] = entries

    return audit


def analyze_metric(
    metric: Metric,
    conditions: Optional[Iterable[str]] = None,
    num_vecs: int = 50,
    num_time_vecs: int = 10,
    observer_mode: str = "sampled",
    audit_points: int = 3,
    optimized_max_speed: float = 0.999,
    flat_tolerance: Optional[float] = None,
    energy_condition_method: str = "standard",
    solver_method: str = "christoffel",
) -> AnalysisResult:
    """
    Run the common pyWarpFactory workflow in one call.

    The workflow solves the stress-energy tensor, transforms it to the Eulerian
    frame, evaluates currently supported energy conditions, and returns a compact
    summary next to the raw arrays.

    observer_mode='sampled' is the conservative default. observer_mode='optimized'
    adds a bounded local observer search at candidate points as an audit layer.
    """
    if observer_mode not in {"sampled", "optimized"}:
        raise ValueError("observer_mode must be 'sampled' or 'optimized'.")
    if energy_condition_method not in {"standard", "warpfactory"}:
        raise ValueError("energy_condition_method must be 'standard' or 'warpfactory'.")

    requested = _normalize_conditions(conditions)
    energy_tensor = solve_energy_tensor(metric, flat_tolerance=flat_tolerance, solver_method=solver_method)
    all_conditions = calculate_energy_conditions(
        energy_tensor,
        metric,
        num_vecs=num_vecs,
        num_time_vecs=num_time_vecs,
        method=energy_condition_method,
    )

    if requested is None:
        selected = all_conditions
    else:
        selected = {}
        for condition in requested:
            selected[condition] = calculate_energy_conditions(
                energy_tensor,
                metric,
                condition=condition,
                num_vecs=num_vecs,
                num_time_vecs=num_time_vecs,
                method=energy_condition_method,
            )

    eulerian = do_frame_transfer(metric, energy_tensor, "Eulerian").tensor
    methodology = {
        "energy_condition_mode": energy_condition_method,
        "observer_mode": observer_mode,
        "num_null_directions": num_vecs,
        "num_timelike_shells": num_time_vecs,
        "timelike_speed_samples": (0.0, 0.5, 0.9, 0.99),
        "flat_tolerance": flat_tolerance,
        "solver_method": solver_method,
        "solver_diagnostics": energy_tensor.params.get("solver_diagnostics", {}),
        "notes": (
            "Sampled maps are finite-direction diagnostics in the Eulerian "
            "orthonormal frame. Optimized mode adds bounded local observer "
            "search at selected candidate points."
        ),
    }
    observer_audit = {}

    if observer_mode == "optimized":
        observer_audit = _optimized_observer_audit(
            eulerian,
            selected,
            audit_points=audit_points,
            max_speed=optimized_max_speed,
        )
        methodology["optimized_max_speed"] = optimized_max_speed
        methodology["optimized_audit_points"] = audit_points

    return AnalysisResult(
        metric=metric,
        energy_tensor=energy_tensor,
        energy_conditions=selected,
        eulerian_energy_tensor=eulerian,
        summary=summarize_energy_conditions(selected),
        methodology=methodology,
        observer_audit=observer_audit,
    )
