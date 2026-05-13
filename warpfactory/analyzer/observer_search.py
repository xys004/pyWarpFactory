from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import numpy as np


CANONICAL_CONDITIONS = {
    "null": "Null",
    "nec": "Null",
    "weak": "Weak",
    "wec": "Weak",
    "strong": "Strong",
    "sec": "Strong",
    "dominant": "Dominant",
    "dec": "Dominant",
}


@dataclass
class ObserverSearchResult:
    """Result of a local optimized observer search."""

    condition: str
    value: float
    observer: np.ndarray
    spatial_momentum: np.ndarray
    iterations: int
    converged: bool

    @property
    def violates(self) -> bool:
        return self.value < 0


def condition_alias(condition: str) -> str:
    key = condition.lower()
    if key not in CANONICAL_CONDITIONS:
        supported = ", ".join(sorted(set(CANONICAL_CONDITIONS.values())))
        raise NotImplementedError(
            f"Energy condition '{condition}' is not implemented. "
            f"Supported conditions: {supported}."
        )
    return CANONICAL_CONDITIONS[key]


def unit_timelike_from_spatial_momentum(momentum: np.ndarray) -> np.ndarray:
    """
    Build a future-directed unit timelike vector u=(sqrt(1+p.p), p).

    With eta=(-,+,+,+), this guarantees eta_ab u^a u^b = -1.
    """
    p = np.asarray(momentum, dtype=float)
    u = np.empty(4, dtype=float)
    u[0] = np.sqrt(1.0 + float(np.dot(p, p)))
    u[1:] = p
    return u


def max_momentum_from_speed(max_speed: float) -> float:
    if max_speed < 0 or max_speed >= 1:
        raise ValueError("max_speed must satisfy 0 <= max_speed < 1.")
    gamma = 1.0 / np.sqrt(1.0 - max_speed**2)
    return gamma * max_speed


def project_to_ball(momentum: np.ndarray, radius: float) -> np.ndarray:
    norm = float(np.linalg.norm(momentum))
    if norm <= radius or norm == 0:
        return momentum
    return momentum * (radius / norm)


def trace_minkowski(T_lower: np.ndarray) -> float:
    return float(-T_lower[0, 0] + T_lower[1, 1] + T_lower[2, 2] + T_lower[3, 3])


def trace_reversed_tensor(T_lower: np.ndarray) -> np.ndarray:
    eta = np.diag([-1.0, 1.0, 1.0, 1.0])
    return T_lower - 0.5 * trace_minkowski(T_lower) * eta


def mixed_tensor_from_lower(T_lower: np.ndarray) -> np.ndarray:
    mixed = T_lower.copy()
    mixed[0, :] = -mixed[0, :]
    return mixed


def contract(T_lower: np.ndarray, vector: np.ndarray) -> float:
    return float(vector @ T_lower @ vector)


def dominant_energy_margin(T_lower: np.ndarray, observer: np.ndarray) -> float:
    rho = contract(T_lower, observer)
    flux = -(mixed_tensor_from_lower(T_lower) @ observer)
    flux_norm = -flux[0] ** 2 + float(np.dot(flux[1:], flux[1:]))
    return float(min(rho, flux[0], -flux_norm))


def condition_objective(T_lower: np.ndarray, condition: str) -> Callable[[np.ndarray], float]:
    canonical = condition_alias(condition)
    if canonical == "Weak":
        return lambda p: contract(T_lower, unit_timelike_from_spatial_momentum(p))
    if canonical == "Strong":
        S_lower = trace_reversed_tensor(T_lower)
        return lambda p: contract(S_lower, unit_timelike_from_spatial_momentum(p))
    if canonical == "Dominant":
        return lambda p: dominant_energy_margin(T_lower, unit_timelike_from_spatial_momentum(p))
    raise NotImplementedError("Null condition optimization is handled separately.")


def fibonacci_directions(count: int) -> np.ndarray:
    indices = np.arange(count, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / count)
    theta = np.pi * (1 + 5**0.5) * indices
    dirs = np.zeros((count, 3), dtype=float)
    dirs[:, 0] = np.cos(theta) * np.sin(phi)
    dirs[:, 1] = np.sin(theta) * np.sin(phi)
    dirs[:, 2] = np.cos(phi)
    return dirs


def default_seed_momenta(max_momentum: float, direction_count: int = 12) -> np.ndarray:
    speeds = (0.0, 0.35, 0.75, 0.95, 1.0)
    directions = fibonacci_directions(direction_count)
    seeds = [np.zeros(3)]
    for fraction in speeds[1:]:
        radius = fraction * max_momentum
        seeds.extend(radius * directions)
    return np.asarray(seeds)


def pattern_search_minimize(
    objective: Callable[[np.ndarray], float],
    seeds: np.ndarray,
    max_radius: float,
    initial_step: Optional[float] = None,
    min_step: float = 1e-5,
    max_iter: int = 200,
) -> tuple:
    """Derivative-free projected coordinate search on a 3-ball."""
    if initial_step is None:
        initial_step = max(max_radius / 3.0, min_step)

    axes = np.eye(3)
    best_p = None
    best_value = np.inf
    iterations = 0

    for seed in seeds:
        p = project_to_ball(np.asarray(seed, dtype=float), max_radius)
        value = objective(p)
        step = initial_step
        local_iter = 0

        while step > min_step and local_iter < max_iter:
            improved = False
            for axis in axes:
                for sign in (-1.0, 1.0):
                    candidate = project_to_ball(p + sign * step * axis, max_radius)
                    candidate_value = objective(candidate)
                    local_iter += 1
                    if candidate_value < value:
                        p = candidate
                        value = candidate_value
                        improved = True

            if not improved:
                step *= 0.5

        iterations += local_iter
        if value < best_value:
            best_value = value
            best_p = p.copy()

    converged = iterations > 0 and best_p is not None
    return best_p, float(best_value), iterations, converged


def optimize_timelike_condition(
    T_lower: np.ndarray,
    condition: str,
    max_speed: float = 0.999,
    direction_count: int = 12,
    initial_step: Optional[float] = None,
    min_step: float = 1e-5,
    max_iter: int = 200,
) -> ObserverSearchResult:
    """
    Find a low-value timelike observer for WEC, SEC, or DEC at one grid point.

    The search is bounded by max_speed. If the objective keeps improving at the
    boundary, the result is a strong diagnostic but not a global proof.
    """
    canonical = condition_alias(condition)
    if canonical == "Null":
        raise NotImplementedError("Optimized NEC search will use a separate null-direction solver.")

    T_lower = np.asarray(T_lower, dtype=float)
    if T_lower.shape != (4, 4):
        raise ValueError(f"T_lower must have shape (4, 4), got {T_lower.shape}.")

    max_radius = max_momentum_from_speed(max_speed)
    seeds = default_seed_momenta(max_radius, direction_count=direction_count)
    objective = condition_objective(T_lower, canonical)
    p, value, iterations, converged = pattern_search_minimize(
        objective,
        seeds,
        max_radius=max_radius,
        initial_step=initial_step,
        min_step=min_step,
        max_iter=max_iter,
    )

    return ObserverSearchResult(
        condition=canonical,
        value=value,
        observer=unit_timelike_from_spatial_momentum(p),
        spatial_momentum=p,
        iterations=iterations,
        converged=converged,
    )


def optimize_timelike_conditions(
    T_lower: np.ndarray,
    conditions: Optional[Iterable[str]] = None,
    **kwargs,
) -> Dict[str, ObserverSearchResult]:
    if conditions is None:
        conditions = ("Weak", "Strong", "Dominant")

    results = {}
    for condition in conditions:
        canonical = condition_alias(condition)
        results[canonical] = optimize_timelike_condition(T_lower, canonical, **kwargs)
    return results
