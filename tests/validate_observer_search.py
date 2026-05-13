import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory import optimize_timelike_conditions


def assert_unit_timelike(observer, tolerance=1e-9):
    norm = -observer[0] ** 2 + float(np.dot(observer[1:], observer[1:]))
    assert abs(norm + 1.0) < tolerance
    assert observer[0] > 0


def validate_minkowski_zero_tensor():
    T = np.zeros((4, 4))
    results = optimize_timelike_conditions(
        T,
        direction_count=4,
        max_speed=0.95,
        max_iter=30,
        min_step=1e-4,
    )

    for condition, result in results.items():
        assert_unit_timelike(result.observer)
        assert abs(result.value) < 1e-9, (condition, result.value)
        assert not result.violates

    print("SUCCESS: optimized observer search keeps zero tensor neutral.")


def validate_known_diagonal_tensor():
    # Covariant stress tensor in a local Minkowski frame:
    # rho=1, pressures=(-2,-2,-2). WEC holds for the comoving observer,
    # while SEC is violated because rho + p_x + p_y + p_z < 0.
    T = np.diag([1.0, -2.0, -2.0, -2.0])
    results = optimize_timelike_conditions(
        T,
        conditions=("WEC", "SEC", "DEC"),
        direction_count=8,
        max_speed=0.9,
        max_iter=40,
        min_step=1e-4,
    )

    for result in results.values():
        assert_unit_timelike(result.observer)

    assert results["Weak"].value < 0.0
    assert results["Strong"].value < 0.0
    assert results["Dominant"].value < 0.0

    print("SUCCESS: optimized observer search finds violations in diagonal test tensor.")
    print(f"WEC minimum: {results['Weak'].value:.4e}")
    print(f"SEC minimum: {results['Strong'].value:.4e}")
    print(f"DEC minimum: {results['Dominant'].value:.4e}")


def validate():
    validate_minkowski_zero_tensor()
    validate_known_diagonal_tensor()


if __name__ == "__main__":
    validate()
