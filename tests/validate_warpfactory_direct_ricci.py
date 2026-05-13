import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from warpfactory.solver.tensor_utils import (
    get_ricci_scalar,
    get_ricci_tensor,
    get_ricci_tensor_warpfactory_direct,
    get_c4_inv,
)


def minkowski_metric(grid_size):
    tensor = np.zeros((4, 4) + tuple(grid_size), dtype=float)
    tensor[0, 0] = -1.0
    tensor[1, 1] = 1.0
    tensor[2, 2] = 1.0
    tensor[3, 3] = 1.0
    return tensor


def constant_boosted_minkowski_metric(grid_size, beta=0.2):
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
    return np.broadcast_to(boosted.reshape((4, 4) + (1,) * len(grid_size)), (4, 4) + tuple(grid_size)).copy()


def rindler_metric(grid_size, grid_scale, acceleration=0.015):
    tensor = minkowski_metric(grid_size)
    x = np.arange(grid_size[1], dtype=float) * grid_scale[1]
    x = x - x.mean()
    lapse = 1.0 + acceleration * x
    tensor[0, 0] = -(lapse.reshape(1, grid_size[1], 1, 1) ** 2)
    return tensor


def assert_ricci_zero(metric, grid_scale, label, tolerance=1e-8, christoffel_tolerance=1e-8):
    direct = get_ricci_tensor_warpfactory_direct(metric, grid_scale)
    christoffel = get_ricci_tensor(metric, grid_scale)

    interior = (slice(None), slice(None), slice(2, -2), slice(2, -2), slice(2, -2), slice(2, -2))
    direct_abs = float(np.max(np.abs(direct[interior])))
    christoffel_abs = float(np.max(np.abs(christoffel[interior])))

    scalar = get_ricci_scalar(direct, get_c4_inv(metric))
    scalar_abs = float(np.max(np.abs(scalar[(slice(2, -2),) * 4])))

    assert direct_abs < tolerance, f"{label}: direct Ricci max {direct_abs}"
    assert christoffel_abs < christoffel_tolerance, f"{label}: Christoffel Ricci max {christoffel_abs}"
    assert scalar_abs < tolerance, f"{label}: direct Ricci scalar max {scalar_abs}"

    print(
        f"{label}: direct={direct_abs:.4e}, christoffel={christoffel_abs:.4e}, scalar={scalar_abs:.4e}"
    )


def main():
    grid_size = (7, 9, 9, 9)
    grid_scale = (0.5, 0.25, 0.25, 0.25)

    assert_ricci_zero(minkowski_metric(grid_size), grid_scale, "Minkowski")
    assert_ricci_zero(
        constant_boosted_minkowski_metric(grid_size),
        grid_scale,
        "constant boosted Minkowski",
    )
    assert_ricci_zero(
        rindler_metric(grid_size, grid_scale),
        grid_scale,
        "Rindler",
        christoffel_tolerance=2e-4,
    )

    print("WarpFactory direct Ricci flat-spacetime checks passed.")


if __name__ == "__main__":
    main()
