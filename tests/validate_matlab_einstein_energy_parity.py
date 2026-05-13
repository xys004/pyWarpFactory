import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from warpfactory.constants import C, G
from warpfactory.solver.tensor_utils import (
    get_c4_inv,
    get_einstein_tensor,
    get_energy_tensor,
    get_ricci_scalar,
)


def matlab_ricci_scalar_reference(ricci_tensor, metric_inverse):
    out = np.zeros(ricci_tensor.shape[2:])
    for mu in range(4):
        for nu in range(4):
            out += metric_inverse[mu, nu] * ricci_tensor[mu, nu]
    return out


def matlab_einstein_tensor_reference(ricci_tensor, ricci_scalar, metric_tensor):
    out = np.zeros_like(metric_tensor)
    for mu in range(4):
        for nu in range(4):
            out[mu, nu] = ricci_tensor[mu, nu] - 0.5 * metric_tensor[mu, nu] * ricci_scalar
    return out


def matlab_energy_tensor_reference(einstein_tensor, metric_inverse):
    covariant_energy = (C**4) / (8.0 * np.pi * G) * einstein_tensor
    out = np.zeros_like(einstein_tensor)
    for mu in range(4):
        for nu in range(4):
            for alpha in range(4):
                for beta in range(4):
                    out[mu, nu] += (
                        covariant_energy[alpha, beta]
                        * metric_inverse[alpha, mu]
                        * metric_inverse[beta, nu]
                    )
    return out


def main():
    rng = np.random.default_rng(240502709)
    shape = (2, 3, 4, 5)

    metric = np.zeros((4, 4) + shape)
    metric += np.diag([-1.0, 1.0, 1.0, 1.0]).reshape((4, 4) + (1,) * len(shape))
    perturb = 0.02 * rng.normal(size=(4, 4) + shape)
    metric += 0.5 * (perturb + np.swapaxes(perturb, 0, 1))
    metric_inverse = get_c4_inv(metric)

    ricci = rng.normal(size=(4, 4) + shape)
    ricci = 0.5 * (ricci + np.swapaxes(ricci, 0, 1))

    scalar_expected = matlab_ricci_scalar_reference(ricci, metric_inverse)
    scalar_actual = get_ricci_scalar(ricci, metric_inverse)
    np.testing.assert_allclose(scalar_actual, scalar_expected, rtol=0, atol=0)

    einstein_expected = matlab_einstein_tensor_reference(ricci, scalar_expected, metric)
    einstein_actual = get_einstein_tensor(ricci, scalar_actual, metric)
    np.testing.assert_allclose(einstein_actual, einstein_expected, rtol=0, atol=0)

    energy_expected = matlab_energy_tensor_reference(einstein_expected, metric_inverse)
    energy_actual = get_energy_tensor(einstein_actual, metric_inverse)
    np.testing.assert_allclose(energy_actual, energy_expected, rtol=1e-13, atol=1e20)

    print("MATLAB Ricci scalar, Einstein tensor, and energy tensor parity checks passed.")


if __name__ == "__main__":
    main()
