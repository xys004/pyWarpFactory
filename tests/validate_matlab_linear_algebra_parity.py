import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from warpfactory.solver.tensor_utils import get_c4_inv


def main():
    rng = np.random.default_rng(240502709)
    shape = (3, 4, 5, 6)

    metric = np.zeros((4, 4) + shape)
    eta = np.diag([-1.0, 1.0, 1.0, 1.0])
    metric += eta.reshape((4, 4) + (1,) * len(shape))

    perturb = 0.03 * rng.normal(size=(4, 4) + shape)
    perturb = 0.5 * (perturb + np.swapaxes(perturb, 0, 1))
    metric += perturb

    inv_metric = get_c4_inv(metric)
    product_t = np.matmul(
        np.moveaxis(metric, [0, 1], [-2, -1]),
        np.moveaxis(inv_metric, [0, 1], [-2, -1]),
    )
    product = np.moveaxis(product_t, [-2, -1], [0, 1])
    identity = np.broadcast_to(np.eye(4).reshape((4, 4) + (1,) * len(shape)), product.shape)

    np.testing.assert_allclose(product, identity, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(inv_metric, np.swapaxes(inv_metric, 0, 1), rtol=1e-12, atol=1e-12)

    print("Metric inverse axis/broadcasting parity checks passed.")


if __name__ == "__main__":
    main()
