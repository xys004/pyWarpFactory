import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory.recipes.fuchs_warp_shell import create_fuchs_constant_warp_shell
from warpfactory.solver.tensor_utils import (
    get_ricci_tensor_christoffel_from_warpfactory_derivatives,
    get_ricci_tensor_warpfactory_direct,
)


def _interior_mask(shape, margin=3):
    mask = np.ones(shape, dtype=bool)
    for axis, size in enumerate(shape):
        if size <= 2 * margin:
            continue
        index = [slice(None)] * len(shape)
        index[axis] = slice(0, margin)
        mask[tuple(index)] = False
        index[axis] = slice(size - margin, size)
        mask[tuple(index)] = False
    return mask


def main():
    metric = create_fuchs_constant_warp_shell(profile="quick")
    metric.tensor[0, 1] = 0.0
    metric.tensor[1, 0] = 0.0

    direct = get_ricci_tensor_warpfactory_direct(metric.tensor, metric.scaling)
    bridge = get_ricci_tensor_christoffel_from_warpfactory_derivatives(metric.tensor, metric.scaling)

    mask = _interior_mask(metric.tensor.shape[2:])
    diff = np.abs(direct - bridge)
    max_error = float(np.max(diff[:, :, mask]))
    print(f"WarpFactory direct vs same-derivative Christoffel bridge max error = {max_error:.4e}")
    assert max_error < 1e-12

    print("WarpFactory direct Ricci bridge equivalence check passed.")


if __name__ == "__main__":
    main()
