import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory.solver.finite_difference import derive_1st_4th_order, derive_2nd_4th_order


def _interior(shape):
    return tuple(slice(2, n - 2) for n in shape)


def validate():
    shape = (7, 11, 12, 13)
    deltas = (0.4, 0.3, 0.25, 0.2)
    t, x, y, z = np.meshgrid(
        *(np.arange(n, dtype=float) * h for n, h in zip(shape, deltas)),
        indexing="ij",
    )

    f = (
        1.7
        + 0.2 * t
        - 0.4 * x
        + 0.3 * y**2
        - 0.15 * z**3
        + 0.11 * x * y
        - 0.07 * y * z
        + 0.05 * x * z
        + 0.09 * t * x
        - 0.08 * t * y
    )

    first_expected = (
        0.2 + 0.09 * x - 0.08 * y,
        -0.4 + 0.11 * y + 0.05 * z + 0.09 * t,
        0.6 * y + 0.11 * x - 0.07 * z - 0.08 * t,
        -0.45 * z**2 - 0.07 * y + 0.05 * x,
    )
    second_expected = {
        (0, 0): np.zeros_like(f),
        (1, 1): np.zeros_like(f),
        (2, 2): 0.6 * np.ones_like(f),
        (3, 3): -0.9 * z,
        (0, 1): 0.09 * np.ones_like(f),
        (0, 2): -0.08 * np.ones_like(f),
        (0, 3): np.zeros_like(f),
        (1, 2): 0.11 * np.ones_like(f),
        (1, 3): 0.05 * np.ones_like(f),
        (2, 3): -0.07 * np.ones_like(f),
    }

    interior = _interior(shape)
    for axis, delta in enumerate(deltas):
        actual = derive_1st_4th_order(f, axis, delta)
        err = float(np.max(np.abs(actual[interior] - first_expected[axis][interior])))
        print(f"d{axis} max interior error = {err:.4e}")
        assert err < 1e-11

    for axis1 in range(4):
        for axis2 in range(axis1, 4):
            actual = derive_2nd_4th_order(f, axis1, axis2, deltas[axis1], deltas[axis2])
            expected = second_expected[(axis1, axis2)]
            err = float(np.max(np.abs(actual[interior] - expected[interior])))
            print(f"d{axis1}{axis2} max interior error = {err:.4e}")
            assert err < 1e-10

            if axis1 != axis2:
                reversed_actual = derive_2nd_4th_order(
                    f, axis2, axis1, deltas[axis2], deltas[axis1]
                )
                symmetry_err = float(np.max(np.abs(actual[interior] - reversed_actual[interior])))
                print(f"d{axis1}{axis2} symmetry error = {symmetry_err:.4e}")
                assert symmetry_err < 1e-12

    print("Manufactured finite-derivative checks passed.")


if __name__ == "__main__":
    validate()
