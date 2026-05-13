import numpy as np
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from warpfactory.solver.finite_difference import (
    derive_1st_4th_order,
    derive_2nd_4th_order,
)


def _idx(axis, idx, ndim):
    sl = [slice(None)] * ndim
    sl[axis] = idx
    return tuple(sl)


def _range(axis, start, stop, ndim):
    sl = [slice(None)] * ndim
    sl[axis] = slice(start, stop)
    return tuple(sl)


def matlab_take_finite_difference1_reference(A, axis, delta, phiphi_flag=False):
    B = np.zeros_like(A)
    if A.shape[axis] < 5:
        return B

    B[_range(axis, 2, -2, A.ndim)] = (
        -(
            A[_range(axis, 4, None, A.ndim)]
            - A[_range(axis, None, -4, A.ndim)]
        )
        + 8
        * (
            A[_range(axis, 3, -1, A.ndim)]
            - A[_range(axis, 1, -3, A.ndim)]
        )
    ) / (12 * delta)

    if phiphi_flag and axis == 2:
        B[_idx(axis, 0, A.ndim)] = 2 * 4
        B[_idx(axis, 1, A.ndim)] = 2 * 3
        B[_idx(axis, -2, A.ndim)] = 2 * (A.shape[axis] - 6)
        B[_idx(axis, -1, A.ndim)] = 2 * (A.shape[axis] - 5)
    else:
        B[_idx(axis, 0, A.ndim)] = B[_idx(axis, 2, A.ndim)]
        B[_idx(axis, 1, A.ndim)] = B[_idx(axis, 2, A.ndim)]
        B[_idx(axis, -2, A.ndim)] = B[_idx(axis, -3, A.ndim)]
        B[_idx(axis, -1, A.ndim)] = B[_idx(axis, -3, A.ndim)]
    return B


def matlab_take_finite_difference2_reference(
    A, axis1, axis2, delta1, delta2=None, phiphi_flag=False
):
    if delta2 is None:
        delta2 = delta1

    B = np.zeros_like(A)
    if A.shape[axis1] < 5 or A.shape[axis2] < 5:
        return B

    if axis1 == axis2:
        axis = axis1
        B[_range(axis, 2, -2, A.ndim)] = (
            -(
                A[_range(axis, 4, None, A.ndim)]
                + A[_range(axis, None, -4, A.ndim)]
            )
            + 16
            * (
                A[_range(axis, 3, -1, A.ndim)]
                + A[_range(axis, 1, -3, A.ndim)]
            )
            - 30 * A[_range(axis, 2, -2, A.ndim)]
        ) / (12 * delta1**2)

        if phiphi_flag and axis == 2:
            B[_idx(axis, 0, A.ndim)] = -2
            B[_idx(axis, 1, A.ndim)] = -2
            B[_idx(axis, -2, A.ndim)] = 2
            B[_idx(axis, -1, A.ndim)] = 2
        else:
            B[_idx(axis, 0, A.ndim)] = B[_idx(axis, 2, A.ndim)]
            B[_idx(axis, 1, A.ndim)] = B[_idx(axis, 2, A.ndim)]
            B[_idx(axis, -2, A.ndim)] = B[_idx(axis, -3, A.ndim)]
            B[_idx(axis, -1, A.ndim)] = B[_idx(axis, -3, A.ndim)]
        return B

    weights = {-2: 1.0, -1: -8.0, 1: 8.0, 2: -1.0}
    interior = [slice(None)] * A.ndim
    interior[axis1] = slice(2, -2)
    interior[axis2] = slice(2, -2)
    target = tuple(interior)

    mixed = np.zeros_like(B[target])
    for offset1, weight1 in weights.items():
        src1 = [slice(None)] * A.ndim
        src1[axis1] = slice(2 + offset1, A.shape[axis1] - 2 + offset1)
        for offset2, weight2 in weights.items():
            src = src1.copy()
            src[axis2] = slice(2 + offset2, A.shape[axis2] - 2 + offset2)
            mixed += weight1 * weight2 * A[tuple(src)]

    B[target] = mixed / (144.0 * delta1 * delta2)
    return B


def main():
    rng = np.random.default_rng(240502709)
    A = rng.normal(size=(7, 8, 9, 10))
    deltas = [0.7, 1.1, 1.3, 1.7]

    for axis, delta in enumerate(deltas):
        expected = matlab_take_finite_difference1_reference(A, axis, delta)
        actual = derive_1st_4th_order(A, axis, delta)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

        expected2 = matlab_take_finite_difference2_reference(A, axis, axis, delta)
        actual2 = derive_2nd_4th_order(A, axis, axis, delta)
        np.testing.assert_allclose(actual2, expected2, rtol=0, atol=0)

    expected_phi_1 = matlab_take_finite_difference1_reference(
        A, 2, deltas[2], phiphi_flag=True
    )
    actual_phi_1 = derive_1st_4th_order(A, 2, deltas[2], phiphi_flag=True)
    np.testing.assert_allclose(actual_phi_1, expected_phi_1, rtol=0, atol=0)

    expected_phi_2 = matlab_take_finite_difference2_reference(
        A, 2, 2, deltas[2], phiphi_flag=True
    )
    actual_phi_2 = derive_2nd_4th_order(A, 2, 2, deltas[2], phiphi_flag=True)
    np.testing.assert_allclose(actual_phi_2, expected_phi_2, rtol=0, atol=0)

    for axis1 in range(4):
        for axis2 in range(axis1 + 1, 4):
            expected = matlab_take_finite_difference2_reference(
                A, axis1, axis2, deltas[axis1], deltas[axis2]
            )
            actual = derive_2nd_4th_order(
                A, axis1, axis2, deltas[axis1], deltas[axis2]
            )
            np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    print("MATLAB finite-difference parity checks passed.")


if __name__ == "__main__":
    main()
