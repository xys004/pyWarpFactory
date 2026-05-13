import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from warpfactory.analyzer.transform import (
    change_tensor_index,
    do_frame_transfer,
    get_eulerian_transformation_matrix,
)
from warpfactory.generator.base import Metric


def make_constant_metric(matrix, grid_size=(2, 2, 2, 2)):
    tensor = np.broadcast_to(
        np.asarray(matrix, dtype=float).reshape((4, 4) + (1,) * len(grid_size)),
        (4, 4) + tuple(grid_size),
    ).copy()
    coords = {axis: np.zeros(grid_size) for axis in ("t", "x", "y", "z")}
    metric = Metric(tensor=tensor, coords=coords, scaling=np.ones(4), name="metric", index="covariant")
    metric.type = "Metric"
    return metric


def make_energy_tensor(matrix, metric, index="contravariant"):
    tensor = np.broadcast_to(
        np.asarray(matrix, dtype=float).reshape((4, 4) + (1,) * len(metric.tensor.shape[2:])),
        metric.tensor.shape,
    ).copy()
    energy = Metric(
        tensor=tensor,
        coords=metric.coords,
        scaling=metric.scaling,
        name="energy",
        index=index,
    )
    energy.type = "Stress-Energy"
    return energy


def assert_matrix_field_close(actual, expected, atol=1e-12):
    expected_field = np.broadcast_to(
        np.asarray(expected).reshape((4, 4) + (1,) * (actual.ndim - 2)),
        actual.shape,
    )
    np.testing.assert_allclose(actual, expected_field, rtol=1e-12, atol=atol)


def test_eulerian_matrix_flattens_metric(metric):
    M = get_eulerian_transformation_matrix(metric.tensor)
    M_batch = np.moveaxis(M, [0, 1], [-2, -1])
    g_batch = np.moveaxis(metric.tensor, [0, 1], [-2, -1])
    flattened = np.matmul(np.matmul(np.swapaxes(M_batch, -1, -2), g_batch), M_batch)
    flattened = np.moveaxis(flattened, [-2, -1], [0, 1])
    assert_matrix_field_close(flattened, np.diag([-1.0, 1.0, 1.0, 1.0]), atol=2e-12)


def test_index_roundtrip(metric):
    contravariant = np.array(
        [
            [2.0, 0.3, -0.2, 0.1],
            [0.3, 4.0, 0.5, -0.4],
            [-0.2, 0.5, 6.0, 0.7],
            [0.1, -0.4, 0.7, 8.0],
        ]
    )
    energy = make_energy_tensor(contravariant, metric, index="contravariant")
    lowered = change_tensor_index(energy, "covariant", metric)
    raised = change_tensor_index(lowered, "contravariant", metric)
    np.testing.assert_allclose(raised.tensor, energy.tensor, rtol=1e-12, atol=1e-12)


def test_minkowski_full_tensor_identity():
    metric = make_constant_metric(np.diag([-1.0, 1.0, 1.0, 1.0]))
    tensor = np.array(
        [
            [5.0, 1.0, -2.0, 3.0],
            [1.0, 7.0, 0.25, -0.5],
            [-2.0, 0.25, 11.0, 0.75],
            [3.0, -0.5, 0.75, 13.0],
        ]
    )
    energy = make_energy_tensor(tensor, metric, index="contravariant")
    transferred = do_frame_transfer(metric, energy)
    assert_matrix_field_close(transferred.tensor, tensor)


def test_shift_metric_matches_manual_tetrad_transform():
    beta = 0.2
    # ADM-like constant shift metric with flat spatial metric:
    # ds^2 = (-1 + beta^2) dt^2 + 2 beta dt dx + dx^2 + dy^2 + dz^2.
    metric_matrix = np.array(
        [
            [-1.0 + beta**2, beta, 0.0, 0.0],
            [beta, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    metric = make_constant_metric(metric_matrix)
    test_eulerian_matrix_flattens_metric(metric)
    test_index_roundtrip(metric)

    tensor = np.array(
        [
            [10.0, 2.0, 0.5, -0.25],
            [2.0, 4.0, 0.1, 0.2],
            [0.5, 0.1, 3.0, -0.3],
            [-0.25, 0.2, -0.3, 2.0],
        ]
    )
    energy = make_energy_tensor(tensor, metric, index="contravariant")
    lowered = change_tensor_index(energy, "covariant", metric)
    M = get_eulerian_transformation_matrix(metric.tensor)
    M_batch = np.moveaxis(M, [0, 1], [-2, -1])
    T_batch = np.moveaxis(lowered.tensor, [0, 1], [-2, -1])
    manual = np.matmul(np.matmul(np.swapaxes(M_batch, -1, -2), T_batch), M_batch)
    manual = np.moveaxis(manual, [-2, -1], [0, 1])
    for i in range(1, 4):
        manual[0, i] *= -1
        manual[i, 0] *= -1

    transferred = do_frame_transfer(metric, energy)
    np.testing.assert_allclose(transferred.tensor, manual, rtol=1e-12, atol=1e-12)


def main():
    test_minkowski_full_tensor_identity()
    test_shift_metric_matches_manual_tetrad_transform()
    print("Frame transfer known-metric checks passed.")


if __name__ == "__main__":
    main()
