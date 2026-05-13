import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory.analyzer.matlab_compat import (
    matlab_change_tensor_index,
    matlab_energy_conditions,
    matlab_eval_metric,
    matlab_even_points_on_sphere,
    matlab_generate_uniform_field,
)
from warpfactory.generator.base import Metric
from warpfactory.generator.minkowski import create_minkowski_metric


def assert_close(value, expected, tolerance=1e-12):
    if not np.allclose(value, expected, atol=tolerance, rtol=0.0):
        raise AssertionError(f"Expected {expected}, got {value}")


def validate_even_points():
    points = matlab_even_points_on_sphere(1.0, 4)
    golden_ratio = (1.0 + 5.0**0.5) / 2.0
    expected = np.zeros((3, 4))
    for i in range(4):
        theta = 2.0 * np.pi * i / golden_ratio
        phi = np.arccos(1.0 - 2.0 * (i + 0.5) / 4)
        expected[:, i] = [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    assert_close(points, expected)

    null_field = matlab_generate_uniform_field("nulllike", 4)
    euclidean_norm = np.sqrt(np.sum(null_field**2, axis=0))
    assert_close(euclidean_norm, np.ones(4))


def validate_index_changes():
    shape = (1, 2, 2, 2)
    metric_tensor = np.zeros((4, 4) + shape)
    for mu, sign in enumerate((-1.0, 1.0, 1.0, 1.0)):
        metric_tensor[mu, mu] = sign
    metric = Metric(metric_tensor, coords={}, scaling=np.ones(4), name="metric", index="covariant")
    metric.type = "metric"

    tensor = np.zeros((4, 4) + shape)
    tensor[0, 1] = 2.0
    tensor[1, 0] = 3.0
    energy = Metric(tensor, coords={}, scaling=np.ones(4), name="energy", index="contravariant")
    energy.type = "Stress-Energy"

    cov = matlab_change_tensor_index(energy, "covariant", metric)
    assert_close(cov.tensor[0, 1], -2.0)
    assert_close(cov.tensor[1, 0], -3.0)

    mixed = matlab_change_tensor_index(energy, "mixedupdown", metric)
    assert_close(mixed.tensor[0, 1], 2.0)
    assert_close(mixed.tensor[1, 0], -3.0)

    mixed_down_up = matlab_change_tensor_index(energy, "mixeddownup", metric)
    assert_close(mixed_down_up.tensor[0, 1], -2.0)
    assert_close(mixed_down_up.tensor[1, 0], 3.0)


def validate_zero_energy_conditions():
    metric = create_minkowski_metric((1, 4, 4, 4), (1.0, 1.0, 1.0, 1.0))
    zero = Metric(np.zeros_like(metric.tensor), metric.coords, metric.scaling, "zero", "contravariant")
    zero.type = "Stress-Energy"
    conditions = matlab_energy_conditions(zero, metric, num_angular_vec=4, num_time_vec=3)
    for name, values in conditions.items():
        assert_close(values, np.zeros_like(values))
        print(f"{name}: zero tensor neutral")


def validate_minkowski_eval():
    metric = create_minkowski_metric((1, 6, 6, 6), (1.0, 1.0, 1.0, 1.0))
    result = matlab_eval_metric(metric, num_angular_vec=4, num_time_vec=3)
    max_energy = float(np.max(np.abs(result.energy_tensor.tensor)))
    max_conditions = max(float(np.max(np.abs(values))) for values in result.energy_conditions.values())
    print(f"Max |T|: {max_energy:.4e}")
    print(f"Max |CE|: {max_conditions:.4e}")
    assert max_energy == 0.0
    assert max_conditions == 0.0


def main():
    validate_even_points()
    validate_index_changes()
    validate_zero_energy_conditions()
    validate_minkowski_eval()
    print("SUCCESS: MATLAB compatibility helpers pass baseline checks.")


if __name__ == "__main__":
    main()
