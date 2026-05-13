import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory.analyzer.energy_conditions import calculate_energy_conditions
from warpfactory.generator.base import Metric
from warpfactory.generator.minkowski import create_minkowski_metric


def make_energy_tensor(diagonal, grid_size=(1, 2, 2, 2)):
    tensor = np.zeros((4, 4) + grid_size)
    for i, value in enumerate(diagonal):
        tensor[i, i] = value
    coords = {axis: np.zeros(grid_size) for axis in ("t", "x", "y", "z")}
    energy = Metric(
        tensor=tensor,
        coords=coords,
        scaling=np.ones(4),
        name="synthetic energy",
        index="contravariant",
    )
    energy.type = "Stress-Energy"
    return energy


def validate_minkowski_zero_both_methods():
    metric = create_minkowski_metric((5, 4, 4, 4), (1.0, 1.0, 1.0, 1.0))
    energy = make_energy_tensor((0.0, 0.0, 0.0, 0.0), grid_size=(5, 4, 4, 4))

    for method in ("standard", "warpfactory"):
        maps = calculate_energy_conditions(energy, metric, num_vecs=4, method=method)
        max_abs = max(float(np.max(np.abs(values))) for values in maps.values())
        assert max_abs == 0.0

    print("SUCCESS: standard and warpfactory CE methods keep zero tensor neutral.")


def validate_diagonal_tensor_methods_are_finite():
    metric = create_minkowski_metric((1, 2, 2, 2), (1.0, 1.0, 1.0, 1.0))
    energy = make_energy_tensor((1.0, -2.0, -2.0, -2.0))

    standard = calculate_energy_conditions(energy, metric, condition="DEC", num_vecs=4, method="standard")
    compatible = calculate_energy_conditions(energy, metric, condition="DEC", num_vecs=4, method="warpfactory")

    assert np.all(np.isfinite(standard))
    assert np.all(np.isfinite(compatible))
    assert np.min(standard) < 0.0

    print("SUCCESS: DEC is finite in standard and warpfactory-compatible methods.")
    print(f"Standard DEC min    : {np.min(standard):.4e}")
    print(f"WarpFactory DEC min : {np.min(compatible):.4e}")


def validate():
    validate_minkowski_zero_both_methods()
    validate_diagonal_tensor_methods_are_finite()


if __name__ == "__main__":
    validate()
