import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory.analyzer.transform import do_frame_transfer
from warpfactory.generator.base import Metric
from warpfactory.generator.minkowski import create_minkowski_metric


def make_tensor(values, grid_size):
    tensor = np.zeros((4, 4) + grid_size)
    for i, value in enumerate(values):
        tensor[i, i] = value
    coords = {axis: np.zeros(grid_size) for axis in ("t", "x", "y", "z")}
    obj = Metric(tensor=tensor, coords=coords, scaling=np.ones(4), name="tensor", index="contravariant")
    obj.type = "Stress-Energy"
    return obj


def validate_minkowski_frame_transfer_identity():
    grid_size = (5, 3, 3, 3)
    metric = create_minkowski_metric(grid_size, (1.0, 1.0, 1.0, 1.0))
    energy = make_tensor((1.0, 2.0, 3.0, 4.0), grid_size)

    transferred = do_frame_transfer(metric, energy, "Eulerian")
    max_diff = float(np.max(np.abs(transferred.tensor - energy.tensor)))

    assert transferred.frame == "Eulerian"
    assert transferred.index == "contravariant"
    assert max_diff < 1e-12

    print("SUCCESS: Minkowski Eulerian frame transfer preserves diagonal tensor.")
    print(f"Max transfer diff: {max_diff:.4e}")


def validate():
    validate_minkowski_frame_transfer_identity()


if __name__ == "__main__":
    validate()
