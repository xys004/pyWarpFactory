import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory import analyze_metric
from warpfactory.generator.minkowski import create_minkowski_metric


def validate():
    grid_size = (5, 8, 8, 8)
    grid_scale = (1.0, 1.0, 1.0, 1.0)

    metric = create_minkowski_metric(grid_size, grid_scale)
    result = analyze_metric(metric, num_vecs=6)

    max_energy_abs = float(np.max(np.abs(result.energy_tensor.tensor)))
    max_euler_abs = float(np.max(np.abs(result.eulerian_energy_tensor)))
    max_condition_abs = max(
        float(np.max(np.abs(values)))
        for values in result.energy_conditions.values()
    )

    print("Minkowski pipeline verification")
    print(f"Max |T^uv|              : {max_energy_abs:.4e}")
    print(f"Max |T^hat(u)hat(v)|    : {max_euler_abs:.4e}")
    print(f"Max |energy conditions| : {max_condition_abs:.4e}")
    print(f"Summary                 : {result.summary}")

    tolerance = 1e-8
    assert max_energy_abs < tolerance
    assert max_euler_abs < tolerance
    assert max_condition_abs < tolerance

    for condition in ("Null", "Weak", "Strong", "Dominant"):
        assert not result.has_violation(condition, tolerance=tolerance)

    print("SUCCESS: Minkowski stays flat and satisfies all sampled conditions.")


if __name__ == "__main__":
    validate()
