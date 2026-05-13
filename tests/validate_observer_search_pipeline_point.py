import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory import analyze_metric, optimize_timelike_conditions
from warpfactory.analyzer.energy_conditions import _lower_indices_minkowski
from warpfactory.generator.alcubierre import create_alcubierre_metric


def validate():
    metric = create_alcubierre_metric(
        grid_size=(5, 10, 10, 10),
        grid_scale=(1.0, 0.2, 0.2, 0.2),
        world_center=(1.0, 1.0, 1.0, 1.0),
        v=0.5,
        R=0.8,
        sigma=4.0,
    )
    result = analyze_metric(metric, num_vecs=6)

    dec_map = result.energy_conditions["Dominant"]
    point = np.unravel_index(np.argmin(dec_map), dec_map.shape)

    T_euler_local = result.eulerian_energy_tensor[(slice(None), slice(None)) + point]
    T_lower_local = _lower_indices_minkowski(T_euler_local)
    optimized = optimize_timelike_conditions(
        T_lower_local,
        conditions=("WEC", "SEC", "DEC"),
        direction_count=8,
        max_speed=0.95,
        max_iter=40,
        min_step=1e-4,
    )

    print("Observer-search pipeline point verification")
    print(f"Selected grid point          : {point}")
    print(f"Sampled DEC at point         : {dec_map[point]:.4e}")
    print(f"Optimized WEC at point       : {optimized['Weak'].value:.4e}")
    print(f"Optimized SEC at point       : {optimized['Strong'].value:.4e}")
    print(f"Optimized DEC at point       : {optimized['Dominant'].value:.4e}")

    assert optimized["Dominant"].value < 0.0
    print("SUCCESS: local optimized observer search detects a DEC violation.")


if __name__ == "__main__":
    validate()
