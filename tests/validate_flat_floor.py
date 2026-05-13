import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory import analyze_metric
from warpfactory.generator.warp_shell import create_warp_shell_metric


def validate():
    metric = create_warp_shell_metric(
        grid_size=(5, 8, 8, 8),
        grid_scale=(1.0, 0.5, 0.5, 0.5),
        world_center=(0.0, 2.0, 2.0, 2.0),
        mass=0.0,
        r_inner=1.0,
        r_outer=2.0,
        r_buff=0.25,
        sigma=4.0,
        smooth_factor=2,
        v_warp=0.0,
        do_warp=False,
    )

    result = analyze_metric(metric, num_vecs=4, flat_tolerance=1e-12)
    diagnostics = result.methodology["solver_diagnostics"]
    max_energy = float(np.max(np.abs(result.energy_tensor.tensor)))
    max_condition = max(float(np.max(np.abs(v))) for v in result.energy_conditions.values())

    assert diagnostics["flat_floor_applied"]
    assert max_energy == 0.0
    assert max_condition == 0.0

    print("SUCCESS: explicit near-flat solver floor zeros roundoff-only WarpShell.")
    print(f"Metric deviation: {diagnostics['max_abs_metric_deviation_from_minkowski']:.4e}")


if __name__ == "__main__":
    validate()
