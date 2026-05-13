import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory.constants import C, G
from warpfactory.recipes import create_fuchs_constant_warp_shell, fuchs_constant_warp_shell_parameters


def validate_original_parameters():
    params = fuchs_constant_warp_shell_parameters(profile="original")

    assert params["R1"] == 10.0
    assert params["R2"] == 20.0
    assert params["Rbuff"] == 0.0
    assert params["factor"] == 1.0 / 3.0
    assert params["v_warp"] == 0.02
    assert params["sigma"] == 0.0
    assert params["smooth_factor"] == 4000
    assert params["grid_size"] == (1, 300, 300, 5)
    assert np.isclose(params["grid_scaling"][0], 1.0 / (1000.0 * C))
    assert np.isclose(params["grid_scaling"][1], 0.2)
    assert np.isclose(params["mass"], 20.0 / (2.0 * G) * C**2 / 3.0)

    print("SUCCESS: Fuchs/WarpFactory original recipe parameters match W1_Warp_Shell.mlx.")


def validate_quick_metric_builds():
    metric = create_fuchs_constant_warp_shell(profile="quick")
    assert metric.tensor.shape == (4, 4, 1, 60, 60, 5)
    assert np.all(np.isfinite(metric.tensor))
    assert np.nanmin(metric.tensor[0, 1]) >= -0.02 - 1e-12
    assert np.nanmax(metric.tensor[0, 1]) <= 1e-12

    print("SUCCESS: quick Fuchs/WarpFactory recipe builds finite metric.")
    print(f"Metric shape: {metric.tensor.shape}")


def validate():
    validate_original_parameters()
    validate_quick_metric_builds()


if __name__ == "__main__":
    validate()
