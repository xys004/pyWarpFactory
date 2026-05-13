import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory import analyze_metric
from warpfactory.analyzer.stress_energy_diagnostics import eulerian_stress_decomposition
from warpfactory.generator.minkowski import create_minkowski_metric
from warpfactory.generator.warp_shell import create_warp_shell_metric


def validate_minkowski_stress_decomposition():
    metric = create_minkowski_metric((5, 8, 8, 8), (1.0, 0.5, 0.5, 0.5))
    result = analyze_metric(metric, num_vecs=4, flat_tolerance=1e-12)
    stress = eulerian_stress_decomposition(result.eulerian_energy_tensor, coords=metric.coords)

    assert np.nanmax(np.abs(stress["rho"])) < 1e-12
    assert np.nanmax(np.abs(stress["flux_norm"])) < 1e-12
    assert np.nanmax(np.abs(stress["pressure_min"])) < 1e-12

    print("SUCCESS: Minkowski stress-energy decomposition is zero.")


def validate_shift_stress_decomposition_finite_and_negative_margin():
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
        v_warp=0.1,
        do_warp=True,
    )
    result = analyze_metric(metric, num_vecs=4, energy_condition_method="warpfactory")
    stress = eulerian_stress_decomposition(result.eulerian_energy_tensor, coords=metric.coords)

    assert np.all(np.isfinite(stress["rho"]))
    assert np.all(np.isfinite(stress["pressure_min"]))
    assert np.nanmin(stress["rho_plus_p_min"]) < 0.0
    assert np.nanmax(stress["flux_norm"]) >= 0.0

    print("SUCCESS: shift stress-energy diagnostic is finite and finds negative rho+p.")
    print(f"rho min       : {np.nanmin(stress['rho']):.4e}")
    print(f"rho+p_min min : {np.nanmin(stress['rho_plus_p_min']):.4e}")


def validate():
    validate_minkowski_stress_decomposition()
    validate_shift_stress_decomposition_finite_and_negative_margin()


if __name__ == "__main__":
    validate()
