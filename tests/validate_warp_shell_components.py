import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory.generator.warp_shell import create_warp_shell_metric


def validate_massless_static_is_minkowski():
    grid_size = (5, 6, 6, 6)
    grid_scale = (1.0, 0.5, 0.5, 0.5)
    center = (0.0, 1.5, 1.5, 1.5)

    metric = create_warp_shell_metric(
        grid_size,
        grid_scale,
        center,
        mass=0.0,
        r_inner=0.75,
        r_outer=1.5,
        r_buff=0.1,
        sigma=4.0,
        smooth_factor=2,
        v_warp=0.0,
        do_warp=False,
    )

    eta = np.diag([-1.0, 1.0, 1.0, 1.0]).reshape(4, 4, 1, 1, 1, 1)
    deviation = float(np.nanmax(np.abs(metric.tensor - eta)))
    assert deviation < 1e-12
    assert np.all(np.isfinite(metric.tensor))

    # MATLAB parity: first x coordinate is 1*dx - center_x, not 0*dx - center_x.
    assert np.isclose(metric.coords["x"][0, 0, 0, 0], grid_scale[1] - center[1])

    print("SUCCESS: massless static WarpShell is component-wise Minkowski.")
    print(f"Max metric deviation: {deviation:.4e}")


def validate_shift_component_when_warp_enabled():
    grid_size = (5, 6, 6, 6)
    grid_scale = (1.0, 0.5, 0.5, 0.5)
    center = (0.0, 1.5, 1.5, 1.5)
    v_warp = 0.2

    metric = create_warp_shell_metric(
        grid_size,
        grid_scale,
        center,
        mass=0.0,
        r_inner=0.75,
        r_outer=1.5,
        r_buff=0.1,
        sigma=4.0,
        smooth_factor=2,
        v_warp=v_warp,
        do_warp=True,
    )

    assert np.allclose(metric.tensor, np.swapaxes(metric.tensor, 0, 1))
    assert np.nanmin(metric.tensor[0, 1]) >= -v_warp - 1e-12
    assert np.nanmax(metric.tensor[0, 1]) <= 1e-12
    assert np.nanmax(np.abs(metric.tensor[0, 2])) < 1e-12
    assert np.nanmax(np.abs(metric.tensor[0, 3])) < 1e-12

    print("SUCCESS: WarpShell shift component is symmetric and bounded.")
    print(f"g_tx range: {np.nanmin(metric.tensor[0, 1]):.4e}, {np.nanmax(metric.tensor[0, 1]):.4e}")


def validate_adm_shift_g00_option():
    grid_size = (5, 6, 6, 6)
    grid_scale = (1.0, 0.5, 0.5, 0.5)
    center = (0.0, 1.5, 1.5, 1.5)
    v_warp = 0.2

    matlab_parity = create_warp_shell_metric(
        grid_size,
        grid_scale,
        center,
        mass=0.0,
        r_inner=0.75,
        r_outer=1.5,
        r_buff=0.1,
        sigma=4.0,
        smooth_factor=2,
        v_warp=v_warp,
        do_warp=True,
    )
    adm_consistent = create_warp_shell_metric(
        grid_size,
        grid_scale,
        center,
        mass=0.0,
        r_inner=0.75,
        r_outer=1.5,
        r_buff=0.1,
        sigma=4.0,
        smooth_factor=2,
        v_warp=v_warp,
        do_warp=True,
        adm_shift_g00=True,
    )

    assert matlab_parity.params["adm_shift_g00"] is False
    assert adm_consistent.params["adm_shift_g00"] is True
    assert np.all(adm_consistent.tensor[0, 0] >= matlab_parity.tensor[0, 0])
    assert np.nanmax(adm_consistent.tensor[0, 0] - matlab_parity.tensor[0, 0]) > 0.0

    print("SUCCESS: optional ADM shift g00 term is available without changing default parity.")


def validate():
    validate_massless_static_is_minkowski()
    validate_shift_component_when_warp_enabled()
    validate_adm_shift_g00_option()


if __name__ == "__main__":
    validate()
