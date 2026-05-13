import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory.analyzer.transform import do_frame_transfer
from warpfactory.generator.warp_shell import create_warp_shell_metric
from warpfactory.solver.solvers import solve_energy_tensor


def validate():
    metric = create_warp_shell_metric(
        grid_size=(5, 10, 10, 10),
        grid_scale=(1.0, 0.5, 0.5, 0.5),
        world_center=(0.0, 2.5, 2.5, 2.5),
        mass=1e26,
        r_inner=1.0,
        r_outer=2.0,
        r_buff=0.25,
        sigma=4.0,
        smooth_factor=2,
        v_warp=0.0,
        do_warp=False,
    )

    energy = solve_energy_tensor(metric)
    eulerian = do_frame_transfer(metric, energy, "Eulerian").tensor
    rho = eulerian[0, 0]

    x = metric.coords["x"]
    y = metric.coords["y"]
    z = metric.coords["z"]
    radius = np.sqrt(x**2 + y**2 + z**2)
    shell_mask = (radius > 1.0) & (radius < 2.0)
    interior_mask = radius < 0.75
    exterior_mask = radius > 2.25

    shell_median = float(np.nanmedian(rho[shell_mask]))
    interior_max_abs = float(np.nanmax(np.abs(rho[interior_mask])))
    exterior_median_abs = float(np.nanmedian(np.abs(rho[exterior_mask])))
    negative_shell_fraction = float(np.mean(rho[shell_mask] < 0))

    print("WarpShell static mass response")
    print(f"Shell median rho             : {shell_median:.4e}")
    print(f"Shell negative fraction      : {negative_shell_fraction:.4e}")
    print(f"Interior max |rho|           : {interior_max_abs:.4e}")
    print(f"Exterior median |rho|        : {exterior_median_abs:.4e}")
    print(f"Solver diagnostics           : {energy.params.get('solver_diagnostics')}")

    assert np.all(np.isfinite(rho))
    assert shell_median > 0.0
    assert negative_shell_fraction < 0.5


if __name__ == "__main__":
    validate()
