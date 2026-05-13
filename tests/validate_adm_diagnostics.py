import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory.analyzer.adm_diagnostics import adm_constraint_terms, lapse_spatial_derivatives
from warpfactory.generator.minkowski import create_minkowski_metric
from warpfactory.generator.warp_shell import create_warp_shell_metric


def validate_minkowski_adm_terms():
    metric = create_minkowski_metric((5, 8, 8, 8), (1.0, 0.5, 0.5, 0.5))
    terms = adm_constraint_terms(metric)

    assert np.nanmax(np.abs(terms["R3"])) < 1e-12
    assert np.nanmax(np.abs(terms["KijKij"])) < 1e-12
    assert np.nanmax(np.abs(terms["hamiltonian"])) < 1e-12
    assert np.nanmax(np.abs(terms["lapse_grad_norm"])) < 1e-12

    print("SUCCESS: Minkowski ADM balance is zero.")


def validate_static_shell_has_spatial_curvature_without_shift_cost():
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
    terms = adm_constraint_terms(metric)
    lapse = lapse_spatial_derivatives(metric)

    assert np.nanmax(np.abs(terms["R3"])) > 0.0
    assert np.nanmax(np.abs(terms["KijKij"])) < 1e-12
    assert np.nanmax(lapse["log_grad_norm"]) > 0.0

    print("SUCCESS: static shell has R3 without shift extrinsic-curvature cost.")
    print(f"R3 abs max     : {np.nanmax(np.abs(terms['R3'])):.4e}")
    print(f"KijKij abs max : {np.nanmax(np.abs(terms['KijKij'])):.4e}")
    print(f"|grad ln alpha|: {np.nanmax(lapse['log_grad_norm']):.4e}")


def validate_pure_shift_has_extrinsic_curvature_on_flat_slice():
    metric = create_warp_shell_metric(
        grid_size=(5, 10, 10, 10),
        grid_scale=(1.0, 0.5, 0.5, 0.5),
        world_center=(0.0, 2.5, 2.5, 2.5),
        mass=0.0,
        r_inner=1.0,
        r_outer=2.0,
        r_buff=0.25,
        sigma=4.0,
        smooth_factor=2,
        v_warp=0.1,
        do_warp=True,
    )
    terms = adm_constraint_terms(metric)

    assert np.nanmax(np.abs(terms["R3"])) < 1e-10
    assert np.nanmax(np.abs(terms["KijKij"])) > 0.0
    assert np.nanmin(terms["K_sq"] - terms["KijKij"]) < 0.0

    print("SUCCESS: pure shift has extrinsic-curvature cost on a flat slice.")
    print(f"R3 abs max     : {np.nanmax(np.abs(terms['R3'])):.4e}")
    print(f"KijKij abs max : {np.nanmax(np.abs(terms['KijKij'])):.4e}")


def validate():
    validate_minkowski_adm_terms()
    validate_static_shell_has_spatial_curvature_without_shift_cost()
    validate_pure_shift_has_extrinsic_curvature_on_flat_slice()


if __name__ == "__main__":
    validate()
