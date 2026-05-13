import math

from warpfactory.constants import C, G
from warpfactory.generator.warp_shell import create_warp_shell_metric


def fuchs_constant_warp_shell_parameters(profile="original", grid_scale_factor=None):
    """
    Parameters from WarpFactory's W1_Warp_Shell.mlx example.

    profile='original' follows the MATLAB notebook as closely as possible.
    profile='quick' keeps the same physical radii/velocity/mass formula but uses
    a thinner plotting slab and smaller smoothing factor for fast smoke tests.
    """
    if profile not in {"original", "quick"}:
        raise ValueError("profile must be 'original' or 'quick'.")

    R1 = 10.0
    R2 = 20.0
    Rbuff = 0.0
    factor = 1.0 / 3.0
    mass = R2 / (2.0 * G) * C**2 * factor
    v_warp = 0.02
    sigma = 0.0
    do_warp = True
    centered = True
    time_scale = 1.0

    if profile == "original":
        space_scale = 5.0
        cartoon_thickness = 5
        smooth_factor = 4000
    else:
        space_scale = 1.0 if grid_scale_factor is None else float(grid_scale_factor)
        cartoon_thickness = 5
        smooth_factor = 200

    if grid_scale_factor is not None and profile == "original":
        space_scale = float(grid_scale_factor)

    if centered:
        grid_size = (
            1,
            math.ceil(2 * (R2 + 10) * space_scale),
            math.ceil(2 * (R2 + 10) * space_scale),
            cartoon_thickness,
        )
    else:
        grid_size = (
            1,
            math.ceil((R2 + 10) * space_scale),
            math.ceil((R2 + 10) * space_scale),
            cartoon_thickness,
        )

    grid_scaling = (
        1.0 / (1000.0 * C),
        1.0 / space_scale,
        1.0 / space_scale,
        1.0 / space_scale,
    )

    if centered:
        world_center = (
            (cartoon_thickness + 1) / 2 * grid_scaling[0],
            (2 * (R2 + 10) * space_scale + 1) / 2 * grid_scaling[1],
            (2 * (R2 + 10) * space_scale + 1) / 2 * grid_scaling[2],
            (cartoon_thickness + 1) / 2 * grid_scaling[3],
        )
    else:
        world_center = (
            (cartoon_thickness + 1) / 2 * grid_scaling[0],
            5.0 * grid_scaling[1],
            5.0 * grid_scaling[2],
            (cartoon_thickness + 1) / 2 * grid_scaling[3],
        )

    return {
        "profile": profile,
        "R1": R1,
        "R2": R2,
        "Rbuff": Rbuff,
        "factor": factor,
        "mass": mass,
        "v_warp": v_warp,
        "sigma": sigma,
        "do_warp": do_warp,
        "smooth_factor": smooth_factor,
        "space_scale": space_scale,
        "time_scale": time_scale,
        "cartoon_thickness": cartoon_thickness,
        "grid_size": grid_size,
        "grid_scaling": grid_scaling,
        "world_center": world_center,
    }


def create_fuchs_constant_warp_shell(profile="original", grid_scale_factor=None):
    params = fuchs_constant_warp_shell_parameters(profile=profile, grid_scale_factor=grid_scale_factor)
    metric = create_warp_shell_metric(
        params["grid_size"],
        params["grid_scaling"],
        params["world_center"],
        mass=params["mass"],
        r_inner=params["R1"],
        r_outer=params["R2"],
        r_buff=params["Rbuff"],
        sigma=params["sigma"],
        smooth_factor=params["smooth_factor"],
        v_warp=params["v_warp"],
        do_warp=params["do_warp"],
    )
    metric.params["fuchs_recipe"] = params
    return metric
