import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from warpfactory.analyzer.spatial_geometry import spatial_geometry_summary
from warpfactory.generator.minkowski import create_minkowski_metric
from warpfactory.generator.warp_shell import create_warp_shell_metric


def validate_minkowski_spatial_geometry():
    metric = create_minkowski_metric((5, 8, 8, 8), (1.0, 0.5, 0.5, 0.5))
    summary = spatial_geometry_summary(metric)

    assert summary["max_abs_gamma_minus_identity"] == 0.0
    assert abs(summary["spatial_ricci_scalar_abs_max"]) < 1e-12

    print("SUCCESS: Minkowski spatial hypersurface is flat.")


def validate_warp_shell_spatial_geometry_is_nonflat():
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
    summary = spatial_geometry_summary(metric)

    assert summary["max_abs_gamma_minus_identity"] > 0.0
    assert summary["spatial_ricci_scalar_abs_max"] > 0.0

    print("SUCCESS: WarpShell mass profile creates a non-flat spatial hypersurface.")
    print(f"gamma-I max : {summary['max_abs_gamma_minus_identity']:.4e}")
    print(f"3-Ricci max : {summary['spatial_ricci_scalar_abs_max']:.4e}")


def validate():
    validate_minkowski_spatial_geometry()
    validate_warp_shell_spatial_geometry_is_nonflat()


if __name__ == "__main__":
    validate()
