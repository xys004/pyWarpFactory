import numpy as np
from warpfactory.generator.base import Metric
from warpfactory.generator.commons import create_grid, get_minkowski_metric

def create_schwarzschild_metric(grid_size, grid_scale, world_center, rs):
    """
    Generates the Schwarzschild metric.
    
    Args:
        grid_size (tuple): (Nt, Nx, Ny, Nz). Nt must be 1.
        grid_scale (tuple): (dt, dx, dy, dz)
        world_center (tuple): (tc, xc, yc, zc)
        rs (float): Schwarzschild radius
        
    Returns:
        Metric object
    """
    if grid_size[0] > 1:
        raise ValueError("The time grid is greater than 1, only a size of 1 can be used for the Schwarzschild solution")
    
    grid = create_grid(grid_size, grid_scale)
    t, x, y, z = grid['t'], grid['x'], grid['y'], grid['z']
    
    # Physical coordinates
    # t_phys = (t + grid_scale[0]) - world_center[0] # Not used for static
    x_phys = (x + grid_scale[1]) - world_center[1]
    y_phys = (y + grid_scale[2]) - world_center[2]
    z_phys = (z + grid_scale[3]) - world_center[3]
    
    epsilon = 1e-10
    
    r = np.sqrt(x_phys**2 + y_phys**2 + z_phys**2) + epsilon
    
    # Initialize with Minkowski
    g = get_minkowski_metric(grid_size)
    
    # Apply Schwarzschild components
    # g_tt = -(1 - rs/r)
    # g_xx = (x^2/(1-rs/r) + y^2 + z^2) / r^2
    # g_yy = (x^2 + y^2/(1-rs/r) + z^2) / r^2
    # g_zz = (x^2 + y^2 + z^2/(1-rs/r)) / r^2
    
    factor = 1.0 - rs/r
    inv_factor = 1.0 / factor
    
    g[0, 0] = -factor
    
    r2 = r**2
    x2, y2, z2 = x_phys**2, y_phys**2, z_phys**2
    
    g[1, 1] = (x2*inv_factor + y2 + z2) / r2
    g[2, 2] = (x2 + y2*inv_factor + z2) / r2
    g[3, 3] = (x2 + y2 + z2*inv_factor) / r2
    
    # Cross terms
    # g_xy = rs/(r^3 - r^2*rs) * x * y
    # term = rs / (r^3 * (1 - rs/r)) = rs / (r^2 * (r - rs)) = rs / (r^2 * r * factor) ?
    # MATLAB: rs/(r^3 - r^2*rs) = rs / (r^2 * (r - rs))
    # My factor is (1 - rs/r) = (r - rs)/r => (r - rs) = r * factor
    # So denominator is r^2 * r * factor = r^3 * factor
    
    cross_common = rs / (r**3 * factor)
    
    g[1, 2] = cross_common * x_phys * y_phys
    g[2, 1] = g[1, 2]
    
    g[1, 3] = cross_common * x_phys * z_phys
    g[3, 1] = g[1, 3]
    
    g[2, 3] = cross_common * y_phys * z_phys
    g[3, 2] = g[2, 3]
    
    metric = Metric(
        tensor=g,
        coords={
            't': t, # Or t_phys? MATLAB doesn't use t for static.
            'x': x_phys,
            'y': y_phys,
            'z': z_phys
        },
        scaling=np.array(grid_scale),
        name="Schwarzschild",
        index="covariant",
        params={
            'grid_size': grid_size,
            'world_center': world_center,
            'rs': rs
        }
    )
    
    return metric
