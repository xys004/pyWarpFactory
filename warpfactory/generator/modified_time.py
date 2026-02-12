import numpy as np
from warpfactory.generator.base import Metric
from warpfactory.generator.commons import create_grid, shape_function_alcubierre, get_minkowski_metric
from warpfactory.constants import C

def create_modified_time_metric(grid_size, grid_scale, world_center, v, R, sigma, A):
    """
    Generates the Modified Time metric.
    
    Args:
        grid_size (tuple): (Nt, Nx, Ny, Nz)
        grid_scale (tuple): (dt, dx, dy, dz)
        world_center (tuple): (tc, xc, yc, zc)
        v (float): Velocity of the warp drive
        R (float): Radius of the warp bubble
        sigma (float): Thickness parameter
        A (float): Lapse rate modification
        
    Returns:
        Metric object
    """
    grid = create_grid(grid_size, grid_scale)
    t, x, y, z = grid['t'], grid['x'], grid['y'], grid['z']
    
    # Physical coordinates
    t_phys = (t + grid_scale[0]) - world_center[0]
    x_phys = (x + grid_scale[1]) - world_center[1]
    y_phys = (y + grid_scale[2]) - world_center[2]
    z_phys = (z + grid_scale[3]) - world_center[3]
    
    # Calculate bubble center position xs(t)
    xs = t_phys * v * C
    
    # Radius
    r = np.sqrt((x_phys - xs)**2 + y_phys**2 + z_phys**2)
    
    # Shape function
    fs = shape_function_alcubierre(r, R, sigma)
    
    # Initialize with Minkowski
    g = get_minkowski_metric(grid_size)
    
    # Add Alcubierre term to dxdt (g_tx)
    # metric.tensor{1,2} = -v*fs
    g[0, 1] = -v * fs
    g[1, 0] = g[0, 1]
    
    # Add dt term modification (g_tt)
    # metric.tensor{1,1} = -((1-fs)+fs/A)^2 + (fs*v)^2
    term1 = (1 - fs) + fs/A
    g[0, 0] = -(term1**2) + (fs * v)**2
    
    metric = Metric(
        tensor=g,
        coords={
            't': t_phys,
            'x': x_phys,
            'y': y_phys,
            'z': z_phys
        },
        scaling=np.array(grid_scale),
        name="Modified Time",
        index="covariant",
        params={
            'grid_size': grid_size,
            'world_center': world_center,
            'velocity': v,
            'R': R,
            'sigma': sigma,
            'A': A
        }
    )
    
    return metric
