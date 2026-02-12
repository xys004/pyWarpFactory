import numpy as np
from warpfactory.generator.base import Metric
from warpfactory.generator.commons import create_grid, shape_function_alcubierre
from warpfactory.constants import C

def create_van_den_broeck_metric(grid_size, grid_scale, world_center, v, R1, sigma1, R2, sigma2, A):
    """
    Generates the Van Den Broeck metric.
    
    Args:
        grid_size (tuple): (Nt, Nx, Ny, Nz)
        grid_scale (tuple): (dt, dx, dy, dz)
        world_center (tuple): (tc, xc, yc, zc)
        v (float): Velocity of the warp drive in factors of c (unscaled?)
                   MATLAB: v_effective = v*(1+A)^2
        R1 (float): Spatial expansion radius
        sigma1 (float): Width factor of spatial expansion
        R2 (float): Shift vector radius
        sigma2 (float): Width factor of shift vector transition
        A (float): Spatial expansion factor
        
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
    # MATLAB: xs = (t*gridScale(1)-worldCenter(1))*v*(1+A)^2*c
    v_eff = v * (1 + A)**2
    xs = t_phys * v_eff * C
    
    # Radius from bubble center
    r = np.sqrt((x_phys - xs)**2 + y_phys**2 + z_phys**2)
    
    # B function value (spatial expansion)
    # B = 1 + shape_function(r, R1, sigma1) * A
    B = 1 + shape_function_alcubierre(r, R1, sigma1) * A
    
    # fs (shift vector shape)
    # fs = shape_function(r, R2, sigma2) * v
    # Note: MATLAB multiplies by 'v' here.
    # Where 'v' is the input parameter, distinct from v_eff? 
    # Yes. metric.tensor{1,2} = -B^2 * fs.
    fs_val = shape_function_alcubierre(r, R2, sigma2) * v
    
    # Metric tensor components
    # g_xx = B^2, g_yy = B^2, g_zz = B^2
    # g_tx = -B^2 * fs
    # g_tt = -(1 - B^2 * fs^2)
    
    g = np.zeros((4, 4) + t.shape)
    
    # Spatial part
    B_sq = B**2
    g[1, 1] = B_sq
    g[2, 2] = B_sq
    g[3, 3] = B_sq
    
    # Shift part (g_tx, g_xt)
    # metric.tensor{1,2} = -B^2 * fs  (t, x)
    # In python: g[0, 1]
    shift_term = -B_sq * fs_val
    g[0, 1] = shift_term
    g[1, 0] = shift_term
    
    # Time part (g_tt)
    # metric.tensor{1,1} = -(1 - B^2 * fs^2)
    # fs is (shape * v). 
    # Check consistency: g_tt = -N^2 + beta_i beta^i ?
    # Here it's explicit formula.
    g[0, 0] = -(1 - B_sq * (fs_val**2))
    
    metric = Metric(
        tensor=g,
        coords={
            't': t_phys,
            'x': x_phys,
            'y': y_phys,
            'z': z_phys
        },
        scaling=np.array(grid_scale),
        name="Van Den Broeck",
        index="covariant",
        params={
            'grid_size': grid_size,
            'world_center': world_center,
            'velocity': v,
            'R1': R1,
            'sigma1': sigma1,
            'R2': R2,
            'sigma2': sigma2,
            'A': A
        }
    )
    
    return metric
