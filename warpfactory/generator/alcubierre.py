import numpy as np
from warpfactory.generator.base import Metric
from warpfactory.generator.commons import create_grid, get_minkowski_metric, shape_function_alcubierre
from warpfactory.constants import C


def create_alcubierre_metric(grid_size, grid_scale, world_center, v, R, sigma):
    """
    Generates the Alcubierre warp drive metric.
    
    Args:
        grid_size (tuple): (Nt, Nx, Ny, Nz) grid points.
        grid_scale (tuple): (dt, dx, dy, dz) grid spacing.
        world_center (tuple): (tc, xc, yc, zc) center of the grid in physical units.
        v (float): Velocity of the warp bubble (fraction of c if not normalized, but here treated as dimensionless scalar multiplier usually).
                  However, MATLAB code multiplies by 'c' explicitly: xs = ... * v * c.
                  So 'v' is likely v/c.
        R (float): Radius of the warp bubble.
        sigma (float): Thickness parameter.
        
    Returns:
        Metric: The generated metric.
    """
    grid = create_grid(grid_size, grid_scale)
    t, x, y, z = grid['t'], grid['x'], grid['y'], grid['z']
    
    # Adjust coordinates relative to world center
    # Note: create_grid returns 0-based coordinates [0, dt, 2dt...]
    # MATLAB: x = i*dx - center_x. i is 1-based index (1..N).
    # Python: i is 0-based index (0..N-1).
    # To match MATLAB exactly:
    # MATLAB x_1 = 1*dx - center_x
    # Python x_0 = 0*dx - center_x (if we just sub center).
    # This implies a shift of dx if we want exact spatial alignment.
    # However, usually physics grids are centered or 0-based.
    # We will assume standard 0-based grid generation is acceptable, but let's shift to match "physical" coordinates centered as requested.
    # Actually, let's just use the coordinates as provided minus center.
    # If explicit parity with MATLAB meshgrid is needed, we might need to offset.
    # MATLAB: 1..N. Python: 0..N-1.
    # x_matlab = (idx + 1) * dx - cx
    # x_python = idx * dx
    # So x_matlab = x_python + dx - cx.
    # Let's adjust for the index shift to be safe?
    # No, typically 0-based is fine if we are self-consistent. 
    # But if we strictly want numerical parity, we should consider that t=0 in Python corresponds to t=1 in MATLAB?
    # Let's just implement the physics: x_phys = coordinate - center.
    # We'll use (coordinate + dx) to match MATLAB's 1-based indexing offset if strict parity is paramount.
    # Let's check the user requirement: "System must yield identical numerical results".
    # This implies we should respect the grid positions.
    # MATLAB: t = 1*dt, 2*dt...
    # Python: t = 0*dt, 1*dt...
    # So Python t array should be (np.arange(Nt) + 1) * dt.
    
    t_phys = (t + grid_scale[0]) - world_center[0]
    x_phys = (x + grid_scale[1]) - world_center[1]
    y_phys = (y + grid_scale[2]) - world_center[2]
    z_phys = (z + grid_scale[3]) - world_center[3]
    
    # Calculate bubble center position xs(t)
    # MATLAB: xs = (t*gridScale(1)-worldCenter(1))*v*c
    # Our t_phys matches (t*gridScale(1)-worldCenter(1)).
    xs = t_phys * v * C
    
    # Calculate radius r from bubble center
    # r = sqrt((x - xs)^2 + y^2 + z^2)
    # usage of x_phys, y_phys, z_phys
    r = np.sqrt((x_phys - xs)**2 + y_phys**2 + z_phys**2)
    
    # Shape function
    fs = shape_function_alcubierre(r, R, sigma)
    
    # Metric Construction (3+1)
    # Alcubierre metric in ADM form:
    # alpha = 1
    # beta = [-v * fs, 0, 0] (shift vector)
    # gamma = diag(1, 1, 1) (flat spatial metric)
    
    # Construct 4x4 metric tensor
    # g_00 = -alpha^2 + beta^2
    # g_0i = beta_i
    # g_ij = gamma_ij
    
    # Components
    # beta_x = -v * fs
    # beta^2 = beta_x^2 + beta_y^2 + beta_z^2 = (-v * fs)^2
    
    beta_x = -v * fs
    
    # Initialize with Minkowski background (diagonal -1, 1, 1, 1)
    g = np.zeros((4, 4) + t.shape)
    
    # Populate g_00
    # g_00 = -1 + (beta_x)^2
    g[0, 0] = -1 + beta_x**2
    
    # Populate g_01 and g_10 (time-x mixing)
    g[0, 1] = beta_x
    g[1, 0] = beta_x
    
    # Populate spatial diagonal (g_xx, g_yy, g_zz) -> all 1
    g[1, 1] = 1.0
    g[2, 2] = 1.0
    g[3, 3] = 1.0
    
    # Create Metric object
    metric = Metric(
        tensor=g,
        coords={
            't': t_phys,
            'x': x_phys,
            'y': y_phys,
            'z': z_phys
        },
        scaling=np.array(grid_scale),
        name="Alcubierre",
        index="covariant",
        params={
            'grid_size': grid_size,
            'world_center': world_center,
            'velocity': v,
            'R': R,
            'sigma': sigma
        }
    )
    
    return metric
