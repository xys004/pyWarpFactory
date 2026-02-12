import numpy as np

def create_grid(grid_size, grid_scale):
    """
    Creates a 4D coordinate grid (t, x, y, z).
    
    Args:
        grid_size (list or tuple): [Nt, Nx, Ny, Nz]
        grid_scale (list or tuple): [dt, dx, dy, dz]
        
    Returns:
        dict: Dictionary containing 't', 'x', 'y', 'z' grids.
    """
    Nt, Nx, Ny, Nz = grid_size
    dt, dx, dy, dz = grid_scale
    
    # Create coordinate ranges
    t_range = np.arange(Nt) * dt
    x_range = np.arange(Nx) * dx
    y_range = np.arange(Ny) * dy
    z_range = np.arange(Nz) * dz
    
    # Create meshgrid. Note indexing='ij' to match matrix notation (and likely MATLAB 3D behavior for consistency)
    # MATLAB's meshgrid is weird with x/y swap for 2D, but for 3D/ND ndgrid is usually preferred for consistency.
    # We will use 'ij' to strictly follow (t, x, y, z) order.
    t, x, y, z = np.meshgrid(t_range, x_range, y_range, z_range, indexing='ij')
    
    return {'t': t, 'x': x, 'y': y, 'z': z}

def get_minkowski_metric(grid_size):
    """
    Returns the Minkowski metric tensor (eta_munu) for a given grid size.
    Signature: (-, +, +, +)
    
    Args:
        grid_size (tuple): (Nt, Nx, Ny, Nz)
        
    Returns:
        np.ndarray: 4x4xNt x Nx x Ny x Nz array
    """
    shape = (4, 4) + tuple(grid_size)
    g = np.zeros(shape)
    
    # Fill diagonal: -1, 1, 1, 1
    # We use broadcasting to fill the entire grid
    ones = np.ones(grid_size)
    g[0, 0] = -ones
    g[1, 1] = ones
    g[2, 2] = ones
    g[3, 3] = ones
    
    return g

def shape_function_alcubierre(r, R, sigma):
    """
    Computes the Alcubierre shape function f(r_s).
    f = (tanh(sigma * (R + r)) - tanh(sigma * (r - R))) / (2 * tanh(sigma * R))
    """
    # Note: MATLAB code used: (tanh(sigma*(R + r)) + tanh(sigma*(R - r))) / ...
    # tanh(A - B) = -tanh(B - A). So tanh(sigma*(R - r)) = -tanh(sigma*(r - R)).
    # Thus (tanh(A) + tanh(B)) becomes (tanh(A) - tanh(-B)).
    # We will stick to the MATLAB implementation formula to ensure parity.
    numerator = np.tanh(sigma * (R + r)) + np.tanh(sigma * (R - r))
    denominator = 2 * np.tanh(sigma * R)
    return numerator / denominator
