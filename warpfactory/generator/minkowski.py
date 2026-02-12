import numpy as np
from warpfactory.generator.base import Metric
from warpfactory.generator.commons import create_grid

def create_minkowski_metric(grid_size, grid_scale):
    """
    Creates a flat Minkowski metric.
    diag(-1, 1, 1, 1).
    """
    Nt, Nx, Ny, Nz = grid_size
    shape = (4, 4, Nt, Nx, Ny, Nz)
    
    tensor = np.zeros(shape)
    
    # Fill diagonal
    # 00: -1
    tensor[0, 0] = -1.0
    # 11, 22, 33: 1
    tensor[1, 1] = 1.0
    tensor[2, 2] = 1.0
    tensor[3, 3] = 1.0
    
    coords = create_grid(grid_size, grid_scale)
    
    return Metric(
        tensor=tensor,
        coords=coords,
        scaling=np.array(grid_scale),
        name="Minkowski",
        index="covariant"
    )
