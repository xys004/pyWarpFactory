import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from warpfactory.generator.modified_time import create_modified_time_metric
from warpfactory.analyzer.scalars import calculate_scalars

def validate():
    print("Testing Modified Time Metric...")
    Nx, Ny, Nz = 20, 20, 20
    Nt = 5
    grid_scale = (1.0, 0.2, 0.2, 0.2)
    grid_size = (Nt, Nx, Ny, Nz)
    center = (grid_scale[0]*1, grid_scale[1]*Nx/2, grid_scale[2]*Ny/2, grid_scale[3]*Nz/2)
    
    from warpfactory.constants import C
    
    v = 0.5 / C
    R = 2.0
    sigma = 8.0
    A = 2.0
    
    metric = create_modified_time_metric(grid_size, grid_scale, center, v, R, sigma, A)
    print("Modified Time Metric created successfully.")
    
    # Check g_tt roughly
    g_tt = metric.tensor[0, 0]
    print(f"Min g_tt: {np.min(g_tt)}")
    print(f"Max g_tt: {np.max(g_tt)}")
    
    if np.any(g_tt != -1):
        print("SUCCESS: g_tt is modified from Minkowski.")
    else:
        print("WARNING: g_tt appears to be Minkowski (-1) everywhere.")

if __name__ == "__main__":
    validate()
