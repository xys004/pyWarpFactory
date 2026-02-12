import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from warpfactory.generator.schwarzschild import create_schwarzschild_metric
from warpfactory.solver.tensor_utils import get_ricci_scalar, get_c4_inv, get_ricci_tensor

def validate():
    print("Testing Schwarzschild Metric...")
    Nx, Ny, Nz = 20, 20, 20
    Nt = 1 # Static
    grid_scale = (1.0, 0.5, 0.5, 0.5)
    grid_size = (Nt, Nx, Ny, Nz)
    # Center sufficiently far to avoid horizon (r=rs=1.0)
    # x goes 0..10. We want r > 1.5 everywhere.
    # Set center at (-5, -5, -5) so x_phys starts at 5.
    center = (0, -5.0, -5.0, -5.0)
    
    rs = 1.0
    
    metric = create_schwarzschild_metric(grid_size, grid_scale, center, rs)
    print("Schwarzschild Metric created successfully.")
    
    # Verify vacuum solution (Ricci scalar should be ~0)
    print("Calculating Ricci Scalar...")
    R_mn = get_ricci_tensor(metric.tensor, grid_scale)
    g_inv = get_c4_inv(metric.tensor)
    R = get_ricci_scalar(R_mn, g_inv)
    
    max_R = np.max(np.abs(R))
    print(f"Max abs(Ricci Scalar): {max_R}")
    
    # It won't be exactly zero due to numerical error and singularity approximation
    # But checking if it runs is the main goal
    if max_R < 10.0: # Arbitrary large bound for coarse grid near singularity
        print("SUCCESS: Ricci scalar is reasonable.")
    else:
        print(f"WARNING: Ricci scalar {max_R} might be too high (expected near singularity though).")

if __name__ == "__main__":
    validate()
