import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from warpfactory.generator.van_den_broeck import create_van_den_broeck_metric
from warpfactory.analyzer.scalars import calculate_scalars

def validate():
    print("Testing Van Den Broeck Metric...")
    Nx, Ny, Nz = 20, 20, 20
    Nt = 5
    grid_scale = (1.0, 0.2, 0.2, 0.2)
    grid_size = (Nt, Nx, Ny, Nz)
    center = (grid_scale[0]*1, grid_scale[1]*Nx/2, grid_scale[2]*Ny/2, grid_scale[3]*Nz/2)
    
    from warpfactory.constants import C
    
    v = 0.5 / C
    R1, sigma1 = 2.0, 4.0
    R2, sigma2 = 1.0, 4.0
    A = 0.5
    
    metric = create_van_den_broeck_metric(grid_size, grid_scale, center, v, R1, sigma1, R2, sigma2, A)
    print("Van Den Broeck Metric created successfully.")
    
    print("Calculating Scalars...")
    scalars = calculate_scalars(metric, grid_scale)
    exp = scalars['expansion']
    
    print(f"Max Expansion: {np.max(exp)}")
    
    if np.any(exp != 0):
        print("SUCCESS: Non-zero expansion detected.")
    else:
        print("WARNING: Expansion is zero everywhere?")

if __name__ == "__main__":
    validate()
