import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from warpfactory.generator.lentz import create_lentz_metric
from warpfactory.analyzer.scalars import calculate_scalars

def validate():
    print("Testing Lentz Metric...")
    # Low resolution for speed
    Nx, Ny, Nz = 20, 20, 20
    Nt = 5
    grid_scale = (1.0, 0.2, 0.2, 0.2)
    grid_size = (Nt, Nx, Ny, Nz)
    center = (grid_scale[0]*1, grid_scale[1]*Nx/2, grid_scale[2]*Ny/2, grid_scale[3]*Nz/2)
    
    from warpfactory.constants import C
    
    # We need small velocity to stay in grid if C is physical
    # v is fraction of C.
    # We want physical speed ~ 1 grid unit per time step? 
    # dx=0.2. dt/dx = 1? No dt=1.0? 
    # grid_scale = (1.0, ...)
    # If we want v_phys = 0.1 m/s (slow drift)
    v = 0.1 / C
    
    metric = create_lentz_metric(grid_size, grid_scale, center, v)
    print("Lentz Metric created successfully.")
    
    print("Calculating Scalars...")
    scalars = calculate_scalars(metric, grid_scale)
    exp = scalars['expansion']
    shear = scalars['shear']
    vort = scalars['vorticity']
    
    print(f"Max Expansion: {np.max(exp)}")
    print(f"Max Shear: {np.max(shear)}")
    print(f"Max Vorticity: {np.max(vort)}")
    
    if np.all(np.isfinite(exp)):
        print("SUCCESS: Scalars are finite.")
    else:
        print("WARNING: Scalars contain NaNs or Infs.")

if __name__ == "__main__":
    validate()
