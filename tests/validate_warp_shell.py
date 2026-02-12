import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from warpfactory.generator.warp_shell import create_warp_shell_metric
from warpfactory.solver.solvers import solve_energy_tensor
from warpfactory.constants import C

def validate():
    print("Testing WarpShell Metric...")
    Nx, Ny, Nz = 20, 20, 20
    Nt = 5
    grid_scale = (1.0, 0.5, 0.5, 0.5)
    grid_size = (Nt, Nx, Ny, Nz)
    
    # Center
    center = (0, grid_scale[1]*Nx/2, grid_scale[2]*Ny/2, grid_scale[3]*Nz/2)
    
    # Params
    mass = 1000.0
    r_inner = 2.0
    r_outer = 4.0
    r_buff = 0.5
    sigma = 4.0
    smooth_factor = 2
    v_warp = 0.5 # Fraction of C?
    
    metric = create_warp_shell_metric(
        grid_size, grid_scale, center, 
        mass, r_inner, r_outer, r_buff, sigma, smooth_factor, 
        v_warp, do_warp=True
    )
    print("WarpShell Metric created.")
    
    # Check values
    print(f"B max: {np.max(metric.params['B'])}")
    print(f"A min: {np.min(metric.params['A'])}")
    
    # Verify finite
    if not np.all(np.isfinite(metric.tensor)):
        print("FAILURE: Metric contains non-finite values.")
        exit(1)
        
    print("Metric values are finite.")
    
    # Solve Energy
    print("Solving Energy Tensor...")
    energy = solve_energy_tensor(metric)
    rho = energy.tensor[0,0]
    print(f"Max Energy Density: {np.max(rho)}")
    
    print("SUCCESS: WarpShell validation passed.")

if __name__ == "__main__":
    validate()
