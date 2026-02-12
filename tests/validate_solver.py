import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from warpfactory.generator.alcubierre import create_alcubierre_metric
from warpfactory.solver.solvers import solve_energy_tensor
from warpfactory.solver.tensor_utils import verify_tensor
from warpfactory.constants import C

def validate():
    print("Testing Solver...")
    Nx, Ny, Nz = 20, 20, 20
    Nt = 5
    grid_scale = (1.0, 0.2, 0.2, 0.2)
    grid_size = (Nt, Nx, Ny, Nz)
    center = (grid_scale[0]*1, grid_scale[1]*Nx/2, grid_scale[2]*Ny/2, grid_scale[3]*Nz/2)
    
    v = 0.5 / C
    R = 2.0
    sigma = 8.0
    
    metric = create_alcubierre_metric(grid_size, grid_scale, center, v, R, sigma)
    print("Metric created.")
    
    # 1. Test Verification
    print("Testing verify_tensor...")
    if verify_tensor(metric):
        print("Metric verification PASSED.")
    else:
        print("Metric verification FAILED.")
        
    # 2. Test Solver
    print("Running Einstein Solver (met2den equivalent)...")
    energy_tensor = solve_energy_tensor(metric)
    
    if energy_tensor.type == "Stress-Energy":
        print("Solver returned correct type.")
    else:
        print(f"Solver returned wrong type: {energy_tensor.type}")
        
    # Check max energy density
    # T^00 is energy density in frame?
    # Actually T_00 is energy density. T^00 approx T_00 in flat space.
    # Alcubierre has negative energy density.
    
    T_00 = energy_tensor.tensor[0, 0]
    min_energy = np.min(T_00)
    print(f"Min Energy Density T^00: {min_energy}")
    
    if min_energy < 0:
        print("SUCCESS: Negative energy detected.")
    else:
        # Note: T^00 might differ from T_00 by sign depending on signature.
        # -+++ signature. u = (1,0,0,0). T_00 = T(u,u).
        # T^00 = g^0a g^0b T_ab = (-1)(-1) T_00 = T_00.
        # So T^00 should be negative for Alcubierre.
        print(f"WARNING: No negative energy? Min={min_energy}. velocity might be too low to see effect?")
        # With v very small, energy density is very small.
        # -v^2 terms.

if __name__ == "__main__":
    validate()
