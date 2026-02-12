import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from warpfactory.generator.alcubierre import create_alcubierre_metric
from warpfactory.solver.solvers import solve_energy_tensor
from warpfactory.analyzer.energy_conditions import calculate_energy_conditions
from warpfactory.constants import C

def validate():
    print("Testing Energy Conditions...")
    Nx, Ny, Nz = 20, 20, 20
    Nt = 5
    grid_scale = (1.0, 0.2, 0.2, 0.2)
    grid_size = (Nt, Nx, Ny, Nz)
    center = (grid_scale[0]*1, grid_scale[1]*Nx/2, grid_scale[2]*Ny/2, grid_scale[3]*Nz/2)
    
    # Needs significant velocity to show violations clearly
    # But for frame transfer formula (1/factor0), g_44 must be fine.
    
    v = 0.5 / C
    R = 2.0
    sigma = 8.0
    
    metric = create_alcubierre_metric(grid_size, grid_scale, center, v, R, sigma)
    print("Metric created.")
    
    # 1. Solve Energy Tensor
    energy_tensor = solve_energy_tensor(metric)
    print("Energy Tensor computed.")
    
    # 2. Calculate NEC
    # Alcubierre drive is known to violate NEC.
    # So we expect negative values in the map.
    
    nec_map = calculate_energy_conditions(energy_tensor, metric, condition="Null")
    min_nec = np.min(nec_map)
    print(f"Minimum NEC evaluation: {min_nec}")
    
    if min_nec < -1e-10: # Tolerance for float
        print("SUCCESS: NEC violation detected (Negative value found).")
    else:
        print("WARNING: No NEC violation detected? Alcubierre should violate it.")
        
    # Plot center slice of NEC
    slice_idx = Nx // 2
    nec_slice = nec_map[Nt//2, slice_idx, :, :]
    
    plt.figure()
    plt.imshow(nec_slice.T, origin='lower')
    plt.colorbar(label='NEC Evaluation')
    plt.title('Null Energy Condition Violation')
    plt.savefig('tests/output/nec_violation.png')
    print("Plot saved to tests/output/nec_violation.png")

if __name__ == "__main__":
    validate()
