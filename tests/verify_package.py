import sys
import os

# Ensure we can import warpfactory (assuming it's installed or in path)
try:
    import warpfactory
    print(f"WarpFactory package found at: {warpfactory.__file__}")
except ImportError:
    print("WarpFactory package NOT found. Please install it.")
    sys.exit(1)

from warpfactory.generator.minkowski import create_minkowski_metric
from warpfactory.solver.solvers import solve_energy_tensor

def verify_package():
    print("Verifying WarpFactory package...")
    
    # 1. Generate
    grid_size = (3, 10, 10, 10)
    grid_scale = (1.0, 1.0, 1.0, 1.0)
    metric = create_minkowski_metric(grid_size, grid_scale)
    print(f"Generated {metric.name} metric.")
    
    # 2. Solve
    # Minkowski should have 0 energy
    energy = solve_energy_tensor(metric)
    max_E = energy.tensor.max()
    print(f"Max energy in Minkowski: {max_E}")
    
    if max_E > 1e-10:
        print("FAILURE: Minkowski energy should be effectively 0.")
        sys.exit(1)
        
    print("SUCCESS: Package verification passed.")

if __name__ == "__main__":
    verify_package()
