import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from warpfactory.generator.alcubierre import create_alcubierre_metric
from warpfactory.algo.adm import decompose_3plus1, reconstruct_3plus1
from warpfactory.constants import C

def validate():
    print("Testing ADM Decomposition...")
    Nx, Ny, Nz = 10, 10, 10
    Nt = 2
    grid_scale = (1.0, 1.0, 1.0, 1.0)
    grid_size = (Nt, Nx, Ny, Nz)
    center = (0, 5, 5, 5)
    
    metric = create_alcubierre_metric(grid_size, grid_scale, center, v=0.5/C, R=2.0, sigma=8.0)
    
    print("Metric created.")
    
    # Decompose
    alpha, beta, gamma = decompose_3plus1(metric)
    print("Decomposed.")
    
    # Reconstruct
    tensor_rec = reconstruct_3plus1(alpha, beta, gamma)
    print("Reconstructed.")
    
    # Compare
    diff = np.abs(metric.tensor - tensor_rec)
    max_diff = np.max(diff)
    print(f"Max reconstruction error: {max_diff}")
    
    if max_diff < 1e-12:
        print("SUCCESS: Reconstruction matches original.")
    else:
        print("FAILURE: Reconstruction mismatch.")
        exit(1)

if __name__ == "__main__":
    validate()
