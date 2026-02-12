import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from warpfactory.generator.alcubierre import create_alcubierre_metric
from warpfactory.visualizer.plotter import plot_tensor, plot_3plus1
from warpfactory.constants import C

def validate():
    print("Testing Visualizer...")
    Nx, Ny, Nz = 20, 20, 20
    Nt = 5
    grid_scale = (1.0, 0.2, 0.2, 0.2)
    grid_size = (Nt, Nx, Ny, Nz)
    center = (grid_scale[0]*1, grid_scale[1]*Nx/2, grid_scale[2]*Ny/2, grid_scale[3]*Nz/2)
    
    # 0.5c
    v = 0.5 / C
    R = 2.0
    sigma = 8.0
    
    metric = create_alcubierre_metric(grid_size, grid_scale, center, v, R, sigma)
    print("Metric created.")
    
    # Create output dir
    output_dir = 'tests/output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Plotting Tensor...")
    # Slice Logic: 
    # planes=(1, 2) means x, y. 
    # slice_locations indices. Center of z is 10. Center of t is 2.
    # slice_locations=[2, 10, 10, 10] -> t=2, x=10, y=10, z=10.
    # But slice_planes are ignored in extraction.
    
    plot_tensor(metric, slice_planes=(1, 2), save_dir=output_dir, filename_prefix="alcubierre_tensor")
    
    print("Plotting 3+1 variables...")
    plot_3plus1(metric, slice_planes=(1, 2), save_dir=output_dir, filename_prefix="alcubierre_adm")
    
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    validate()
