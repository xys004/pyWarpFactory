import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from warpfactory.generator.alcubierre import create_alcubierre_metric
from warpfactory.analyzer.energy_conditions import calculate_energy_conditions
from warpfactory.solver.tensor_utils import get_energy_tensor

def validate():
    print("Initializing Alcubierre Metric...")
    # Parameters matching a standard test case
    # Grid: coords t, x, y, z
    # Let's use a modest grid for speed
    # MATLAB examples often use 100x100x100
    Nx, Ny, Nz = 30, 30, 30
    Nt = 7 # Increased for finite difference stability (stencil width 5)
    
    # 0.1 spacing
    grid_scale = (1.0, 0.2, 0.2, 0.2)
    grid_size = (Nt, Nx, Ny, Nz)
    
    # Center
    center = (grid_scale[0]*1, grid_scale[1]*Nx/2, grid_scale[2]*Ny/2, grid_scale[3]*Nz/2)
    
    # Physics params
    v = 0.5 # 0.5c
    R = 2.0
    sigma = 8.0 # Steepness
    
    metric = create_alcubierre_metric(grid_size, grid_scale, center, v, R, sigma)
    print("Metric created.")
    
    print("Calculating Energy Conditions...")
    results, T_euler = calculate_energy_conditions(metric.tensor, grid_scale)
    pass
    
    # Extract data from the middle time slice
    t_slice = 1
    z_slice = Nz // 2
    
    # T_euler is (4, 4, t, x, y, z)
    # Energy Density is T_00 (in Eulerian frame)
    energy_density = T_euler[0, 0, t_slice, :, :, z_slice]
    
    print(f"Max Energy Density: {np.max(energy_density)}")
    print(f"Min Energy Density: {np.min(energy_density)}")
    
    # Alcubierre drive should have negative energy density in the shell
    if np.min(energy_density) < 0:
        print("SUCCESS: Negative energy density detected (expected for Alcubierre drive).")
    else:
        print("WARNING: No negative energy density detected. Physics might be wrong.")
        
    # Plotting
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(energy_density.T, origin='lower', extent=[0, Nx*grid_scale[1], 0, Ny*grid_scale[2]])
    plt.colorbar(label='Energy Density T_00')
    plt.title(f'Energy Density (z={z_slice*grid_scale[3]:.1f})')
    
    # Plot Null Energy Condition Violation
    # results['Null'] contains min(T_ab k^a k^b)
    # If negative, it violates.
    nec = results['Null'][t_slice, :, :, z_slice]
    plt.subplot(2, 2, 2)
    plt.imshow(nec.T, origin='lower')
    plt.colorbar(label='NEC Violation (<0)')
    plt.title('Null Energy Condition')
    
    # Plot Weak
    wec = results['Weak'][t_slice, :, :, z_slice]
    plt.subplot(2, 2, 3)
    plt.imshow(wec.T, origin='lower')
    plt.colorbar(label='WEC Violation (<0)')
    plt.title('Weak Energy Condition')
    
    # Plot Expansion (Tr(K)?) - Not implemented yet, just show check status
    print(f"Min NEC: {np.min(nec)}")
    print(f"Min WEC: {np.min(wec)}")
    
    output_file = 'alcubierre_validation_plot.png'
    plt.savefig(output_file)
    print(f"Plots saved to {output_file}")

if __name__ == "__main__":
    validate()
