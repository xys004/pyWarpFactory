import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from warpfactory.generator.alcubierre import create_alcubierre_metric
from warpfactory.analyzer.energy_conditions import calculate_energy_conditions


def validate():
    print("Initializing Alcubierre Metric...")
    # Parameters matching a standard test case
    # Grid: coords t, x, y, z
    Nx, Ny, Nz = 30, 30, 30
    Nt = 7  # Nt >= 5 required for 4th-order finite differences

    # Grid spacing (code units, C=1 assumed in this test)
    grid_scale = (1.0, 0.2, 0.2, 0.2)
    grid_size  = (Nt, Nx, Ny, Nz)

    # World center (physical coords of grid center)
    center = (grid_scale[0] * 1, grid_scale[1] * Nx / 2,
              grid_scale[2] * Ny / 2, grid_scale[3] * Nz / 2)

    # Warp drive parameters
    v     = 0.5   # bubble velocity (fraction of C – keep small to stay on grid)
    R     = 2.0   # bubble radius (grid units)
    sigma = 8.0   # shell steepness

    metric = create_alcubierre_metric(grid_size, grid_scale, center, v, R, sigma)
    print("Metric created.")

    print("Calculating Energy Conditions (full pipeline)...")
    results, T_euler = calculate_energy_conditions(metric.tensor, grid_scale)

    # Extract a representative 2-D slice (middle time step, mid-z plane)
    t_slice = Nt // 2
    z_slice = Nz // 2

    # T_euler is (4, 4, Nt, Nx, Ny, Nz)
    # Energy density seen by a local observer = T^{00} in Eulerian frame
    energy_density = T_euler[0, 0, t_slice, :, :, z_slice]

    print(f"Max Energy Density : {np.max(energy_density):.4e}")
    print(f"Min Energy Density : {np.min(energy_density):.4e}")

    if np.min(energy_density) < 0:
        print("SUCCESS: Negative energy density detected (expected for Alcubierre drive).")
    else:
        print("WARNING: No negative energy density detected. Check physics / units.")

    # ----------------------------------------------------------------
    # Plotting
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    extent = [0, Nx * grid_scale[1], 0, Ny * grid_scale[2]]

    im0 = axes[0].imshow(energy_density.T, origin='lower', extent=extent)
    plt.colorbar(im0, ax=axes[0], label='T^{00} (energy density)')
    axes[0].set_title(f'Energy Density  (t={t_slice}, z={z_slice})')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')

    nec = results['Null'][t_slice, :, :, z_slice]
    im1 = axes[1].imshow(nec.T, origin='lower', extent=extent)
    plt.colorbar(im1, ax=axes[1], label='NEC  (<0 = violated)')
    axes[1].set_title('Null Energy Condition')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')

    wec = results['Weak'][t_slice, :, :, z_slice]
    im2 = axes[2].imshow(wec.T, origin='lower', extent=extent)
    plt.colorbar(im2, ax=axes[2], label='WEC  (<0 = violated)')
    axes[2].set_title('Weak Energy Condition')
    axes[2].set_xlabel('x'); axes[2].set_ylabel('y')

    print(f"Min NEC value : {np.min(nec):.4e}")
    print(f"Min WEC value : {np.min(wec):.4e}")

    plt.tight_layout()
    output_file = 'alcubierre_validation_plot.png'
    plt.savefig(output_file, dpi=120)
    print(f"Plots saved to {output_file}")


if __name__ == "__main__":
    validate()
