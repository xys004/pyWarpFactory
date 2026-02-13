import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
# Assuming we run from StaticWarpBubbles/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add warpfactory path if needed (Assuming parent folder)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from static_bubbles.generator import create_static_bubble_metric
from static_bubbles.analyzer import analyze_static_bubble
from warpfactory.constants import C

def demo_static_bubble():
    print("Generating 2025 Static Bubble Demo...")
    
    # 1. Define Density Profile (Double Shell Example)
    # rho(r) = A * (r/R)^2 * exp(-(r-R)^2 / sigma^2)
    # This creates a shell with zero density at the center.
    
    R = 5.0
    sigma = 1.0
    Amp = 0.5
    
    def rho_func(r):
        return Amp * (r/R)**2 * np.exp(-(r - R)**2 / sigma**2)
    
    # 2. Generate Metric (to get grid and beta)
    grid_size = (3, 60, 60, 60)
    grid_scale = (1.0, 0.2, 0.2, 0.2) # dt, dx, dy, dz
    center = (0, 6.0, 6.0, 6.0)
    
    metric = create_static_bubble_metric(
        grid_size, grid_scale, center,
        rho_profile=rho_func
    )
    
    # 3. Analyze Energy Conditions
    # We analyze on the 3D grid
    coords = metric.coords
    x = coords['x'] - center[1]
    y = coords['y'] - center[2]
    z = coords['z'] - center[3]
    r_3d = np.sqrt(x**2 + y**2 + z**2)
    
    analysis = analyze_static_bubble(rho_func, r_3d)
    
    # 4. Visualization
    # Extract 1D slice for clean plotting
    # Slice along X axis
    sl_y = grid_size[2] // 2
    sl_z = grid_size[3] // 2
    
    r_slice = x[0, :, sl_y, sl_z] # 1D array of x coordinates (relative to center)
    # Filter for positive r to match radial plot
    mask = r_slice >= 0
    r_plot = r_slice[mask]
    
    # Get Beta from metric parameters (it stores the 1D profile used)
    # Or interpolate from 3D metric. 
    # Let's use the analytic beta profile stored in params if available, or re-calculate.
    # The metric params has 'beta_r' and 'r_samples'.
    
    if 'beta_r' in metric.params:
        beta_vals = metric.params['beta_r']
        r_vals = metric.params['r_samples']
        # Interpolate to r_plot
        beta_plot = np.interp(r_plot, r_vals, beta_vals)
    else:
        # Fallback if params missing
        beta_plot = np.zeros_like(r_plot) 
    
    # Get Energy Conditions
    rho_plot = analysis['rho'][0, :, sl_y, sl_z][mask]
    nec_plot = analysis['NEC'][0, :, sl_y, sl_z][mask]
    dec_plot = analysis['DEC'][0, :, sl_y, sl_z][mask]
    
    # Plot 1: Metric Functions
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Radius r')
    ax1.set_ylabel('Energy Density rho', color=color)
    ax1.plot(r_plot, rho_plot, color=color, label='rho(r)')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Shift Beta(r)', color=color)  # we already handled the x-label with ax1
    ax2.plot(r_plot, beta_plot, color=color, linestyle='--', label='beta(r)')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(1.0, color='gray', linestyle=':', label='Horizon Limit')
    
    plt.title('Static Bubble: Density and Shift')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # Output to current dir (examples/)
    output_dir = os.path.dirname(__file__)
    plt.savefig(os.path.join(output_dir, 'static_bubble_metric.png'))
    print(f"Saved metric plot to {os.path.join(output_dir, 'static_bubble_metric.png')}")
    
    # Plot 2: Energy Conditions
    plt.figure(figsize=(10, 6))
    plt.plot(r_plot, rho_plot, label='rho (WEC)')
    plt.plot(r_plot, nec_plot, label='rho + p_perp (NEC)')
    plt.plot(r_plot, dec_plot, label='rho - |p_perp| (DEC)')
    
    plt.axhline(0, color='black', linewidth=0.5)
    plt.xlabel('Radius r')
    plt.ylabel('Energy Condition Value')
    plt.title('Energy Conditions (Positive = Satisfied)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'static_bubble_ec.png'))
    print(f"Saved EC plot to {os.path.join(output_dir, 'static_bubble_ec.png')}")

if __name__ == "__main__":
    demo_static_bubble()
