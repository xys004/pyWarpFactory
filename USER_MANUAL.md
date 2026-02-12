# pyWarpFactory User Manual

## Introduction
WarpFactory is a toolkit for analyzing warp drive spacetimes. This Python version is a translation of the original MATLAB WarpFactory, designed to be more accessible and easy to integrate with the Python scientific ecosystem.

## Installation

### Requirements
- Python 3.8+
- numpy
- matplotlib
- scipy (optional, but recommended for advanced interpolation)

### Setup
1. Clone the repository.
2. Ensure the `warpfactory` package is in your PYTHONPATH.

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/WarpFactory
```

## Quick Start

### Creating an Alcubierre Warp Drive
```python
from warpfactory.generator.alcubierre import create_alcubierre_metric
from warpfactory.constants import C

# Define grid
grid_size = (Nt, Nx, Ny, Nz) = (5, 50, 50, 50)
grid_scale = (dt, dx, dy, dz) = (1.0, 0.2, 0.2, 0.2)
center = (0, 5.0, 5.0, 5.0)

# Parameters
v = 0.5 / C  # Velocity
R = 2.0      # Radius
sigma = 8.0  # Wall thickness

# Create Metric
metric = create_alcubierre_metric(grid_size, grid_scale, center, v, R, sigma)
print(f"Created metric: {metric.name}")
```

### Analysis: Computing Energy density
```python
from warpfactory.solver.solvers import solve_energy_tensor
from warpfactory.analyzer.energy_conditions import calculate_energy_conditions

# Compute Energy Tensor (Einstein Solver)
energy_tensor = solve_energy_tensor(metric)

# Check Null Energy Condition
nec_map = calculate_energy_conditions(energy_tensor, metric, condition="Null")

print(f"Min NEC value: {nec_map.min()}")
# Negative values indicate NEC violation (required for warp drives!)
```

### Visualization
```python
from warpfactory.visualizer.plotter import plot_tensor

# Plot Energy Density (T_00) at center slice
plot_tensor(
    energy_tensor, 
    component_indices=(0, 0), 
    slice_dim='z', 
    slice_idx=Nz//2, 
    time_idx=Nt//2,
    save_path="energy_density.png"
)
```

## Advanced Usage

### Warp Shell Metric
The Warp Shell is a more physically detailed metric with defined shell mass and finite thickness.

```python
from warpfactory.generator.warp_shell import create_warp_shell_metric

metric = create_warp_shell_metric(
    grid_size, grid_scale, center,
    mass=1000.0,
    r_inner=2.0, r_outer=4.0, r_buff=0.5,
    sigma=4.0, smooth_factor=2,
    v_warp=0.1, do_warp=True
)
```

### ADM Decomposition
You can decompose time-dependent metrics into 3+1 ADM variables (lapse $\alpha$, shift $\beta$, spatial metric $\gamma$).

```python
from warpfactory.algo.adm import decompose_3plus1

alpha, beta, gamma = decompose_3plus1(metric)
# alpha: scalar field
# beta: [beta_x, beta_y, beta_z]
# gamma: [[g_xx, g_xy, ...], ...]
```

### Frame Transfer
Transform tensors into the local Eulerian frame (free-falling observer).

```python
from warpfactory.analyzer.transform import do_frame_transfer

energy_eulerian = do_frame_transfer(metric, energy_tensor, frame="Eulerian")
# energy_eulerian.tensor is now in the orthonormal basis
```

## API Overview

### Generators (`warpfactory.generator`)
- `alcubierre.py`: Classic Alcubierre drive.
- `lentz.py`: Lentz solution.
- `van_den_broeck.py`: Volume expanding metric.
- `warp_shell.py`: Finite shell model.
- `schwarzschild.py`: Static black hole.
- `modified_time.py`: Modified time rate metric.
- `minkowski.py`: Flat space.

### Solvers (`warpfactory.solver`)
- `solvers.py`: Main entry point (`solve_energy_tensor`).
- `finite_difference.py`: 4th order finite difference derivatives.

### Analyzer (`warpfactory.analyzer`)
- `energy_conditions.py`: NEC, WEC, SEC, DEC evaluations.
- `transform.py`: Frame transformations and index manipulation.
- `momentum_flow.py`: Flow line tracking.
- `scalars.py`: Expansion and Shear scalars.

### Visualizer (`warpfactory.visualizer`)
- `plotter.py`: 2D slice plotting.

## Differences from MATLAB Version
- **0-based indexing**: Python uses 0-based indexing (t, x, y, z are indices 0, 1, 2, 3).
- **Class-based**: `Metric` is a dataclass object, not a struct.
- **Explicit ordering**: Tensor dimensions are `(4, 4, Nt, Nx, Ny, Nz)` for efficient broadcasting. MATLAB often put 4x4 at the end.
