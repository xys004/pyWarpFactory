# pyWarpFactory

A Python toolkit for analyzing warp drive spacetimes, ported from the original [MATLAB WarpFactory](https://github.com/NerdsWithAttitudes/WarpFactory).

![Alcubierre Metric](https://raw.githubusercontent.com/NerdsWithAttitudes/WarpFactory/main/Visualizer/images/AlcubierreMomentumFlow.gif)

## Overview

WarpFactory provides a numerical framework to analyze the physicality of spacetime metrics, specifically focusing on warp drives. It includes:

- **Metric Generators**: Alcubierre, Lentz, Warp Shell, Schwarzschild, etc.
- **Einstein Solver**: Numerical 3D solver for the Stress-Energy Tensor ($G_{\mu\nu} = 8\pi T_{\mu\nu}$).
- **Energy Conditions**: Point-wise evaluation of Null, Weak, Strong, and Dominant Energy Conditions.
- **ADM Decomposition**: 3+1 splitting of spacetime into Lapse, Shift, and Spatial Metric.
- **Visualizer**: Tools to plot tensor components and scalar fields.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Google Colab

You can run WarpFactory directly in Google Colab.

1. Upload the `WarpFactory_Colab.ipynb` file to Colab.
2. Upload the `warpfactory` folder to your Colab runtime (or clone your fork of the repository).
3. Run the notebook to install and verify the toolkit.

## Quick Start

```python
from warpfactory.generator.alcubierre import create_alcubierre_metric
from warpfactory.solver.solvers import solve_energy_tensor
from warpfactory.analyzer.energy_conditions import calculate_energy_conditions
from warpfactory.constants import C

# 1. Create a Metric
grid_size = (5, 50, 50, 50)
grid_scale = (1.0, 0.2, 0.2, 0.2)
center = (0, 5.0, 5.0, 5.0)
metric = create_alcubierre_metric(grid_size, grid_scale, center, v=0.5/C, R=2.0, sigma=8.0)

# 2. Solve Einstein Field Equations
energy_tensor = solve_energy_tensor(metric)

# 3. Analyze Energy Conditions
nec_map = calculate_energy_conditions(energy_tensor, metric, condition="Null")

if nec_map.min() < 0:
    print("Optimization: Negative Energy Detected (Warp Drive Active!)")
```

## Documentation

See [USER_MANUAL.md](USER_MANUAL.md) for detailed instructions and examples.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Credits

Original WarpFactory Team: Christopher Helmerich, Jared Fuchs, et al.
Python Port: Nelson (Agentic AI).
