# pyWarpFactory

A Python-first toolkit for analyzing warp drive spacetimes, inspired by and
partly ported from the original [MATLAB WarpFactory](https://github.com/NerdsWithAttitudes/WarpFactory).

This repository is not the original WarpFactory codebase. Its default physical
engine is the independent `christoffel` solver; the MATLAB-style
`warpfactory_direct` path is kept as a legacy-audit/compatibility route and is
not recommended for physical energy-condition conclusions.

![Alcubierre Metric](https://raw.githubusercontent.com/NerdsWithAttitudes/WarpFactory/main/Visualizer/images/AlcubierreMomentumFlow.gif)

## Overview

WarpFactory provides a numerical framework to analyze the physicality of spacetime metrics, specifically focusing on warp drives. It includes:

- **Metric Generators**: Alcubierre, Lentz, Warp Shell, Schwarzschild, etc.
- **Einstein Solver**: Numerical 3D solver for the Stress-Energy Tensor ($G_{\mu\nu} = 8\pi T_{\mu\nu}$).
- **Energy Conditions**: Point-wise evaluation of Null, Weak, Strong, and Dominant Energy Conditions. Weak, Strong, and Dominant currently use sampled timelike observers.
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

## Local and Cloud Jobs

The same job runner works locally and on Vertex AI:

```bash
python -m warpfactory.cloud.run_job --execution-target local --profile quick --static
```

For Google Cloud Vertex AI setup and GPU/CPU job submission, see
[docs/vertex_ai.md](docs/vertex_ai.md). Cloud credentials are not stored in this
repository.

## Quick Start

### One-call analysis

```python
from warpfactory import analyze_metric
from warpfactory.generator.alcubierre import create_alcubierre_metric

metric = create_alcubierre_metric(
    grid_size=(5, 50, 50, 50),
    grid_scale=(1.0, 0.2, 0.2, 0.2),
    world_center=(0, 5.0, 5.0, 5.0),
    v=0.5,
    R=2.0,
    sigma=8.0,
)

result = analyze_metric(metric)
print(result.summary)
print(result.has_violation("Null"))
print(result.has_violation("Dominant"))
```

### Observer modes

```python
# Conservative default: finite sampled observers, WarpFactory-style diagnostics.
sampled = analyze_metric(metric, observer_mode="sampled")

# Audit mode: keep sampled maps, then run bounded optimized observer searches
# at the most critical candidate points.
audited = analyze_metric(metric, observer_mode="optimized", audit_points=3)
print(audited.observer_audit)
```

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

## Result Hygiene

Some historical Fuchs W1 outputs were generated before the `compact_sigmoid`
orientation fix. Do not use those pre-fix artifacts for physical conclusions;
see [outputs/DEPRECATED_PRE_SWARP_FIX.md](outputs/DEPRECATED_PRE_SWARP_FIX.md)
for the affected runs and the valid post-fix replacement result.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Credits

Original WarpFactory Team: Christopher Helmerich, Jared Fuchs, et al.
pyWarpFactory development: Nelson, with Codex-assisted implementation and audit.
