from warpfactory.generator.alcubierre import create_alcubierre_metric
from warpfactory.solver.solvers import solve_energy_tensor
from warpfactory.analyzer.energy_conditions import calculate_energy_conditions
from warpfactory.constants import C

print("--- pyWarpfactory Example ---")
print("Input:")
print("Generating Alcubierre Metric with velocity v=0.5c on a 5x50x50x50 grid...")
grid_size = (5, 50, 50, 50)
grid_scale = (1.0, 0.2, 0.2, 0.2)
center = (0, 5.0, 5.0, 5.0)
metric = create_alcubierre_metric(grid_size, grid_scale, center, v=0.5/C, R=2.0, sigma=8.0)

print("Running 3D Numerical Solver for the Einstein Field Equations (Stress-Energy Tensor)...")
energy_tensor = solve_energy_tensor(metric)

print("Analyzing Energy Conditions (Null Energy Condition)...")
nec_map = calculate_energy_conditions(energy_tensor, metric, condition="Null")

print("\nOutput:")
if nec_map.min() < 0:
    print(f"Result: NEGATIVE ENERGY DETECTED.")
    print(f"Minimum Energy Density (NEC violation peak): {nec_map.min():.4e}")
else:
    print("Result: No negative energy detected.")
print("Evaluation Complete.")
