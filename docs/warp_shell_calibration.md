# Warp Shell Calibration Notes

This note tracks the current status of the Python Warp Shell port against the
constant-velocity subluminal Warp Shell target from Fuchs et al.

## Current verification path

1. Build a `Comoving Warp Shell` metric with `create_warp_shell_metric`.
2. Solve the stress-energy tensor using `solve_energy_tensor`.
3. Transform to the Eulerian orthonormal frame.
4. Evaluate NEC/WEC/SEC/DEC with sampled observers.
5. Optionally run bounded optimized observer searches at candidate points.

The user-facing command is:

```bash
python examples/verify_physical_warp_shell.py --grid 8 --num-vecs 4 --audit-points 1
```

The W1 WarpFactory recipe extracted from `Examples/4 Warp Shell/W1_Warp_Shell.mlx`
is available as:

```bash
python examples/reproduce_fuchs_constant_warp_shell.py --profile quick
```

The exact MATLAB notebook parameters are encoded in:

```python
from warpfactory.recipes import fuchs_constant_warp_shell_parameters
params = fuchs_constant_warp_shell_parameters(profile="original")
```

Key original values:

- `R1 = 10`
- `R2 = 20`
- `factor = 1/3`
- `mass = R2/(2*G)*c^2*factor`
- `vWarp = 0.02`
- `sigma = 0`
- `smoothFactor = 4000`
- `spaceScale = 5`
- `gridSize = (1, 300, 300, 5)`
- `gridScaling = (1/(1000*c), 1/5, 1/5, 1/5)`

A small sweep is available with:

```bash
python examples/sweep_warp_shell_parameters.py --grid 10 --num-vecs 4
```

WarpFactory-compatible energy-condition conventions can be selected with:

```bash
python examples/sweep_warp_shell_parameters.py --energy-condition-method warpfactory
```

This is especially important for DEC because the MATLAB implementation evaluates
the dominant condition using nulllike vectors and the Minkowski norm of the flux,
then flips the sign so negative means violation.

The Fuchs-specific calibration sweep is:

```bash
python Examples/calibrate_fuchs_shell.py --profile original --velocities 0,0.02
```

It reports both global summaries and radial shell-only summaries. This matters
because the paper explicitly discusses numerical error floors in the vacuum and
boundary regions; the radial material shell is the first place to check before
interpreting global finite-difference noise as a physical violation.

## What is now controlled

- The massless static Warp Shell is component-wise Minkowski to roundoff.
- The solver has an explicit `flat_tolerance` option for near-flat smoke tests.
- `flat_tolerance` is opt-in and reported through `solver_diagnostics`.
- The Warp Shell grid coordinates now match MATLAB's `i*dx - center` convention.
- The warp shift component is symmetric and bounded by `[-v_warp, 0]`.
- A `warpfactory` energy-condition method is available for convention parity,
  especially DEC.
- `examples/export_warp_shell_radials.py` exports `r_sample`, `rho`, `P`, `M`,
  `A`, and `B` as `.npz` for MATLAB/Python parity checks.
- `tools/matlab_export_w1_reference.m`, `tools/convert_w1_mat_to_npz.py`, and
  `tools/compare_w1_reference.py` define the MATLAB-to-Python parity workflow
  for the full W1 recipe.
- `Examples/calibrate_fuchs_shell.py` sweeps Fuchs/W1 mass factors, smoothing,
  buffer radius, density/pressure smoothing ratio, and warp velocity.
- `summarize_energy_conditions(..., mask=..., tolerance=...)` can now summarize
  physical subregions separately from global numerical noise.

## Current nontrivial behavior

For the original W1 resolution (`gridSize = (1, 300, 300, 5)`) and WarpFactory
observer sampling, the static matter shell with `vWarp = 0` has no NEC
violations inside the material shell region when summarized with the radial mask
`R1 <= r <= R2`. This is the first successful reproduction milestone: the
positive-mass shell base is behaving like the physical shell described in the
paper.

Adding the published shift magnitude `beta_warp = 0.02` still produces shell
region violations in the Python pipeline. A velocity sweep at original
resolution shows no shell NEC violations through about `vWarp = 0.002`, small
violations by `vWarp = 0.005`, and large shell-region violations at `vWarp =
0.01` and `vWarp = 0.02`. This isolates the remaining mismatch to the shifted
Warp Shell stage rather than the static shell construction.

The public paper states two parameter details that are now exposed as knobs:

- The density/pressure smoothing span ratio is reported as approximately `1.72`;
  the public MATLAB metric file currently uses `1.79`.
- The compact sigmoid definition includes a positive buffer `R_b > 0`; the local
  W1 `.mlx` recipe uses `Rbuff = 0`.

In the current Python tests, changing these knobs did not recover the published
`beta_warp = 0.02` result, so the next calibration target is the shift-induced
stress-energy calculation itself.

Likely next checks:

- Compare Python and MATLAB stress-energy components for the shifted shell, not
  just the energy-condition maps.
- Compare Python `do_frame_transfer` against MATLAB `doFrameTransfer` on the
  same static shell.
- Compare `getEnergyConditions` maps against MATLAB with
  `energy_condition_method="warpfactory"`.
- Validate the TOV pressure/lapse profile against MATLAB arrays (`A`, `B`, `P`,
  `M`) for identical parameters.
- Audit the treatment of the `g01` shift term in the solver against the paper's
  Eq. (26), including whether the publication figures used a full 3D grid rather
  than the W1 cartoon slab with five z-slices.

## Interpretation

The current scripts are calibration tools, not yet a reproduction of the
published physical Warp Shell. They are designed to fail transparently when the
Python port does not yet match the intended metric/observer evaluation.

As of this calibration checkpoint, the Fuchs/WarpFactory static shell base is
reproduced in the material shell region, but the shifted `beta_warp = 0.02` Warp
Shell is not. The next comparison must therefore focus on the shifted
stress-energy tensor components and the exact solver behavior for the `g01`
addition, rather than broad parameter guessing.
