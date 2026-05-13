# Fuchs / WarpFactory Reproduction Status

This note records the current Python parity status against the local
WarpFactory `Examples/4 Warp Shell/W1_Warp_Shell.mlx` recipe.

## Reference Recipe

The local MATLAB W1 recipe uses:

- `R1 = 10`
- `R2 = 20`
- `Rbuff = 0`
- `factor = 1/3`
- `m = R2/(2*G)*c^2*factor`
- `vWarp = 0.02`
- `sigma = 0`
- `smoothFactor = 4000`
- `spaceScale = 5`
- `gridSize = [1, 300, 300, 5]`
- `gridScaling = [1/(1000*c), 1/5, 1/5, 1/5]`

The Python recipe is implemented in
`warpfactory.recipes.fuchs_warp_shell`.

## Current Cloud Checks

Both checks below were run on Google Compute Engine with the packaged
pyWarpFactory source, not from an editable local checkout. This verifies the
installable package path as well as the numerical pipeline.

### W1 Original, Static Control

Command shape:

```powershell
tools\run_compute_job.ps1 `
  -InstanceName pywarpfactory-original-static `
  -MachineType e2-standard-4 `
  -Profile original `
  -Static `
  -NumVecs 4 `
  -OutputUri gs://warpopt-data/pywarpfactory/runs/compute-original-static
```

Material-shell result (`R1 <= r <= R2`):

| Condition | Min | Violating fraction |
| --- | ---: | ---: |
| Null | `1.8257e+39` | `0.0` |
| Weak | `1.8257e+39` | `0.0` |
| Strong | `9.9695e+38` | `0.0` |
| Dominant | `3.8285e+39` | `0.0` |

Interpretation: the static positive-mass shell base is reproduced in the
material region. Global/exterior violations are still dominated by numerical
vacuum/boundary behavior, so material-shell masks are mandatory for Fuchs
comparisons.

### W1 Original, Shifted

Command shape:

```powershell
tools\run_compute_job.ps1 `
  -InstanceName pywarpfactory-original-shift `
  -MachineType e2-standard-4 `
  -Profile original `
  -VWarp 0.02 `
  -NumVecs 4 `
  -OutputUri gs://warpopt-data/pywarpfactory/runs/compute-original-shift
```

Material-shell result (`R1 <= r <= R2`):

| Condition | Min | Violating fraction |
| --- | ---: | ---: |
| Null | `-1.1419e+40` | `0.2710` |
| Weak | `-1.1419e+40` | `0.2710` |
| Strong | `-1.1419e+40` | `0.2833` |
| Dominant | `-1.4882e+40` | `0.3068` |

Interpretation: the remaining mismatch is not the static shell construction.
It is isolated to the shifted Warp Shell stage, specifically the interaction
between the `g_tx = -ShiftMatrix*vWarp` term, stress-energy construction,
Eulerian frame transfer, and energy-condition evaluation.

## Local MATLAB Parity Notes

Direct inspection of the local MATLAB source confirms that the Python metric
builder currently follows the public WarpFactory construction:

- `metricGet_WarpShellComoving.m` sets `Metric.tensor{1,2}` to
  `-ShiftMatrix*vWarp` when the original value is zero.
- The MATLAB metric builder does not add an ADM `beta_i beta^i` correction to
  `g_tt` in this function.
- `getEnergyConditions.m` first transfers the energy tensor to the Eulerian
  frame, then evaluates finite sampled null/timelike fields.
- The MATLAB dominant condition uses null vectors and a Minkowski flux norm,
  followed by a sign flip so negative values mean violation.

## Next Parity Targets

The Python-side component audit is now available:

```powershell
python Examples\audit_fuchs_w1_components.py `
  --profile original `
  --case both `
  --num-vecs 4 `
  --critical-points 3 `
  --output-dir outputs\fuchs_w1_component_audit_original_arrays `
  --save-arrays
```

This writes:

- `outputs/fuchs_w1_component_audit_original_arrays/fuchs_w1_component_audit.json`
- `outputs/fuchs_w1_component_audit_original_arrays/fuchs_w1_component_audit_arrays.npz`

The original-profile audit confirms that the shifted violations are not caused
by negative shell density. In the material shell, the shifted case keeps
`rho >= 0` and `rho + p_min >= 0`, but develops a large Eulerian flux term:

| Case | shell `rho` min | shell `T01` range | shell `flux_norm` max | shell `rho+p_min` min |
| --- | ---: | ---: | ---: | ---: |
| static | `5.8722e+39` | `0` | `0` | `3.5017e+39` |
| shifted | `5.7907e+39` | `[-1.8750e+40, 1.3901e+40]` | `1.8750e+40` | `3.5022e+39` |

The most negative shifted shell NEC points occur around `r = 11.72`, with
`T00 = 1.3909e+40` and `T01 = -1.7467e+40`. This strongly suggests that the
remaining Fuchs mismatch is in the shifted momentum/flux sector: stress-energy
construction, Eulerian frame transfer, or condition contraction after the
`g_tx` term is added.

After porting MATLAB `getEvenPointsOnSphere.m` exactly into the
WarpFactory-compatible Python energy-condition sampler, the qualitative result
is unchanged. The shifted shell still violates sampled CE in the material
region, now with the exact MATLAB angular field:

| Case | shell NEC min | shell NEC violating fraction | strongest-point radius | strongest-point `T00` | strongest-point `T01` |
| --- | ---: | ---: | ---: | ---: | ---: |
| static | `1.8224e+39` | `0.0` | `20.0` | `5.8742e+39` | `0` |
| shifted | `-1.0012e+40` | `0.2796` | `10.28` | `9.7587e+39` | `-1.4544e+40` |

This removes sampled-direction mismatch as the primary explanation.

## MATLAB-Compatible Python Lane

A first explicit MATLAB-compatibility layer now exists in
`warpfactory.analyzer.matlab_compat`. It ports/exposes:

- `getEvenPointsOnSphere.m`
- `generateUniformField.m`
- `changeTensorIndex.m`
- `doFrameTransfer.m` through the existing explicit Cholesky implementation
- `getEnergyConditions.m` in WarpFactory-compatible mode
- `evalMetric.m` core products through `matlab_eval_metric`

Baseline validation:

```powershell
python tests\validate_matlab_compat.py
```

Current result:

- zero stress-energy tensors remain neutral for all CE;
- Minkowski remains exactly zero through the MATLAB-compatible lane;
- MATLAB vector-field and index-convention checks pass.

The W1 audit can use this lane:

```powershell
python Examples\audit_fuchs_w1_components.py `
  --profile original `
  --case static `
  --analysis-mode matlab_compat `
  --solver-method warpfactory_direct `
  --num-vecs 4
```

Important: this lane is still experimental for W1. With
`solver_method=warpfactory_direct`, the original static shell currently gives
clean material-shell NEC/SEC but a small WEC negative fraction and a large DEC
negative fraction. The independent Christoffel lane remains the one that
reproduces the static material shell cleanly across NEC/WEC/SEC/DEC.

This means the next literal porting target is narrower than before:
`ricciT.m`, `takeFiniteDifference1.m`, `takeFiniteDifference2.m`, and the DEC
branch of `getEnergyConditions.m`.

The matching MATLAB-side export is prepared as:

```matlab
run('tools/matlab_export_w1_component_audit.m')
```

Convert it with:

```powershell
python tools\convert_w1_mat_to_npz.py `
  tools\w1_component_audit_reference.mat `
  --output tools\w1_component_audit_reference.npz
```

Then compare against Python:

```powershell
python tools\compare_w1_component_audit.py `
  outputs\fuchs_w1_component_audit_original_arrays\fuchs_w1_component_audit_arrays.npz `
  tools\w1_component_audit_reference.npz
```

Remaining parity targets:

1. Run the MATLAB W1 component export from the same local source and compare
   Python arrays component by component.
2. Compare shifted stress-energy components before energy-condition contraction:
   `T_hat00`, `T_hat01`, spatial stress, trace, and flux norm.
3. Compare Python `do_frame_transfer` against MATLAB `doFrameTransfer` on the
   same shifted tensor.
4. Compare the Ricci / Einstein tensor path against MATLAB `met2den.m` for the
   shifted case, with special attention to finite-difference boundary copying.
5. Only after component parity, increase observer coverage beyond `num_vecs=4`
   or run optimized observer audits.
