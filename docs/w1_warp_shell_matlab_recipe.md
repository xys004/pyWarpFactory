# W1 Warp Shell MATLAB Recipe

Extracted from `Examples/4 Warp Shell/W1_Warp_Shell.mlx`.

```matlab
%% Shell Metric
spaceScale = 5;
timeScale = 1;
tryGPU = 1;
centered = 1;
cartoonThickness = 5;

R1 = 10;
Rbuff = 0;
R2 = 20;

if centered == 1
    gridSize = ceil([1,2*(R2+10)*spaceScale,2*(R2+10)*spaceScale,cartoonThickness]);
else
    gridSize = ceil([1,(R2+10)*spaceScale,(R2+10)*spaceScale,cartoonThickness]);
end

factor = 1/3;
m = R2/(2*G)*c^2*factor;

vWarp = 0.02; % in betas

sigma = 0;
doWarp = 1;

gridScaling = [1/(timeScale*spaceScale*((vWarp)*c+1)),1/spaceScale,1/spaceScale,1/spaceScale];
gridScaling(1) = 1/(1000*c);

if centered == 1
    worldCenter = [(cartoonThickness+1)/2,(2*(R2+10)*spaceScale+1)/2,(2*(R2+10)*spaceScale+1)/2,(cartoonThickness+1)/2].*gridScaling;
else
    worldCenter = [(cartoonThickness+1)/2,5,5,(cartoonThickness+1)/2].*gridScaling;
end

smoothFactor = 4000;

[Metric_ConstantWarp] = metricGet_WarpShellComoving(gridSize,worldCenter,m,R1,R2,Rbuff,sigma,smoothFactor,vWarp,doWarp,gridScaling);
ConstantWarp = evalMetric(Metric_ConstantWarp,1,1);
```

Python equivalent:

```python
from warpfactory.recipes import create_fuchs_constant_warp_shell
from warpfactory import analyze_metric

metric = create_fuchs_constant_warp_shell(profile="original")
result = analyze_metric(metric, energy_condition_method="warpfactory")
```

For a fast smoke test:

```bash
python examples/w1_warp_shell_python.py --profile quick
```

The `quick` profile preserves the W1 physical radii, mass formula, velocity,
and metric family, but uses a smaller grid/smoothing setup so the workflow can
run on ordinary laptops. It is not a publication reproduction.

## MATLAB Reference Export

If MATLAB is available, export the original W1 reference arrays:

```matlab
run('tools/matlab_export_w1_reference.m')
```

This writes:

```text
tools/w1_reference.mat
```

Convert it to NumPy format:

```bash
python tools/convert_w1_mat_to_npz.py tools/w1_reference.mat --output tools/w1_reference.npz
```

Run the Python W1 workflow and save arrays:

```bash
python examples/w1_warp_shell_python.py --profile original --output-dir outputs/w1_original
```

Compare Python against MATLAB:

```bash
python tools/compare_w1_reference.py outputs/w1_original/w1_python_arrays.npz tools/w1_reference.npz
```

The comparison includes metric tensor components, Eulerian stress-energy tensor,
energy-condition maps, and radial profiles.
