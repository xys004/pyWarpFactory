# Local and Vertex AI execution

pyWarpFactory uses one portable job runner for both local and cloud execution:

```bash
python -m warpfactory.cloud.run_job --profile quick --v-warp 0.02
```

The same module is used by Vertex AI Custom Jobs with
`--execution-target=vertex`. Credentials are intentionally not stored in the
repository. Use `gcloud auth login`, `gcloud auth application-default login`, or
a service account configured outside the repo.

## Local execution

Local runs do not need Google Cloud:

```powershell
.\tools\run_local_job.ps1 `
  -Config configs\fuchs_w1_quick_local.json `
  -NumVecs 8 `
  -OutputDir outputs\local_smoke
```

Equivalent direct Python command:

```powershell
python -m warpfactory.cloud.run_job `
  --config configs\fuchs_w1_quick_local.json `
  --execution-target local `
  --num-vecs 8 `
  --local-output-dir outputs\local_smoke
```

Both write `summary.json` locally and print the same payload to stdout.

## Job Configs

Reusable job definitions live in `configs/`. A config is plain JSON:

```json
{
  "execution_target": "local",
  "recipe": "fuchs-w1",
  "profile": "quick",
  "static": true,
  "num_vecs": 12,
  "energy_condition_method": "warpfactory",
  "solver_method": "christoffel",
  "local_output_dir": "outputs/fuchs_w1_quick_local"
}
```

Command-line flags override config values, so the same config can be used for a
small local smoke run or a larger Vertex run.

## Submit to Vertex AI

Create or choose an Artifact Registry Docker repository first, then run:

```powershell
$env:GOOGLE_CLOUD_PROJECT = "YOUR_PROJECT_ID"
$env:PYWARPFACTORY_OUTPUT_URI = "gs://YOUR_BUCKET/pywarpfactory/runs/fuchs-w1"

.\tools\submit_vertex_job.ps1 `
  -Config configs\fuchs_w1_original_vertex.json `
  -ProjectId $env:GOOGLE_CLOUD_PROJECT `
  -Region us-central1 `
  -ServiceAccount "astrumdrivetechnologies@astrumdrive.com" `
  -ArtifactImageUri "us-central1-docker.pkg.dev/YOUR_PROJECT_ID/pywarpfactory/vertex-job:latest" `
  -NumVecs 40 `
  -SaveArrays
```

To inspect the command without submitting anything:

```powershell
.\tools\submit_vertex_job.ps1 `
  -DryRun `
  -Config configs\fuchs_w1_original_vertex.json `
  -ProjectId "YOUR_PROJECT_ID" `
  -ArtifactImageUri "us-central1-docker.pkg.dev/YOUR_PROJECT_ID/pywarpfactory/vertex-job:latest"
```

Before the first real submission, run the preflight check:

```powershell
.\tools\check_cloud_prereqs.ps1 `
  -ProjectId "YOUR_PROJECT_ID" `
  -OutputUri "gs://YOUR_BUCKET/pywarpfactory/runs/fuchs-w1"
```

The script uses `gcloud ai custom-jobs create` with the same job runner, a
prebuilt GPU executor image, and Vertex autopackaging. The relevant gcloud fields are
`local-package-path`, `python-module`, `executor-image-uri`, and
`output-image-uri`.

## Submit Without Local Docker

If Docker Desktop is not installed, build a source package and submit it through
Cloud Storage:

```powershell
python setup.py sdist
gcloud storage cp dist\pywarpfactory-1.0.0.tar.gz `
  gs://warpopt-data/pywarpfactory/packages/pywarpfactory-1.0.0.tar.gz

.\tools\submit_vertex_package_job.ps1 `
  -ProjectId warpopt `
  -Region us-central1 `
  -PackageUri gs://warpopt-data/pywarpfactory/packages/pywarpfactory-1.0.0.tar.gz `
  -OutputUri gs://warpopt-data/pywarpfactory/runs/fuchs-w1-package-smoke `
  -Static `
  -NumVecs 8
```

This uses `--python-package-uris` and avoids the local `docker build` step.

## CPU-only run

For a cheaper CPU run, use a CPU executor image and omit the accelerator:

```powershell
.\tools\submit_vertex_job.ps1 `
  -ArtifactImageUri "us-central1-docker.pkg.dev/YOUR_PROJECT_ID/pywarpfactory/vertex-job:cpu" `
  -ExecutorImageUri "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-15.py310:latest" `
  -MachineType "n1-standard-16" `
  -AcceleratorType "" `
  -AcceleratorCount 0
```

If `AcceleratorType` is empty or `AcceleratorCount` is zero, the script omits
the accelerator fields.

## Current GPU status

This wiring gets pyWarpFactory onto Vertex AI GPU machines. The numerical core is
still mostly NumPy/SciPy, so it will benefit first from larger machines and
parallel job sweeps. True GPU speedups require a second stage: porting the heavy
finite-difference, inverse, Ricci, and observer-contraction kernels to a backend
such as CuPy or JAX.

Recommended next implementation stage:

- Add a backend module that selects NumPy or CuPy.
- Move finite differences and tensor contractions onto that backend.
- Keep analysis outputs as NumPy arrays for plotting and serialization.
- Use Vertex AI for large parameter sweeps while the GPU kernels mature.
