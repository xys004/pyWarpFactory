param(
    [string]$Config,
    [string]$ProjectId = $env:GOOGLE_CLOUD_PROJECT,
    [string]$Region = "us-central1",
    [string]$DisplayName = "pywarpfactory-fuchs-w1",
    [string]$ServiceAccount = "astrumdrivetechnologies@astrumdrive.com",
    [string]$ArtifactImageUri,
    [string]$OutputUri = $env:PYWARPFACTORY_OUTPUT_URI,
    [string]$GcloudExe = $env:GCLOUD_EXE,
    [string]$MachineType = "n1-standard-8",
    [string]$AcceleratorType = "NVIDIA_TESLA_T4",
    [int]$AcceleratorCount = 1,
    [string]$ExecutorImageUri = "us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-15.py310:latest",
    [string]$Profile = "quick",
    [switch]$Static,
    [double]$VWarp = 0.02,
    [int]$NumVecs = 40,
    [switch]$DryRun,
    [switch]$SaveArrays
)

$ErrorActionPreference = "Stop"

if (-not $ProjectId) {
    throw "ProjectId is required. Pass -ProjectId or set GOOGLE_CLOUD_PROJECT."
}

if (-not $ArtifactImageUri) {
    throw "ArtifactImageUri is required, e.g. ${Region}-docker.pkg.dev/${ProjectId}/pywarpfactory/vertex-job:latest"
}

if (-not $GcloudExe) {
    $gcloudCommand = Get-Command gcloud -ErrorAction SilentlyContinue
    if ($gcloudCommand) {
        $GcloudExe = $gcloudCommand.Source
    }
}
if (-not $GcloudExe) {
    $candidate = "C:\Users\Nelson\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
    if (Test-Path -LiteralPath $candidate -ErrorAction SilentlyContinue) {
        $GcloudExe = $candidate
    }
}
if (-not $GcloudExe -and -not $DryRun) {
    throw "gcloud was not found. Pass -GcloudExe or set GCLOUD_EXE."
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$workerPoolFields = @(
    "machine-type=$MachineType",
    "replica-count=1",
    "executor-image-uri=$ExecutorImageUri",
    "output-image-uri=$ArtifactImageUri",
    "python-module=warpfactory.cloud.run_job",
    "local-package-path=$repoRoot",
    "requirements=numpy>=1.20.0;scipy>=1.7.0;matplotlib>=3.4.0;google-cloud-storage>=2.16.0",
    "extra-dirs=warpfactory;configs"
)

if ($AcceleratorType -and $AcceleratorCount -gt 0) {
    $workerPoolFields += "accelerator-type=$AcceleratorType"
    $workerPoolFields += "accelerator-count=$AcceleratorCount"
}

$workerPoolSpec = $workerPoolFields -join ","

$jobArgs = @(
    "--execution-target=vertex"
)

if ($Config) {
    $vertexConfigPath = $Config -replace "\\", "/"
    $jobArgs += "--config=$vertexConfigPath"
}

if (-not $Config -or $PSBoundParameters.ContainsKey("Profile")) {
    $jobArgs += "--profile=$Profile"
}
if (-not $Config -or $PSBoundParameters.ContainsKey("VWarp")) {
    $jobArgs += "--v-warp=$VWarp"
}
if (-not $Config -or $PSBoundParameters.ContainsKey("NumVecs")) {
    $jobArgs += "--num-vecs=$NumVecs"
}
if (-not $Config) {
    $jobArgs += "--energy-condition-method=warpfactory"
    $jobArgs += "--solver-method=christoffel"
}

if ($Static -or (-not $Config -and $PSBoundParameters.ContainsKey("Static"))) {
    $jobArgs += "--static"
}
if ($SaveArrays) {
    $jobArgs += "--save-arrays"
}
if ($OutputUri) {
    $jobArgs += "--output-uri=$OutputUri"
}

$gcloudArgs = @(
    "ai", "custom-jobs", "create",
    "--region=$Region",
    "--display-name=$DisplayName",
    "--service-account=$ServiceAccount",
    "--worker-pool-spec=$workerPoolSpec",
    "--args=$($jobArgs -join ',')"
)

if ($DryRun) {
    Write-Host "Dry run: Vertex AI job would be submitted with:"
    Write-Host ""
    $gcloudDisplay = if ($GcloudExe) { $GcloudExe } else { "gcloud" }
    Write-Host "$gcloudDisplay config set project $ProjectId"
    Write-Host ($gcloudDisplay + " " + ($gcloudArgs -join " "))
    Write-Host ""
    Write-Host "Worker pool spec:"
    Write-Host $workerPoolSpec
    Write-Host ""
    Write-Host "Job args:"
    $jobArgs | ForEach-Object { Write-Host "  $_" }
    exit 0
}

& $GcloudExe config set project $ProjectId
& $GcloudExe @gcloudArgs
