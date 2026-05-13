param(
    [string]$ProjectId = $env:GOOGLE_CLOUD_PROJECT,
    [string]$Region = "us-central1",
    [string]$DisplayName = "pywarpfactory-package-smoke",
    [string]$ServiceAccount = "astra-vertex-sa@warpopt.iam.gserviceaccount.com",
    [string]$PackageUri = "gs://warpopt-data/pywarpfactory/packages/pywarpfactory-1.0.0.tar.gz",
    [string]$OutputUri = $env:PYWARPFACTORY_OUTPUT_URI,
    [string]$GcloudExe = $env:GCLOUD_EXE,
    [string]$MachineType = "n1-standard-4",
    [string]$ExecutorImageUri = "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-15.py310:latest",
    [string]$Profile = "quick",
    [switch]$Static,
    [double]$VWarp = 0.02,
    [int]$NumVecs = 8,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if (-not $ProjectId) {
    throw "ProjectId is required. Pass -ProjectId or set GOOGLE_CLOUD_PROJECT."
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

$workerPoolSpec = "machine-type=$MachineType,replica-count=1,executor-image-uri=$ExecutorImageUri,python-module=warpfactory.cloud.run_job"
$jobArgs = @(
    "--execution-target=vertex",
    "--profile=$Profile",
    "--v-warp=$VWarp",
    "--num-vecs=$NumVecs",
    "--energy-condition-method=warpfactory",
    "--solver-method=christoffel"
)
if ($Static) {
    $jobArgs += "--static"
}
if ($OutputUri) {
    $jobArgs += "--output-uri=$OutputUri"
}

$gcloudArgs = @(
    "ai", "custom-jobs", "create",
    "--project=$ProjectId",
    "--region=$Region",
    "--display-name=$DisplayName",
    "--service-account=$ServiceAccount",
    "--python-package-uris=$PackageUri",
    "--worker-pool-spec=$workerPoolSpec",
    "--args=$($jobArgs -join ',')"
)

if ($DryRun) {
    $gcloudDisplay = if ($GcloudExe) { $GcloudExe } else { "gcloud" }
    Write-Host "Dry run: Vertex AI package job would be submitted with:"
    Write-Host ($gcloudDisplay + " " + ($gcloudArgs -join " "))
    Write-Host ""
    Write-Host "Job args:"
    $jobArgs | ForEach-Object { Write-Host "  $_" }
    exit 0
}

& $GcloudExe @gcloudArgs
