param(
    [string]$ProjectId = "warpopt",
    [string]$Zone = "us-central1-a",
    [string]$InstanceName = "pywarpfactory-smoke",
    [string]$MachineType = "e2-standard-2",
    [string]$ServiceAccount = "astra-vertex-sa@warpopt.iam.gserviceaccount.com",
    [string]$PackageUri = "gs://warpopt-data/pywarpfactory/packages/pywarpfactory-1.0.0.tar.gz",
    [string]$OutputUri = "gs://warpopt-data/pywarpfactory/runs/compute-smoke",
    [string]$Profile = "quick",
    [switch]$Static,
    [double]$VWarp = 0.02,
    [int]$NumVecs = 4,
    [switch]$SaveArrays,
    [string]$GcloudExe = $env:GCLOUD_EXE,
    [string]$PythonExe = $env:PYWARPFACTORY_PYTHON,
    [switch]$DryRun,
    [switch]$NoCreatePackage
)

$ErrorActionPreference = "Stop"

function Invoke-CheckedNative {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
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

if (-not $PythonExe) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $PythonExe = $pythonCommand.Source
    }
}
if (-not $PythonExe) {
    $pyCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCommand) {
        $PythonExe = $pyCommand.Source
    }
}
if (-not $PythonExe -and -not $NoCreatePackage) {
    throw "Python was not found. Pass -PythonExe, set PYWARPFACTORY_PYTHON, or use -NoCreatePackage."
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$startupTemplate = Join-Path $PSScriptRoot "compute_startup_pywarpfactory.sh"
$startupOutDir = Join-Path $repoRoot "outputs"
New-Item -ItemType Directory -Force -Path $startupOutDir | Out-Null
$startupFile = Join-Path $startupOutDir "$InstanceName-startup.sh"

$staticValue = if ($Static) { "true" } else { "false" }
$startup = Get-Content -Raw -Path $startupTemplate
$startup = $startup.Replace("__PACKAGE_URI__", $PackageUri)
$startup = $startup.Replace("__OUTPUT_URI__", $OutputUri)
$startup = $startup.Replace("__PROFILE__", $Profile)
$startup = $startup.Replace("__STATIC_FLAG__", $staticValue)
$startup = $startup.Replace("__V_WARP__", [string]$VWarp)
$startup = $startup.Replace("__NUM_VECS__", [string]$NumVecs)
$startup = $startup.Replace("__SAVE_ARRAYS__", $(if ($SaveArrays) { "true" } else { "false" }))
Set-Content -Path $startupFile -Value $startup -Encoding UTF8

if (-not $NoCreatePackage) {
    Push-Location $repoRoot
    try {
        & $PythonExe setup.py sdist
        if ($LASTEXITCODE -ne 0) {
            throw "Package build failed with exit code ${LASTEXITCODE}."
        }
        Invoke-CheckedNative $GcloudExe storage cp "dist\pywarpfactory-1.0.0.tar.gz" $PackageUri --project=$ProjectId
    }
    finally {
        Pop-Location
    }
}

$createArgs = @(
    "compute", "instances", "create", $InstanceName,
    "--project=$ProjectId",
    "--zone=$Zone",
    "--machine-type=$MachineType",
    "--image-family=debian-12",
    "--image-project=debian-cloud",
    "--boot-disk-size=50GB",
    "--boot-disk-type=pd-balanced",
    "--service-account=$ServiceAccount",
    "--scopes=cloud-platform",
    "--metadata-from-file=startup-script=$startupFile",
    "--labels=app=pywarpfactory,job=compute-smoke"
)

if ($DryRun) {
    $gcloudDisplay = if ($GcloudExe) { $GcloudExe } else { "gcloud" }
    Write-Host "Dry run: Compute VM would be created with:"
    Write-Host ($gcloudDisplay + " " + ($createArgs -join " "))
    Write-Host ""
    Write-Host "Startup script: $startupFile"
    exit 0
}

Invoke-CheckedNative $GcloudExe @createArgs

Write-Host ""
Write-Host "VM created. Follow logs with:"
Write-Host "$GcloudExe compute instances get-serial-port-output $InstanceName --project=$ProjectId --zone=$Zone --port=1"
Write-Host ""
Write-Host "The VM will shut itself down after the pyWarpFactory job completes."
