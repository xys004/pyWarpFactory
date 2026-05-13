param(
    [string]$ProjectId = $env:GOOGLE_CLOUD_PROJECT,
    [string]$OutputUri = $env:PYWARPFACTORY_OUTPUT_URI,
    [string]$GcloudExe = $env:GCLOUD_EXE,
    [string]$ExpectedAccount = "astrumdrivetechnologies@astrumdrive.com"
)

$ErrorActionPreference = "Stop"

function Pass($Message) {
    Write-Host "[OK] $Message"
}

function Warn($Message) {
    Write-Host "[WARN] $Message"
}

function Fail($Message) {
    Write-Host "[FAIL] $Message"
    $script:HasFailure = $true
}

$HasFailure = $false

$gcloud = $null
if ($GcloudExe) {
    if (Test-Path -LiteralPath $GcloudExe -ErrorAction SilentlyContinue) {
        $gcloud = $GcloudExe
    } else {
        Fail "GcloudExe does not exist: $GcloudExe"
    }
} else {
    $gcloudCommand = Get-Command gcloud -ErrorAction SilentlyContinue
    if ($gcloudCommand) {
        $gcloud = $gcloudCommand.Source
    } else {
        $candidate = "C:\Users\Nelson\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
        if (Test-Path -LiteralPath $candidate -ErrorAction SilentlyContinue) {
            $gcloud = $candidate
        }
    }
}

if (-not $gcloud) {
    Fail "gcloud is not available. Install Google Cloud SDK, open a configured shell, or pass -GcloudExe."
    exit 1
}
Pass "gcloud found at $gcloud"

$version = & $gcloud --version 2>$null | Select-Object -First 1
if ($version) {
    Pass $version
}

$activeAccount = & $gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>$null
if ($activeAccount) {
    Pass "Active gcloud account: $activeAccount"
    if ($ExpectedAccount -and $activeAccount -ne $ExpectedAccount) {
        Warn "Expected $ExpectedAccount, but active account is $activeAccount."
    }
} else {
    Fail "No active gcloud account. Run: gcloud auth login"
}

if (-not $ProjectId) {
    $ProjectId = & $gcloud config get-value project 2>$null
}
if ($ProjectId) {
    Pass "Project: $ProjectId"
} else {
    Fail "No project configured. Set GOOGLE_CLOUD_PROJECT or run: gcloud config set project PROJECT_ID"
}

if ($OutputUri) {
    if ($OutputUri -notlike "gs://*") {
        Fail "OutputUri must start with gs://"
    } else {
        Pass "Output URI: $OutputUri"
        $bucket = $OutputUri.Substring(5).Split("/")[0]
        $bucketCheck = & $gcloud storage buckets describe "gs://$bucket" --format="value(name)" 2>$null
        if ($bucketCheck) {
            Pass "Output bucket exists: gs://$bucket"
        } else {
            Warn "Could not verify bucket gs://$bucket. Create it or check permissions."
        }
    }
} else {
    Warn "PYWARPFACTORY_OUTPUT_URI is not set. Vertex job will only write local container artifacts unless --output-uri is passed."
}

if ($ProjectId) {
    $services = @(
        "aiplatform.googleapis.com",
        "artifactregistry.googleapis.com",
        "storage.googleapis.com"
    )
    foreach ($service in $services) {
        $enabled = & $gcloud services list --enabled --project=$ProjectId --filter="config.name=$service" --format="value(config.name)" 2>$null
        if ($enabled) {
            Pass "API enabled: $service"
        } else {
            Warn "API may not be enabled: $service"
        }
    }
}

if ($HasFailure) {
    exit 1
}

Pass "Cloud preflight completed."
