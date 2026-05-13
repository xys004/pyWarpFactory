<#
Sets the current PowerShell session PATH for this pyWarpFactory checkout.

This does not modify the global Windows environment. It only makes the known
local Anaconda env and Google Cloud SDK visible to commands launched from this
terminal/Codex session.
#>

$ErrorActionPreference = "Stop"

$AstraEnv = Join-Path $env:USERPROFILE "anaconda3\envs\astra"
$GcloudSdk = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk"

$pathsToPrepend = @(
    $AstraEnv,
    (Join-Path $AstraEnv "Scripts"),
    (Join-Path $AstraEnv "Library\bin"),
    (Join-Path $GcloudSdk "bin")
) | Where-Object { Test-Path -LiteralPath $_ }

$reversedPaths = $pathsToPrepend.Clone()
[array]::Reverse($reversedPaths)

foreach ($path in $reversedPaths) {
    $env:PATH = "$path;$env:PATH"
}

$env:PYWARPFACTORY_PYTHON = Join-Path $AstraEnv "python.exe"

Write-Host "Session PATH updated for pyWarpFactory."
Write-Host "Python:" (& python --version)
Write-Host "Python path:" (& python -c "import sys; print(sys.executable)")
Write-Host "gcloud:" (& gcloud --version | Select-Object -First 1)
