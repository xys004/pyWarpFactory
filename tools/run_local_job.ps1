param(
    [string]$Config,
    [string]$Profile = "quick",
    [switch]$Static,
    [double]$VWarp = 0.02,
    [int]$NumVecs = 40,
    [string]$OutputDir = "outputs\local_job",
    [string]$PythonExe = $env:PYWARPFACTORY_PYTHON,
    [switch]$SaveArrays
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

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
if (-not $PythonExe) {
    throw "Python was not found. Pass -PythonExe or set PYWARPFACTORY_PYTHON."
}

$argsList = @(
    "-m", "warpfactory.cloud.run_job",
    "--execution-target=local"
)

if ($Config) {
    $argsList += "--config=$Config"
}

if (-not $Config -or $PSBoundParameters.ContainsKey("Profile")) {
    $argsList += "--profile=$Profile"
}
if (-not $Config -or $PSBoundParameters.ContainsKey("VWarp")) {
    $argsList += "--v-warp=$VWarp"
}
if (-not $Config -or $PSBoundParameters.ContainsKey("NumVecs")) {
    $argsList += "--num-vecs=$NumVecs"
}
if (-not $Config -or $PSBoundParameters.ContainsKey("OutputDir")) {
    $argsList += "--local-output-dir=$OutputDir"
}
if (-not $Config) {
    $argsList += "--energy-condition-method=warpfactory"
    $argsList += "--solver-method=christoffel"
}

if ($Static -or (-not $Config -and $PSBoundParameters.ContainsKey("Static"))) {
    $argsList += "--static"
}
if ($SaveArrays) {
    $argsList += "--save-arrays"
}

Push-Location $repoRoot
try {
    & $PythonExe @argsList
}
finally {
    Pop-Location
}
