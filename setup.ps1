[CmdletBinding()]
param(
    [ValidateSet("core", "vision", "workbench", "data", "training", "full")]
    [string]$Profile = "workbench",
    [switch]$WithData,
    [switch]$BuildZarr,
    [switch]$AcceptDatasetLicense,
    [string]$DataDir = "data\ParaCAD_download",
    [string]$ZarrOutput = "ParaCAD_full_v3.zarr"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "[BLOCKED] uv is not installed. Install it from https://docs.astral.sh/uv/getting-started/installation/ then rerun .\setup.ps1."
    exit 2
}

$extrasByProfile = @{
    "core" = @()
    "vision" = @("vision")
    "workbench" = @("vision", "workbench")
    "data" = @("data")
    "training" = @("vision", "copilot-training")
    "full" = @("vision", "workbench", "data", "copilot-training")
}

$extras = [System.Collections.Generic.List[string]]::new()
foreach ($extra in $extrasByProfile[$Profile]) { $extras.Add($extra) }
if ($WithData -and -not $extras.Contains("data")) { $extras.Add("data") }

$sync = [System.Collections.Generic.List[string]]::new()
$sync.Add("sync")
foreach ($extra in $extras) {
    $sync.Add("--extra")
    $sync.Add($extra)
}
if ($Profile -eq "full") { $sync.Add("--group"); $sync.Add("dev") }

Write-Host "[INFO] Installing uv-managed dependencies for profile '$Profile'..."
& uv @sync
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& uv run python .\project_setup.py doctor --profile $Profile --data-dir $DataDir --zarr-output $ZarrOutput
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($WithData) {
    if (-not $AcceptDatasetLicense) {
        Write-Host "[BLOCKED] Raw ParaCAD download was not started. Re-run with -AcceptDatasetLicense after reviewing the upstream ModelScope dataset terms."
        exit 2
    }
    $fetch = @("run", "python", ".\project_setup.py", "data", "fetch", "--data-dir", $DataDir, "--zarr-output", $ZarrOutput, "--accept-dataset-license")
    if ($BuildZarr) { $fetch += "--build-zarr" }
    & uv @fetch
    exit $LASTEXITCODE
}

Write-Host "[NEXT] Start the workbench with: uv run streamlit run .\streamlit_app.py"
Write-Host "[NOTICE] Native optional components (Ollama, Tesseract, NVIDIA driver/CUDA) are reported above and are never installed by this script."

