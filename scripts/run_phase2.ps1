# Windows PowerShell version of scripts/run_phase2.sh: same four steps, same logs, resumable.
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File scripts\run_phase2.ps1
# Override the interpreter with:  $env:PY = "C:\path\to\python.exe"

$ErrorActionPreference = "Continue"
$py = if ($env:PY) { $env:PY } else { ".venv\Scripts\python.exe" }

$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

New-Item -ItemType Directory -Force -Path "output\logs" | Out-Null

$steps = @(
    @{ log = "phase2_main.log";     args = @("-m", "exploratory.phase2_grid", "--models", "ols,enet,rf") },
    @{ log = "phase2_within.log";   args = @("-m", "exploratory.phase2_grid", "--models", "ols,enet,rf", "--levels", "L2", "--predictor-set", "within") },
    @{ log = "phase2_constant.log"; args = @("-m", "exploratory.phase2_grid", "--models", "ols,rf", "--levels", "L2", "--predictor-set", "constant") },
    @{ log = "phase2_results.log";  args = @("-m", "exploratory.phase2_results") }
)

foreach ($step in $steps) {
    & $py @($step.args) 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath "output\logs\$($step.log)" -Append
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Step failed (exit $LASTEXITCODE): $($step.args -join ' ')"
        exit $LASTEXITCODE
    }
}
