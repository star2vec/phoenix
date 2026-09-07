# Amendment 1: train the relabeled model, seed 0 then seed 1, sequentially.
# Run from the phoenix repo root in PowerShell:
#     .\scripts\train_relabel.ps1 -Python ..\RRR\.venv\Scripts\python.exe
# Checkpoints:  ckpts\seed0\relabel\best.pt   (+ latest_state.pt, epoch_NNN.pt)
#               ckpts\seed1\relabel\best.pt
# Logs/metrics: results\seed0\relabel\{train.out, metrics.jsonl, best.json}
#               results\seed1\relabel\{...}
# Resumable: rerun the same command after an interruption; train.py picks up
# from latest_state.pt.

param(
    [string]$Python = "..\RRR\.venv\Scripts\python.exe",
    [string]$Device = "cuda",
    [int[]]$Seeds = @(0, 1)
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

foreach ($seed in $Seeds) {
    $run = "seed$seed/relabel"
    $outDir = "results\seed$seed\relabel"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    New-Item -ItemType Directory -Force -Path "ckpts\seed$seed\relabel" | Out-Null
    Write-Host "=== training $run on $Device, log -> $outDir\train.out ==="
    & $Python -u src\phoenix\train.py --seed $seed --run-name $run --device $Device `
        --data-dir data\relabel --fixed-early-stages --full-task-patience 15 --save-every 10 `
        2>&1 | Tee-Object -FilePath "$outDir\train.out" -Append
    if ($LASTEXITCODE -ne 0) { throw "train.py exited with $LASTEXITCODE for $run" }
    if (-not (Test-Path "ckpts\seed$seed\relabel\best.pt")) { throw "no best.pt for $run" }
    Write-Host "=== $run done: ckpts\seed$seed\relabel\best.pt ==="
}
Write-Host "Both seeds trained. Copy ckpts\seed*\relabel\ and results\seed*\relabel\ back (see LAPTOP.md)."
