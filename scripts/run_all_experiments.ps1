# Run All Experiments
# Execute centralized, federated, and cross-day robustness experiments

param(
    [int]$Seed = 42,
    [switch]$SkipCentralized,
    [switch]$SkipFederated,
    [switch]$SkipCrossDay
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "="*80 -ForegroundColor Magenta
Write-Host "mmWave MIMO Federated Learning - Full Experiment Suite" -ForegroundColor Magenta
Write-Host "="*80 -ForegroundColor Magenta
Write-Host ""
Write-Host "This will run all experiments from the proposal:" -ForegroundColor Yellow
Write-Host "  1. Centralized training (2 models × 3 days = 6 runs)" -ForegroundColor Yellow
Write-Host "  2. Federated learning (2 models × 3 days = 6 runs)" -ForegroundColor Yellow
Write-Host "  3. Cross-day robustness (2 models × 3 train days = 6 runs)" -ForegroundColor Yellow
Write-Host ""
Write-Host "Total: ~18 training runs" -ForegroundColor Yellow
Write-Host "Estimated time: 2-4 hours (depends on hardware)" -ForegroundColor Yellow
Write-Host ""

$Continue = Read-Host "Continue? (y/n)"
if ($Continue -ne "y") {
    Write-Host "Aborted." -ForegroundColor Red
    exit 0
}

Write-Host ""

# 1. Centralized experiments
if (-not $SkipCentralized) {
    Write-Host ""
    Write-Host "="*80 -ForegroundColor Cyan
    Write-Host "STEP 1/3: Centralized Training" -ForegroundColor Cyan
    Write-Host "="*80 -ForegroundColor Cyan
    Write-Host ""

    & "$ScriptDir\run_centralized_experiments.ps1" -Seed $Seed

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Centralized experiments failed"
        exit $LASTEXITCODE
    }
} else {
    Write-Host "Skipping centralized experiments (--SkipCentralized)" -ForegroundColor Yellow
}

# 2. Federated experiments
if (-not $SkipFederated) {
    Write-Host ""
    Write-Host "="*80 -ForegroundColor Cyan
    Write-Host "STEP 2/3: Federated Learning" -ForegroundColor Cyan
    Write-Host "="*80 -ForegroundColor Cyan
    Write-Host ""

    & "$ScriptDir\run_federated_experiments.ps1" -Seed $Seed

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Federated experiments failed"
        exit $LASTEXITCODE
    }
} else {
    Write-Host "Skipping federated experiments (--SkipFederated)" -ForegroundColor Yellow
}

# 3. Cross-day robustness
if (-not $SkipCrossDay) {
    Write-Host ""
    Write-Host "="*80 -ForegroundColor Cyan
    Write-Host "STEP 3/3: Cross-Day Robustness" -ForegroundColor Cyan
    Write-Host "="*80 -ForegroundColor Cyan
    Write-Host ""

    & "$ScriptDir\run_cross_day_experiments.ps1" -Seed $Seed

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Cross-day experiments failed"
        exit $LASTEXITCODE
    }
} else {
    Write-Host "Skipping cross-day experiments (--SkipCrossDay)" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "="*80 -ForegroundColor Magenta
Write-Host "ALL EXPERIMENTS COMPLETED SUCCESSFULLY!" -ForegroundColor Magenta
Write-Host "="*80 -ForegroundColor Magenta
Write-Host ""
Write-Host "Results locations:" -ForegroundColor Green
Write-Host "  - Centralized:  outputs/centralized/" -ForegroundColor Green
Write-Host "  - Federated:    outputs/federated/" -ForegroundColor Green
Write-Host "  - Cross-day:    outputs/cross_day/" -ForegroundColor Green
Write-Host "  - Cross-day summary: outputs/cross_day_summary/" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Review evaluation reports in each output directory" -ForegroundColor Cyan
Write-Host "  2. Check confusion matrices and per-class metrics" -ForegroundColor Cyan
Write-Host "  3. Compare centralized vs. federated performance" -ForegroundColor Cyan
Write-Host "  4. Analyze domain shift patterns in cross-day results" -ForegroundColor Cyan
Write-Host ""
