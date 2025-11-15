# Cross-Day Robustness Experiments
# Train on each day, test on all days to measure domain shift

param(
    [int]$Seed = 42,
    [int]$Epochs = 50,
    [int]$BatchSize = 32,
    [double]$LearningRate = 0.001,
    [double]$Dropout = 0.3,
    [string]$Normalize = "zscore",
    [double]$ValRatio = 0.2,
    [int]$Patience = 10
)

# Ensure virtual environment is activated
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
$VenvPython = Join-Path $RootDir ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment not found at: $VenvPython"
    Write-Error "Please create the virtual environment first."
    exit 1
}

Write-Host "="*80 -ForegroundColor Cyan
Write-Host "Cross-Day Robustness Experiments" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

$Models = @("baseline", "improved")
$TrainDays = @(0, 1, 2)

foreach ($Model in $Models) {
    foreach ($TrainDay in $TrainDays) {
        Write-Host ""
        Write-Host ("-"*80) -ForegroundColor Yellow
        Write-Host "Cross-day: Train $Model on day$TrainDay, test on all days" -ForegroundColor Yellow
        Write-Host ("-"*80) -ForegroundColor Yellow
        Write-Host ""

        $OutputDir = "outputs/cross_day"

        & $VenvPython src/cross_day_robustness.py `
            --train_day $TrainDay `
            --model $Model `
            --epochs $Epochs `
            --batch_size $BatchSize `
            --lr $LearningRate `
            --dropout $Dropout `
            --normalize $Normalize `
            --val_ratio $ValRatio `
            --patience $Patience `
            --seed $Seed `
            --output_dir $OutputDir

        if ($LASTEXITCODE -ne 0) {
            Write-Error "Cross-day experiment failed for $Model trained on day$TrainDay"
            exit $LASTEXITCODE
        }

        Write-Host ""
        Write-Host "Completed: $Model trained on day$TrainDay" -ForegroundColor Green
        Write-Host ""
    }
}

Write-Host ""
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "Aggregating cross-day results..." -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

& $VenvPython src/aggregate_cross_day.py `
    --input_dir "outputs/cross_day" `
    --output_dir "outputs/cross_day_summary"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Cross-day aggregation failed"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "All cross-day experiments completed!" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved in: outputs/cross_day/" -ForegroundColor Green
Write-Host "Summary saved in: outputs/cross_day_summary/" -ForegroundColor Green
