# Centralized Training Experiments
# Run baseline and improved CNN models on all days

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
Write-Host "Centralized Training Experiments" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

$Models = @("baseline", "improved")
$Days = @(0, 1, 2)

foreach ($Model in $Models) {
    foreach ($Day in $Days) {
        Write-Host ""
        Write-Host ("-"*80) -ForegroundColor Yellow
        Write-Host "Training $Model model on day$Day" -ForegroundColor Yellow
        Write-Host ("-"*80) -ForegroundColor Yellow
        Write-Host ""

        $OutputDir = "outputs/centralized/day${Day}_${Model}"

        & $VenvPython src/train_centralized.py `
            --day $Day `
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
            Write-Error "Training failed for $Model on day$Day"
            exit $LASTEXITCODE
        }

        Write-Host ""
        Write-Host "Completed: $Model on day$Day" -ForegroundColor Green
        Write-Host "Output: $OutputDir" -ForegroundColor Green
        Write-Host ""
    }
}

Write-Host ""
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "All centralized experiments completed!" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved in: outputs/centralized/" -ForegroundColor Green
