# Federated Learning Experiments
# Run FedAvg with baseline and improved models on all days

param(
    [int]$Seed = 42,
    [int]$Rounds = 20,
    [int]$LocalEpochs = 5,
    [int]$BatchSize = 32,
    [double]$LearningRate = 0.001,
    [double]$Dropout = 0.3,
    [string]$Normalize = "zscore",
    [int]$Patience = 5
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
Write-Host "Federated Learning Experiments" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""

$Models = @("baseline", "improved")
$Days = @(0, 1, 2)

foreach ($Model in $Models) {
    foreach ($Day in $Days) {
        Write-Host ""
        Write-Host ("-"*80) -ForegroundColor Yellow
        Write-Host "Federated training: $Model model on day$Day (9 clients)" -ForegroundColor Yellow
        Write-Host ("-"*80) -ForegroundColor Yellow
        Write-Host ""

        $OutputDir = "outputs/federated/day${Day}_${Model}"

        & $VenvPython src/train_federated.py `
            --day $Day `
            --model $Model `
            --rounds $Rounds `
            --local_epochs $LocalEpochs `
            --batch_size $BatchSize `
            --lr $LearningRate `
            --dropout $Dropout `
            --normalize $Normalize `
            --patience $Patience `
            --seed $Seed `
            --output_dir $OutputDir

        if ($LASTEXITCODE -ne 0) {
            Write-Error "Federated training failed for $Model on day$Day"
            exit $LASTEXITCODE
        }

        Write-Host ""
        Write-Host "Completed: Federated $Model on day$Day" -ForegroundColor Green
        Write-Host "Output: $OutputDir" -ForegroundColor Green
        Write-Host ""
    }
}

Write-Host ""
Write-Host "="*80 -ForegroundColor Cyan
Write-Host "All federated experiments completed!" -ForegroundColor Cyan
Write-Host "="*80 -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved in: outputs/federated/" -ForegroundColor Green
