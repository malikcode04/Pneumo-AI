# Pneumo AI - Professional Startup Script
$ErrorActionPreference = "Stop"

try {
    Write-Host "Checking system..." -ForegroundColor Cyan
    
    # Check if we should skip install (optimization)
    if (Test-Path "venv") {
        Write-Host "Virtual environment detected." -ForegroundColor Gray
    }

    # Install dependencies only if flag provided or critical libs missing
    # For now, we'll use a simple check to see if streamlit is installed
    try {
        python -c "import streamlit" 2>$null
        $dependencies_installed = $true
    } catch {
        $dependencies_installed = $false
    }

    if (-not $dependencies_installed) {
        Write-Host "Installing dependencies... (this may take a few minutes)" -ForegroundColor Yellow
        pip install -r requirements.txt
    } else {
        Write-Host "Dependencies appear to be installed. Skipping heavy install." -ForegroundColor Green
        Write-Host "To force reinstall, delete 'venv' or run 'pip install -r requirements.txt' manually." -ForegroundColor Gray
    }

    Write-Host "Starting Pneumo AI App..." -ForegroundColor Cyan
    python -m streamlit run app/streamlit_app.py
}
catch {
    Write-Host "An error occurred during startup:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
