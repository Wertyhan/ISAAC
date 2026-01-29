<#
.SYNOPSIS
    Start ISAAC - Backend API + Chainlit Frontend

.DESCRIPTION
    Launches both the FastAPI backend and Chainlit frontend in parallel.
    Press Ctrl+C to stop all services.

.EXAMPLE
    .\start.ps1
#>

$ErrorActionPreference = "Stop"

# Colors
$Host.UI.RawUI.WindowTitle = "ISAAC - Intelligent System Architecture Advisor"

Write-Host ""
Write-Host "  ====================================" -ForegroundColor Yellow
Write-Host "           I S A A C" -ForegroundColor Yellow
Write-Host "  ====================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Intelligent System Architecture" -ForegroundColor Cyan
Write-Host "  Advisor and Consultant" -ForegroundColor Cyan
Write-Host ""

# Configuration
$API_HOST = "127.0.0.1"
$API_PORT = 8000
$CHAINLIT_PORT = 8501

# Activate virtual environment if exists
$venvPath = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "[*] Activating virtual environment..." -ForegroundColor Gray
    & $venvPath
}

# Check if ports are already in use
function Test-PortInUse {
    param([int]$Port)
    $connection = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    return $null -ne $connection
}

if (Test-PortInUse $API_PORT) {
    Write-Host "[!] Warning: Port $API_PORT is already in use" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[1/2] Starting Backend API on http://${API_HOST}:${API_PORT}" -ForegroundColor Green
Write-Host "[2/2] Starting Chainlit UI on http://${API_HOST}:${CHAINLIT_PORT}" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop all services" -ForegroundColor DarkGray
Write-Host ""

# Start both processes
$backendJob = Start-Job -ScriptBlock {
    Set-Location $using:PSScriptRoot
    if (Test-Path ".venv\Scripts\python.exe") {
        & ".venv\Scripts\python.exe" -m uvicorn isaac_api.main:app --host $using:API_HOST --port $using:API_PORT
    } else {
        python -m uvicorn isaac_api.main:app --host $using:API_HOST --port $using:API_PORT
    }
}

$frontendJob = Start-Job -ScriptBlock {
    Set-Location $using:PSScriptRoot
    # Wait for backend to start
    Start-Sleep -Seconds 2
    if (Test-Path ".venv\Scripts\chainlit.exe") {
        & ".venv\Scripts\chainlit.exe" run ui/app.py --host $using:API_HOST --port $using:CHAINLIT_PORT
    } else {
        chainlit run ui/app.py --host $using:API_HOST --port $using:CHAINLIT_PORT
    }
}

# Wait a moment then open browser
Start-Sleep -Seconds 3
Write-Host "[*] Opening browser..." -ForegroundColor Cyan
Start-Process "http://${API_HOST}:${CHAINLIT_PORT}"

# Stream output from both jobs
try {
    while ($true) {
        # Get and display backend output
        $backendOutput = Receive-Job -Job $backendJob -ErrorAction SilentlyContinue
        if ($backendOutput) {
            $backendOutput | ForEach-Object { Write-Host "[API] $_" -ForegroundColor Blue }
        }

        # Get and display frontend output
        $frontendOutput = Receive-Job -Job $frontendJob -ErrorAction SilentlyContinue
        if ($frontendOutput) {
            $frontendOutput | ForEach-Object { Write-Host "[UI]  $_" -ForegroundColor Magenta }
        }

        # Check if jobs are still running
        if ($backendJob.State -eq "Failed") {
            Write-Host "[!] Backend crashed!" -ForegroundColor Red
            Receive-Job -Job $backendJob
        }
        if ($frontendJob.State -eq "Failed") {
            Write-Host "[!] Frontend crashed!" -ForegroundColor Red
            Receive-Job -Job $frontendJob
        }

        Start-Sleep -Milliseconds 500
    }
}
finally {
    Write-Host ""
    Write-Host "[*] Stopping services..." -ForegroundColor Yellow
    Stop-Job -Job $backendJob -ErrorAction SilentlyContinue
    Stop-Job -Job $frontendJob -ErrorAction SilentlyContinue
    Remove-Job -Job $backendJob -Force -ErrorAction SilentlyContinue
    Remove-Job -Job $frontendJob -Force -ErrorAction SilentlyContinue
    Write-Host "[*] ISAAC stopped." -ForegroundColor Green
}
