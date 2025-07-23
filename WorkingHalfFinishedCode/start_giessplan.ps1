# Gießplan Launcher PowerShell Script
# This script provides an alternative way to start the Gießplan application

param(
    [switch]$Force,  # Force start without checks
    [switch]$Verbose # Enable verbose output
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Gießplan Launcher (PowerShell)" -ForegroundColor Green
Write-Host "==============================" -ForegroundColor Green
Write-Host ""

function Write-VerboseOutput {
    param([string]$Message)
    if ($Verbose) {
        Write-Host "[VERBOSE] $Message" -ForegroundColor Gray
    }
}

function Test-PythonInstallation {
    Write-VerboseOutput "Testing Python installation..."
    
    $pythonCommands = @("python", "py", "python3")
    
    foreach ($cmd in $pythonCommands) {
        try {
            $version = & $cmd --version 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ Python found: $version" -ForegroundColor Green
                Write-VerboseOutput "Python command: $cmd"
                return $cmd
            }
        }
        catch {
            Write-VerboseOutput "Command '$cmd' not found"
        }
    }
    
    Write-Host "❌ Python not found in PATH" -ForegroundColor Red
    return $null
}

function Test-RequiredFiles {
    Write-VerboseOutput "Checking required files..."
    
    $requiredFiles = @("launcher.py", "main.py", "gui.py", "data.py")
    $missingFiles = @()
    
    foreach ($file in $requiredFiles) {
        $filePath = Join-Path $ScriptDir $file
        if (Test-Path $filePath) {
            Write-VerboseOutput "✅ Found: $file"
        } else {
            Write-VerboseOutput "❌ Missing: $file"
            $missingFiles += $file
        }
    }
    
    if ($missingFiles.Count -gt 0) {
        Write-Host "❌ Missing required files:" -ForegroundColor Red
        foreach ($file in $missingFiles) {
            Write-Host "   - $file" -ForegroundColor Red
        }
        return $false
    }
    
    Write-Host "✅ All required files found" -ForegroundColor Green
    return $true
}

function Start-Launcher {
    param([string]$PythonCommand)
    
    Write-Host "Starting Gießplan Launcher..." -ForegroundColor Yellow
    
    try {
        $launcherPath = Join-Path $ScriptDir "launcher.py"
        Set-Location $ScriptDir
        
        if ($Verbose) {
            & $PythonCommand $launcherPath
        } else {
            & $PythonCommand $launcherPath 2>$null
        }
        
        Write-Host "Launcher finished." -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Error starting launcher: $_" -ForegroundColor Red
        throw
    }
}

function Install-Python {
    Write-Host "Python is not installed or not found in PATH." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To install Python:" -ForegroundColor Cyan
    Write-Host "1. Go to: https://www.python.org/downloads/" -ForegroundColor Cyan
    Write-Host "2. Download the latest Python version" -ForegroundColor Cyan
    Write-Host "3. Run the installer as Administrator" -ForegroundColor Cyan
    Write-Host "4. ✅ IMPORTANT: Check 'Add Python to PATH'" -ForegroundColor Yellow
    Write-Host "5. Choose 'Install Now'" -ForegroundColor Cyan
    Write-Host "6. Restart this script after installation" -ForegroundColor Cyan
    Write-Host ""
    
    $response = Read-Host "Open Python download page? (y/n)"
    if ($response -match "^[Yy]") {
        Start-Process "https://www.python.org/downloads/"
    }
}

# Main execution
try {
    Set-Location $ScriptDir
    
    if (!$Force) {
        # Check if Python is installed
        $pythonCmd = Test-PythonInstallation
        if (-not $pythonCmd) {
            Install-Python
            exit 1
        }
        
        # Check if required files exist
        if (-not (Test-RequiredFiles)) {
            Write-Host ""
            Write-Host "Cannot start application due to missing files." -ForegroundColor Red
            Read-Host "Press Enter to exit"
            exit 1
        }
        
        Write-Host ""
    } else {
        Write-Host "Force mode enabled - skipping checks" -ForegroundColor Yellow
        $pythonCmd = "python"
    }
    
    # Start the launcher
    Start-Launcher $pythonCmd
    
} catch {
    Write-Host ""
    Write-Host "❌ Critical error: $_" -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Read-Host "Press Enter to exit"
