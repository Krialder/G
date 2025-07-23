# PowerShell script to create desktop shortcut for Gießplan
# This script creates a desktop shortcut using PowerShell COM objects

param(
    [switch]$Force  # Force creation even if shortcut exists
)

Write-Host "Gießplan Desktop Shortcut Creator (PowerShell)" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""

# Get script directory and target files
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BatchFile = Join-Path $ScriptDir "start_giessplan.bat"
$ShortcutName = "Giessplan.lnk"

Write-Host "Script directory: $ScriptDir" -ForegroundColor Cyan
Write-Host "Target batch file: $BatchFile" -ForegroundColor Cyan

# Check if batch file exists
if (-not (Test-Path $BatchFile)) {
    Write-Host "❌ Error: start_giessplan.bat not found!" -ForegroundColor Red
    Write-Host "Please make sure all files are in the same folder." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Get desktop path
try {
    # Try multiple methods to get desktop path
    $DesktopPath = $null
    
    # Method 1: Environment variable
    $DesktopPath = [Environment]::GetFolderPath("Desktop")
    
    # Method 2: Registry (if method 1 fails)
    if (-not $DesktopPath -or -not (Test-Path $DesktopPath)) {
        $RegKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
        $DesktopPath = (Get-ItemProperty -Path $RegKey -Name "Desktop" -ErrorAction SilentlyContinue).Desktop
    }
    
    # Method 3: Common paths (if methods 1&2 fail)
    if (-not $DesktopPath -or -not (Test-Path $DesktopPath)) {
        $CommonPaths = @(
            "$env:USERPROFILE\Desktop",
            "$env:USERPROFILE\OneDrive\Desktop"
        )
        
        foreach ($Path in $CommonPaths) {
            if (Test-Path $Path) {
                $DesktopPath = $Path
                break
            }
        }
    }
    
    if (-not $DesktopPath -or -not (Test-Path $DesktopPath)) {
        throw "Could not find accessible desktop folder"
    }
    
    Write-Host "✅ Desktop found: $DesktopPath" -ForegroundColor Green
    
} catch {
    Write-Host "❌ Error finding desktop folder: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Attempted paths:" -ForegroundColor Yellow
    Write-Host "  - [Environment]::GetFolderPath('Desktop')" -ForegroundColor Gray
    Write-Host "  - Registry: HKCU:\...\Shell Folders\Desktop" -ForegroundColor Gray
    Write-Host "  - $env:USERPROFILE\Desktop" -ForegroundColor Gray
    Write-Host "  - $env:USERPROFILE\OneDrive\Desktop" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Please create the shortcut manually:" -ForegroundColor Cyan
    Write-Host "1. Right-click on start_giessplan.bat" -ForegroundColor Cyan
    Write-Host "2. Select 'Send to > Desktop (create shortcut)'" -ForegroundColor Cyan
    Read-Host "Press Enter to exit"
    exit 1
}

# Define shortcut path
$ShortcutPath = Join-Path $DesktopPath $ShortcutName

# Check if shortcut already exists
if ((Test-Path $ShortcutPath) -and -not $Force) {
    $Response = Read-Host "Shortcut already exists. Overwrite? (y/n)"
    if ($Response -notmatch "^[Yy]") {
        Write-Host "Operation cancelled." -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 0
    }
}

# Create the shortcut
try {
    Write-Host "Creating desktop shortcut..." -ForegroundColor Yellow
    
    # Create WScript.Shell COM object
    $WshShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WshShell.CreateShortcut($ShortcutPath)
    
    # Configure shortcut properties
    $Shortcut.TargetPath = $BatchFile
    $Shortcut.WorkingDirectory = $ScriptDir
    $Shortcut.Description = "Giessplan - Rotkreuz-Institut BBW"
    $Shortcut.IconLocation = "shell32.dll,21"  # Folder icon
    
    # Save the shortcut
    $Shortcut.Save()
    
    # Verify creation
    if (Test-Path $ShortcutPath) {
        Write-Host ""
        Write-Host "✅ Desktop shortcut created successfully!" -ForegroundColor Green
        Write-Host "📍 Location: $ShortcutPath" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "You can now start Giessplan by double-clicking the desktop icon." -ForegroundColor White
    } else {
        throw "Shortcut file was not created"
    }
    
} catch {
    Write-Host ""
    Write-Host "❌ Failed to create desktop shortcut: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Alternative methods:" -ForegroundColor Cyan
    Write-Host "1. Right-click on start_giessplan.bat and select 'Send to > Desktop'" -ForegroundColor Cyan
    Write-Host "2. Try running this script as Administrator" -ForegroundColor Cyan
    Write-Host "3. Create shortcut manually to: $BatchFile" -ForegroundColor Cyan
    
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Read-Host "Press Enter to exit"
