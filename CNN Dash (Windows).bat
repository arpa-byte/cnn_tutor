@echo off
setlocal
cd /d "%~dp0"

powershell -NoProfile -Command "$root=(Get-Location).Path; $shell=New-Object -ComObject WScript.Shell; $shortcut=$shell.CreateShortcut((Join-Path $root 'CNN Dash (Windows).lnk')); $shortcut.TargetPath=(Join-Path $root 'CNN Dash (Windows).bat'); $shortcut.WorkingDirectory=$root; $shortcut.IconLocation=(Join-Path $root 'assets\cnn-dash-windows.ico'); $shortcut.Save()" >nul 2>&1

py -3 --version >nul 2>&1
if not errorlevel 1 (
    py -3 start.py
) else (
    python --version >nul 2>&1
    if errorlevel 1 (
        echo Python 3.10 or newer is required. Install it and run this launcher again.
        pause
        exit /b 1
    )
    python start.py
)
