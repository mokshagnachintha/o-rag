@echo off
title Fix Docker — Install Ubuntu WSL2
color 0E

echo.
echo  ======================================================
echo   Docker Fix: Install Ubuntu WSL2
echo  ======================================================
echo.
echo  Docker Desktop needs WSL2 to work on your PC.
echo  This script installs Ubuntu on WSL2 (free, ~1 min).
echo.
echo  IMPORTANT: Run this as Administrator!
echo  Right-click the file and choose "Run as administrator"
echo.

:: Check if running as admin
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo  ERROR: Please right-click this file and choose
    echo         "Run as administrator"
    echo.
    pause
    exit /b 1
)

echo  Installing Ubuntu (WSL2)...
wsl --install -d Ubuntu

echo.
echo  ======================================================
echo  Done! Now:
echo    1. Restart your PC when prompted
echo    2. After restart, Ubuntu will finish setup
echo    3. Open Docker Desktop - it should now work
echo    4. Run BUILD_APK.bat to build the Android APK
echo  ======================================================
echo.
pause
