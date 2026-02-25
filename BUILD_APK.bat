@echo off
title Build Offline RAG — Android APK
color 0B

echo.
echo  ======================================================
echo   Offline RAG — Android APK Builder
echo  ======================================================
echo.
echo  This will build an Android APK using Docker + Buildozer.
echo  FIRST RUN: 30-90 minutes.  Subsequent runs: ~5 minutes.
echo.
echo  Requires: Docker Desktop (already installed on this PC)
echo.
pause

:: Run the PowerShell build script
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_android.ps1"

echo.
if %ERRORLEVEL% EQU 0 (
    color 0A
    echo  SUCCESS!
    echo  Look for OfflineRAG-debug.apk in this folder.
) else (
    color 0C
    echo  BUILD FAILED — scroll up to see the error.
)
echo.
pause
