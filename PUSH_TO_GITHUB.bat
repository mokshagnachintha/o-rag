@echo off
title Push to GitHub — Trigger APK Build
color 0B

echo.
echo  ======================================================
echo   Push to GitHub → GitHub Actions builds the APK
echo  ======================================================
echo.

:: Check git is installed
where git >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Git is not installed.
    echo  Download from: https://git-scm.com/download/win
    pause
    exit /b 1
)

:: Get or prompt for GitHub repo URL
set /p REPO_URL="Enter your GitHub repo URL (e.g. https://github.com/yourname/offline-rag): "

if "%REPO_URL%"=="" (
    echo  No URL entered.  Exiting.
    pause
    exit /b 1
)

echo.
echo  Initialising git and pushing...
echo.

cd /d "%~dp0"

:: Init git if not already
if not exist ".git" (
    git init -b main
    echo  Git repo initialised.
)

:: Set remote
git remote remove origin 2>nul
git remote add origin %REPO_URL%

:: Stage everything (GGUF files excluded by .gitignore)
git add -A

:: Commit
git commit -m "Offline RAG app build %DATE% %TIME%" --allow-empty

:: Push
git push -u origin main --force

if errorlevel 1 (
    color 0C
    echo.
    echo  PUSH FAILED.
    echo.
    echo  If you see "Authentication failed":
    echo    1. Go to GitHub.com → Settings → Developer settings
    echo       → Personal access tokens → Tokens (classic)
    echo    2. Create a token with 'repo' scope
    echo    3. Use that token as your password when git prompts you
    echo.
) else (
    color 0A
    echo.
    echo  ====================================================
    echo   PUSHED SUCCESSFULLY!
    echo  ====================================================
    echo.
    echo   GitHub Actions is now building your APK.
    echo.
    echo   To download the APK:
    echo     1. Go to: %REPO_URL%/actions
    echo     2. Click the latest "Build Android APK" workflow
    echo     3. When it finishes, click "OfflineRAG-debug-apk"
    echo        under Artifacts to download the APK
    echo.
    echo   Typical build time: 30-60 minutes (first run)
    echo   After that (with cache): ~10 minutes
    echo.
    start "" "%REPO_URL%/actions"
)

pause
