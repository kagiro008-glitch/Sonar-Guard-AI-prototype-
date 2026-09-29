@echo off
REM  CLI shortcut: analyze a single image/XTF, run benchmark, or export models
chcp 65001 >nul 2>&1
title SonarGuard AI — Command-Line Tools
cd /d "%~dp0"
call ".venv\Scripts\activate.bat"

:MENU
cls
echo ============================================================
echo   SonarGuard AI — CLI Tools Menu
echo ============================================================
echo.
echo   [1] Analyze a specific sample  (e.g. sss_mixed_03.png)
echo   [2] Analyze the HR Chennai XTF  (chennai_hr_400khz_seedA.xtf)
echo   [3] Run full benchmark suite    (runs=5 warmup=1)
echo   [4] Generate synthetic XTF log  (custom size)
echo   [5] Generate realistic 400 kHz Chennai-coast survey XTF
echo   [6] List / Download public XTF sonar datasets (NOAA, marine-geo, ...)
echo   [7] Export models to ONNX/TensorRT
echo   [8] Run XTF parser self-tests
echo   [0] Exit
echo.
set /p choice="Enter choice [0-8]: "

if "%choice%"=="1" (
    set /p file="Enter sample filename (e.g. sss_adv_mixed_03.png): "
    echo.
    python -m app.cli "data\samples\%file%"
    echo.
    pause
    goto MENU
)
if "%choice%"=="2" (
    echo.
    python -m app.cli --xtf data\samples\chennai_hr_400khz_seedA.xtf
    echo.
    pause
    goto MENU
)
if "%choice%"=="3" (
    echo.
    python -m app.cli --benchmark --benchmark-runs 5 --benchmark-warmup 1
    echo.
    echo ^Report saved to: data\outputs\benchmark_last.json
    pause
    goto MENU
)
if "%choice%"=="4" (
    set /p pings="How many sonar pings [default 256]: "
    if "%pings%"=="" set pings=256
    set /p samples="Samples per side [default 512]: "
    if "%samples%"=="" set samples=512
    set /p seed="RNG seed [default 7]: "
    if "%seed%"=="" set seed=7
    echo.
    python -m app.xtf_parser data\samples\demo_log_001.xtf --pings %pings% --samples %samples% --seed %seed%
    echo.
    pause
    goto MENU
)
if "%choice%"=="5" (
    set /p pings="How many sonar pings [default 640]: "
    if "%pings%"=="" set pings=640
    set /p samples="Samples per side [default 1280]: "
    if "%samples%"=="" set samples=1280
    set /p seed="RNG seed [default 42]: "
    if "%seed%"=="" set seed=42
    set /p name="Output filename [default chennai_custom.xtf]: "
    if "%name%"=="" set name=chennai_custom.xtf
    echo.
    python -m app.xtf_parser data\samples\%name% --pings %pings% --samples %samples% --seed %seed% --realistic
    echo.
    pause
    goto MENU
)
if "%choice%"=="6" (
    :DLSUB
    cls
    echo   ==============================================
    echo     Public XTF Dataset Manager
    echo   ==============================================
    echo.
    echo   [A] List all available dataset keys & descriptions
    echo   [B] Fetch chennai-survey-sim    (realistic 400kHz Chennai coast simulation, generated locally — instant)
    echo   [C] Fetch noaa-hudson-small     (NOAA 2009 Hudson River — attempts FTP download, falls back to Chennai sim)
    echo   [Z] Return to main menu
    echo.
    set /p dl="Choice [A/B/C/Z]: "
    if /i "%dl%"=="A" python scripts\download_public_data.py list
    if /i "%dl%"=="B" python scripts\download_public_data.py get chennai-survey-sim --force
    if /i "%dl%"=="C" python scripts\download_public_data.py get noaa-hudson-small
    if /i "%dl%"=="Z" goto MENU
    echo.
    pause
    goto DLSUB
)
if "%choice%"=="7" (
    echo.
    python -m app.cli --export-trt
    echo.
    echo ^Manifest saved to: weights\export\manifest.json
    pause
    goto MENU
)
if "%choice%"=="8" (
    echo.
    python -m app.xtf_parser --self-test
    echo.
    pause
    goto MENU
)
if "%choice%"=="0" exit /b 0

goto MENU
