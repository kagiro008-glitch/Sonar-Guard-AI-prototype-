@echo off
REM ============================================================
REM  SonarGuard AI -- One-Click Start Launcher for Windows CMD
REM ============================================================

chcp 65001 >nul 2>&1
title SonarGuard AI -- Web Dashboard

cd /d "%~dp0"
if errorlevel 1 goto :FAIL_CD

echo.
echo ------------------------------------------------------------
echo   Working directory: %cd%
echo   Script location:   %~dp0
echo ------------------------------------------------------------
echo.

REM -------- Preflight: kill any stale uvicorn/python already on our port --------
echo [PREFLIGHT] Checking for stale uvicorn on port 8000 / 8001...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000 " ^| findstr /i "LISTENING"') do (
    echo   Killing stale PID %%a holding port 8000...
    taskkill /F /PID %%a /T >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8001 " ^| findstr /i "LISTENING"') do (
    echo   Killing stale PID %%a holding port 8001...
    taskkill /F /PID %%a /T >nul 2>&1
)
timeout /t 1 /nobreak >nul

REM -------- Preflight: check Python is available --------
where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo [PREFLIGHT] ERROR: python.exe was not found on PATH.
    echo   Please install Python 3.10+ from https://www.python.org/downloads/
    echo   and make sure you check "Add Python to PATH" during install.
    goto :END
)
echo [PREFLIGHT] python.exe is on PATH.

REM -------- Preflight: check .venv exists --------
if not exist ".venv\Scripts\activate.bat" (
    echo [1/3] First-time setup: creating Python virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Could not create .venv.
        echo   TIP: Try running CMD as Administrator, or check antivirus and permissions.
        goto :END
    )
    echo [1/3] Virtual environment created OK.
) else (
    echo [PREFLIGHT] .venv already exists, skipping creation.
)

REM -------- Activate venv --------
echo [PREFLIGHT] Activating virtual environment...
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: .venv\Scripts\activate.bat is missing -- the venv seems incomplete.
    echo   Delete the .venv folder and re-run this script to rebuild it.
    goto :END
)
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo WARNING: activate.bat returned non-zero errorlevel -- continuing anyway.
)
echo [PREFLIGHT] Virtual environment activated.

REM -------- Install requirements only if the environment is incomplete --------
if not exist ".venv\Lib\site-packages\fastapi" (
    echo [2/3] First-time setup: installing dependencies -- takes about 2 min.
    python -m pip install --upgrade pip >nul 2>&1
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: pip install failed. Check requirements.txt or internet access.
        goto :END
    )
    echo [2/3] Dependencies installed OK.
) else (
    echo [PREFLIGHT] Python dependencies already installed, skipping pip install.
)

REM -------- Generate synthetic samples only when the sample bundle is missing --------
if not exist "data\samples\sss_mixed_03.png" (
    echo [3/3] First-time setup: generating synthetic sonar samples...
    if not exist "data\samples" mkdir "data\samples" >nul 2>&1
    python scripts\generate_synthetic_sonar.py
    if errorlevel 1 (
        echo WARNING: Basic synthetic sample generator failed, continuing anyway.
    ) else (
        python scripts\generate_synthetic_sonar.py --advanced --seed 42 --count 3 --with-masks >nul 2>&1
    )
) else (
    echo [PREFLIGHT] Sample bundle already present, skipping synthetic generation.
)

REM -------- Select the first available local port --------
set PORT=
for /f %%P in ('powershell -NoProfile -Command "$ports = 8000..8002; foreach ($port in $ports) { $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $port); try { $listener.Start(); $listener.Stop(); Write-Output $port; exit 0 } catch { try { $listener.Stop() } catch {} } }; exit 1"') do set PORT=%%P
if not defined PORT (
    echo [ERROR] Ports 8000, 8001, and 8002 are unavailable.
    goto :END
)

REM -------- Header --------
echo.
echo ============================================================
echo   SonarGuard AI is starting...
echo   Dashboard URL: http://127.0.0.1:%PORT%
echo   This window runs the server. Press Ctrl+C here to stop it.
echo ============================================================
echo.

REM -------- Open the default browser as soon as the dashboard responds --------
start "" /b powershell.exe -NoProfile -WindowStyle Hidden -Command "$url='http://127.0.0.1:%PORT%'; for($i=0; $i -lt 600; $i++){ try { $response=Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 1; if($response.StatusCode -ge 200){ Start-Process $url; exit 0 } } catch {}; Start-Sleep -Milliseconds 500 }; exit 1"

echo.
title SonarGuard AI -- Running on port %PORT%
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port %PORT%
set "SERVER_EXIT_CODE=%errorlevel%"
echo.
echo SonarGuard server stopped with exit code %SERVER_EXIT_CODE%. Closing launcher.
exit /b %SERVER_EXIT_CODE%

:FAIL_CD
echo ERROR: Failed to change directory to %~dp0
echo   unusual issue -- check permissions on this folder.

:END
echo.
echo ------------------------------------------------------------
echo   SonarGuard AI launcher finished. You can close this window.
echo ------------------------------------------------------------
echo.
pause
exit /b
