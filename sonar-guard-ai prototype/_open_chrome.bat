@echo off
REM ============================================================
REM  Helper -- open SonarGuard dashboard in Chrome or fallback
REM  Usage: call _open_chrome.bat <URL> <timeout_seconds>
REM ============================================================
setlocal EnableExtensions EnableDelayedExpansion
set "URL=%~1"
set "WAIT_SECONDS=%~2"
if "%URL%"=="" set "URL=http://127.0.0.1:8000"
if "%WAIT_SECONDS%"=="" set "WAIT_SECONDS=120"

REM Wait for the dashboard before opening it; run this helper with START /B.
powershell -NoProfile -Command "$url='%URL%'; for ($i=0; $i -lt %WAIT_SECONDS%; $i++) { try { $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2; if ($response.StatusCode -ge 200) { exit 0 } } catch {}; Start-Sleep -Seconds 1 }; exit 1" >nul 2>&1
if errorlevel 1 (
   echo [WARN] Dashboard did not become ready within %WAIT_SECONDS% seconds. Open %URL% manually.
   goto :DONE
)
echo [READY] Opening %URL%

REM 1) chrome.exe via PATH
where chrome.exe >nul 2>&1
if not errorlevel 1 (
    start "" chrome.exe "%URL%"
    goto :DONE
)

REM 2) 64-bit program files
if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" (
    start "" "%ProgramFiles%\Google\Chrome\Application\chrome.exe" "%URL%"
    goto :DONE
)

REM 3) 32-bit program files
if exist "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" (
    start "" "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" "%URL%"
    goto :DONE
)

REM 4) Per-user install
if exist "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe" (
    start "" "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe" "%URL%"
    goto :DONE
)

REM 5) Last resort -- default browser
start "" "%URL%"

:DONE
endlocal
exit /b
