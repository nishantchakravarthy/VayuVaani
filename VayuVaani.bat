@echo off
setlocal EnableExtensions
title VayuVaani
cd /d "%~dp0"
set "HERE=%~dp0"
set "VPY=%HERE%.venv\Scripts\python.exe"
set "VPYW=%HERE%.venv\Scripts\pythonw.exe"
set "LOG=%HERE%setup_log.txt"
set "PKGS=numpy pandas scikit-learn lightgbm requests python-dotenv joblib openpyxl gTTS"
set "PKGS_312=numpy==2.1.3 scipy==1.14.1 pandas==2.2.3 scikit-learn==1.5.2 lightgbm==4.5.0 requests python-dotenv joblib openpyxl gTTS"
set "PY312=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"

echo.
echo   VayuVaani - Bengaluru air quality forecast
echo   ------------------------------------------
echo.

rem ---- Already set up? Start straight away. ----
if exist "%VPY%" (
    "%VPY%" -c "import vayuvaani.app.server" >nul 2>nul && goto start
    echo   The last setup did not finish. Setting up again...
    echo.
)

echo   First run: setting things up. This takes 3-10 minutes and needs internet.
echo   Details are written to setup_log.txt in this folder.
echo.
echo VayuVaani setup %date% %time% > "%LOG%"
if exist "%HERE%.tools" rmdir /s /q "%HERE%.tools"
if exist "%HERE%.python" rmdir /s /q "%HERE%.python"

rem ---- Official Python 3.12 already installed? Use it. ----
if exist "%PY312%" goto use312

rem ---- Try 1: the Python already on this computer ----
python -c "import sys; assert sys.version_info >= (3, 10)" >nul 2>nul || goto get312
python --version >> "%LOG%" 2>&1
echo   [1/2] Installing packages...
call :install_with python "%PKGS%"
if errorlevel 1 goto get312
call :check
if not errorlevel 1 goto installed

:get312
rem ---- Try 2: official Python 3.12 with long-established package versions.
rem      Windows Smart App Control can block very new package builds. ----
echo   Windows blocked some packages built for your Python version.
echo   Installing the official Python 3.12 with winget (one time only)...
winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements >> "%LOG%" 2>&1
if not exist "%PY312%" goto nopython

:use312
"%PY312%" --version >> "%LOG%" 2>&1
echo   [1/2] Installing packages with Python 3.12...
call :install_with "%PY312%" "%PKGS_312%"
if errorlevel 1 goto fail
call :check
if errorlevel 1 goto blocked

:installed
echo   [2/2] Checking the fast model engine (LightGBM)...
"%VPY%" -c "import lightgbm" >> "%LOG%" 2>&1 || (
    echo         LightGBM could not run here. The app will use the built-in engine instead.
    "%VPY%" -m pip uninstall -y lightgbm >> "%LOG%" 2>&1
)
call :shortcut
echo.
echo   Setup finished. A "VayuVaani" shortcut was added to your desktop.
echo.

:start
if not exist "%HERE%.icon_v3" (call :shortcut & ie4uinit.exe -show >nul 2>nul & echo.> "%HERE%.icon_v3")
"%VPY%" -c "import gtts" >nul 2>nul || (
    echo   Adding voice support for alerts ^(one time^)...
    "%VPY%" -m pip install --only-binary=:all: "gTTS>=2.5" >> "%LOG%" 2>&1
)
rem Open the app in its own window, with no console left behind.
if exist "%VPYW%" (start "" "%VPYW%" -m vayuvaani.app & exit)
"%VPY%" -m vayuvaani.app
if errorlevel 1 goto fail
goto :eof


rem ---- %1 = python.exe (quoted), %2 = package list (quoted) ----
:install_with
echo ---- installing with %~1 ---- >> "%LOG%"
if exist "%HERE%.venv" rmdir /s /q "%HERE%.venv"
"%~1" -m venv "%HERE%.venv" >> "%LOG%" 2>&1 || exit /b 1
"%VPY%" -m pip install --upgrade pip >> "%LOG%" 2>&1
"%VPY%" -m pip install --only-binary=:all: %~2 >> "%LOG%" 2>&1 || exit /b 1
"%VPY%" -m pip install --no-deps -e "%HERE%." >> "%LOG%" 2>&1 || exit /b 1
exit /b 0

:check
echo ---- import check ---- >> "%LOG%"
"%VPY%" -c "import numpy, scipy, pandas, sklearn, vayuvaani.app.server; print('import check ok')" >> "%LOG%" 2>&1 || exit /b 1
exit /b 0

:shortcut
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$s = (New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop') + '\VayuVaani.lnk');" ^
  "$w = '%VPYW%'; if (Test-Path $w) { $s.TargetPath = $w; $s.Arguments = '-m vayuvaani.app' } else { $s.TargetPath = '%HERE%VayuVaani.bat' }; $s.WorkingDirectory = '%HERE%'; $s.IconLocation = '%HERE%VayuVaani.ico,0';" ^
  "$s.Description = 'VayuVaani air quality forecast'; $s.Save()" >nul 2>nul
exit /b 0

:nopython
echo   Could not install Python 3.12 automatically.
echo.
echo   1. Install Python 3.12 from https://www.python.org/downloads/
echo      (tick "Add python.exe to PATH" on the first screen)
echo   2. Then double-click VayuVaani.bat again.
echo.
start "" "https://www.python.org/downloads/"
pause
exit /b 1

:blocked
echo.
echo   The packages installed, but Windows blocked them from running.
echo   This is usually Smart App Control. See setup_log.txt for details.
echo.
pause
exit /b 1

:fail
echo.
echo   Setup could not finish. See setup_log.txt in this folder for details.
echo   To start again from scratch, delete the ".venv" folder and re-run.
echo.
pause
exit /b 1
