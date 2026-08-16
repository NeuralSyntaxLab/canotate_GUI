@echo off
REM Windows: double-click this in Explorer to start Cannotate.
REM
REM A console window stays open alongside the GUI and shows any progress or
REM error output. To launch without it, double-click Cannotate.vbs instead.

cd /d "%~dp0"

where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    py launch.py
    goto :done
)

where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    python launch.py
    goto :done
)

echo No Python interpreter was found.
echo Install Python 3 from https://www.python.org/downloads/
echo Remember to tick "Add python.exe to PATH" in the installer.
pause
exit /b 1

:done
REM Only pause on failure, so a clean exit closes the window straight away.
if %ERRORLEVEL% neq 0 pause
