@echo off
REM Windows: double-click this in Explorer to start Cannotate.
REM
REM A console window stays open alongside the GUI and shows any progress or
REM error output. To launch without it, double-click Cannotate.vbs instead.

cd /d "%~dp0"

REM The interpreter setup\install_windows.bat recorded. Preferred over
REM anything on PATH, and the only thing that works on a PC with no system
REM Python at all -- which is the machine the setup script exists for.
REM launch.py does its own checking once it is running; this just starts it.
set "CFG_PY="
if exist "launcher.cfg" (
    for /f "usebackq tokens=1,* delims==" %%A in (`findstr /b /i "python" "launcher.cfg"`) do (
        for /f "tokens=* delims= " %%C in ("%%B") do set "CFG_PY=%%C"
    )
)
if defined CFG_PY if exist "%CFG_PY%" (
    "%CFG_PY%" launch.py
    goto :done
)

REM Only Python 3 will do. A bare "python" is still Python 2 on some older
REM machines, and Python 2 dies on this project's source with a confusing
REM "Non-ASCII character" SyntaxError rather than saying it is too old.
where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    py -3 launch.py
    goto :done
)

where python >nul 2>&1
if errorlevel 1 goto :nopython

python -c "import sys; sys.exit(0 if sys.version_info[0] >= 3 else 1)" >nul 2>&1
if errorlevel 1 goto :nopython

python launch.py
goto :done

:nopython
echo Cannotate is not set up on this PC yet.
echo.
echo Open the "setup" folder next to this file and double-click
echo     install_windows.bat
echo.
echo That installs everything Cannotate needs, including the neural-network
echo features. It takes a few minutes and needs no administrator rights.
pause
exit /b 1

:done
REM Only pause on failure, so a clean exit closes the window straight away.
if %ERRORLEVEL% neq 0 pause
