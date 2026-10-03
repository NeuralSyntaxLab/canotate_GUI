@echo off
REM Cannotate -- one-time setup for Windows. Double-click this in Explorer.
REM
REM It builds Cannotate its own private Python in %LOCALAPPDATA%\Cannotate and
REM installs everything the tool needs, including the deep-learning packages
REM behind the NN buttons. Nothing is installed system-wide, nothing is put on
REM PATH, and no administrator rights are needed.
REM
REM Safe to run again: it rebuilds the environment from scratch, which is the
REM fix if anything ever breaks.
REM
REM NOTE: this mirrors setup/install_mac.command, which was verified end to
REM end. This file has NOT been run on a Windows machine. If it fails, the
REM error it prints is the thing to report.

setlocal enabledelayedexpansion

for %%I in ("%~dp0..") do set "PROJECT_DIR=%%~fI"
if "%CANNOTATE_HOME%"=="" set "CANNOTATE_HOME=%LOCALAPPDATA%\Cannotate"
set "ENV_DIR=%CANNOTATE_HOME%\env"
set "MICROMAMBA=%CANNOTATE_HOME%\micromamba.exe"
set "MAMBA_ROOT_PREFIX=%CANNOTATE_HOME%\root"

REM conda-forge's pytorch here is the CPU build, as on macOS. A lab machine
REM with an NVIDIA card would want the CUDA build instead -- that is a
REM deliberate choice to keep the download small, not an oversight.
set "CONDA_PACKAGES=python=3.11 tk numpy scipy matplotlib pytorch onnxruntime scikit-learn pandas scikit-image"
set "PIP_PACKAGES=soundfile sounddevice"

echo.
echo Setting up Cannotate
echo    Project:    %PROJECT_DIR%
echo    Installing: %CANNOTATE_HOME%  (nothing else on this PC is touched)
echo.

REM ---- 1. a package manager that needs no Python to run ----------------
echo [1/5] Fetching the package manager
if exist "%MICROMAMBA%" (
    echo    already present
) else (
    if not exist "%CANNOTATE_HOME%" mkdir "%CANNOTATE_HOME%"
    REM curl.exe ships with Windows 10 1803 and later.
    curl -fL -o "%MICROMAMBA%" "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64.exe"
    if errorlevel 1 (
        echo.
        echo    Could not download the package manager.
        echo    Check that this PC is online, then run this file again.
        goto :failed
    )
    echo    downloaded
)

REM ---- 2. a private Python, rebuilt from scratch every run -------------
echo.
echo [2/5] Building Cannotate's own Python (about 250 MB to download)
echo       This is the slow part - several minutes on a first run.
if exist "%ENV_DIR%" (
    echo       Removing the previous environment first.
    rmdir /s /q "%ENV_DIR%"
)
"%MICROMAMBA%" create -y -p "%ENV_DIR%" -c conda-forge %CONDA_PACKAGES%
if errorlevel 1 goto :failed
set "PYTHON=%ENV_DIR%\python.exe"
if not exist "%PYTHON%" (
    echo    Python was not created at %PYTHON%
    goto :failed
)
"%PYTHON%" --version

REM ---- 3. the audio libraries ------------------------------------------
echo.
echo [3/5] Adding the audio libraries
"%PYTHON%" -m pip install --quiet --disable-pip-version-check %PIP_PACKAGES%
if errorlevel 1 goto :failed
echo    soundfile and sounddevice installed

REM ---- 4. tell the launcher which Python to use ------------------------
REM launch.py reads this and skips searching the machine for an interpreter.
echo.
echo [4/5] Pointing Cannotate at it
> "%PROJECT_DIR%\launcher.cfg" echo [cannotate]
>>"%PROJECT_DIR%\launcher.cfg" echo python = %PYTHON%
echo    wrote launcher.cfg

REM ---- 5. prove it actually works --------------------------------------
echo.
echo [5/5] Checking that everything works
"%PYTHON%" "%PROJECT_DIR%\smoke_test.py"
if errorlevel 1 (
    echo.
    echo    Cannotate is installed but something is not working.
    echo    The failing checks are listed above. Send them to whoever gave you
    echo    Cannotate - the tool may still open, but do not trust the NN buttons.
    goto :failed
)

echo.
echo Done - Cannotate is ready.
echo    Open it by double-clicking Cannotate.vbs in:
echo        %PROJECT_DIR%
echo.
echo    You never need to run this setup again unless something breaks.
echo.
pause
exit /b 0

:failed
echo.
echo Setup did not finish.
echo.
echo Nothing on this PC was changed outside %CANNOTATE_HOME%,
echo so it is safe to delete that folder and run this again:
echo.
echo     rmdir /s /q "%CANNOTATE_HOME%"
echo.
echo If it keeps failing, send whatever is printed above to whoever
echo gave you Cannotate.
echo.
pause
exit /b 1
