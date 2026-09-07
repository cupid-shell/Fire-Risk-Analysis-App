@echo off
setlocal
echo Starting FRAT (Fire Risk Analysis Tool)...

:: Detect Python interpreter and its environment
:: 1. Dedicated project Conda environment
if exist "C:\ANACONDA3\envs\thesis-seismic\python.exe" (
    set "ENV_DIR=C:\ANACONDA3\envs\thesis-seismic"
    set "PYTHON=C:\ANACONDA3\envs\thesis-seismic\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "ENV_DIR=%~dp0.venv"
    set "PYTHON=%~dp0.venv\Scripts\python.exe"
) else if exist "%~dp0venv\Scripts\python.exe" (
    set "ENV_DIR=%~dp0venv"
    set "PYTHON=%~dp0venv\Scripts\python.exe"
) else if defined CONDA_PREFIX (
    set "ENV_DIR=%CONDA_PREFIX%"
    set "PYTHON=%CONDA_PREFIX%\python.exe"
) else (
    set "ENV_DIR="
    set "PYTHON=python"
)

:: If a specific environment directory was found, ensure its DLL & binary paths take precedence
if defined ENV_DIR (
    if exist "%ENV_DIR%\Library\bin" (
        set "PATH=%ENV_DIR%\Library\bin;%ENV_DIR%\Library\usr\bin;%ENV_DIR%\Library\mingw-w64\bin;%ENV_DIR%\Scripts;%PATH%"
    ) else if exist "%ENV_DIR%\Scripts" (
        set "PATH=%ENV_DIR%\Scripts;%PATH%"
    )
)

echo Using Python: %PYTHON%
"%PYTHON%" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.fileWatcherType none --browser.gatherUsageStats false
pause
