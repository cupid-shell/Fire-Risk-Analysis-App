@echo off
setlocal
echo Starting FRAT (Fire Risk Analysis Tool)...

:: 1. Optional user-specific local override (git-ignored)
if exist "%~dp0.env.bat" call "%~dp0.env.bat"

:: 2. If PYTHON is already set by local override or environment, use it
if defined PYTHON goto setup_path

:: 3. Check for active Conda or virtual environment
if defined CONDA_PREFIX (
    set "ENV_DIR=%CONDA_PREFIX%"
    set "PYTHON=%CONDA_PREFIX%\python.exe"
    goto setup_path
)
if defined VIRTUAL_ENV (
    set "ENV_DIR=%VIRTUAL_ENV%"
    set "PYTHON=%VIRTUAL_ENV%\Scripts\python.exe"
    goto setup_path
)

:: 4. Check for standard project-local virtual environments (.venv / venv)
if exist "%~dp0.venv\Scripts\python.exe" (
    set "ENV_DIR=%~dp0.venv"
    set "PYTHON=%~dp0.venv\Scripts\python.exe"
    goto setup_path
)
if exist "%~dp0venv\Scripts\python.exe" (
    set "ENV_DIR=%~dp0venv"
    set "PYTHON=%~dp0venv\Scripts\python.exe"
    goto setup_path
)

:: 5. Fallback to system Python on PATH
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PYTHON=python"
    goto run
)

echo [ERROR] Python was not found on your system or PATH.
echo Please install Python/Conda and activate your environment,
echo or create a virtual environment (.venv) in this project folder.
pause
exit /b 1

:setup_path
:: If running in Conda or virtualenv on Windows, ensure DLLs and binaries take precedence
if defined ENV_DIR (
    if exist "%ENV_DIR%\Library\bin" (
        set "PATH=%ENV_DIR%\Library\bin;%ENV_DIR%\Library\usr\bin;%ENV_DIR%\Library\mingw-w64\bin;%ENV_DIR%\Scripts;%PATH%"
    ) else if exist "%ENV_DIR%\Scripts" (
        set "PATH=%ENV_DIR%\Scripts;%PATH%"
    )
)

:run
echo Using Python: %PYTHON%
"%PYTHON%" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.fileWatcherType none --browser.gatherUsageStats false
pause
