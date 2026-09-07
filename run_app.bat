@echo off
echo Starting FRAT (Fire Risk Analysis Tool)...
set PATH=C:\ANACONDA3\envs\thesis-seismic\Library\bin;C:\ANACONDA3\envs\thesis-seismic\Library\usr\bin;C:\ANACONDA3\envs\thesis-seismic\Library\mingw-w64\bin;C:\ANACONDA3\envs\thesis-seismic\Scripts;%PATH%
set PYTHON="C:\ANACONDA3\envs\thesis-seismic\python.exe"
if not exist %PYTHON% (
    set PYTHON=python
)
%PYTHON% -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.fileWatcherType none --browser.gatherUsageStats false
pause
