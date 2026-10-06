@echo off
rem Lung Volume Change Heatmap — development launcher
rem Requires a Python environment with the packages in requirements.txt
rem (e.g. conda create -n lungvc python=3.11 && pip install -r requirements.txt)

set PYTHONW=pythonw.exe
where %PYTHONW% >nul 2>nul || set PYTHONW=python.exe

start "" %PYTHONW% "%~dp0app_pyct6.py"
