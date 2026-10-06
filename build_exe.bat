@echo off
echo ===== Lung Volume Change Heatmap - Build EXE =====
echo.

REM Clean previous build
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
if exist "*.spec" del /q "*.spec"

echo [1/3] Building EXE with PyInstaller ...

pyinstaller --onefile --noconsole --icon="logo.png" ^
    --add-data "logo.png;." ^
    --name "LungVolumeChange" ^
    --hidden-import PySide6.QtCore ^
    --hidden-import PySide6.QtWidgets ^
    --hidden-import PySide6.QtGui ^
    --hidden-import SimpleITK ^
    --hidden-import pydicom ^
    --hidden-import numpy ^
    --collect-submodules matplotlib ^
    --collect-submodules PyCt6 ^
    --collect-data PyCt6 ^
    app_pyct6.py

if errorlevel 1 (
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

echo.
echo [2/3] Build successful!
echo [3/3] Output: dist\LungVolumeChange.exe
echo.
dir dist\LungVolumeChange.exe 2>nul && (
    echo Size:
    for %%A in (dist\LungVolumeChange.exe) do echo %%~zA bytes
)
echo.
echo Run: dist\LungVolumeChange.exe
pause
