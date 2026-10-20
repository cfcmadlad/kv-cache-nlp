@echo off
setlocal
cd /d "%~dp0"

set "PY="
for /f "delims=" %%i in ('python -c "import sys; assert sys.version_info >= (3, 10); print(sys.executable)" 2^>nul') do set "PY=%%i"
if not defined PY for /f "delims=" %%i in ('py -3 -c "import sys; assert sys.version_info >= (3, 10); print(sys.executable)" 2^>nul') do set "PY=%%i"
if defined PY goto have_python

echo Python 3.10 or newer was not found. Installing Python 3.12 with winget...
winget --version >nul 2>&1 || goto no_winget
winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if defined PY goto have_python
echo Python was installed. Close this window, open a new one and run run.bat again.
goto fail

:have_python
echo Using Python: %PY%

"%PY%" -c "import torch, numpy, tokenizers, tqdm" >nul 2>&1 || goto use_venv
set "VPY=%PY%"
goto run_demo

:use_venv
set "VENV=%LOCALAPPDATA%\kv-cache-nlp\venv"
set "VPY=%VENV%\Scripts\python.exe"
if exist "%VPY%" goto have_venv
echo Creating a Python environment in %VENV% (one time only)...
"%PY%" -m venv "%VENV%" || goto venv_failed
:have_venv
"%VPY%" -c "import torch, numpy, tokenizers, tqdm" >nul 2>&1 && goto run_demo
"%VPY%" -c "import torch" >nul 2>&1 && goto have_torch
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
nvidia-smi >nul 2>&1 && set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
echo Installing PyTorch from %TORCH_INDEX% (one time only; the GPU build is about 2.5 GB)...
"%VPY%" -m pip install torch --index-url %TORCH_INDEX% || goto pip_failed
:have_torch
echo Installing the other requirements...
"%VPY%" -m pip install -r requirements.txt || goto pip_failed

:run_demo
"%VPY%" scripts\demo.py
if errorlevel 1 goto fail
exit /b 0

:no_winget
echo winget is not available on this computer. Install Python 3.12 from
echo https://www.python.org/downloads/ (tick "Add python.exe to PATH"), then run run.bat again.
goto fail
:venv_failed
echo Could not create the Python environment with %PY%.
goto fail
:pip_failed
echo Installing packages failed; see the message above. Check the internet connection and run
echo run.bat again. If an antivirus (for example Norton) blocked it, allow it in the antivirus
echo history first. If it keeps failing, delete the folder %LOCALAPPDATA%\kv-cache-nlp\venv and retry.
goto fail
:fail
pause
exit /b 1
