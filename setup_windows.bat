@echo off
REM BridgeIDS one-shot setup for Windows 10/11 (cmd or PowerShell)
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% -c "import sys; assert sys.version_info >= (3,10), 'Python 3.10+ required'" || goto :error
%PY% -m venv .venv || goto :error
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip wheel
pip install -r requirements.txt || goto :error
pip install -e . || goto :error
python tests\test_core.py || goto :error
echo.
echo Setup complete. Activate with:  .venv\Scripts\activate
echo Next:  run_all.bat        (experiments + report)
echo        demo_stream.bat    (live dynamic detection demo)
goto :eof
:error
echo Setup failed - see the message above. & exit /b 1
