@echo off
REM Live dynamic-detection demo. Usage: demo_stream.bat [campaign|zero_day|domain_shift|shuffle] [edge|forge|nsl]
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
set SCEN=%1& if "%SCEN%"=="" set SCEN=campaign
set DS=%2& if "%DS%"=="" set DS=edge
set ART=stream_artifacts& if "%SCEN%"=="domain_shift" set ART=bench_forge_only
start "BridgeIDS detector" cmd /k python -m bridgeids.run serve --artifacts %ART%
timeout /t 8 /nobreak >nul
start "" http://127.0.0.1:8000
python -m bridgeids.run replay --scenario %SCEN% --dataset %DS% --rate 400 --n 20000
echo Replay finished. Close the "BridgeIDS detector" window to stop the server.
