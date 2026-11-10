@echo off
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
python -m bridgeids.run prepare || exit /b 1
python -m bridgeids.run exp all || exit /b 1
python -m bridgeids.run report || exit /b 1
echo Done. Open reports\results_dashboard.html and reports\paper_draft.md
