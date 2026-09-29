@echo off
REM Starts the Cat vs Dog classifier web app using the project virtual environment.
setlocal
cd /d "%~dp0"
call ..\.venv\Scripts\activate.bat
python app.py
endlocal
