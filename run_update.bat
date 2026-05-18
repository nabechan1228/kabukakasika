@echo off
cd /d "%~dp0"
echo [%date% %time%] Starting stock update... >> backend\update.log

cd backend
call .venv\Scripts\activate.bat
python update_db.py >> update.log 2>&1

echo [%date% %time%] Stock update completed. >> update.log
echo -------------------------------------------------- >> update.log
