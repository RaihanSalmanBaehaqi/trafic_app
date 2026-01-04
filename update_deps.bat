@echo off
setlocal

cd /d D:\DATA_SAINS\TUBES\trafic_app

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment belum ada. Jalankan setup.bat dulu.
  pause
  exit /b 1
)

call .venv\Scripts\activate

echo [INFO] Update dependencies...
pip install -r requirements.txt

echo.
echo [OK] Dependencies updated.
pause
endlocal
