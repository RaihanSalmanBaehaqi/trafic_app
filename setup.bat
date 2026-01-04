@echo off
setlocal

REM Pindah ke folder project (aman walau beda drive)
cd /d D:\DATA_SAINS\TUBES\trafic_app

REM Cek apakah venv sudah ada
if exist ".venv\Scripts\python.exe" (
  echo [INFO] Virtual environment sudah ada: .venv
) else (
  echo [INFO] Membuat virtual environment: .venv
  python -m venv .venv
)

REM Aktifkan venv
call .venv\Scripts\activate

REM Upgrade pip (opsional tapi bagus)
python -m pip install --upgrade pip

REM Install requirements
echo [INFO] Install dependencies dari requirements.txt ...
pip install -r requirements.txt

echo.
echo [OK] Setup selesai.
echo Jalankan aplikasi dengan run.bat
echo.
pause
endlocal
