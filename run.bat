@echo off
setlocal

cd /d D:\DATA_SAINS\TUBES\trafic_app

REM Aktifkan venv
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment belum ada. Jalankan setup.bat dulu.
  pause
  exit /b 1
)

call .venv\Scripts\activate

REM (Opsional) bersihkan cache streamlit jika ingin selalu fresh
REM streamlit cache clear

REM Jalankan streamlit pakai interpreter venv
python -m streamlit run app.py

endlocal
