@echo off
REM Convenience script to run the app on Windows
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
