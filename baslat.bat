@echo off
REM Sanal ortamı aktif et
call venv\Scripts\activate

REM Streamlit uygulamasını başlat
streamlit run app.py

pause