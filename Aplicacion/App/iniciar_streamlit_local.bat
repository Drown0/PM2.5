@echo off
:: Inicia la aplicación Streamlit de forma desatendida en el puerto 8501
cd /d "%~dp0"
python -m streamlit run app.py --server.headless true --server.port 8501
