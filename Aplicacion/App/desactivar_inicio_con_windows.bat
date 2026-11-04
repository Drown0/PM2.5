@echo off
chcp 65001 >nul
set "SHORTCUT=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Streamlit_AirePredict.lnk"

if exist "%SHORTCUT%" (
    del "%SHORTCUT%"
    echo [OK] Inicio automatico con Windows desactivado exitosamente.
) else (
    echo [INFO] No habia ningun acceso directo configurado en la carpeta de Inicio.
)

call "%~dp0detener_streamlit_local.bat"
