@echo off
chcp 65001 >nul
echo Buscando y cerrando procesos de Streamlit en el puerto 8501...
set count=0
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8501" ^| findstr "LISTENING"') do (
    echo Deteniendo proceso PID: %%a
    taskkill /F /PID %%a >nul 2>&1
    set /a count+=1
)
if %count% gtr 0 (
    echo [OK] Servidor Streamlit local detenido con éxito.
) else (
    echo [INFO] No habia ningun servidor Streamlit activo en el puerto 8501.
)
timeout /t 3 >nul
