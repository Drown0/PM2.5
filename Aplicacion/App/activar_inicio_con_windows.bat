@echo off
chcp 65001 >nul
echo Configurando Streamlit para iniciar siempre en segundo plano con Windows...

set "TARGET_DIR=%~dp0"
set "VBS_FILE=%TARGET_DIR%iniciar_segundo_plano_silencioso.vbs"
set "SHORTCUT=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Streamlit_AirePredict.lnk"

powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT%'); $s.TargetPath = '%VBS_FILE%'; $s.WorkingDirectory = '%TARGET_DIR%'; $s.Description = 'Inicia AirePredict Streamlit en segundo plano'; $s.Save()"

if exist "%SHORTCUT%" (
    echo.
    echo [EXITO] Acceso directo creado en la carpeta de Inicio de Windows:
    echo "%SHORTCUT%"
    echo.
    echo A partir de ahora, cada vez que enciendas o inicies sesion en tu PC,
    echo Streamlit estara corriendo siempre en segundo plano en http://localhost:8501.
    echo.
    echo Iniciando ahora mismo por primera vez...
    wscript "%VBS_FILE%"
    echo [OK] Servidor iniciado. Puedes abrir http://localhost:8501 en tu navegador.
) else (
    echo [ERROR] No se pudo crear el acceso directo de inicio automatico.
)

pause
