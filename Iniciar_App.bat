@echo off
TITLE Plan Minero - IA Console
COLOR 0A

echo ==================================================
echo      INICIANDO PLAN MINERO 2026-2029 (IA)
echo ==================================================
echo.

:: 1. Check Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python no encontrado. Instala Python y marcalo en PATH.
    pause
    exit
)

:: 2. Launch Streamlit in Background
echo [1/3] Iniciando Servidor Neural...
start /B python -m streamlit run app.py --server.address 0.0.0.0 --server.headless true --theme.base "dark"

:: 3. Wait for Server Warmup
echo [2/3] Cargando Cerebro Digital...
timeout /t 5 >nul

:: 4. Launch Browser in App Mode
echo [3/3] Abriendo Interfaz Grafica...
:: Try Chrome
start chrome --app=http://localhost:8501
if %ERRORLEVEL% NEQ 0 (
    :: Fallback to Edge
    start msedge --app=http://localhost:8501
)

echo.
echo ==================================================
echo   SISTEMA ACTIVO. NO CIERRES ESTA VENTANA NEGRA.
echo ==================================================
echo   Para cerrar la app, cierra esta ventana.
echo ==================================================

:: Keep alive
cmd /k
