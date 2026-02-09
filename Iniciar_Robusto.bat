@echo off
TITLE Plan Minero - IA Console (Modo Robusto)
COLOR 0B

echo ==================================================
echo   INICIANDO PLAN MINERO - MODO TOLERANCIA A FALLOS
echo ==================================================
echo.
echo [INFO] Este lanzador monitorizara la aplicacion constantemente.
echo [INFO] Si se "duerme" o falla, se reiniciara sola.
echo.

:LOOP
echo [LAUNCHER] Iniciando Watchdog...
python watchdog.py
echo [LAUNCHER] Watchdog terminado inesperadamente. Reiniciando en 5 segundos...
timeout /t 5
GOTO LOOP
