@echo off
title Lanzador Plan Minero
cd /d c:\PROYECTOS\Proyecto_Plan_Minero
echo Iniciando Aplicacion Plan Minero...
streamlit run app.py --server.port 8505
pause
