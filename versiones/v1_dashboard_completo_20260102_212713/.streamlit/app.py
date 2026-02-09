import streamlit as st
import base64
from PIL import Image
import io
import os
import pandas as pd

# --- CONFIGURACIÓN ---
st.set_page_config(layout="wide", page_title="Plan Minero 2026", page_icon="💎")

# --- CSS VISUAL (ESTILO SCI-FI / INDUSTRIAL) ---
# Este bloque define el look exacto de "simulada_deseable.jpg"
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&display=swap');

    /* FONDO OSCURO */
    [data-testid="stAppViewContainer"] {
        background-color: #0b1120;
        background-image: linear_gradient(rgba(11, 17, 32, 0.92), rgba(11, 17, 32, 0.96)); 
        /* Nota: La imagen de fondo se cargará dinámicamente si existe */
    }

    /* TIPOGRAFÍA GLOBAL */
    html, body, p, div, span, h1, h2, h3, h4 {
        font-family: 'Rajdhani', sans-serif !important;
        color: white;
    }

    /* TARJETAS CON BRACKETS CIAN */
    .kpi-card {
        background-color: rgba(13, 20, 35, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 2px;
        padding: 15px;
        position: relative;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 15px;
    }

    /* ESQUINAS DE NEÓN (BRACKETS) */
    .kpi-card::before {
        content: ""; position: absolute; top: 0; left: 0; width: 10px; height: 10px;
        border-top: 2px solid #06b6d4; border-left: 2px solid #06b6d4;
    }
    .kpi-card::after {
        content: ""; position: absolute; bottom: 0; right: 0; width: 10px; height: 10px;
        border-bottom: 2px solid #06b6d4; border-right: 2px solid #06b6d4;
    }

    /* TEXTOS DE LA TARJETA */
    .card-label { color: #94a3b8; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
    .card-val { font-size: 38px; font-weight: 700; color: white; line-height: 1; text-shadow: 0 0 20px rgba(6, 182, 212, 0.2); }
    .card-unit { font-size: 16px; color: #fca5a5; margin-left: 5px; }
    
    /* HEADER */
    .main-title { font-size: 34px; font-weight: 800; text-transform: uppercase; text-align: right; margin: 0; }
    .sub-title { font-size: 20px; color: #94a3b8; text-transform: uppercase; text-align: right; margin: 0; }
</style>
""", unsafe_allow_html=True)

# --- LAYOUT PRINCIPAL ---
c1, c2 = st.columns([2, 8])
with c1:
    # Intenta cargar el logo si existe
    if os.path.exists("logo.png"):
        st.image("logo.png", width=160)
    else:
        st.write("📂 (Sube logo.png)")
with c2:
    st.markdown("""
        <div>
            <div class="main-title">PLAN MINERO ESTRATÉGICO</div>
            <div class="sub-title">CICLO 2026 - 2030</div>
        </div>
    """, unsafe_allow_html=True)

# --- SECCIÓN KPIs (Placeholder Visual) ---
st.markdown("#### ❖ KEY PERFORMANCE INDICATORS")

# Función auxiliar para dibujar tarjetas HTML
def card(label, value, unit):
    return f"""
    <div class="kpi-card">
        <div class="card-label">{label}</div>
        <div class="card-val">{value}<span class="card-unit">{unit}</span></div>
        <div style="font-size:12px; color:#06b6d4;">↗ PROYECTADO</div>
    </div>
    """

# Grilla 5x2 (Dummy Data para empezar)
r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
with r1c1: st.markdown(card("Cobre Fino", "420", "KTON"), unsafe_allow_html=True)
with r1c2: st.markdown(card("Mov. F03", "420", "KTON"), unsafe_allow_html=True)
with r1c3: st.markdown(card("Mov. F05", "420", "KTON"), unsafe_allow_html=True)
with r1c4: st.markdown(card("Remanejo", "420", "KTON"), unsafe_allow_html=True)
with r1c5: st.markdown(card("Mov. Total", "150", "MT"), unsafe_allow_html=True)

r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5)
with r2c1: st.markdown(card("Trat. Planta", "420", "KTON"), unsafe_allow_html=True)
with r2c2: st.markdown(card("Ley CuT", "0.85", "%"), unsafe_allow_html=True)
with r2c3: st.markdown(card("Recuperación", "82.5", "%"), unsafe_allow_html=True)
with r2c4: st.markdown(card("Costo Mina", "3.0", "$/t"), unsafe_allow_html=True)
with r2c5: st.markdown(card("Costo Planta", "15.0", "$/t"), unsafe_allow_html=True)

st.write("---")
st.write("🔹 Visual Analytics & AI Copilot ready for connection...")