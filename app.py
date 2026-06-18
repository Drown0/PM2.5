import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import requests
import io
import os
from datetime import datetime, timedelta
import warnings

# Configuración de página
st.set_page_config(page_title="Predictor MP2.5 - Parque O'Higgins", layout="wide")
warnings.filterwarnings('ignore')

st.title("🌬️ Predictor de Calidad del Aire (MP2.5)")
st.markdown("""
Esta herramienta predice la contaminación de mañana en la estación Parque O'Higgins.
**Sistema de Datos:** Prioriza conexión en tiempo real con SINCA; usa base de datos local como respaldo.
""")

# Cargar Modelo y Features
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('modelo_final_mp25.joblib')
        features = joblib.load('features_list.joblib')
        return model, features
    except Exception as e:
        st.error(f"Error al cargar archivos del modelo: {e}")
        return None, None

model, features_list = load_resources()

# --- FUNCIÓN DE DATOS HÍBRIDA (VIVO + LOCAL) ---
def get_hybrid_data():
    url = "https://sinca.mma.gob.cl/cgi-bin/ap_ex_csv.cgi?id=83&param=MP25&type=diario"
    local_file = "datos_respaldo.csv"
    
    # 1. INTENTO VIVO (SINCA)
    try:
        response = requests.get(url, verify=False, timeout=8)
        if response.status_code == 200 and "FECHA" in response.text:
            df = pd.read_csv(io.StringIO(response.text), sep=';', decimal=',').tail(15)
            df.columns = [c.strip() for c in df.columns]
            df['MP25'] = df['Registros validados'].fillna(df.get('Registros preliminares', np.nan)).fillna(df.get('Registros no validados', np.nan))
            df = df.dropna(subset=['MP25'])
            vals = df['MP25'].tolist()
            if len(vals) >= 3:
                return vals[-1], vals[-2], vals[-3], np.mean(vals[-7:]), "En Vivo (SINCA)"
    except:
        pass

    # 2. INTENTO LOCAL (CSV en GitHub)
    if os.path.exists(local_file):
        try:
            df_local = pd.read_csv(local_file, sep=';', decimal=',').tail(15)
            df_local.columns = [c.strip() for c in df_local.columns]
            df_local['MP25'] = df_local['Registros validados'].fillna(df_local.get('Registros preliminares', np.nan))
            df_local = df_local.dropna(subset=['MP25'])
            vals = df_local['MP25'].tolist()
            if len(vals) >= 3:
                return vals[-1], vals[-2], vals[-3], np.mean(vals[-7:]), "Respaldo Local (CSV)"
        except:
            pass

    # 3. ÚLTIMO RECURSO
    return 30.0, 25.0, 20.0, 22.0, "Valores por Defecto"

# --- LÓGICA DE SESIÓN ---
if 'l1' not in st.session_state:
    l1, l2, l3, l7, source = get_hybrid_data()
    st.session_state.l1, st.session_state.l2, st.session_state.l3, st.session_state.l7, st.session_state.source = l1, l2, l3, l7, source

# Mostrar estado de conexión
if "Vivo" in st.session_state.source:
    st.success(f"✅ Conectado al SINCA. Datos actualizados en tiempo real.")
elif "Respaldo" in st.session_state.source:
    st.info(f"📂 Usando base de datos histórica (CSV). El servidor del SINCA no está disponible.")
else:
    st.warning("⚠️ Sin conexión a datos. Usando valores de prueba.")

# Sidebar
st.sidebar.header("Ajuste de Variables")
if st.sidebar.button("🔄 Intentar Sincronizar"):
    l1, l2, l3, l7, source = get_hybrid_data()
    st.session_state.l1, st.session_state.l2, st.session_state.l3, st.session_state.l7, st.session_state.source = l1, l2, l3, l7, source
    st.rerun()

lag_1 = st.sidebar.number_input("MP2.5 Hoy (µg/m³)", value=float(st.session_state.l1))
lag_2 = st.sidebar.number_input("MP2.5 Ayer (µg/m³)", value=float(st.session_state.l2))
lag_3 = st.sidebar.number_input("MP2.5 hace 2 días", value=float(st.session_state.l3))
lag_7_avg = st.sidebar.number_input("Promedio 7 días", value=float(st.session_state.l7))

# --- PREDICCIÓN ---
def prepare_input(l1, l2, l3, l7_avg):
    rolling_3 = np.mean([l1, l2, l3])
    rolling_std_3 = np.std([l1, l2, l3])
    diff_1 = l1 - l2
    mañana = datetime.now() + timedelta(days=1)
    input_data = pd.DataFrame([{
        'lag_1': l1, 'lag_2': l2, 'lag_3': l3,
        'rolling_mean_3': rolling_3, 'rolling_std_3': rolling_std_3,
        'rolling_mean_7': l7_avg, 'diff_1': diff_1,
        'mes_sin': np.sin(2 * np.pi * mañana.month / 12),
        'mes_cos': np.cos(2 * np.pi * mañana.month / 12),
        'Es_FinDeSemana': 1 if mañana.weekday() >= 5 else 0
    }])
    return input_data

if model:
    input_df = prepare_input(lag_1, lag_2, lag_3, lag_7_avg)
    prediccion = model.predict(input_df)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Predicción de Mañana")
        st.metric("MP2.5", f"{prediccion:.2f} µg/m³")
        if prediccion <= 50: st.success("BUENO")
        elif prediccion <= 79: st.warning("REGULAR")
        elif prediccion <= 109: st.error("ALERTA")
        elif prediccion <= 169: st.error("PRE-EMERGENCIA")
        else: st.error("EMERGENCIA")

    with col2:
        tendencia = pd.DataFrame({
            'Día': ['-2 días', 'Ayer', 'Hoy', 'MAÑANA'],
            'MP2.5': [lag_3, lag_2, lag_1, prediccion]
        })
        fig = px.line(tendencia, x='Día', y='MP2.5', markers=True, title="Tendencia")
        fig.add_hline(y=50, line_dash="dot", line_color="green", annotation_text="Límite Bueno")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption(f"Fuente de datos actual: {st.session_state.source}")
