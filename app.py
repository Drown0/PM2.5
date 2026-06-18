import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import requests
import io
from datetime import datetime, timedelta

# Configuración de página
st.set_page_config(page_title="Predictor MP2.5 - Parque O'Higgins", layout="wide")

st.title("🌬️ Predicción Automática de Calidad del Aire (MP2.5)")
st.markdown("""
Esta herramienta se conecta en tiempo real al **SINCA** para predecir la contaminación de mañana 
en la estación Parque O'Higgins usando un modelo de **Stacking Ensemble**.
""")

# Cargar Modelo y Features
@st.cache_resource
def load_resources():
    model = joblib.load('modelo_final_mp25.joblib')
    features = joblib.load('features_list.joblib')
    return model, features

model, features_list = load_resources()

# --- FUNCIÓN DE SCRAPING MEJORADA ---
def get_live_sinca_data():
    # Intentamos la URL de respaldo que suele ser más estable para scripts
    url = "https://sinca.mma.gob.cl/cgi-bin/ap_ex_csv.cgi?id=83&param=MP25&type=diario"
    try:
        response = requests.get(url, verify=False, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text), sep=';', decimal=',').tail(10)
            df['MP25'] = df['Registros validados'].fillna(df['Registros preliminares']).fillna(df['Registros no validados'])
            df = df.dropna(subset=['MP25'])
            vals = df['MP25'].tolist()
            # Retornamos los últimos 3 días y el promedio de la semana
            return vals[-1], vals[-2], vals[-3], np.mean(vals[-7:]), True
    except Exception as e:
        return 30.0, 25.0, 20.0, 22.0, False

# --- LÓGICA DE DATOS ---
if 'data_fetched' not in st.session_state:
    st.session_state.l1, st.session_state.l2, st.session_state.l3, st.session_state.l7, st.session_state.success = get_live_sinca_data()
    st.session_state.data_fetched = True

if st.session_state.success:
    st.success(f"✅ Datos sincronizados exitosamente desde el SINCA (Última actualización: {datetime.now().strftime('%H:%M:%S')})")
else:
    st.warning("⚠️ No se pudo conectar con el servidor del SINCA. Usando valores históricos por defecto.")

# Sidebar
st.sidebar.header("Control de Datos")
if st.sidebar.button("🔄 Sincronizar con SINCA ahora"):
    st.session_state.l1, st.session_state.l2, st.session_state.l3, st.session_state.l7, st.session_state.success = get_live_sinca_data()
    st.rerun()

lag_1 = st.sidebar.number_input("MP2.5 Hoy (µg/m³)", value=float(st.session_state.l1))
lag_2 = st.sidebar.number_input("MP2.5 Ayer (µg/m³)", value=float(st.session_state.l2))
lag_3 = st.sidebar.number_input("MP2.5 hace 2 días (µg/m³)", value=float(st.session_state.l3))
lag_7_avg = st.sidebar.number_input("Promedio última semana", value=float(st.session_state.l7))

# --- PREDICCIÓN AUTOMÁTICA ---
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

input_df = prepare_input(lag_1, lag_2, lag_3, lag_7_avg)
prediccion = model.predict(input_df)[0]

# --- VISUALIZACIÓN ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Pronóstico para Mañana")
    st.metric("MP2.5 Predicho", f"{prediccion:.2f} µg/m³")
    
    if prediccion <= 50:
        st.success("Estado esperado: BUENO")
    elif prediccion <= 79:
        st.warning("Estado esperado: REGULAR")
    elif prediccion <= 109:
        st.error("Estado esperado: ALERTA")
    elif prediccion <= 169:
        st.error("Estado esperado: PRE-EMERGENCIA")
    else:
        st.error("Estado esperado: EMERGENCIA")
    
    st.info("""
    **Sugerencia:**
    El modelo utiliza los niveles actuales para anticipar condiciones de mala ventilación.
    """)

with col2:
    tendencia = pd.DataFrame({
        'Día': ['Hace 2 días', 'Ayer', 'Hoy', 'MAÑANA (Pred)'],
        'MP2.5': [lag_3, lag_2, lag_1, prediccion]
    })
    fig = px.line(tendencia, x='Día', y='MP2.5', title="Evolución de la Contaminación", markers=True)
    fig.update_layout(yaxis_title="µg/m³")
    st.plotly_chart(fig)

st.markdown("---")
st.caption("Proyecto Final - Minería de Datos | Datos: SINCA (Parque O'Higgins) | Modelo: Stacking Regressor")
