import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import requests
import io
from datetime import datetime, timedelta
import warnings

# Configuración de página
st.set_page_config(page_title="Predictor MP2.5 - Parque O'Higgins", layout="wide")

# Ignorar advertencias de SSL para el SINCA
warnings.filterwarnings('ignore')

st.title("🌬️ Predicción Automática de Calidad del Aire (MP2.5)")
st.markdown("""
Esta herramienta se conecta en tiempo real al **SINCA** para predecir la contaminación de mañana 
en la estación Parque O'Higgins usando un modelo de **Stacking Ensemble (XGBoost + LightGBM + CatBoost)**.
""")

# Cargar Modelo y Features
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('modelo_final_mp25.joblib')
        features = joblib.load('features_list.joblib')
        return model, features
    except Exception as e:
        st.error(f"Error al cargar los archivos del modelo: {e}")
        return None, None

model, features_list = load_resources()

# --- FUNCIÓN DE SCRAPING MEJORADA Y ROBUSTA ---
def get_live_sinca_data():
    url = "https://sinca.mma.gob.cl/cgi-bin/ap_ex_csv.cgi?id=83&param=MP25&type=diario"
    # Valores por defecto en caso de cualquier falla
    default_vals = (30.0, 25.0, 20.0, 22.0, False)
    
    try:
        response = requests.get(url, verify=False, timeout=15)
        if response.status_code == 200 and "FECHA" in response.text:
            df = pd.read_csv(io.StringIO(response.text), sep=';', decimal=',').tail(15)
            # Limpiar nombres de columnas por si traen espacios
            df.columns = [c.strip() for c in df.columns]
            
            # Consolidar MP2.5
            # Intentamos detectar las columnas correctas basándonos en el formato conocido
            col_val = 'Registros validados'
            col_pre = 'Registros preliminares'
            col_no = 'Registros no validados'
            
            if col_val in df.columns:
                df['MP25'] = df[col_val].fillna(df.get(col_pre, np.nan)).fillna(df.get(col_no, np.nan))
            else:
                # Si las columnas cambiaron, buscamos la primera columna numérica después de la hora
                df['MP25'] = df.iloc[:, 2] 
            
            df = df.dropna(subset=['MP25'])
            vals = df['MP25'].tolist()
            
            # Verificar que tengamos suficientes datos para desempaquetar
            if len(vals) >= 3:
                l1 = vals[-1]
                l2 = vals[-2]
                l3 = vals[-3]
                l7 = np.mean(vals[-7:]) if len(vals) >= 7 else np.mean(vals)
                return float(l1), float(l2), float(l3), float(l7), True
        
        return default_vals
    except Exception:
        return default_vals

# --- LÓGICA DE DATOS CON SESSION STATE ---
if 'l1' not in st.session_state:
    l1, l2, l3, l7, success = get_live_sinca_data()
    st.session_state.l1, st.session_state.l2, st.session_state.l3, st.session_state.l7, st.session_state.success = l1, l2, l3, l7, success

if st.session_state.success:
    st.success(f"✅ Datos sincronizados con el SINCA (Actualización: {datetime.now().strftime('%H:%M:%S')})")
else:
    st.warning("⚠️ El servidor del SINCA no respondió. Se cargaron valores de referencia.")

# Sidebar para entradas manuales y control
st.sidebar.header("Panel de Control")
if st.sidebar.button("🔄 Forzar Sincronización"):
    l1, l2, l3, l7, success = get_live_sinca_data()
    st.session_state.l1, st.session_state.l2, st.session_state.l3, st.session_state.l7, st.session_state.success = l1, l2, l3, l7, success
    st.rerun()

lag_1 = st.sidebar.number_input("MP2.5 Hoy (µg/m³)", value=float(st.session_state.l1))
lag_2 = st.sidebar.number_input("MP2.5 Ayer (µg/m³)", value=float(st.session_state.l2))
lag_3 = st.sidebar.number_input("MP2.5 hace 2 días (µg/m³)", value=float(st.session_state.l3))
lag_7_avg = st.sidebar.number_input("Promedio última semana", value=float(st.session_state.l7))

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

if model is not None:
    input_df = prepare_input(lag_1, lag_2, lag_3, lag_7_avg)
    prediccion = model.predict(input_df)[0]

    # --- VISUALIZACIÓN ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Pronóstico para Mañana")
        st.metric("Concentración Predicha", f"{prediccion:.2f} µg/m³")
        
        # Categorización Normativa Chilena
        if prediccion <= 50:
            st.success("Estado: BUENO")
            st.write("Aire de calidad óptima. Sin restricciones.")
        elif prediccion <= 79:
            st.warning("Estado: REGULAR")
            st.write("Calidad aceptable, pero precaución en grupos sensibles.")
        elif prediccion <= 109:
            st.error("Estado: ALERTA")
            st.write("Posibles restricciones. Evite ejercicio intenso.")
        elif prediccion <= 169:
            st.error("Estado: PRE-EMERGENCIA")
            st.write("Restricción vehicular y prohibición de humos visibles.")
        else:
            st.error("Estado: EMERGENCIA")
            st.write("Condición crítica. Máxima restricción sanitaria.")

    with col2:
        tendencia = pd.DataFrame({
            'Día': ['Hace 2 días', 'Ayer', 'Hoy', 'MAÑANA (Pred)'],
            'MP2.5': [lag_3, lag_2, lag_1, prediccion]
        })
        fig = px.line(tendencia, x='Día', y='MP2.5', title="Trayectoria de la Calidad del Aire", markers=True)
        fig.update_layout(yaxis_title="µg/m³", hovermode="x")
        fig.add_hline(y=50, line_dash="dot", annotation_text="Límite Bueno", line_color="green")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.error("No se pudo cargar el modelo predictivo. Verifique que los archivos .joblib estén en el repositorio.")

st.markdown("---")
st.caption("Proyecto Final - Minería de Datos | Alumno: [Tu Nombre] | Fuente: SINCA MMA | Modelo: Stacking Ensemble v1.2")
