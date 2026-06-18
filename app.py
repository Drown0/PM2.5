import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# Configuración de página
st.set_page_config(page_title="Predicctor MP2.5 - Parque O'Higgins", layout="wide")

st.title("🌬️ Predicción de Calidad del Aire (MP2.5)")
st.markdown("""
Esta herramienta utiliza un modelo de **Stacking Ensemble (XGBoost + LightGBM + CatBoost)** 
para predecir la concentración de Material Particulado 2.5 en la estación Parque O'Higgins.
""")

# Cargar Modelo y Features
@st.cache_resource
def load_resources():
    model = joblib.load('modelo_final_mp25.joblib')
    features = joblib.load('features_list.joblib')
    return model, features

model, features_list = load_resources()

# Función para obtener datos reales actuales (Auto-relleno)
def get_current_data():
    url = "https://sinca.mma.gob.cl/cgi-bin/ap_ex_csv.cgi?id=83&param=MP25&type=diario"
    try:
        import requests
        import io
        response = requests.get(url, verify=False, timeout=5)
        df = pd.read_csv(io.StringIO(response.text), sep=';', decimal=',').tail(7)
        df['MP25'] = df['Registros validados'].fillna(df['Registros preliminares']).fillna(df['Registros no validados'])
        vals = df['MP25'].tolist()
        return vals[-1], vals[-2], vals[-3], np.mean(vals)
    except:
        return 30.0, 25.0, 20.0, 22.0 # Valores por defecto si falla el SINCA

curr_l1, curr_l2, curr_l3, curr_avg = get_current_data()

# Sidebar para entradas
st.sidebar.header("Entrada de Datos")
st.sidebar.info("Los valores se han auto-completado con datos reales del SINCA (si están disponibles).")

lag_1 = st.sidebar.number_input("MP2.5 de Hoy (µg/m³)", min_value=0.0, value=float(curr_l1))
lag_2 = st.sidebar.number_input("MP2.5 de Ayer (µg/m³)", min_value=0.0, value=float(curr_l2))
lag_3 = st.sidebar.number_input("MP2.5 de hace 2 días (µg/m³)", min_value=0.0, value=float(curr_l3))
lag_7_avg = st.sidebar.number_input("Promedio MP2.5 última semana (µg/m³)", min_value=0.0, value=float(curr_avg))

# Preparar datos para el modelo
def prepare_input(l1, l2, l3, l7_avg):
    # Calcular variables derivadas
    rolling_3 = np.mean([l1, l2, l3])
    rolling_std_3 = np.std([l1, l2, l3])
    diff_1 = l1 - l2
    
    # Fecha de mañana
    mañana = datetime.now() + timedelta(days=1)
    mes = mañana.month
    dia_sem = mañana.weekday()
    
    input_data = pd.DataFrame([{
        'lag_1': l1,
        'lag_2': l2,
        'lag_3': l3,
        'rolling_mean_3': rolling_3,
        'rolling_std_3': rolling_std_3,
        'rolling_mean_7': l7_avg,
        'diff_1': diff_1,
        'mes_sin': np.sin(2 * np.pi * mes / 12),
        'mes_cos': np.cos(2 * np.pi * mes / 12),
        'Es_FinDeSemana': 1 if dia_sem >= 5 else 0
    }])
    return input_data

if st.sidebar.button("Predecir"):
    input_df = prepare_input(lag_1, lag_2, lag_3, lag_7_avg)
    prediccion = model.predict(input_df)[0]
    
    # Mostrar resultados
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Predicción MP2.5 Mañana", f"{prediccion:.2f} µg/m³")
        
        # Categorización según normativa chilena (aproximada)
        if prediccion <= 50:
            st.success("Estado: BUENO")
        elif prediccion <= 79:
            st.warning("Estado: REGULAR")
        elif prediccion <= 109:
            st.error("Estado: ALERTA")
        elif prediccion <= 169:
            st.error("Estado: PRE-EMERGENCIA")
        else:
            st.error("Estado: EMERGENCIA")

    with col2:
        # Gráfico simple de tendencia
        tendencia = pd.DataFrame({
            'Día': ['Hace 2 días', 'Ayer', 'Hoy', 'MAÑANA (Pred)'],
            'MP2.5': [lag_3, lag_2, lag_1, prediccion]
        })
        fig = px.line(tendencia, x='Día', y='MP2.5', title="Tendencia de Contaminación")
        st.plotly_chart(fig)

st.markdown("---")
st.caption("Proyecto Final - Minería de Datos | Datos fuente: SINCA (Estación Parque O'Higgins)")
