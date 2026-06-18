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

# Cargar Modelo y Features
@st.cache_resource
def load_resources():
    try:
        # Obtener la ruta absoluta de la carpeta donde está este script
        base_path = os.path.dirname(__file__)
        model_path = os.path.join(base_path, 'modelo_final_mp25.joblib')
        features_path = os.path.join(base_path, 'features_list.joblib')

        model = joblib.load(model_path)
        features = joblib.load(features_path)
        return model, features
    except Exception as e:
        st.error(f"Error al cargar los archivos del modelo: {e}")
        return None, None


model, features_list = load_resources()

# --- FUNCIÓN DE DATOS HÍBRIDA (VIVO + LOCAL) ---
def get_hybrid_data():
    url = "https://sinca.mma.gob.cl/cgi-bin/ap_ex_csv.cgi?id=83&param=MP25&type=diario"
    
    # Obtener la ruta absoluta de la carpeta donde está este script
    base_path = os.path.dirname(__file__)
    local_file = os.path.join(base_path, "datos_respaldo.csv")
    
    # 1. INTENTO VIVO (SINCA)
    try:
        response = requests.get(url, verify=False, timeout=8)
        if response.status_code == 200 and "FECHA" in response.text:
            df = pd.read_csv(io.StringIO(response.text), sep=';', decimal=',').tail(15)
            df.columns = [c.strip() for c in df.columns]
            df['MP25'] = df['Registros validados'].fillna(df.get('Registros preliminares', np.nan)).fillna(df.get('Registros no validados', np.nan))
            df = df.dropna(subset=['MP25'])
            
            # Obtener fecha del último registro
            last_date_str = str(df.iloc[-1]['FECHA (YYMMDD)']).zfill(6)
            last_date = datetime.strptime(last_date_str, '%y%m%d')
            
            vals = df['MP25'].tolist()
            if len(vals) >= 3:
                return vals[-1], vals[-2], vals[-3], np.mean(vals[-7:]), "En Vivo (SINCA)", last_date
    except:
        pass

    # 2. INTENTO LOCAL (CSV en GitHub)
    if os.path.exists(local_file):
        try:
            df_local = pd.read_csv(local_file, sep=';', decimal=',').tail(15)
            df_local.columns = [c.strip() for c in df_local.columns]
            df_local['MP25'] = df_local['Registros validados'].fillna(df_local.get('Registros preliminares', np.nan))
            df_local = df_local.dropna(subset=['MP25'])
            
            last_date_str = str(df_local.iloc[-1]['FECHA (YYMMDD)']).zfill(6)
            last_date = datetime.strptime(last_date_str, '%y%m%d')
            
            vals = df_local['MP25'].tolist()
            if len(vals) >= 3:
                return vals[-1], vals[-2], vals[-3], np.mean(vals[-7:]), "Respaldo Local (CSV)", last_date
        except:
            pass

    # 3. ÚLTIMO RECURSO
    return 30.0, 25.0, 20.0, 22.0, "Valores por Defecto", datetime.now()

# --- LÓGICA DE SESIÓN ---
if 'l1' not in st.session_state:
    l1, l2, l3, l7, source, last_dt = get_hybrid_data()
    st.session_state.l1, st.session_state.l2, st.session_state.l3, st.session_state.l7, st.session_state.source, st.session_state.last_dt = l1, l2, l3, l7, source, last_dt

# Sidebar
st.sidebar.header("Ajuste de Variables")
if st.sidebar.button("🔄 Intentar Sincronizar"):
    for key in list(st.session_state.keys()): del st.session_state[key]
    st.rerun()

lag_1 = st.sidebar.number_input("MP2.5 Hoy (µg/m³)", value=float(st.session_state.l1))
lag_2 = st.sidebar.number_input("MP2.5 Ayer (µg/m³)", value=float(st.session_state.l2))
lag_3 = st.sidebar.number_input("MP2.5 hace 2 días", value=float(st.session_state.l3))
lag_7_avg = st.sidebar.number_input("Promedio 7 días", value=float(st.session_state.l7))

# --- DETECTAR SI LOS DATOS FUERON MODIFICADOS ---
is_manual = (lag_1 != st.session_state.l1 or lag_2 != st.session_state.l2 or 
             lag_3 != st.session_state.l3 or lag_7_avg != st.session_state.l7)

# Mostrar estado y fecha
fecha_pred = st.session_state.last_dt + timedelta(days=1)
if is_manual:
    st.warning(f"⚠️ **Escenario Simulado:** Predicción basada en datos ingresados manualmente.")
else:
    source_msg = "✅ **Sincronizado con SINCA**" if "Vivo" in st.session_state.source else "📂 **Usando Respaldo Histórico**"
    st.info(f"{source_msg} | Predicción automática para el día: **{fecha_pred.strftime('%d/%m/%Y')}**")

# --- PREDICCIÓN ---
def prepare_input(l1, l2, l3, l7_avg):
    rolling_3 = np.mean([l1, l2, l3])
    rolling_std_3 = np.std([l1, l2, l3])
    diff_1 = l1 - l2
    input_data = pd.DataFrame([{
        'lag_1': l1, 'lag_2': l2, 'lag_3': l3,
        'rolling_mean_3': rolling_3, 'rolling_std_3': rolling_std_3,
        'rolling_mean_7': l7_avg, 'diff_1': diff_1,
        'mes_sin': np.sin(2 * np.pi * fecha_pred.month / 12),
        'mes_cos': np.cos(2 * np.pi * fecha_pred.month / 12),
        'Es_FinDeSemana': 1 if fecha_pred.weekday() >= 5 else 0
    }])
    return input_data

if model:
    input_df = prepare_input(lag_1, lag_2, lag_3, lag_7_avg)
    prediccion = model.predict(input_df)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Resultado del Modelo")
        st.metric("MP2.5 Predicho", f"{prediccion:.2f} µg/m³")
        if prediccion <= 50: st.success("Estado: BUENO")
        elif prediccion <= 79: st.warning("Estado: REGULAR")
        elif prediccion <= 109: st.error("Estado: ALERTA")
        elif prediccion <= 169: st.error("Estado: PRE-EMERGENCIA")
        else: st.error("Estado: EMERGENCIA")
        
        st.caption(f"Fecha de los datos de entrada: {st.session_state.last_dt.strftime('%d/%m/%Y')}")

    with col2:
        tendencia = pd.DataFrame({
            'Día': ['-2 días', 'Ayer', 'Hoy', 'MAÑANA'],
            'MP2.5': [lag_3, lag_2, lag_1, prediccion]
        })
        fig = px.line(tendencia, x='Día', y='MP2.5', markers=True, title="Evolución de la Concentración")
        fig.add_hline(y=50, line_dash="dot", line_color="green", annotation_text="Norma")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption(f"Fuente de datos original: {st.session_state.source} | Modelo: Stacking Ensemble")
