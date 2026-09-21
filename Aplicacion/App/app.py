# -*- coding: utf-8 -*-
"""
Plataforma Ciudadana de Monitoreo y Alerta Temprana de Material Particulado Fino (MP2.5)
Estación Parque O'Higgins (SINCA D14/273) - Santiago de Chile
Orientada a la Comunidad: Pronósticos Preventivos (24h, 48h y 72h) y Protocolo de Salud PPDA
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import requests
import io
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(
    page_title="Alerta Temprana MP2.5 - Parque O'Higgins",
    layout="wide",
    page_icon="🌬️"
)

# --- CARGA DE RECURSOS PREDICTIVOS ---
@st.cache_resource
def load_resources():
    try:
        base_path = os.path.dirname(__file__)
        models_dir = os.path.join(base_path, 'modelos')
        if not os.path.exists(models_dir):
            models_dir = base_path

        m24_path = os.path.join(models_dir, 'modelo_final_mp25_tuneado_24h.joblib')
        m48_path = os.path.join(models_dir, 'modelo_final_mp25_tuneado_48h.joblib')
        m72_path = os.path.join(models_dir, 'modelo_final_mp25_tuneado_72h.joblib')
        features_path = os.path.join(models_dir, 'features_list.joblib')

        if not os.path.exists(m24_path):
            m24_path = os.path.join(models_dir, 'modelo_final_mp25.joblib')

        m24 = joblib.load(m24_path)
        m48 = joblib.load(m48_path) if os.path.exists(m48_path) else m24
        m72 = joblib.load(m72_path) if os.path.exists(m72_path) else m24
        features = joblib.load(features_path) if os.path.exists(features_path) else None

        return m24, m48, m72, features
    except Exception as e:
        st.error(f"Error al cargar los modelos predictivos: {e}")
        return None, None, None, None

model_24h, model_48h, model_72h, features_list = load_resources()

# --- INGESTA EN VIVO / RESPALDO LOCAL ---
@st.cache_data(ttl=1800)
def fetch_sinca_raw():
    today = datetime.now()
    today_str = today.strftime('%y%m%d')
    from_date = (today - timedelta(days=45)).strftime('%y%m%d')
    url_base = f"https://sinca.mma.gob.cl/cgi-bin/APUB-MMA/apub.tsindico2.cgi?outtype=xcl&from={from_date}&to={today_str}&path=/usr/airviro/data/CONAMA/&lang=esp&rsrc=&macropath=&macro=./RM/D14/Cal/PM25//PM25.diario.diario.ic"

    try:
        r = requests.get(url_base, verify=False, timeout=10)
        if r.status_code == 200 and "FECHA" in r.text:
            return r.text
    except Exception:
        pass
    return None

def get_hybrid_data():
    base_path = os.path.dirname(__file__)
    local_file = os.path.join(base_path, "datos_respaldo.csv")

    # 1. Respaldo local
    if os.path.exists(local_file):
        try:
            df_local = pd.read_csv(local_file, sep=';', decimal=',')
            if 'Fecha' in df_local.columns:
                df_local['Fecha'] = pd.to_datetime(df_local['Fecha'])
                df_local = df_local.sort_values('Fecha').reset_index(drop=True)
                last_dt = df_local.iloc[-1]['Fecha']
                return df_local, "Sistema de Monitoreo SINCA (Sincronizado)", last_dt
        except Exception:
            pass

    # 2. En vivo SINCA
    try:
        csv_pm25 = fetch_sinca_raw()
        if csv_pm25:
            df25 = pd.read_csv(io.StringIO(csv_pm25), sep=';', decimal=',', na_values=['', ' ', 'NaN'], dtype={'FECHA (YYMMDD)': str})
            df25.columns = [c.strip() for c in df25.columns]
            col_val, col_pre, col_no_val = 'Registros validados', 'Registros preliminares', 'Registros no validados'
            df25['MP25'] = df25[col_val].fillna(df25.get(col_pre, np.nan)).fillna(df25.get(col_no_val, np.nan))
            df25 = df25.dropna(subset=['MP25'])
            df25['FECHA_STR'] = df25['FECHA (YYMMDD)'].astype(str).str.split('.').str[0].str.zfill(6)
            df25['Fecha'] = pd.to_datetime('20' + df25['FECHA_STR'], format='%Y%m%d', errors='coerce')
            df25 = df25.sort_values('Fecha').reset_index(drop=True).tail(30)
            last_dt = df25.iloc[-1]['Fecha']
            return df25, "En Vivo (Red Oficial SINCA - MMA)", last_dt
    except Exception:
        pass

    # 3. Fallback
    fechas = [datetime.now() - timedelta(days=i) for i in reversed(range(15))]
    vals25 = [14.0, 18.0, 15.0, 22.0, 29.0, 25.0, 19.0, 15.0, 12.0, 18.0, 14.0, 11.0, 16.0, 20.0, 18.0]
    df_def = pd.DataFrame({'Fecha': fechas, 'MP25': vals25})
    return df_def, "Respaldo Local de Contingencia", datetime.now() - timedelta(days=1)

# --- ESTADO DE SESIÓN ---
if 'historico_df' not in st.session_state:
    df_hist, source, last_dt = get_hybrid_data()
    st.session_state.historico_df = df_hist
    st.session_state.source = source
    st.session_state.last_dt = last_dt
    
    vals_pm25 = df_hist['MP25'].dropna().tolist()
    st.session_state.l0 = float(vals_pm25[-1]) if len(vals_pm25) >= 1 else 18.0
    st.session_state.l1 = float(vals_pm25[-2]) if len(vals_pm25) >= 2 else 15.0
    st.session_state.l2 = float(vals_pm25[-3]) if len(vals_pm25) >= 3 else 14.0
    st.session_state.l3 = float(vals_pm25[-4]) if len(vals_pm25) >= 4 else 12.0
    st.session_state.l7 = float(vals_pm25[-8]) if len(vals_pm25) >= 8 else float(vals_pm25[0])

# --- CLASIFICACIÓN SEGÚN NORMA PPDA DE CHILE ---
def clasificar_norma(val):
    if val <= 50.0:
        return "BUENO", "🟢", "#28a745", "Calidad del aire favorable. Sin restricciones para la población ni actividades al aire libre."
    elif val <= 79.0:
        return "REGULAR", "🟡", "#ffc107", "Aceptable para la mayoría. Personas sensibles (asma, niños, adultos mayores) deben moderar esfuerzos físicos prolongados."
    elif val <= 109.0:
        return "ALERTA", "🟠", "#fd7e14", "Riesgo moderado a alto. Se recomienda suspender clases de educación física en colegios y limitar ejercicio intenso al aire libre."
    elif val <= 169.0:
        return "PRE-EMERGENCIA", "🔴", "#dc3545", "Condición crítica. Prohibición de humos visibles y calefactores a leña, restricción vehicular e industrias. Evitar salir."
    else:
        return "EMERGENCIA", "🟣", "#6f42c1", "Condición extrema. Prohibición total de actividad física al aire libre. Población general debe permanecer en interiores."

# --- BARRA LATERAL CIUDADANA ---
st.sidebar.header("🌬️ Estación Parque O'Higgins")
st.sidebar.caption("Santiago Centro, Región Metropolitana")

if st.sidebar.button("🔄 Actualizar Datos en Tiempo Real", use_container_width=True):
    fetch_sinca_raw.clear()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Simulación Ciudadana")
modo_simulacion = st.sidebar.checkbox("Simular Otro Nivel de Contaminación", value=False)

if modo_simulacion:
    in_l0 = st.sidebar.slider("Nivel actual de MP2.5 (µg/m³)", min_value=5.0, max_value=220.0, value=float(st.session_state.l0), step=1.0)
    in_l1 = in_l0 * 0.9
    in_l2 = in_l0 * 0.85
    in_l3 = in_l0 * 0.8
    in_l7 = in_l0 * 0.95
    st.sidebar.caption("Calculando proyecciones preventivas según el valor seleccionado.")
else:
    in_l0 = st.session_state.l0
    in_l1 = st.session_state.l1
    in_l2 = st.session_state.l2
    in_l3 = st.session_state.l3
    in_l7 = st.session_state.l7

base_dt = st.session_state.last_dt
dt_24h = base_dt + timedelta(days=1)
dt_48h = base_dt + timedelta(days=2)
dt_72h = base_dt + timedelta(days=3)

# --- CONSTRUCCIÓN DEL VECTOR PREDICTIVO ---
def construir_features(l0, l1, l2, l3, l7, dt_target, feat_cols):
    recent_pm25 = [l7, l3, l2, l1, l0]
    rolling_3 = np.mean([l0, l1, l2])
    rolling_7 = np.mean(recent_pm25)
    rolling_std_7 = np.std(recent_pm25, ddof=1) if len(recent_pm25) > 1 else 1.0
    rolling_min_7 = np.min(recent_pm25)
    rolling_max_7 = np.max(recent_pm25)
    diff_1 = l0 - l1
    
    mes = dt_target.month
    dia_ano = dt_target.timetuple().tm_yday
    dia_semana = dt_target.weekday()
    es_fin_de_semana = 1 if dia_semana in [5, 6] else 0
    
    mes_sin = np.sin(2 * np.pi * mes / 12)
    mes_cos = np.cos(2 * np.pi * mes / 12)
    dia_ano_sin = np.sin(2 * np.pi * dia_ano / 365.25)
    dia_ano_cos = np.cos(2 * np.pi * dia_ano / 365.25)
    
    co = 0.45 * (l0 / 20.0)
    no2 = 18.0 * (l0 / 20.0)
    temp = 10.0
    wspd = 1.35
    
    fila = {
        'MP25_t': l0,
        'MP25_lag1': l1,
        'MP25_lag2': l2,
        'MP25_lag3': l3,
        'MP25_lag7': l7,
        'MP25_roll_mean_3': rolling_3,
        'MP25_roll_mean_7': rolling_7,
        'MP25_roll_std_7': rolling_std_7,
        'MP25_roll_min_7': rolling_min_7,
        'MP25_roll_max_7': rolling_max_7,
        'MP25_diff1': diff_1,
        'CO': co,
        'NOX': no2 * 1.5,
        'NO2': no2,
        'NO': no2 * 0.5,
        'O3': 22.0,
        'TEMP_mean': temp + 6.0,
        'TEMP_min': temp,
        'TEMP_max': temp + 12.0,
        'RHUM_mean': 55.0,
        'WSPD_mean': wspd,
        'WSPD_max': wspd * 2.2,
        'CO_lag1': co,
        'NO2_lag1': no2,
        'WSPD_mean_lag1': wspd,
        'TEMP_min_lag1': temp,
        'mes_sin': mes_sin,
        'mes_cos': mes_cos,
        'dia_ano_sin': dia_ano_sin,
        'dia_ano_cos': dia_ano_cos,
        'dia_semana': dia_semana,
        'es_fin_de_semana': es_fin_de_semana
    }
    
    df_feat = pd.DataFrame([fila])
    if feat_cols:
        for c in feat_cols:
            if c not in df_feat.columns:
                df_feat[c] = 0.0
        df_feat = df_feat[feat_cols]
    return df_feat

# Inferencia
if model_24h and model_48h and model_72h:
    feat_24 = construir_features(in_l0, in_l1, in_l2, in_l3, in_l7, dt_24h, features_list)
    feat_48 = construir_features(in_l0, in_l1, in_l2, in_l3, in_l7, dt_48h, features_list)
    feat_72 = construir_features(in_l0, in_l1, in_l2, in_l3, in_l7, dt_72h, features_list)

    pred_24 = max(0.0, float(model_24h.predict(feat_24)[0]))
    pred_48 = max(0.0, float(model_48h.predict(feat_48)[0]))
    pred_72 = max(0.0, float(model_72h.predict(feat_72)[0]))
else:
    pred_24, pred_48, pred_72 = in_l0 * 0.95, in_l0 * 0.92, in_l0 * 0.90

# --- ENCABEZADO CIUDADANO ---
st.title("🌬️ Alerta Temprana de Calidad del Aire")
st.markdown("### **Estación Parque O'Higgins — Santiago de Chile**")
st.caption(f"ℹ️ {st.session_state.source} | Última medición oficial: **{base_dt.strftime('%d/%m/%Y')}** | Nivel Registrado: **{in_l0:.1f} µg/m³**")

# PESTAÑAS CIUDADANAS
tab1, tab2 = st.tabs([
    "🎯 Pronóstico y Semáforo de Calidad del Aire (24h, 48h, 72h)",
    "📜 Protocolos Normativos y Consejos de Salud (PPDA)"
])

# -------------------------------------------------------------
# PESTAÑA 1: PRONÓSTICOS Y RECOMENDACIONES
# -------------------------------------------------------------
with tab1:
    st.subheader("Pronóstico Preventivo para los Próximos 3 Días")
    st.markdown("Anticípate a los episodios críticos de contaminación ambiental para planificar actividades al aire libre, clases escolares y deportes:")

    col1, col2, col3 = st.columns(3)

    # 24 Horas
    cat_24, icon_24, col_24, rec_24 = clasificar_norma(pred_24)
    delta_24 = pred_24 - in_l0
    with col1:
        st.markdown(f"#### ⏱️ Mañana ({dt_24h.strftime('%d/%m')})")
        st.metric(
            label="Concentración Proyectada",
            value=f"{pred_24:.1f} µg/m³",
            delta=f"{delta_24:+.1f} vs Hoy",
            delta_color="inverse"
        )
        st.markdown(f"**Estado del Aire:** {icon_24} `{cat_24}`")
        st.info(rec_24)

    # 48 Horas
    cat_48, icon_48, col_48, rec_48 = clasificar_norma(pred_48)
    delta_48 = pred_48 - in_l0
    with col2:
        st.markdown(f"#### ⏱️ Pasado Mañana ({dt_48h.strftime('%d/%m')})")
        st.metric(
            label="Concentración Proyectada",
            value=f"{pred_48:.1f} µg/m³",
            delta=f"{delta_48:+.1f} vs Hoy",
            delta_color="inverse"
        )
        st.markdown(f"**Estado del Aire:** {icon_48} `{cat_48}`")
        st.info(rec_48)

    # 72 Horas
    cat_72, icon_72, col_72, rec_72 = clasificar_norma(pred_72)
    delta_72 = pred_72 - in_l0
    with col3:
        st.markdown(f"#### ⏱️ En 3 Días ({dt_72h.strftime('%d/%m')})")
        st.metric(
            label="Concentración Proyectada",
            value=f"{pred_72:.1f} µg/m³",
            delta=f"{delta_72:+.1f} vs Hoy",
            delta_color="inverse"
        )
        st.markdown(f"**Estado del Aire:** {icon_72} `{cat_72}`")
        st.info(rec_72)

    st.markdown("---")
    st.subheader("📈 Evolución de la Calidad del Aire y Zonas de Riesgo")

    df_plot_hist = st.session_state.historico_df.tail(10).copy()
    fechas_hist = [base_dt - timedelta(days=len(df_plot_hist)-1-i) for i in range(len(df_plot_hist))]
    valores_hist = df_plot_hist['MP25'].tolist()

    etiquetas_x = [f.strftime('%d/%m') for f in fechas_hist] + [
        f"{dt_24h.strftime('%d/%m')} (Mañana)",
        f"{dt_48h.strftime('%d/%m')} (Pasado)",
        f"{dt_72h.strftime('%d/%m')} (+3 Días)"
    ]

    serie_hist = valores_hist + [None, None, None]
    serie_pred = [None] * (len(valores_hist) - 1) + [valores_hist[-1], pred_24, pred_48, pred_72]

    fig = go.Figure()

    # Zonas coloreadas del PPDA de fondo
    max_grafico = max(max(valores_hist), pred_24, pred_48, pred_72, 120.0) + 15
    fig.add_hrect(y0=0, y1=50, fillcolor="#28a745", opacity=0.12, line_width=0, annotation_text="Zona Buena (0 - 50)", annotation_position="top left")
    fig.add_hrect(y0=50, y1=80, fillcolor="#ffc107", opacity=0.12, line_width=0, annotation_text="Zona Regular (51 - 79)", annotation_position="top left")
    fig.add_hrect(y0=80, y1=110, fillcolor="#fd7e14", opacity=0.15, line_width=0, annotation_text="Zona Alerta (80 - 109)", annotation_position="top left")
    fig.add_hrect(y0=110, y1=170, fillcolor="#dc3545", opacity=0.18, line_width=0, annotation_text="Zona Pre-Emergencia (110 - 169)", annotation_position="top left")
    if max_grafico > 170:
        fig.add_hrect(y0=170, y1=max_grafico, fillcolor="#6f42c1", opacity=0.20, line_width=0, annotation_text="Emergencia (≥ 170)", annotation_position="top left")

    # Serie histórica
    fig.add_trace(go.Scatter(
        x=etiquetas_x, y=serie_hist,
        mode='lines+markers', name='Registros Observados (SINCA)',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))

    # Serie pronosticada
    fig.add_trace(go.Scatter(
        x=etiquetas_x, y=serie_pred,
        mode='lines+markers', name='Pronóstico Preventivo',
        line=dict(color='#d62728', width=3, dash='dash'),
        marker=dict(size=11, symbol='diamond')
    ))

    fig.update_layout(
        title="Curva de Tendencia y Pronóstico vs. Umbrales Sanitarios Ambientales",
        xaxis_title="Fecha",
        yaxis_title="Concentración MP2.5 (µg/m³)",
        template="plotly_white",
        height=480,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# PESTAÑA 2: PROTOCOLOS NORMATIVOS Y SALUD
# -------------------------------------------------------------
with tab2:
    st.subheader("📜 Niveles de Alerta Sanitaria Ambiental (Norma Chilena)")
    st.markdown("De acuerdo al **Plan de Prevención y Descontaminación Atmosférica (PPDA, D.S. N° 31/2016)**, estos son los niveles oficiales de calidad del aire y las acciones recomendadas:")

    col_izq, col_der = st.columns(2)

    with col_izq:
        st.markdown("""
        ### 🟢 1. Nivel Bueno (0 a 50 µg/m³)
        * **Calidad del aire:** Favorable.
        * **Recomendación:** Actividades normales para toda la población. Condiciones ideales para deportes y ventilación de hogares.

        ---

        ### 🟡 2. Nivel Regular (51 a 79 µg/m³)
        * **Calidad del aire:** Moderada.
        * **Recomendación:** Grupos sensibles (niños, embarazadas, personas con asma o EPOC y adultos mayores) deben evitar esfuerzos físicos intensos y prolongados al aire libre.

        ---

        ### 🟠 3. Nivel Alerta (80 a 109 µg/m³)
        * **Calidad del aire:** Dañina para grupos vulnerables.
        * **Recomendación Escolar:** Modificar o suspender clases de educación física en colegios.
        * **Medidas Comunitarias:** Prohibición de humos visibles y restricción voluntaria del vehículo particular.
        """)

    with col_der:
        st.markdown("""
        ### 🔴 4. Nivel Pre-Emergencia (110 a 169 µg/m³)
        * **Calidad del aire:** Dañina para la salud de toda la población.
        * **Medidas Obligatorias:**
          * Prohibición absoluta de calefactores a leña y derivados en toda la cuenca.
          * Paralización de grandes fuentes industriales estacionarias.
          * Restricción vehicular extendida a vehículos con y sin sello verde.
          * Suspensión total de actividades deportivas escolares y masivas al aire libre.

        ---

        ### 🟣 5. Nivel Emergencia (≥ 170 µg/m³)
        * **Calidad del aire:** Condición crítica extrema.
        * **Medidas Obligatorias:**
          * Máxima restricción vehicular e industrial.
          * Se prohíbe toda actividad física al aire libre.
          * Se aconseja a toda la comunidad permanecer en interiores con ventanas cerradas y purificadores de aire.
        """)

    st.markdown("---")
    st.subheader("💡 Consejos Prácticos para la Ciudadanía")
    st.markdown("""
    1. **Ventilación del Hogar:** Prefiere ventilar tu casa durante las horas de la tarde (14:00 a 17:00 hrs), cuando el viento y la radiación solar dispersan los contaminantes hacia capas altas de la atmósfera.
    2. **Calefacción Limpia:** Evita estufas a leña o parafina en días de Alerta o Pre-emergencia; opta por climatización eléctrica o gas licuado.
    3. **Protección en Desplazamientos:** Si te trasladas en bicicleta o caminas cerca de avenidas de alto tráfico en días fríos matutinos, considera el uso de mascarillas con filtro tipo KN95.
    """)

st.markdown("---")
st.caption("Plataforma de Monitoreo Preventivo MP2.5 | Estación Parque O'Higgins (SINCA MMA) | Universidad Bernardo O'Higgins")
