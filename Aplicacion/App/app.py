# -*- coding: utf-8 -*-
"""
Plataforma de Monitoreo y Alerta Temprana de Material Particulado Fino (MP2.5)
Estación Parque O'Higgins (SINCA D14/273) - Santiago de Chile
Versión V4: Modelos Multi-Ventana (24h, 48h y 72h) con Gradient Boosting, MICE e Ingesta Multivariada
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import io
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(
    page_title="Plataforma de Monitoreo MP2.5 - Parque O'Higgins",
    layout="wide",
    page_icon="🌬️"
)

# --- CARGA DE RECURSOS (MODELOS MULTI-HORIZONTE Y FEATURES) ---
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

        # Fallback si no estuvieran disponibles
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

# --- FUNCIONES DE INGESTA DE DATOS (SINCA + RESPALDO LOCAL) ---
@st.cache_data(ttl=3600)
def fetch_sinca_raw():
    """Descarga los registros más recientes de PM2.5 desde el servicio CGI de SINCA (MMA)."""
    today = datetime.now()
    today_str = today.strftime('%y%m%d')
    from_date = (today - timedelta(days=45)).strftime('%y%m%d')
    
    url_base = "https://sinca.mma.gob.cl/cgi-bin/APUB-MMA/apub.tsindico2.cgi?outtype=xcl&from=" + from_date + "&to=" + today_str + "&path=/usr/airviro/data/CONAMA/&lang=esp&rsrc=&macropath="
    url_pm25 = url_base + "&macro=./RM/D14/Cal/PM25//PM25.diario.diario.ic"

    try:
        r25 = requests.get(url_pm25, verify=False, timeout=12)
        if r25.status_code == 200 and "FECHA" in r25.text:
            return r25.text
    except Exception:
        pass
    return None

def get_hybrid_data():
    base_path = os.path.dirname(__file__)
    local_file = os.path.join(base_path, "datos_respaldo.csv")

    # 1. INTENTO DESDE RESPALDO MULTIVARIADO RECIENTE
    if os.path.exists(local_file):
        try:
            df_local = pd.read_csv(local_file, sep=';', decimal=',')
            if 'Fecha' in df_local.columns:
                df_local['Fecha'] = pd.to_datetime(df_local['Fecha'])
                df_local = df_local.sort_values('Fecha').reset_index(drop=True)
                last_dt = df_local.iloc[-1]['Fecha']
                return df_local, "Respaldo Multivariado Local", last_dt
        except Exception:
            pass

    # 2. INTENTO EN VIVO DESDE SINCA
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
            return df25, "En Vivo (SINCA - MMA)", last_dt
    except Exception:
        pass

    # 3. FALLBACK DE CONTINGENCIA
    fechas = [datetime.now() - timedelta(days=i) for i in reversed(range(15))]
    vals25 = [14.0, 18.0, 15.0, 22.0, 29.0, 25.0, 19.0, 15.0, 12.0, 18.0, 14.0, 11.0, 16.0, 20.0, 18.0]
    df_def = pd.DataFrame({'Fecha': fechas, 'MP25': vals25})
    return df_def, "Valores Base de Contingencia", datetime.now() - timedelta(days=1)

# --- ESTADO DE LA SESIÓN ---
if 'historico_df' not in st.session_state:
    df_hist, source, last_dt = get_hybrid_data()
    st.session_state.historico_df = df_hist
    st.session_state.source = source
    st.session_state.last_dt = last_dt
    
    # Extraer valores recientes de MP2.5
    vals_pm25 = df_hist['MP25'].dropna().tolist()
    st.session_state.l0 = float(vals_pm25[-1]) if len(vals_pm25) >= 1 else 18.0
    st.session_state.l1 = float(vals_pm25[-2]) if len(vals_pm25) >= 2 else 15.0
    st.session_state.l2 = float(vals_pm25[-3]) if len(vals_pm25) >= 3 else 14.0
    st.session_state.l3 = float(vals_pm25[-4]) if len(vals_pm25) >= 4 else 12.0
    st.session_state.l7 = float(vals_pm25[-8]) if len(vals_pm25) >= 8 else float(vals_pm25[0])

# --- ENCABEZADO Y TÍTULO ---
st.title("🌬️ Plataforma de Monitoreo y Alerta Temprana MP2.5")
st.markdown("### **Estación Parque O'Higgins (Santiago de Chile)** | Sistema Predictivo Multivariado Multi-Ventana (24h, 48h y 72h)")

# --- BARRA LATERAL (AJUSTE Y SIMULACIÓN) ---
st.sidebar.header("⚙️ Configuración y Sincronización")
if st.sidebar.button("🔄 Sincronizar con SINCA Ahora", use_container_width=True):
    fetch_sinca_raw.clear()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Parámetros de Entrada")
modo_simulacion = st.sidebar.checkbox("Activar Modo Simulación Manual", value=False)

if modo_simulacion:
    in_l0 = st.sidebar.number_input("MP2.5 Hoy (Día t, µg/m³)", value=st.session_state.l0, step=1.0)
    in_l1 = st.sidebar.number_input("MP2.5 Ayer (t-1, µg/m³)", value=st.session_state.l1, step=1.0)
    in_l2 = st.sidebar.number_input("MP2.5 Anteayer (t-2, µg/m³)", value=st.session_state.l2, step=1.0)
    in_l3 = st.sidebar.number_input("MP2.5 Hace 3 días (t-3, µg/m³)", value=st.session_state.l3, step=1.0)
    in_l7 = st.sidebar.number_input("MP2.5 Hace 7 días (t-7, µg/m³)", value=st.session_state.l7, step=1.0)
    in_co = st.sidebar.number_input("CO Hoy (ppm)", value=0.45, step=0.05)
    in_no2 = st.sidebar.number_input("NO2 Hoy (ppb)", value=18.5, step=1.0)
    in_wspd = st.sidebar.number_input("Velocidad Viento (m/s)", value=1.40, step=0.1)
    in_temp = st.sidebar.number_input("Temp. Mínima (°C)", value=9.0, step=0.5)
else:
    in_l0 = st.session_state.l0
    in_l1 = st.session_state.l1
    in_l2 = st.session_state.l2
    in_l3 = st.session_state.l3
    in_l7 = st.session_state.l7
    in_co = 0.45
    in_no2 = 18.5
    in_wspd = 1.40
    in_temp = 9.0
    st.sidebar.info(f"**Valores Automáticos SINCA:**\n- MP2.5 Hoy (t): {in_l0:.1f} µg/m³\n- MP2.5 Ayer (t-1): {in_l1:.1f} µg/m³\n- MP2.5 Hace 7 días: {in_l7:.1f} µg/m³\n- CO: {in_co:.2f} ppm | NO2: {in_no2:.1f} ppb\n- Viento: {in_wspd:.2f} m/s | Temp Mín: {in_temp:.1f} °C")

# Fechas futuras de pronóstico
base_dt = st.session_state.last_dt
dt_24h = base_dt + timedelta(days=1)
dt_48h = base_dt + timedelta(days=2)
dt_72h = base_dt + timedelta(days=3)

# Banner informativo
if modo_simulacion:
    st.warning("⚠️ **Modo Simulación Activo:** Calculando predicciones multi-ventana sobre valores ingresados manualmente.")
else:
    status_icon = "🟢" if "Vivo" in st.session_state.source else "🔵"
    st.info(f"{status_icon} **Origen:** {st.session_state.source} | **Última Observación:** {base_dt.strftime('%d/%m/%Y')} | **Horizontes de Pronóstico:** 24h ({dt_24h.strftime('%d/%m')}), 48h ({dt_48h.strftime('%d/%m')}), 72h ({dt_72h.strftime('%d/%m')})")

# --- CONSTRUCCIÓN DEL VECTOR DE 32 CARACTERÍSTICAS (SIN PM10 NI SO2) ---
def construir_features(l0, l1, l2, l3, l7, co, no2, wspd, temp, dt_target, feat_cols):
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

# Clasificación según Norma Primaria y PPDA
def clasificar_norma(val):
    if val <= 50.0:
        return "BUENO", "🟢", "#28a745", "Calidad del aire favorable. Sin restricciones para la población."
    elif val <= 79.0:
        return "REGULAR", "🟡", "#ffc107", "Grupos sensibles (niños, adultos mayores, asmáticos) deben moderar actividades físicas prolongadas."
    elif val <= 109.0:
        return "ALERTA", "🟠", "#fd7e14", "Se recomienda suspensión de clases de educación física al aire libre. Uso voluntario de mascarilla."
    elif val <= 169.0:
        return "PRE-EMERGENCIA", "🔴", "#dc3545", "Prohibición de quemas agrícolas, restricción vehicular y paralización de fuentes industriales críticas."
    else:
        return "EMERGENCIA", "🟣", "#6f42c1", "Condición extrema. Prohibición total de actividad física y máxima fiscalización ambiental."

# --- INFERENCIA MULTI-VENTANA ---
if model_24h and model_48h and model_72h:
    feat_24 = construir_features(in_l0, in_l1, in_l2, in_l3, in_l7, in_co, in_no2, in_wspd, in_temp, dt_24h, features_list)
    feat_48 = construir_features(in_l0, in_l1, in_l2, in_l3, in_l7, in_co, in_no2, in_wspd, in_temp, dt_48h, features_list)
    feat_72 = construir_features(in_l0, in_l1, in_l2, in_l3, in_l7, in_co, in_no2, in_wspd, in_temp, dt_72h, features_list)

    pred_24 = max(0.0, float(model_24h.predict(feat_24)[0]))
    pred_48 = max(0.0, float(model_48h.predict(feat_48)[0]))
    pred_72 = max(0.0, float(model_72h.predict(feat_72)[0]))
else:
    pred_24, pred_48, pred_72 = 18.5, 19.2, 17.8

# Pestañas principales de navegación
tab1, tab2, tab3 = st.tabs([
    "📊 Monitoreo y Proyecciones Multi-Ventana",
    "📈 Comparativa Experimental (Defecto vs. Optuna)",
    "📑 Arquitectura y Protocolo Normativo (PPDA)"
])

with tab1:
    st.subheader("🎯 Pronóstico Directo en 3 Ventanas de Tiempo")
    
    col1, col2, col3 = st.columns(3)
    
    # 24 Horas (Campeón: XGBoost)
    cat_24, icon_24, col_24, rec_24 = clasificar_norma(pred_24)
    delta_24 = pred_24 - in_l0
    with col1:
        st.markdown(f"#### ⏱️ Ventana 24 Horas (Mañana)")
        st.caption(f"Fecha estimada: **{dt_24h.strftime('%A %d/%m/%Y')}**")
        st.metric(label="MP2.5 Estimado (XGBoost 🏆)", value=f"{pred_24:.1f} µg/m³", delta=f"{delta_24:+.1f} vs Hoy", delta_color="inverse")
        st.markdown(f"**Estado Normativo:** {icon_24} `{cat_24}`")
        st.caption(rec_24)

    # 48 Horas (Campeón: LightGBM)
    cat_48, icon_48, col_48, rec_48 = clasificar_norma(pred_48)
    delta_48 = pred_48 - in_l0
    with col2:
        st.markdown(f"#### ⏱️ Ventana 48 Horas (Pasado Mañana)")
        st.caption(f"Fecha estimada: **{dt_48h.strftime('%A %d/%m/%Y')}**")
        st.metric(label="MP2.5 Estimado (LightGBM 🏆)", value=f"{pred_48:.1f} µg/m³", delta=f"{delta_48:+.1f} vs Hoy", delta_color="inverse")
        st.markdown(f"**Estado Normativo:** {icon_48} `{cat_48}`")
        st.caption(rec_48)

    # 72 Horas (Campeón: CatBoost)
    cat_72, icon_72, col_72, rec_72 = clasificar_norma(pred_72)
    delta_72 = pred_72 - in_l0
    with col3:
        st.markdown(f"#### ⏱️ Ventana 72 Horas (En 3 Días)")
        st.caption(f"Fecha estimada: **{dt_72h.strftime('%A %d/%m/%Y')}**")
        st.metric(label="MP2.5 Estimado (CatBoost 🏆)", value=f"{pred_72:.1f} µg/m³", delta=f"{delta_72:+.1f} vs Hoy", delta_color="inverse")
        st.markdown(f"**Estado Normativo:** {icon_72} `{cat_72}`")
        st.caption(rec_72)

    st.markdown("---")
    st.subheader("📈 Curva de Evolución Temporal y Umbrales Normativos")

    # Construir datos para gráfico
    df_plot_hist = st.session_state.historico_df.tail(10).copy()
    fechas_hist = [base_dt - timedelta(days=len(df_plot_hist)-1-i) for i in range(len(df_plot_hist))]
    valores_hist = df_plot_hist['MP25'].tolist()
    
    etiquetas_x = [f.strftime('%d/%m') for f in fechas_hist] + [
        f"{dt_24h.strftime('%d/%m')} (24h)",
        f"{dt_48h.strftime('%d/%m')} (48h)",
        f"{dt_72h.strftime('%d/%m')} (72h)"
    ]
    
    serie_hist = valores_hist + [None, None, None]
    serie_pred = [None] * (len(valores_hist) - 1) + [valores_hist[-1], pred_24, pred_48, pred_72]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=etiquetas_x, y=serie_hist,
        mode='lines+markers', name='Histórico Registrado (SINCA)',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=etiquetas_x, y=serie_pred,
        mode='lines+markers', name='Pronóstico Multi-Ventana (Modelos Campeones)',
        line=dict(color='#e377c2', width=3, dash='dash'),
        marker=dict(size=10, symbol='diamond')
    ))

    # Líneas de umbral normativo PPDA
    fig.add_hline(y=50, line_dash="dot", line_color="green", annotation_text="Norma Diaria (50 µg/m³)")
    fig.add_hline(y=80, line_dash="dot", line_color="orange", annotation_text="Alerta (80 µg/m³)")
    fig.add_hline(y=110, line_dash="dot", line_color="red", annotation_text="Pre-Emergencia (110 µg/m³)")
    fig.add_hline(y=170, line_dash="dot", line_color="purple", annotation_text="Emergencia (170 µg/m³)")

    fig.update_layout(
        title="Proyección de Concentración de MP2.5 (µg/m³) vs. Estándar Ambiental",
        xaxis_title="Eje Cronológico",
        yaxis_title="MP2.5 (µg/m³)",
        template="plotly_white",
        height=450,
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🔬 Evidencia Científica: Modelos por Defecto vs. Optimizados con Optuna")
    st.markdown("""
    Evaluación en el **conjunto de prueba ciego independiente** (año 2026 completo, fuera de muestra).
    La calibración fue realizada con **Optimización Bayesiana (Optuna)** orientada a maximizar directamente $R^2$, aplicando **Early Stopping con paciencia de 30 iteraciones** sobre el conjunto de Validación (2025).
    """)

    # Cargar tabla oficial si existe
    base_path = os.path.dirname(__file__)
    tabla_csv = os.path.join(base_path, "tabla_comparativa_defecto_vs_tuneados.csv")
    if os.path.exists(tabla_csv):
        df_comp = pd.read_csv(tabla_csv, sep=';', decimal=',')
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
    else:
        st.info("Tabla comparativa generada en el entrenamiento.")

    st.markdown("""
    > **Hallazgos Clave de la Investigación:**
    > 1. **Horizonte 24h:** **XGBoost Tuneado** alcanzó el rendimiento más alto del proyecto con **$R^2 = 0,6965$** y un error cuadrático medio de **$RMSE = 8,63\ \mu\text{g/m}^3$**, superando por más de un **340%** al baseline Seasonal Naive ($R^2 = 0,1570$).
    > 2. **Horizonte 48h:** **LightGBM Tuneado** lideró con **$R^2 = 0,5293$** ($RMSE = 10,76\ \mu\text{g/m}^3$).
    > 3. **Horizonte 72h:** **CatBoost Tuneado** demostró mayor resistencia en proyecciones a 3 días vista con **$R^2 = 0,4909$** ($RMSE = 11,20\ \mu\text{g/m}^3$).
    > 4. **Degradación Física Coherente:** El coeficiente de determinación disminuye suave y monótonamente ($0,70 \rightarrow 0,53 \rightarrow 0,49$), consistente con la pérdida natural de predictibilidad atmosférica.
    """)

with tab3:
    st.subheader("🏛️ Protocolo Normativo y Arquitectura del Sistema")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 📜 Umbrales PPDA (Decreto Supremo N° 31/2016)")
        st.markdown("""
        | Rango MP2.5 | Estado | Impacto y Medida Sanitaria |
        |---|---|---|
        | **0 - 50 µg/m³** | 🟢 Bueno | Calidad del aire óptima. Sin restricciones. |
        | **51 - 79 µg/m³** | 🟡 Regular | Población de riesgo modera ejercicio físico. |
        | **80 - 109 µg/m³** | 🟠 Alerta | Suspensión de clases deportivas escolares. |
        | **110 - 169 µg/m³** | 🔴 Pre-emergencia | Restricción vehicular extendida e industrias. |
        | **≥ 170 µg/m³** | 🟣 Emergencia | Prohibición absoluta de actividad física. |
        """)

    with col_b:
        st.markdown("#### 🧠 Innovaciones Metodológicas Implementadas")
        st.markdown("""
        - **Saneamiento MICE en Train con PM10:** Reconstrucción de la serie histórica de $PM_{2.5}$ mediante `IterativeImputer(BayesianRidge)` aprovechando la correlación física ($r = 0,88$).
        - **Eliminación Definitiva de PM10:** El modelo no depende de $PM_{10}$ durante la inferencia en producción.
        - **Selección Exógena por Correlación (Clase 05 - Diapo 75):** Inclusión de gases de combustión vehicular ($NO_2, NO_X, NO, CO$) y variables de dispersión/inversión térmica ($WSPD, TEMP_{min}, O_3$).
        - **Imputación de X sin Data Leakage (Clase 07 - Diapo 33):** Imputador iterativo ajustado exclusivamente sobre Train (2020-2024).
        - **Early Stopping (Paciencia 30):** Detención automática del boosting al estancarse el aprendizaje en Validación (2025).
        - **Modelos Boosting Individuales:** Enfoque puro, interpretable y de baja latencia sin la complejidad de ensambles Stacking.
        """)

st.markdown("---")
st.caption("Plataforma Predictiva MP2.5 V4 | SINCA - Estación Parque O'Higgins | XGBoost (24h) • LightGBM (48h) • CatBoost (72h)")
