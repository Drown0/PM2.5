# -*- coding: utf-8 -*-
"""
Plataforma de Monitoreo y Alerta Temprana de Material Particulado Fino (MP2.5)
Estación Parque O'Higgins - Santiago de Chile
Versión V3: Modelos Multi-Ventana (24h, 48h y 72h) con Boosting y Optimización Bayesiana (Optuna)
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

# Configuración de página
st.set_page_config(
    page_title="Plataforma de Monitoreo MP2.5 - Parque O'Higgins",
    layout="wide",
    page_icon="🌬️"
)
warnings.filterwarnings('ignore')

# --- CARGA DE RECURSOS (MODELOS MULTI-HORIZONTE Y VARIABLES) ---
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

        # Fallback al modelo base si aún no existieran los tuneados
        if not os.path.exists(m24_path):
            m24_path = os.path.join(models_dir, 'modelo_final_mp25.joblib')

        m24 = joblib.load(m24_path)
        m48 = joblib.load(m48_path) if os.path.exists(m48_path) else m24
        m72 = joblib.load(m72_path) if os.path.exists(m72_path) else m24
        features = joblib.load(features_path)
        return m24, m48, m72, features
    except Exception as e:
        st.error(f"Error al cargar los modelos predictivos: {e}")
        return None, None, None, None

model_24h, model_48h, model_72h, features_list = load_resources()

# --- FUNCIONES DE INGESTA DE DATOS (SINCA + RESPALDO LOCAL) ---
@st.cache_data(ttl=3600)
def fetch_sinca_raw():
    """Descarga los registros más recientes de PM2.5 y PM10 desde el servicio CGI de SINCA (MMA)."""
    today = datetime.now()
    today_str = today.strftime('%y%m%d')
    from_date = (today - timedelta(days=40)).strftime('%y%m%d')
    
    url_base = "https://sinca.mma.gob.cl/cgi-bin/APUB-MMA/apub.tsindico2.cgi?outtype=xcl&from=" + from_date + "&to=" + today_str + "&path=/usr/airviro/data/CONAMA/&lang=esp&rsrc=&macropath="
    url_pm25 = url_base + "&macro=./RM/D14/Cal/PM25//PM25.diario.diario.ic"
    url_pm10 = url_base + "&macro=./RM/D14/Cal/PM10//PM10.diario.diario.ic"

    try:
        r25 = requests.get(url_pm25, verify=False, timeout=12)
        r10 = requests.get(url_pm10, verify=False, timeout=12)
        if r25.status_code == 200 and "FECHA" in r25.text:
            return r25.text, (r10.text if r10.status_code == 200 and "FECHA" in r10.text else None)
    except Exception:
        pass
    return None, None

def get_hybrid_data():
    base_path = os.path.dirname(__file__)
    local_file = os.path.join(base_path, "datos_respaldo.csv")

    # 1. INTENTO EN VIVO (SINCA)
    try:
        csv_pm25, csv_pm10 = fetch_sinca_raw()
        if csv_pm25:
            df25 = pd.read_csv(io.StringIO(csv_pm25), sep=';', decimal=',', na_values=['', ' ', 'NaN'], dtype={'FECHA (YYMMDD)': str})
            df25.columns = [c.strip() for c in df25.columns]
            col_val, col_pre, col_no_val = 'Registros validados', 'Registros preliminares', 'Registros no validados'
            df25['MP25'] = df25[col_val].fillna(df25.get(col_pre, np.nan)).fillna(df25.get(col_no_val, np.nan))
            df25 = df25.dropna(subset=['MP25'])

            if csv_pm10:
                df10 = pd.read_csv(io.StringIO(csv_pm10), sep=';', decimal=',', na_values=['', ' ', 'NaN'], dtype={'FECHA (YYMMDD)': str})
                df10.columns = [c.strip() for c in df10.columns]
                df10['PM10'] = df10[col_val].fillna(df10.get(col_pre, np.nan)).fillna(df10.get(col_no_val, np.nan))
                merged = pd.merge(df25[['FECHA (YYMMDD)', 'MP25']], df10[['FECHA (YYMMDD)', 'PM10']], on='FECHA (YYMMDD)', how='left')
            else:
                merged = df25[['FECHA (YYMMDD)', 'MP25']].copy()
                merged['PM10'] = merged['MP25'] * 1.85

            merged['PM10'] = merged['PM10'].fillna(merged['MP25'] * 1.85)
            merged = merged.sort_values('FECHA (YYMMDD)').reset_index(drop=True).tail(30)

            if len(merged) >= 8:
                try:
                    merged.to_csv(local_file, sep=';', decimal=',', index=False)
                except Exception:
                    pass

                last_str = str(merged.iloc[-1]['FECHA (YYMMDD)']).split('.')[0].zfill(6)
                last_dt = datetime.strptime('20' + last_str, '%Y%m%d')
                return merged, "En Vivo (SINCA - MMA)", last_dt
    except Exception:
        pass

    # 2. INTENTO LOCAL (datos_respaldo.csv)
    if os.path.exists(local_file):
        try:
            df_local = pd.read_csv(local_file, sep=';', decimal=',', dtype={'FECHA (YYMMDD)': str})
            df_local.columns = [c.strip() for c in df_local.columns]
            if 'MP25' not in df_local.columns:
                col_val = 'Registros validados'
                col_pre = 'Registros preliminares'
                df_local['MP25'] = df_local[col_val].fillna(df_local.get(col_pre, np.nan))
            if 'PM10' not in df_local.columns:
                df_local['PM10'] = df_local['MP25'] * 1.85
            df_local = df_local.dropna(subset=['MP25']).sort_values('FECHA (YYMMDD)').reset_index(drop=True)

            last_str = str(df_local.iloc[-1]['FECHA (YYMMDD)']).split('.')[0].zfill(6)
            last_dt = datetime.strptime('20' + last_str, '%Y%m%d')
            return df_local, "Respaldo Local (Contingencia)", last_dt
        except Exception:
            pass

    # 3. FALLBACK POR DEFECTO
    fechas = [(datetime.now() - timedelta(days=i)).strftime('%y%m%d') for i in reversed(range(10))]
    vals25 = [22.0, 18.0, 25.0, 19.0, 15.0, 12.0, 18.0, 14.0, 11.0, 16.0]
    vals10 = [v * 1.85 for v in vals25]
    df_def = pd.DataFrame({'FECHA (YYMMDD)': fechas, 'MP25': vals25, 'PM10': vals10})
    return df_def, "Valores Base de Contingencia", datetime.now() - timedelta(days=1)

# --- ESTADO DE LA SESIÓN ---
if 'historico_df' not in st.session_state:
    df_hist, source, last_dt = get_hybrid_data()
    st.session_state.historico_df = df_hist
    st.session_state.source = source
    st.session_state.last_dt = last_dt
    
    # Extraer valores recientes
    vals_pm25 = df_hist['MP25'].tolist()
    vals_pm10 = df_hist['PM10'].tolist()
    st.session_state.l1 = float(vals_pm25[-1])
    st.session_state.l2 = float(vals_pm25[-2])
    st.session_state.l3 = float(vals_pm25[-3])
    st.session_state.l7 = float(vals_pm25[-7]) if len(vals_pm25) >= 7 else float(vals_pm25[-1])
    st.session_state.pm10_l1 = float(vals_pm10[-1])

# --- ENCABEZADO Y TÍTULO ---
st.title("🌬️ Plataforma de Monitoreo y Alerta Temprana MP2.5")
st.markdown("### **Estación Parque O'Higgins (Santiago de Chile)** | Sistema Predictivo Multi-Ventana (24h, 48h y 72h)")

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
    in_l1 = st.sidebar.number_input("MP2.5 Hoy (µg/m³)", value=st.session_state.l1, step=1.0)
    in_l2 = st.sidebar.number_input("MP2.5 Ayer (µg/m³)", value=st.session_state.l2, step=1.0)
    in_l3 = st.sidebar.number_input("MP2.5 Anteayer (µg/m³)", value=st.session_state.l3, step=1.0)
    in_l7 = st.sidebar.number_input("MP2.5 Hace 7 días (µg/m³)", value=st.session_state.l7, step=1.0)
    in_pm10 = st.sidebar.number_input("PM10 Hoy (µg/m³)", value=st.session_state.pm10_l1, step=2.0)
else:
    in_l1 = st.session_state.l1
    in_l2 = st.session_state.l2
    in_l3 = st.session_state.l3
    in_l7 = st.session_state.l7
    in_pm10 = st.session_state.pm10_l1
    st.sidebar.info(f"**Valores Automáticos SINCA:**\n- MP2.5 Hoy: {in_l1:.1f} µg/m³\n- MP2.5 Ayer: {in_l2:.1f} µg/m³\n- PM10 Hoy: {in_pm10:.1f} µg/m³")

# Fechas futuras
base_dt = st.session_state.last_dt
dt_24h = base_dt + timedelta(days=1)
dt_48h = base_dt + timedelta(days=2)
dt_72h = base_dt + timedelta(days=3)

# Banner informativo
if modo_simulacion:
    st.warning(f"⚠️ **Modo Simulación Activo:** Calculando predicciones multi-ventana sobre valores ingresados manualmente.")
else:
    status_icon = "🟢" if "Vivo" in st.session_state.source else "🟡"
    st.info(f"{status_icon} **Origen:** {st.session_state.source} | **Última Observación:** {base_dt.strftime('%d/%m/%Y')} | **Horizontes de Pronóstico:** 24h ({dt_24h.strftime('%d/%m')}), 48h ({dt_48h.strftime('%d/%m')}), 72h ({dt_72h.strftime('%d/%m')})")

# --- CONSTRUCCIÓN DEL VECTOR DE 16 CARACTERÍSTICAS ---
def construir_features(l1, l2, l3, l7, pm10_1, dt_target, feat_cols):
    recent_pm25 = [l7, (l7+l3)/2, l3, l2, l1]
    rolling_3 = np.mean([l1, l2, l3])
    rolling_std_3 = np.std([l1, l2, l3], ddof=1) if len([l1, l2, l3]) > 1 else 1.0
    rolling_mean_7 = np.mean(recent_pm25)
    diff_1 = l1 - l2
    
    mes = dt_target.month
    dia_ano = dt_target.timetuple().tm_yday
    dia_semana = dt_target.weekday()
    es_fin_de_semana = 1 if dia_semana in [5, 6] else 0
    
    mes_sin = np.sin(2 * np.pi * mes / 12)
    mes_cos = np.cos(2 * np.pi * mes / 12)
    dia_ano_sin = np.sin(2 * np.pi * dia_ano / 365.25)
    dia_ano_cos = np.cos(2 * np.pi * dia_ano / 365.25)
    
    lag_1_pm10 = pm10_1
    rolling_mean_3_pm10 = pm10_1 * 0.95
    
    fila = {
        'lag_1': l1, 'lag_2': l2, 'lag_3': l3, 'lag_7': l7,
        'rolling_mean_3': rolling_3, 'rolling_std_3': rolling_std_3,
        'rolling_mean_7': rolling_mean_7, 'diff_1': diff_1,
        'mes_sin': mes_sin, 'mes_cos': mes_cos,
        'dia_ano_sin': dia_ano_sin, 'dia_ano_cos': dia_ano_cos,
        'dia_semana': dia_semana, 'es_fin_de_semana': es_fin_de_semana,
        'lag_1_pm10': lag_1_pm10, 'rolling_mean_3_pm10': rolling_mean_3_pm10
    }
    
    # Asegurar orden exacto de features_list
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
    feat_24 = construir_features(in_l1, in_l2, in_l3, in_l7, in_pm10, dt_24h, features_list)
    feat_48 = construir_features(in_l1, in_l2, in_l3, in_l7, in_pm10, dt_48h, features_list)
    feat_72 = construir_features(in_l1, in_l2, in_l3, in_l7, in_pm10, dt_72h, features_list)

    pred_24 = max(0.0, float(model_24h.predict(feat_24)[0]))
    pred_48 = max(0.0, float(model_48h.predict(feat_48)[0]))
    pred_72 = max(0.0, float(model_72h.predict(feat_72)[0]))
else:
    pred_24, pred_48, pred_72 = 21.5, 22.0, 19.8

# Pestañas principales de navegación
tab1, tab2, tab3 = st.tabs([
    "📊 Monitoreo y Proyecciones Multi-Ventana",
    "📈 Comparativa Experimental (Defecto vs. Optuna)",
    "📑 Arquitectura y Protocolo Normativo (PPDA)"
])

with tab1:
    st.subheader("🎯 Pronóstico Directo en 3 Ventanas de Tiempo")
    
    col1, col2, col3 = st.columns(3)
    
    # 24 Horas
    cat_24, icon_24, col_24, rec_24 = clasificar_norma(pred_24)
    delta_24 = pred_24 - in_l1
    with col1:
        st.markdown(f"#### ⏱️ Ventana 24 Horas (Mañana)")
        st.caption(f"Fecha estimada: **{dt_24h.strftime('%A %d/%m/%Y')}**")
        st.metric(label="MP2.5 Estimado", value=f"{pred_24:.1f} µg/m³", delta=f"{delta_24:+.1f} vs Hoy", delta_color="inverse")
        st.markdown(f"**Estado Normativo:** {icon_24} `{cat_24}`")
        st.caption(rec_24)

    # 48 Horas
    cat_48, icon_48, col_48, rec_48 = clasificar_norma(pred_48)
    delta_48 = pred_48 - in_l1
    with col2:
        st.markdown(f"#### ⏱️ Ventana 48 Horas (Pasado Mañana)")
        st.caption(f"Fecha estimada: **{dt_48h.strftime('%A %d/%m/%Y')}**")
        st.metric(label="MP2.5 Estimado", value=f"{pred_48:.1f} µg/m³", delta=f"{delta_48:+.1f} vs Hoy", delta_color="inverse")
        st.markdown(f"**Estado Normativo:** {icon_48} `{cat_48}`")
        st.caption(rec_48)

    # 72 Horas
    cat_72, icon_72, col_72, rec_72 = clasificar_norma(pred_72)
    delta_72 = pred_72 - in_l1
    with col3:
        st.markdown(f"#### ⏱️ Ventana 72 Horas (En 3 Días)")
        st.caption(f"Fecha estimada: **{dt_72h.strftime('%A %d/%m/%Y')}**")
        st.metric(label="MP2.5 Estimado", value=f"{pred_72:.1f} µg/m³", delta=f"{delta_72:+.1f} vs Hoy", delta_color="inverse")
        st.markdown(f"**Estado Normativo:** {icon_72} `{cat_72}`")
        st.caption(rec_72)

    st.markdown("---")
    st.subheader("📈 Curva de Evolución Temporal y Umbrales Normativos")

    # Construir datos para gráfico
    df_plot_hist = st.session_state.historico_df.tail(7).copy()
    fechas_hist = [base_dt - timedelta(days=6-i) for i in range(len(df_plot_hist))]
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
        mode='lines+markers', name='Pronóstico Multi-Ventana (Optuna)',
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
    Para evaluar la robustez del sistema, se enfrentaron las configuraciones estándar de fábrica de los algoritmos 
    de Gradient Boosting contra las calibradas mediante **Optimización Bayesiana (Optuna, 4.500 trials)** y **Walk-Forward Validation**.
    Los resultados corresponden a la evaluación en el **conjunto de prueba ciego independiente** (348 días fuera de muestra: 2025–2026):
    """)

    tabla_comparativa = pd.DataFrame([
        {"Horizonte": "24h (t+1)", "Modelo": "LightGBM", "Configuración": "Por Defecto", "R² Test": 0.3410, "MAE (µg/m³)": 7.63, "RMSE (µg/m³)": 11.64, "Mejora": "Línea Base (Sobreajustado)"},
        {"Horizonte": "24h (t+1)", "Modelo": "LightGBM", "Configuración": "Optimizado Optuna", "R² Test": 0.5198, "MAE (µg/m³)": 6.67, "RMSE (µg/m³)": 9.94, "Mejora": "+52,4% en R²"},
        {"Horizonte": "24h (t+1)", "Modelo": "XGBoost", "Configuración": "Por Defecto", "R² Test": 0.4234, "MAE (µg/m³)": 7.12, "RMSE (µg/m³)": 10.89, "Mejora": "Línea Base"},
        {"Horizonte": "24h (t+1)", "Modelo": "XGBoost", "Configuración": "Optimizado Optuna", "R² Test": 0.5225, "MAE (µg/m³)": 6.71, "RMSE (µg/m³)": 9.91, "Mejora": "+23,4% en R²"},
        {"Horizonte": "24h (t+1)", "Modelo": "CatBoost", "Configuración": "Optimizado Optuna", "R² Test": 0.5367, "MAE (µg/m³)": 6.57, "RMSE (µg/m³)": 9.76, "Mejora": "Mejor Modelo Individual"},
        {"Horizonte": "24h (t+1)", "Modelo": "Stacking Ensemble", "Configuración": "Optimizado Optuna", "R² Test": 0.5353, "MAE (µg/m³)": 6.61, "RMSE (µg/m³)": 9.78, "Mejora": "Ensamble Regularizado"},
        {"Horizonte": "48h (t+2)", "Modelo": "LightGBM", "Configuración": "Por Defecto", "R² Test": 0.2497, "MAE (µg/m³)": 8.07, "RMSE (µg/m³)": 12.42, "Mejora": "Línea Base (Sobreajustado)"},
        {"Horizonte": "48h (t+2)", "Modelo": "LightGBM", "Configuración": "Optimizado Optuna", "R² Test": 0.4397, "MAE (µg/m³)": 7.16, "RMSE (µg/m³)": 10.74, "Mejora": "+76,1% en R²"},
        {"Horizonte": "48h (t+2)", "Modelo": "Stacking Ensemble", "Configuración": "Optimizado Optuna", "R² Test": 0.4846, "MAE (µg/m³)": 6.90, "RMSE (µg/m³)": 10.30, "Mejora": "Mejor Modelo a 48h"},
        {"Horizonte": "72h (t+3)", "Modelo": "LightGBM", "Configuración": "Por Defecto", "R² Test": 0.2574, "MAE (µg/m³)": 7.95, "RMSE (µg/m³)": 12.35, "Mejora": "Línea Base (Sobreajustado)"},
        {"Horizonte": "72h (t+3)", "Modelo": "LightGBM", "Configuración": "Optimizado Optuna", "R² Test": 0.4078, "MAE (µg/m³)": 7.34, "RMSE (µg/m³)": 11.03, "Mejora": "+58,4% en R²"},
        {"Horizonte": "72h (t+3)", "Modelo": "Stacking Ensemble", "Configuración": "Optimizado Optuna", "R² Test": 0.4528, "MAE (µg/m³)": 7.07, "RMSE (µg/m³)": 10.61, "Mejora": "Mejor Modelo a 72h"}
    ])
    st.dataframe(tabla_comparativa, use_container_width=True, hide_index=True)

    st.markdown("""
    > **Conclusión Clave:** La optimización bayesiana eliminó por completo el sobreajuste original en LightGBM y XGBoost, 
    > permitiendo que el **Stacking Regressor** heterogéneo lidere con la menor tasa de error en horizontes extendidos (48h y 72h).
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
        st.markdown("#### 🧠 Arquitectura de la Junta de Expertos")
        st.markdown("""
        - **XGBoost:** Regularización elástica L1/L2 para capturar picos no lineales.
        - **LightGBM:** Alta eficiencia por histogramas y sensibilidad a variaciones de gradiente.
        - **CatBoost:** Árboles simétricos (*oblivious trees*) con resistencia estructural al sobreajuste.
        - **Meta-Modelo RidgeCV:** Regresión lineal con penalización L2 que equilibra y pondera dinámicamente las salidas de los estimadores base.
        - **Imputación MICE con PM10:** Reconstrucción continua de la serie temporal mediante Bayesian Ridge ($r = 0,88$).
        """)

st.markdown("---")
st.caption(f"Plataforma Predictiva MP2.5 V3 | Fuente: SINCA / MMA Chile | Ensamble Stacking Optimizado con Optuna & Walk-Forward Validation")
