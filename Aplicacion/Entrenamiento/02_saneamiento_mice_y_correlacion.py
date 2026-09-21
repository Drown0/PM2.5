"""
Saneamiento MICE, Selección por Correlación (Clase 05) e Imputación de X (Clase 07)
Proyecto: Sistema Predictivo Multivariado PM2.5 - Estación Parque O'Higgins (SINCA D14)

Objetivos:
1. Reconstruir lagunas de MP2.5 en TRAIN (2020-2024) con MICE (IterativeImputer) usando PM10.
2. Eliminar definitivamente PM10 para que los modelos predictivos NO dependan de él.
3. Generar Matriz y Gráficos de Correlación según Clase 05 - diapo 75 para respaldar la selección de variables.
4. Imputar variables explicativas (X) con IterativeImputer según Clase 07 - diapo 33 (Fit solo en Train, Transform en Val y Test para evitar Data Leakage).
5. Construir matriz de características autoregresivas, exógenas y multi-horizonte (24h, 48h, 72h).
6. Exportar particiones anuales estrictas:
   - Train: 2020 a 2024
   - Validation: 2025
   - Test: 2026 (hasta la fecha actual)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

warnings.filterwarnings('ignore')

# Configurar estilo visual profesional para gráficos académicos
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300
})


def ejecutar_saneamiento_y_correlacion():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datos_dir = os.path.join(base_dir, "datos")
    raw_path = os.path.join(datos_dir, "sinca_multivariado_2020_presente.csv")
    
    doc_graficos_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "..", "Documentos", "Version 2 Experimentacion y Modelos", "graficos"))
    os.makedirs(doc_graficos_dir, exist_ok=True)
    
    print("=" * 80)
    print("PASO 2: SANEAMIENTO MICE, CORRELACION (CLASE 05) E IMPUTACION X (CLASE 07)")
    print("=" * 80)
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"No se encontró el archivo de datos sinópticos: {raw_path}")
    
    # 1. Cargar datos multivariados
    df = pd.read_csv(raw_path, sep=';', decimal=',')
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.sort_values('Fecha').reset_index(drop=True)
    
    print(f"[CARGA] Total registros: {len(df):,} días ({df['Fecha'].min().strftime('%Y-%m-%d')} a {df['Fecha'].max().strftime('%Y-%m-%d')})")
    
    # Identificar particiones anuales requeridas
    # Train: 2020-01-01 a 2024-12-31
    # Val:   2025-01-01 a 2025-12-31
    # Test:  2026-01-01 a la fecha actual
    idx_train = df[(df['Fecha'] >= '2020-01-01') & (df['Fecha'] <= '2024-12-31')].index
    idx_val = df[(df['Fecha'] >= '2025-01-01') & (df['Fecha'] <= '2025-12-31')].index
    idx_test = df[df['Fecha'] >= '2026-01-01'].index
    
    print(f"  - Train      (2020-2024): {len(idx_train):,} días")
    print(f"  - Validation (2025):      {len(idx_val):,} días")
    print(f"  - Test       (2026):      {len(idx_test):,} días")
    
    # 2. Análisis de Correlación (Clase 05 - diapo 75)
    # Seleccionar variables químicas y meteorológicas
    cols_analisis = [
        'MP25', 'PM10', 'CO', 'NOX', 'NO2', 'NO', 'O3',
        'TEMP_mean', 'TEMP_min', 'TEMP_max', 'RHUM_mean', 'WSPD_mean', 'WSPD_max'
    ]
    
    # Filtrar solo columnas presentes
    cols_analisis = [c for c in cols_analisis if c in df.columns]
    
    # Calcular correlaciones de Pearson y Spearman
    corr_pearson = df[cols_analisis].corr(method='pearson')
    corr_spearman = df[cols_analisis].corr(method='spearman')
    
    print("\n" + "-" * 80)
    print("CORRELACIONES CON MP2.5 (CLASE 05 - DIAPO 75):")
    corr_mp25_p = corr_pearson['MP25'].drop('MP25').sort_values(ascending=False)
    corr_mp25_s = corr_spearman['MP25'].drop('MP25').sort_values(ascending=False)
    
    df_corrs = pd.DataFrame({
        'Pearson (r)': corr_mp25_p,
        'Spearman (rho)': corr_mp25_s
    })
    for var, row in df_corrs.iterrows():
        print(f"  * {var:<12}: Pearson = {row['Pearson (r)']:>+6.3f} | Spearman = {row['Spearman (rho)']:>+6.3f}")
    
    # Generar Gráfico de Correlación de Alta Calidad (Heatmap + Barplot)
    fig, (ax_heat, ax_bar) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1.2, 1]})
    
    # Heatmap
    mask = np.triu(np.ones_like(corr_pearson, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        corr_pearson, mask=mask, cmap=cmap, vmin=-1.0, vmax=1.0, center=0,
        annot=True, fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": .8},
        ax=ax_heat
    )
    ax_heat.set_title("Matriz de Correlación de Pearson - Estación Parque O'Higgins (D14)\n(Variables Químicas y Meteorológicas SINCA)", fontsize=13, fontweight='bold', pad=12)
    
    # Barplot de correlaciones con MP2.5
    bar_data = corr_mp25_p.reset_index()
    bar_data.columns = ['Variable', 'Correlacion']
    colors = ['#d73027' if c > 0.6 else '#fc8d59' if c > 0 else '#4575b4' for c in bar_data['Correlacion']]
    
    bars = ax_bar.barh(bar_data['Variable'], bar_data['Correlacion'], color=colors, edgecolor='black', height=0.65)
    ax_bar.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax_bar.axvline(0.5, color='gray', linewidth=0.7, linestyle=':', label='Umbral Relevancia (|r| ≥ 0.50)')
    ax_bar.axvline(-0.5, color='gray', linewidth=0.7, linestyle=':')
    ax_bar.set_xlim(-0.8, 1.0)
    ax_bar.set_xlabel("Coeficiente de Correlación de Pearson (r) con MP2.5", fontweight='bold')
    ax_bar.set_title("Fuerza de Asociación con MP2.5 (Clase 05 - Diapo 75)\n'Variables con correlación confirmada para modelado'", fontsize=13, fontweight='bold', pad=12)
    ax_bar.legend(loc='lower right')
    
    for bar in bars:
        width = bar.get_width()
        pos_x = width + 0.02 if width >= 0 else width - 0.08
        ax_bar.text(pos_x, bar.get_y() + bar.get_height()/2, f"{width:+.2f}", va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    corr_img_path = os.path.join(doc_graficos_dir, "matriz_correlacion_clase05.png")
    plt.savefig(corr_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[GRAFICO] Gráfico de correlación guardado en: {corr_img_path}")
    
    # 3. RECONSTRUCCIÓN DE TARGET MP2.5 EN TRAIN CON PM10 VÍA MICE
    print("\n" + "-" * 80)
    print("RECONSTRUCCIÓN DE TARGET MP2.5 MEDIANTE MICE EN TRAIN (2020-2024):")
    
    # Covariables temporales para el MICE del target
    df['mes'] = df['Fecha'].dt.month
    df['dia_ano'] = df['Fecha'].dt.dayofyear
    df['dia_semana'] = df['Fecha'].dt.dayofweek
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    df['dia_ano_sin'] = np.sin(2 * np.pi * df['dia_ano'] / 365.25)
    df['dia_ano_cos'] = np.cos(2 * np.pi * df['dia_ano'] / 365.25)
    df['lag_1_mp25'] = df['MP25'].shift(1)
    df['lead_1_mp25'] = df['MP25'].shift(-1)
    
    features_target_mice = [
        'MP25', 'PM10', 'lag_1_mp25', 'lead_1_mp25',
        'mes_sin', 'mes_cos', 'dia_ano_sin', 'dia_ano_cos', 'dia_semana'
    ]
    
    df_train_mice = df.loc[idx_train, features_target_mice].copy()
    nulos_train_mp25 = df_train_mice['MP25'].isna().sum()
    print(f"  - Registros nulos de MP2.5 en Train: {nulos_train_mp25} ({nulos_train_mp25/len(idx_train)*100:.2f}%)")
    
    # Ajustar IterativeImputer SOLO con datos de Train para reconstruir MP2.5
    imputer_target = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=30,
        random_state=42,
        min_value=0.0
    )
    matriz_target_imp = imputer_target.fit_transform(df_train_mice)
    
    # Asignar valores reconstruidos
    df['MP25_Original'] = df['MP25']
    df['MP25_Reconstruido'] = df['MP25']
    df.loc[idx_train, 'MP25_Reconstruido'] = matriz_target_imp[:, 0]
    df['Fue_Imputado_Target'] = 0
    df.loc[idx_train, 'Fue_Imputado_Target'] = df.loc[idx_train, 'MP25_Original'].isna().astype(int)
    
    # En Train usamos la serie continua reconstruida
    df.loc[idx_train, 'MP25'] = df.loc[idx_train, 'MP25_Reconstruido']
    print(f"  [OK] Reconstrucción de Train finalizada. Nulos remanentes en Train MP2.5: {df.loc[idx_train, 'MP25'].isna().sum()}")
    
    # Para Val y Test: si bien interpolamos para rezagos autoregresivos continuos,
    # guardamos MP25_Real para asegurar que la evaluación final sea sobre datos reales del sensor
    df['MP25_Real'] = df['MP25_Original']
    
    # Llenar huecos puntuales en Val y Test SOLO para alimentar los lags autoregresivos (no para el target de evaluación)
    df['MP25_Continuo'] = df['MP25'].ffill().bfill()
    
    # 4. ELIMINACIÓN DEFINITIVA DE PM10 Y SO2
    print("\n" + "-" * 80)
    print("ELIMINACIÓN DE PM10 Y SO2 DEL CONJUNTO PREDICTIVO:")
    print("  * 'PM10' se elimina completamente del pipeline para evitar dependencia en inferencia futura.")
    print("  * 'SO2' se elimina por carecer de registros válidos en la estación Parque O'Higgins (100% nulo).")
    df = df.drop(columns=['PM10', 'SO2'], errors='ignore')
    
    # 5. SELECCIÓN DE VARIABLES EXÓGENAS (CLASE 05)
    # Variables retenidas por correlación confirmada:
    # Químicas: CO, NOX, NO2, NO, O3
    # Meteorológicas: TEMP_mean, TEMP_min, TEMP_max, RHUM_mean, WSPD_mean, WSPD_max
    exog_vars = [
        'CO', 'NOX', 'NO2', 'NO', 'O3',
        'TEMP_mean', 'TEMP_min', 'TEMP_max', 'RHUM_mean', 'WSPD_mean', 'WSPD_max'
    ]
    
    # 6. IMPUTACIÓN DE VARIABLES EXPLICATIVAS (X) CON ITERATIVE IMPUTER (CLASE 07 - DIAPO 33)
    # Regla de Oro: Fit SOLO en Train, transform en Val y Test (Evitar Data Leakage)
    print("\n" + "-" * 80)
    print("IMPUTACIÓN DE VARIABLES EXÓGENAS (X) CON ITERATIVE IMPUTER (CLASE 07 - DIAPO 33):")
    print(f"  * Variables a imputar: {exog_vars}")
    print(f"  * Nulos en Train antes de imputar:\n{df.loc[idx_train, exog_vars].isna().sum().to_dict()}")
    
    imputer_x = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=25,
        random_state=42,
        min_value=0.0
    )
    
    # Fit y transform SOLO en Train
    X_train_exog_imp = imputer_x.fit_transform(df.loc[idx_train, exog_vars])
    X_val_exog_imp   = imputer_x.transform(df.loc[idx_val, exog_vars])
    X_test_exog_imp  = imputer_x.transform(df.loc[idx_test, exog_vars])
    
    # Asignar variables imputadas sin data leakage
    df.loc[idx_train, exog_vars] = X_train_exog_imp
    df.loc[idx_val, exog_vars]   = X_val_exog_imp
    df.loc[idx_test, exog_vars]  = X_test_exog_imp
    
    print(f"  [OK] Imputación de X completada sin Data Leakage.")
    print(f"  * Nulos totales remanentes en variables exógenas: {df[exog_vars].isna().sum().sum()}")
    
    # 7. CONSTRUCCIÓN DE CARACTERÍSTICAS (FEATURE ENGINEERING)
    print("\n" + "-" * 80)
    print("CONSTRUCCIÓN DE CARACTERÍSTICAS TEMPORALES, AUTOREGRESIVAS Y METEOROLÓGICAS:")
    
    # Concentración observada en el día t (origen del pronóstico)
    df['MP25_t'] = df['MP25_Continuo']
    
    # Lags del contaminante objetivo (MP25) respecto al día t
    df['MP25_lag1'] = df['MP25_Continuo'].shift(1)
    df['MP25_lag2'] = df['MP25_Continuo'].shift(2)
    df['MP25_lag3'] = df['MP25_Continuo'].shift(3)
    df['MP25_lag7'] = df['MP25_Continuo'].shift(7)
    
    # Medias móviles y dispersión hasta el día t
    df['MP25_roll_mean_3'] = df['MP25_Continuo'].rolling(window=3, min_periods=1).mean()
    df['MP25_roll_mean_7'] = df['MP25_Continuo'].rolling(window=7, min_periods=1).mean()
    df['MP25_roll_std_7']  = df['MP25_Continuo'].rolling(window=7, min_periods=1).std().fillna(0)
    df['MP25_roll_min_7']  = df['MP25_Continuo'].rolling(window=7, min_periods=1).min()
    df['MP25_roll_max_7']  = df['MP25_Continuo'].rolling(window=7, min_periods=1).max()
    
    # Momentum (Tendencia inmediata entre hoy día t y ayer t-1)
    df['MP25_diff1'] = df['MP25_t'] - df['MP25_lag1']
    
    # Lags de variables exógenas relevantes conocidas al momento del pronóstico (día t)
    df['CO_lag1']        = df['CO'].shift(1)
    df['NO2_lag1']       = df['NO2'].shift(1)
    df['WSPD_mean_lag1'] = df['WSPD_mean'].shift(1)
    df['TEMP_min_lag1']  = df['TEMP_min'].shift(1)
    
    # Calendario y factores cíclicos
    df['es_fin_de_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
    
    # Construcción de Targets Directos Multi-Horizonte (24h, 48h, 72h)
    # Pronóstico a 24h: MP25 en t+1
    # Pronóstico a 48h: MP25 en t+2
    # Pronóstico a 72h: MP25 en t+3
    df['target_24h'] = df['MP25_Continuo'].shift(-1)
    df['target_48h'] = df['MP25_Continuo'].shift(-2)
    df['target_72h'] = df['MP25_Continuo'].shift(-3)
    
    # Targets reales del sensor (para validación y test estricto sobre datos reales)
    df['target_24h_real'] = df['MP25_Real'].shift(-1)
    df['target_48h_real'] = df['MP25_Real'].shift(-2)
    df['target_72h_real'] = df['MP25_Real'].shift(-3)
    
    # Eliminar primeros 7 registros por desfase de lags iniciales
    df_clean = df.iloc[7:].copy().reset_index(drop=True)
    
    # Actualizar índices de particiones
    df_train = df_clean[(df_clean['Fecha'] >= '2020-01-08') & (df_clean['Fecha'] <= '2024-12-31')].reset_index(drop=True)
    df_val   = df_clean[(df_clean['Fecha'] >= '2025-01-01') & (df_clean['Fecha'] <= '2025-12-31')].reset_index(drop=True)
    df_test  = df_clean[df_clean['Fecha'] >= '2026-01-01'].reset_index(drop=True)
    
    # Guardar particiones listas para modelado
    path_train = os.path.join(datos_dir, "dataset_train.csv")
    path_val   = os.path.join(datos_dir, "dataset_val.csv")
    path_test  = os.path.join(datos_dir, "dataset_test.csv")
    path_full  = os.path.join(datos_dir, "dataset_preparado_completo.csv")
    
    df_train.to_csv(path_train, sep=';', decimal=',', index=False)
    df_val.to_csv(path_val, sep=';', decimal=',', index=False)
    df_test.to_csv(path_test, sep=';', decimal=',', index=False)
    df_clean.to_csv(path_full, sep=';', decimal=',', index=False)
    
    print("\n" + "=" * 80)
    print("DATASETS PROCESADOS Y PARTICIONADOS EXITOSAMENTE:")
    print(f"  * Train (2020-2024):       {len(df_train):,} filas | Guardado en: {path_train}")
    print(f"  * Validation (2025):       {len(df_val):,} filas   | Guardado en: {path_val}")
    print(f"  * Test (2026-actual):      {len(df_test):,} filas  | Guardado en: {path_test}")
    print(f"  * Completo (2020-2026):    {len(df_clean):,} filas | Guardado en: {path_full}")
    print(f"  * Total columnas:          {df_clean.shape[1]}")
    print("=" * 80)
    
    # Listado de variables de entrada para los modelos
    feature_cols = [
        'MP25_t', 'MP25_lag1', 'MP25_lag2', 'MP25_lag3', 'MP25_lag7',
        'MP25_roll_mean_3', 'MP25_roll_mean_7', 'MP25_roll_std_7', 'MP25_roll_min_7', 'MP25_roll_max_7',
        'MP25_diff1',
        'CO', 'NOX', 'NO2', 'NO', 'O3',
        'TEMP_mean', 'TEMP_min', 'TEMP_max', 'RHUM_mean', 'WSPD_mean', 'WSPD_max',
        'CO_lag1', 'NO2_lag1', 'WSPD_mean_lag1', 'TEMP_min_lag1',
        'mes_sin', 'mes_cos', 'dia_ano_sin', 'dia_ano_cos', 'dia_semana', 'es_fin_de_semana'
    ]
    print(f"\n[FEATURES] Total variables explicativas (X): {len(feature_cols)}")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:>2}. {col}")


if __name__ == "__main__":
    ejecutar_saneamiento_y_correlacion()
