"""
Evaluacion y Benchmarking de Modelos de Boosting por Defecto
Proyecto: Sistema Predictivo de Calidad del Aire (PM2.5) - Estacion Parque O'Higgins
Modelos evaluados: LightGBM, XGBoost, CatBoost y Stacking Ensemble (Parametros por Defecto)
Horizontes temporales: 24h, 48h y 72h
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

def calcular_mape(y_true, y_pred):
    # Proteccion contra division por cero en concentraciones muy bajas (< 1.0 ug/m3)
    denom = np.where(y_true < 1.0, 1.0, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

def ejecutar_benchmark():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datos_dir = os.path.join(base_dir, "datos")
    data_path = os.path.join(datos_dir, "sinca_pm25_2020_presente_imputado.csv")
    
    docs_dir = r"C:\Users\Usuario\Documents\Personal\Universidad\Semestres\8° Semestre\Titulacion\Documentos\Comparación de datos"
    graficos_dir = os.path.join(docs_dir, "graficos")
    os.makedirs(graficos_dir, exist_ok=True)
    
    print("=" * 70)
    print("INICIANDO BENCHMARKING DE MODELOS BOOSTING POR DEFECTO")
    print("=" * 70)
    
    # 1. Cargar datos imputados limpios
    df = pd.read_csv(data_path, sep=';', decimal=',')
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.sort_values('Fecha').reset_index(drop=True)
    print(f"[DATOS] Dataset cargado: {len(df)} dias ({df['Fecha'].min().date()} al {df['Fecha'].max().date()})")
    
    # 2. Ingenieria de Caracteristicas
    df['lag_1'] = df['MP25'].shift(1)
    df['lag_2'] = df['MP25'].shift(2)
    df['lag_3'] = df['MP25'].shift(3)
    df['lag_7'] = df['MP25'].shift(7)
    
    df['rolling_mean_3'] = df['MP25'].shift(1).rolling(window=3).mean()
    df['rolling_std_3'] = df['MP25'].shift(1).rolling(window=3, min_periods=2).std(ddof=1)
    df['rolling_mean_7'] = df['MP25'].shift(1).rolling(window=7).mean()
    
    df['diff_1'] = df['lag_1'] - df['lag_2']
    
    df['mes'] = df['Fecha'].dt.month
    df['dia_ano'] = df['Fecha'].dt.dayofyear
    df['dia_semana'] = df['Fecha'].dt.dayofweek
    df['es_fin_de_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
    
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    df['dia_ano_sin'] = np.sin(2 * np.pi * df['dia_ano'] / 365.25)
    df['dia_ano_cos'] = np.cos(2 * np.pi * df['dia_ano'] / 365.25)
    
    # Covariable fisica PM10 (disponible en la serie)
    if 'PM10' in df.columns:
        df['lag_1_pm10'] = df['PM10'].shift(1)
        df['rolling_mean_3_pm10'] = df['PM10'].shift(1).rolling(window=3).mean()
    
    # Targets para 24h (t+1), 48h (t+2) y 72h (t+3)
    df['target_24h'] = df['MP25'].shift(-1)
    df['target_48h'] = df['MP25'].shift(-2)
    df['target_72h'] = df['MP25'].shift(-3)
    
    # Limpiar nulos inducidos por lags iniciales y shifts finales
    df_clean = df.dropna().reset_index(drop=True)
    print(f"[FEATURE ENGINEERING] Registros efectivos para modelado: {len(df_clean)}")
    
    feature_cols = [
        'lag_1', 'lag_2', 'lag_3', 'lag_7',
        'rolling_mean_3', 'rolling_std_3', 'rolling_mean_7',
        'diff_1', 'mes_sin', 'mes_cos', 'dia_ano_sin', 'dia_ano_cos',
        'dia_semana', 'es_fin_de_semana'
    ]
    if 'lag_1_pm10' in df_clean.columns:
        feature_cols.extend(['lag_1_pm10', 'rolling_mean_3_pm10'])
        
    X = df_clean[feature_cols]
    fechas = df_clean['Fecha']
    
    # 3. Particion Cronologica Temporal: 70% Train, 15% Val, 15% Test
    n = len(df_clean)
    idx_train = int(n * 0.70)
    idx_val = int(n * 0.85)
    
    split_info = {
        'Train': (0, idx_train, fechas.iloc[0].strftime('%d/%m/%Y'), fechas.iloc[idx_train-1].strftime('%d/%m/%Y')),
        'Val': (idx_train, idx_val, fechas.iloc[idx_train].strftime('%d/%m/%Y'), fechas.iloc[idx_val-1].strftime('%d/%m/%Y')),
        'Test': (idx_val, n, fechas.iloc[idx_val].strftime('%d/%m/%Y'), fechas.iloc[n-1].strftime('%d/%m/%Y'))
    }
    
    print("\n" + "-" * 70)
    print("PARTICION TEMPORAL CRONOLOGICA:")
    for split_name, (s, e, d_start, d_end) in split_info.items():
        print(f"  - {split_name:<10}: {e - s:>4} dias ({d_start} al {d_end}) [{((e-s)/n)*100:.1f}%]")
    print("-" * 70)
    
    horizontes = {
        '24h (t+1)': 'target_24h',
        '48h (t+2)': 'target_48h',
        '72h (t+3)': 'target_72h'
    }
    
    resultados = []
    predicciones_guardadas = {}
    
    for h_label, h_target in horizontes.items():
        print(f"\n>>> ENTRENANDO HORIZONTE: {h_label} <<<")
        y = df_clean[h_target]
        
        X_train, y_train = X.iloc[:idx_train], y.iloc[:idx_train]
        X_val, y_val = X.iloc[idx_train:idx_val], y.iloc[idx_train:idx_val]
        X_test, y_test = X.iloc[idx_val:], y.iloc[idx_val:]
        
        modelos_default = {
            'LightGBM': LGBMRegressor(n_estimators=200, learning_rate=0.05, verbosity=-1, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
            'CatBoost': CatBoostRegressor(n_estimators=200, learning_rate=0.05, depth=5, silent=True, random_state=42)
        }
        
        # Entrenar estimadores base
        trained_base = {}
        for m_name, model in modelos_default.items():
            model.fit(X_train, y_train)
            trained_base[m_name] = model
            
            # Predicciones
            pred_train = model.predict(X_train)
            pred_val = model.predict(X_val)
            pred_test = model.predict(X_test)
            
            for set_name, y_true, y_pred in [('Train', y_train, pred_train), ('Val', y_val, pred_val), ('Test', y_test, pred_test)]:
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                mape = calcular_mape(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                
                resultados.append({
                    'Horizonte': h_label,
                    'Modelo': m_name,
                    'Particion': set_name,
                    'RMSE': rmse,
                    'MAE': mae,
                    'MAPE': mape,
                    'R2': r2
                })
            
            if h_label == '24h (t+1)':
                predicciones_guardadas[f"{m_name}_train"] = pred_train
                predicciones_guardadas[f"{m_name}_test"] = pred_test
                
        # Stacking Ensemble
        stack_est = [
            ('lgb', LGBMRegressor(n_estimators=200, learning_rate=0.05, verbosity=-1, random_state=42)),
            ('xgb', XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)),
            ('cat', CatBoostRegressor(n_estimators=200, learning_rate=0.05, depth=5, silent=True, random_state=42))
        ]
        stack = StackingRegressor(estimators=stack_est, final_estimator=RidgeCV(), cv=KFold(5, shuffle=False))
        stack.fit(X_train, y_train)
        
        pred_stk_train = stack.predict(X_train)
        pred_stk_val = stack.predict(X_val)
        pred_stk_test = stack.predict(X_test)
        
        for set_name, y_true, y_pred in [('Train', y_train, pred_stk_train), ('Val', y_val, pred_stk_val), ('Test', y_test, pred_stk_test)]:
            resultados.append({
                'Horizonte': h_label,
                'Modelo': 'Stacking Ensemble',
                'Particion': set_name,
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'MAE': mean_absolute_error(y_true, y_pred),
                'MAPE': calcular_mape(y_true, y_pred),
                'R2': r2_score(y_true, y_pred)
            })
            
        if h_label == '24h (t+1)':
            predicciones_guardadas["Stacking Ensemble_train"] = pred_stk_train
            predicciones_guardadas["Stacking Ensemble_test"] = pred_stk_test
            predicciones_guardadas["y_train"] = y_train.values
            predicciones_guardadas["y_test"] = y_test.values
            predicciones_guardadas["fechas_train"] = fechas.iloc[:idx_train].values
            predicciones_guardadas["fechas_test"] = fechas.iloc[idx_val:].values
            
    df_res = pd.DataFrame(resultados)
    csv_metricas = os.path.join(docs_dir, "metricas_modelos_default.csv")
    df_res.to_csv(csv_metricas, index=False, sep=';', decimal=',')
    print(f"\n[OK] Metricas consolidadas guardadas en: {csv_metricas}")
    
    # 4. GENERAR GRAFICOS DE ALTA RESOLUCION
    print("\n" + "=" * 70)
    print("GENERANDO GRAFICOS COMPARATIVOS DE ALTA CALIDAD...")
    print("=" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    colors = {'LightGBM': '#2ca02c', 'XGBoost': '#d62728', 'CatBoost': '#ff7f0e', 'Stacking Ensemble': '#1f77b4'}
    
    # GRAFICO 1: Prediccion vs Real en Entrenamiento (Serie Temporal 24h)
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
    f_train = pd.to_datetime(predicciones_guardadas["fechas_train"])
    y_tr = predicciones_guardadas["y_train"]
    
    # Mostramos un zoom de 1 año completo de entrenamiento (ej: todo 2023) para que los detalles se aprecien claramente
    mask_zoom = (f_train >= '2023-01-01') & (f_train <= '2023-12-31')
    f_zoom = f_train[mask_zoom]
    y_zoom = y_tr[mask_zoom]
    
    ax.plot(f_zoom, y_zoom, label='Dato Real (SINCA Parque O\'Higgins)', color='black', linewidth=1.8, alpha=0.85)
    for m in ['LightGBM', 'XGBoost', 'CatBoost', 'Stacking Ensemble']:
        pred_z = predicciones_guardadas[f"{m}_train"][mask_zoom]
        ax.plot(f_zoom, pred_z, label=f'Predicción {m} (Default)', color=colors[m], linewidth=1.2, alpha=0.8)
        
    ax.axhline(50, color='orange', linestyle='--', linewidth=1, label='Umbral Norma Diaria (50 ug/m3)')
    ax.axhline(80, color='red', linestyle=':', linewidth=1, label='Umbral Alerta Ambiental (80 ug/m3)')
    ax.set_title("Ajuste de Modelos Boosting por Defecto en Entrenamiento (Zoom Anual 2023 - Horizonte 24h)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Fecha", fontsize=11)
    ax.set_ylabel("Concentración MP2.5 (ug/m3)", fontsize=11)
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    p1_path = os.path.join(graficos_dir, "01_prediccion_vs_real_entrenamiento_24h.png")
    plt.savefig(p1_path)
    plt.close()
    print(f"[GRAFICO 1] Guardado: {p1_path}")
    
    # GRAFICO 2: Prediccion vs Real en Test (Prueba Ciega 2025-2026)
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
    f_test = pd.to_datetime(predicciones_guardadas["fechas_test"])
    y_te = predicciones_guardadas["y_test"]
    
    ax.plot(f_test, y_te, label='Dato Real Test (Ciego)', color='black', linewidth=1.8, alpha=0.85)
    for m in ['LightGBM', 'XGBoost', 'CatBoost', 'Stacking Ensemble']:
        pred_te = predicciones_guardadas[f"{m}_test"]
        ax.plot(f_test, pred_te, label=f'Predicción {m}', color=colors[m], linewidth=1.2, alpha=0.8)
        
    ax.axhline(50, color='orange', linestyle='--', linewidth=1, label='Norma Diaria (50 ug/m3)')
    ax.axhline(80, color='red', linestyle=':', linewidth=1, label='Alerta (80 ug/m3)')
    ax.set_title("Generalización en Conjunto de Prueba Ciega (2025 - 2026 - Horizonte 24h)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Fecha", fontsize=11)
    ax.set_ylabel("Concentración MP2.5 (ug/m3)", fontsize=11)
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    p2_path = os.path.join(graficos_dir, "02_prediccion_vs_real_test_24h.png")
    plt.savefig(p2_path)
    plt.close()
    print(f"[GRAFICO 2] Guardado: {p2_path}")
    
    # GRAFICO 3: Diagramas de Dispersión Real vs Predicho (Scatter Plots 2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    model_list = ['LightGBM', 'XGBoost', 'CatBoost', 'Stacking Ensemble']
    
    for idx, (m, ax) in enumerate(zip(model_list, axes.flatten())):
        y_te = predicciones_guardadas["y_test"]
        pred_te = predicciones_guardadas[f"{m}_test"]
        r2_val = r2_score(y_te, pred_te)
        mae_val = mean_absolute_error(y_te, pred_te)
        
        ax.scatter(y_te, pred_te, alpha=0.5, color=colors[m], edgecolors='none', s=25)
        # Linea 1:1 perfecta
        max_val = max(np.max(y_te), np.max(pred_te)) + 5
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, label='Ajuste Ideal (1:1)')
        
        ax.set_title(f"{m} (Default)\nR² Test: {r2_val:.4f} | MAE: {mae_val:.2f} ug/m3", fontsize=11, fontweight='bold')
        ax.set_xlabel("MP2.5 Real (ug/m3)", fontsize=10)
        ax.set_ylabel("MP2.5 Predicho (ug/m3)", fontsize=10)
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper left', fontsize=9)
        
    plt.suptitle("Dispersión Real vs. Predicho en Prueba Ciega (Horizonte 24h)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    p3_path = os.path.join(graficos_dir, "03_dispersion_real_vs_predicho.png")
    plt.savefig(p3_path)
    plt.close()
    print(f"[GRAFICO 3] Guardado: {p3_path}")
    
    # GRAFICO 4: Distribucion de Residuos (Errores e Histograma)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    for m in model_list:
        residuos = predicciones_guardadas["y_test"] - predicciones_guardadas[f"{m}_test"]
        ax1.hist(residuos, bins=30, alpha=0.45, label=m, color=colors[m], density=True)
        ax2.scatter(predicciones_guardadas[f"{m}_test"], residuos, alpha=0.4, label=m, color=colors[m], s=18)
        
    ax1.axvline(0, color='black', linestyle='--', linewidth=1.5)
    ax1.set_title("Densidad de Probabilidad de los Errores (Residuos)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Error de Predicción: Real - Predicho (ug/m3)", fontsize=10)
    ax1.set_ylabel("Densidad", fontsize=10)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.5)
    ax2.set_title("Homocedasticidad: Residuos vs. Valores Predichos", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Valor Predicho MP2.5 (ug/m3)", fontsize=10)
    ax2.set_ylabel("Error Residual (ug/m3)", fontsize=10)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle("Diagnóstico de Residuos y Distribución de Errores (Horizonte 24h - Test)", fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    p4_path = os.path.join(graficos_dir, "04_residuos_y_distribucion_errores.png")
    plt.savefig(p4_path)
    plt.close()
    print(f"[GRAFICO 4] Guardado: {p4_path}")
    
    # GRAFICO 5: Comparativa de Metricas Bar Chart (24h)
    df_24_test = df_res[(df_res['Horizonte'] == '24h (t+1)') & (df_res['Particion'] == 'Test')].set_index('Modelo')
    df_24_train = df_res[(df_res['Horizonte'] == '24h (t+1)') & (df_res['Particion'] == 'Train')].set_index('Modelo')
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=300)
    metricas = ['R2', 'MAE', 'RMSE', 'MAPE']
    titulos = ['R² (Mayor es mejor)', 'MAE (Menor es mejor)', 'RMSE (Menor es mejor)', 'MAPE % (Menor es mejor)']
    
    x = np.arange(len(model_list))
    width = 0.35
    
    for ax, m_key, tit in zip(axes, metricas, titulos):
        vals_train = [df_24_train.loc[m, m_key] for m in model_list]
        vals_test = [df_24_test.loc[m, m_key] for m in model_list]
        
        ax.bar(x - width/2, vals_train, width, label='Train', color='#4a90e2', alpha=0.85)
        ax.bar(x + width/2, vals_test, width, label='Test', color='#e74c3c', alpha=0.85)
        
        ax.set_title(tit, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_list, rotation=25, ha='right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        if m_key == 'R2':
            ax.set_ylim(0, 1.0)
            ax.legend(loc='lower right', fontsize=9)
        else:
            ax.legend(loc='upper right', fontsize=9)
            
    plt.suptitle("Comparativa de Desempeño: Train vs. Test (Modelos Boosting por Defecto - 24h)", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    p5_path = os.path.join(graficos_dir, "05_comparativa_metricas_train_vs_test.png")
    plt.savefig(p5_path)
    plt.close()
    print(f"[GRAFICO 5] Guardado: {p5_path}")
    
    # GRAFICO 6: Degradacion Temporal Multi-Horizonte (24h vs 48h vs 72h)
    df_test_all = df_res[df_res['Particion'] == 'Test']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    horiz_keys = ['24h (t+1)', '48h (t+2)', '72h (t+3)']
    
    for m in model_list:
        sub_df = df_test_all[df_test_all['Modelo'] == m].set_index('Horizonte')
        r2_vals = [sub_df.loc[h, 'R2'] for h in horiz_keys]
        mae_vals = [sub_df.loc[h, 'MAE'] for h in horiz_keys]
        
        ax1.plot(horiz_keys, r2_vals, marker='o', linewidth=2, label=m, color=colors[m])
        ax2.plot(horiz_keys, mae_vals, marker='s', linewidth=2, label=m, color=colors[m])
        
    ax1.set_title("Evolución del R² según el Horizonte de Pronóstico", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Coeficiente R²", fontsize=10)
    ax1.set_ylim(0, 0.8)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower left', fontsize=9)
    
    ax2.set_title("Evolución del Error Absoluto Medio (MAE)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("MAE (ug/m3)", fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left', fontsize=9)
    
    plt.suptitle("Impacto del Alcance Temporal en la Capacidad Predictiva (Prueba Ciega)", fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    p6_path = os.path.join(graficos_dir, "06_degradacion_multi_horizonte_24_48_72.png")
    plt.savefig(p6_path)
    plt.close()
    print(f"[GRAFICO 6] Guardado: {p6_path}")
    
    print("\n" + "=" * 70)
    print("RESUMEN DE METRICAS EN TEST (24 HORAS):")
    print("=" * 70)
    print(df_24_test[['R2', 'MAE', 'RMSE', 'MAPE']].to_string())
    print("=" * 70)
    
    return df_res

if __name__ == "__main__":
    ejecutar_benchmark()
