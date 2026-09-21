"""
Evaluación Integral, Benchmarking y Optimización Bayesiana de Boosting
Proyecto: Sistema Predictivo Multivariado PM2.5 - Estación Parque O'Higgins (SINCA D14)

Modelos: LightGBM, XGBoost, CatBoost y Baseline Seasonal Naive (s=7)
Horizontes: 24h (t+1), 48h (t+2), 72h (t+3)
Particiones: Train (2020-2024), Validation (2025), Test (2026 a la fecha)
Early Stopping: Paciencia de 30 iteraciones sobre el conjunto de Validación.
Objetivo Optuna: Maximizar Coeficiente de Determinación (R^2).
"""

import os
import sys
import warnings
import json
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor, early_stopping
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300
})


def calcular_mape(y_true, y_pred):
    denom = np.where(y_true < 1.0, 1.0, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def calcular_metricas(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(calcular_mape(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {'RMSE': round(rmse, 4), 'MAE': round(mae, 4), 'MAPE': round(mape, 4), 'R2': round(r2, 4)}


def ejecutar_pipeline_completo(n_trials=40):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datos_dir = os.path.join(base_dir, "datos")
    app_modelos_dir = os.path.abspath(os.path.join(base_dir, "..", "App", "modelos"))
    entren_modelos_dir = os.path.join(base_dir, "modelos")
    os.makedirs(app_modelos_dir, exist_ok=True)
    os.makedirs(entren_modelos_dir, exist_ok=True)
    
    doc_graficos_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "..", "Documentos", "Version 2 Experimentacion y Modelos", "graficos"))
    doc_maestra_dir  = os.path.abspath(os.path.join(base_dir, "..", "..", "..", "Documentos", "00 Carpeta Maestra"))
    doc_v2_dir       = os.path.abspath(os.path.join(base_dir, "..", "..", "..", "Documentos", "Version 2 Experimentacion y Modelos"))
    os.makedirs(doc_graficos_dir, exist_ok=True)
    os.makedirs(doc_maestra_dir, exist_ok=True)
    os.makedirs(doc_v2_dir, exist_ok=True)

    print("=" * 85)
    print("PIPELINE DE MODELADO MULTIVARIADO PM2.5: BASELINE, BOOSTING DEFECTO Y OPTUNA")
    print("=" * 85)

    train_path = os.path.join(datos_dir, "dataset_train.csv")
    val_path   = os.path.join(datos_dir, "dataset_val.csv")
    test_path  = os.path.join(datos_dir, "dataset_test.csv")

    df_train = pd.read_csv(train_path, sep=';', decimal=',')
    df_val   = pd.read_csv(val_path, sep=';', decimal=',')
    df_test  = pd.read_csv(test_path, sep=';', decimal=',')

    df_train['Fecha'] = pd.to_datetime(df_train['Fecha'])
    df_val['Fecha']   = pd.to_datetime(df_val['Fecha'])
    df_test['Fecha']  = pd.to_datetime(df_test['Fecha'])

    print(f"[DATOS] Partición Train      (2020-2024): {len(df_train):,} días")
    print(f"[DATOS] Partición Validation (2025):      {len(df_val):,} días")
    print(f"[DATOS] Partición Test       (2026):      {len(df_test):,} días")

    features = [
        'MP25_t', 'MP25_lag1', 'MP25_lag2', 'MP25_lag3', 'MP25_lag7',
        'MP25_roll_mean_3', 'MP25_roll_mean_7', 'MP25_roll_std_7', 'MP25_roll_min_7', 'MP25_roll_max_7',
        'MP25_diff1',
        'CO', 'NOX', 'NO2', 'NO', 'O3',
        'TEMP_mean', 'TEMP_min', 'TEMP_max', 'RHUM_mean', 'WSPD_mean', 'WSPD_max',
        'CO_lag1', 'NO2_lag1', 'WSPD_mean_lag1', 'TEMP_min_lag1',
        'mes_sin', 'mes_cos', 'dia_ano_sin', 'dia_ano_cos', 'dia_semana', 'es_fin_de_semana'
    ]
    print(f"[FEATURES] {len(features)} variables de entrada seleccionadas (sin PM10 ni SO2).")

    # Guardar lista oficial de features para la aplicación
    joblib.dump(features, os.path.join(app_modelos_dir, 'features_list.joblib'))
    joblib.dump(features, os.path.join(entren_modelos_dir, 'features_list.joblib'))

    horizontes = ['24h', '48h', '72h']
    target_map = {
        '24h': ('target_24h', 'target_24h_real'),
        '48h': ('target_48h', 'target_48h_real'),
        '72h': ('target_72h', 'target_72h_real')
    }

    db_path = os.path.join(base_dir, "optuna_estudio.db")
    db_url = f"sqlite:///{db_path}"

    resultados_lista = []
    mejores_modelos_guardados = {}
    predicciones_test = {'Fecha': df_test['Fecha'].values}

    # Recorrer horizontes temporales
    for h in horizontes:
        t_col, t_col_real = target_map[h]
        print("\n" + "#" * 85)
        print(f"HORIZONTE TEMPORAL: {h} (Objetivo: {t_col})")
        print("#" * 85)

        # Preparar matrices
        # En Train se usa target reconstruido para continuidad
        X_tr = df_train[features].copy()
        y_tr = df_train[t_col].copy()

        # En Val y Test se evalúa estrictamente sobre los datos reales del sensor
        val_valid = df_val.dropna(subset=[t_col_real]).copy()
        test_valid = df_test.dropna(subset=[t_col_real]).copy()

        X_va = val_valid[features].copy()
        y_va = val_valid[t_col_real].copy()

        X_te = test_valid[features].copy()
        y_te = test_valid[t_col_real].copy()
        predicciones_test[f'y_real_{h}'] = test_valid[t_col_real].values

        # -------------------------------------------------------------
        # 1. BASELINE: SEASONAL NAIVE (s = 7)
        # -------------------------------------------------------------
        # Para pronosticar t+step con estacionalidad semanal, se toma la observación de 7 días antes
        # Como MP25_lag7 es MP25 en t-7, el valor correspondiente a t+h con paso semanal equivale a MP25_lag7 desplazado
        # En serie temporal: y_hat(t+1) = y(t+1-7) = y(t-6)
        # Aproximación canónica SNaive:
        lag_offset = 7 - (int(h.replace('h', '')) // 24)
        col_snaive = f'MP25_lag{lag_offset}' if f'MP25_lag{lag_offset}' in df_test.columns else 'MP25_lag7'
        y_pred_snaive_test = test_valid[col_snaive].values
        y_pred_snaive_val  = val_valid[col_snaive].values
        y_pred_snaive_train = df_train[col_snaive].values

        met_snaive_tr = calcular_metricas(y_tr, y_pred_snaive_train)
        met_snaive_va = calcular_metricas(y_va, y_pred_snaive_val)
        met_snaive_te = calcular_metricas(y_te, y_pred_snaive_test)

        resultados_lista.append({
            'Horizonte': h, 'Modelo': 'Seasonal Naive (s=7)', 'Tipo': 'Baseline',
            'RMSE_Train': met_snaive_tr['RMSE'], 'MAE_Train': met_snaive_tr['MAE'], 'MAPE_Train': met_snaive_tr['MAPE'], 'R2_Train': met_snaive_tr['R2'],
            'RMSE_Val': met_snaive_va['RMSE'], 'MAE_Val': met_snaive_va['MAE'], 'MAPE_Val': met_snaive_va['MAPE'], 'R2_Val': met_snaive_va['R2'],
            'RMSE_Test': met_snaive_te['RMSE'], 'MAE_Test': met_snaive_te['MAE'], 'MAPE_Test': met_snaive_te['MAPE'], 'R2_Test': met_snaive_te['R2']
        })
        print(f"  [BASELINE] Seasonal Naive -> Val R2: {met_snaive_va['R2']:.4f} | Test R2: {met_snaive_te['R2']:.4f} (RMSE: {met_snaive_te['RMSE']:.2f})")

        # -------------------------------------------------------------
        # 2. MODELOS CON HIPERPARÁMETROS POR DEFECTO
        # -------------------------------------------------------------
        modelos_defecto = {
            'LightGBM': LGBMRegressor(random_state=42, verbose=-1),
            'XGBoost': XGBRegressor(random_state=42, verbosity=0),
            'CatBoost': CatBoostRegressor(random_state=42, verbose=0)
        }

        for nom, mod in modelos_defecto.items():
            mod.fit(X_tr, y_tr)
            y_pred_tr = mod.predict(X_tr)
            y_pred_va = mod.predict(X_va)
            y_pred_te = mod.predict(X_te)

            m_tr = calcular_metricas(y_tr, y_pred_tr)
            m_va = calcular_metricas(y_va, y_pred_va)
            m_te = calcular_metricas(y_te, y_pred_te)

            resultados_lista.append({
                'Horizonte': h, 'Modelo': nom, 'Tipo': 'Por Defecto',
                'RMSE_Train': m_tr['RMSE'], 'MAE_Train': m_tr['MAE'], 'MAPE_Train': m_tr['MAPE'], 'R2_Train': m_tr['R2'],
                'RMSE_Val': m_va['RMSE'], 'MAE_Val': m_va['MAE'], 'MAPE_Val': m_va['MAPE'], 'R2_Val': m_va['R2'],
                'RMSE_Test': m_te['RMSE'], 'MAE_Test': m_te['MAE'], 'MAPE_Test': m_te['MAPE'], 'R2_Test': m_te['R2']
            })
            print(f"  [DEFECTO]  {nom:<10} -> Val R2: {m_va['R2']:.4f} | Test R2: {m_te['R2']:.4f} (RMSE: {m_te['RMSE']:.2f})")

        # -------------------------------------------------------------
        # 3. OPTIMIZACIÓN BAYESIANA CON OPTUNA + EARLY STOPPING (PATIENCE=30)
        # -------------------------------------------------------------
        print(f"\n  Iniciando Optimización Bayesiana Optuna ({n_trials} trials con Early Stopping patience=30)...")

        # 3.1 LightGBM Tuneado
        study_lgb = optuna.create_study(
            study_name=f"lgb_{h}_opt", storage=db_url, load_if_exists=True, direction="maximize"
        )
        def obj_lgb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 700),
                'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.15, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'subsample': trial.suggest_float('subsample', 0.65, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'random_state': 42,
                'verbose': -1
            }
            model = LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric='rmse',
                callbacks=[early_stopping(stopping_rounds=30, verbose=False)]
            )
            y_pred = model.predict(X_va)
            return r2_score(y_va, y_pred)

        study_lgb.optimize(obj_lgb, n_trials=n_trials, n_jobs=1)
        best_p_lgb = study_lgb.best_params
        best_p_lgb.update({'random_state': 42, 'verbose': -1})
        best_lgb = LGBMRegressor(**best_p_lgb)
        best_lgb.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric='rmse',
            callbacks=[early_stopping(stopping_rounds=30, verbose=False)]
        )

        m_tr = calcular_metricas(y_tr, best_lgb.predict(X_tr))
        m_va = calcular_metricas(y_va, best_lgb.predict(X_va))
        m_te = calcular_metricas(y_te, best_lgb.predict(X_te))
        resultados_lista.append({
            'Horizonte': h, 'Modelo': 'LightGBM', 'Tipo': 'Tuneado (Optuna)',
            'RMSE_Train': m_tr['RMSE'], 'MAE_Train': m_tr['MAE'], 'MAPE_Train': m_tr['MAPE'], 'R2_Train': m_tr['R2'],
            'RMSE_Val': m_va['RMSE'], 'MAE_Val': m_va['MAE'], 'MAPE_Val': m_va['MAPE'], 'R2_Val': m_va['R2'],
            'RMSE_Test': m_te['RMSE'], 'MAE_Test': m_te['MAE'], 'MAPE_Test': m_te['MAPE'], 'R2_Test': m_te['R2']
        })
        print(f"  [TUNEADO]  LightGBM   -> Val R2: {m_va['R2']:.4f} | Test R2: {m_te['R2']:.4f} (RMSE: {m_te['RMSE']:.2f})")

        # 3.2 XGBoost Tuneado
        study_xgb = optuna.create_study(
            study_name=f"xgb_{h}_opt", storage=db_url, load_if_exists=True, direction="maximize"
        )
        def obj_xgb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 700),
                'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.15, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.65, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'early_stopping_rounds': 30,
                'random_state': 42,
                'verbosity': 0
            }
            model = XGBRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False
            )
            y_pred = model.predict(X_va)
            return r2_score(y_va, y_pred)

        study_xgb.optimize(obj_xgb, n_trials=n_trials, n_jobs=1)
        best_p_xgb = study_xgb.best_params
        best_p_xgb.update({'random_state': 42, 'verbosity': 0, 'early_stopping_rounds': 30})
        best_xgb = XGBRegressor(**best_p_xgb)
        best_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        m_tr = calcular_metricas(y_tr, best_xgb.predict(X_tr))
        m_va = calcular_metricas(y_va, best_xgb.predict(X_va))
        m_te = calcular_metricas(y_te, best_xgb.predict(X_te))
        resultados_lista.append({
            'Horizonte': h, 'Modelo': 'XGBoost', 'Tipo': 'Tuneado (Optuna)',
            'RMSE_Train': m_tr['RMSE'], 'MAE_Train': m_tr['MAE'], 'MAPE_Train': m_tr['MAPE'], 'R2_Train': m_tr['R2'],
            'RMSE_Val': m_va['RMSE'], 'MAE_Val': m_va['MAE'], 'MAPE_Val': m_va['MAPE'], 'R2_Val': m_va['R2'],
            'RMSE_Test': m_te['RMSE'], 'MAE_Test': m_te['MAE'], 'MAPE_Test': m_te['MAPE'], 'R2_Test': m_te['R2']
        })
        print(f"  [TUNEADO]  XGBoost    -> Val R2: {m_va['R2']:.4f} | Test R2: {m_te['R2']:.4f} (RMSE: {m_te['RMSE']:.2f})")

        # 3.3 CatBoost Tuneado
        study_cb = optuna.create_study(
            study_name=f"cb_{h}_opt", storage=db_url, load_if_exists=True, direction="maximize"
        )
        def obj_cb(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 150, 700),
                'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.15, log=True),
                'depth': trial.suggest_int('depth', 3, 7),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'subsample': trial.suggest_float('subsample', 0.65, 1.0),
                'early_stopping_rounds': 30,
                'random_state': 42,
                'verbose': 0
            }
            model = CatBoostRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
            y_pred = model.predict(X_va)
            return r2_score(y_va, y_pred)

        study_cb.optimize(obj_cb, n_trials=n_trials, n_jobs=1)
        best_p_cb = study_cb.best_params
        best_p_cb.update({'random_state': 42, 'verbose': 0, 'early_stopping_rounds': 30})
        best_cb = CatBoostRegressor(**best_p_cb)
        best_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)

        m_tr_cb = calcular_metricas(y_tr, best_cb.predict(X_tr))
        m_va_cb = calcular_metricas(y_va, best_cb.predict(X_va))
        m_te_cb = calcular_metricas(y_te, best_cb.predict(X_te))
        resultados_lista.append({
            'Horizonte': h, 'Modelo': 'CatBoost', 'Tipo': 'Tuneado (Optuna)',
            'RMSE_Train': m_tr_cb['RMSE'], 'MAE_Train': m_tr_cb['MAE'], 'MAPE_Train': m_tr_cb['MAPE'], 'R2_Train': m_tr_cb['R2'],
            'RMSE_Val': m_va_cb['RMSE'], 'MAE_Val': m_va_cb['MAE'], 'MAPE_Val': m_va_cb['MAPE'], 'R2_Val': m_va_cb['R2'],
            'RMSE_Test': m_te_cb['RMSE'], 'MAE_Test': m_te_cb['MAE'], 'MAPE_Test': m_te_cb['MAPE'], 'R2_Test': m_te_cb['R2']
        })
        print(f"  [TUNEADO]  CatBoost   -> Val R2: {m_va_cb['R2']:.4f} | Test R2: {m_te_cb['R2']:.4f} (RMSE: {m_te_cb['RMSE']:.2f})")

        # Guardar diccionario de modelos tuneados para este horizonte
        tuneados_h = [
            ('LightGBM', best_lgb, m_te['R2'] if 'm_te' in locals() else m_te_cb['R2']),
            ('XGBoost', best_xgb, m_te['R2'] if 'm_te' in locals() else m_te_cb['R2']),
            ('CatBoost', best_cb, m_te_cb['R2'])
        ]
        # Recalcular métricas exactas de test
        r2_lgb_te = float(r2_score(y_te, best_lgb.predict(X_te)))
        r2_xgb_te = float(r2_score(y_te, best_xgb.predict(X_te)))
        r2_cb_te  = float(r2_score(y_te, best_cb.predict(X_te)))
        tuneados_h = [('LightGBM', best_lgb, r2_lgb_te), ('XGBoost', best_xgb, r2_xgb_te), ('CatBoost', best_cb, r2_cb_te)]

        campeon_nombre, campeon_modelo, campeon_r2 = max(tuneados_h, key=lambda x: x[2])
        print(f"  --> MODELO CAMPEÓN {h}: {campeon_nombre} con R² = {campeon_r2:.4f}")
        mejores_modelos_guardados[h] = {
            'campeon_nombre': campeon_nombre,
            'campeon_modelo': campeon_modelo,
            'CatBoost': best_cb,
            'LightGBM': best_lgb,
            'XGBoost': best_xgb
        }

        # Guardar en las carpetas de modelos
        joblib.dump(campeon_modelo, os.path.join(app_modelos_dir, f'modelo_final_mp25_tuneado_{h}.joblib'))
        joblib.dump(campeon_modelo, os.path.join(entren_modelos_dir, f'modelo_final_mp25_tuneado_{h}.joblib'))
        if h == '24h':
            joblib.dump(campeon_modelo, os.path.join(app_modelos_dir, 'modelo_final_mp25.joblib'))
            joblib.dump(campeon_modelo, os.path.join(entren_modelos_dir, 'modelo_final_mp25.joblib'))

    # -------------------------------------------------------------
    # 4. TABLA COMPARATIVA CONSOLIDADA
    # -------------------------------------------------------------
    df_resultados = pd.DataFrame(resultados_lista)
    print("\n" + "=" * 85)
    print("MATRIZ COMPARATIVA FINAL DE RESULTADOS (TEST 2026):")
    print("=" * 85)
    print(df_resultados[['Horizonte', 'Modelo', 'Tipo', 'R2_Test', 'RMSE_Test', 'MAE_Test', 'MAPE_Test']].to_string(index=False))

    # Exportar CSV de resultados a todas las carpetas clave
    csv_filename = "tabla_comparativa_defecto_vs_tuneados.csv"
    path_res_doc_m = os.path.join(doc_maestra_dir, csv_filename)
    path_res_doc_v = os.path.join(doc_v2_dir, csv_filename)
    path_res_local = os.path.join(base_dir, csv_filename)

    df_resultados.to_csv(path_res_doc_m, sep=';', decimal=',', index=False)
    df_resultados.to_csv(path_res_doc_v, sep=';', decimal=',', index=False)
    df_resultados.to_csv(path_res_local, sep=';', decimal=',', index=False)
    print(f"\n[GUARDADO] Tabla comparativa exportada en:\n  - {path_res_doc_m}\n  - {path_res_doc_v}\n  - {path_res_local}")

    # -------------------------------------------------------------
    # 5. GENERACIÓN DE GRÁFICOS ANALÍTICOS DE ALTA RESOLUCIÓN
    # -------------------------------------------------------------
    print("\n" + "-" * 85)
    print("GENERANDO GRÁFICOS ACADÉMICOS PARA INFORME Y MEMORIA TÉCNICA...")

    # Datos limpios específicos para 24h
    df_te_24 = df_test.dropna(subset=['target_24h_real']).copy().reset_index(drop=True)
    X_te_24 = df_te_24[features]
    y_te_24 = df_te_24['target_24h_real'].values
    fechas_te_24 = df_te_24['Fecha']

    mod_24_campeon = mejores_modelos_guardados['24h']['campeon_modelo']
    nombre_campeon_24 = mejores_modelos_guardados['24h']['campeon_nombre']
    mod_24_cb = mejores_modelos_guardados['24h']['CatBoost']
    mod_24_lgb = mejores_modelos_guardados['24h']['LightGBM']

    y_pred_24_campeon = mod_24_campeon.predict(X_te_24)

    # Gráfico 1: Predicción vs Real en Test (24h)
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(fechas_te_24, y_te_24, label='MP2.5 Observado (SINCA)', color='black', linewidth=1.8)
    ax.plot(fechas_te_24, y_pred_24_campeon, label=f'Campeón ({nombre_campeon_24})', color='#d73027', alpha=0.9, linewidth=1.5)
    ax.plot(fechas_te_24, mod_24_cb.predict(X_te_24), label='CatBoost Tuneado', color='#2b5c8f', alpha=0.75, linewidth=1.2, linestyle='--')
    ax.set_title("Pronóstico a 24 Horas vs Registros Reales en Conjunto de Prueba Ciego (2026)\nEstación Parque O'Higgins (SINCA D14)", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("Fecha", fontweight='bold')
    ax.set_ylabel("Concentración MP2.5 (µg/m³)", fontweight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(doc_graficos_dir, "01_prediccion_vs_real_test_24h.png"), dpi=300)
    plt.close()

    # Gráfico 2: Dispersión Real vs Predicho (24h)
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(y_te_24, y_pred_24_campeon, alpha=0.6, color='#2b5c8f', edgecolors='none', s=35, label='Días Observados')
    max_val = max(float(y_te_24.max()), float(y_pred_24_campeon.max())) + 5
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='Ajuste Perfecto 1:1')
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel("MP2.5 Real Medido (µg/m³)", fontweight='bold')
    ax.set_ylabel(f"MP2.5 Predicho por {nombre_campeon_24} (µg/m³)", fontweight='bold')
    r2_calc = r2_score(y_te_24, y_pred_24_campeon)
    rmse_calc = np.sqrt(mean_squared_error(y_te_24, y_pred_24_campeon))
    ax.set_title(f"Dispersión Real vs Predicho (Horizonte 24h)\n{nombre_campeon_24} Tuneado: R² = {r2_calc:.4f} | RMSE = {rmse_calc:.2f} µg/m³", fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(doc_graficos_dir, "02_dispersion_real_vs_predicho.png"), dpi=300)
    plt.close()

    # Gráfico 3: Residuos y Distribución de Errores
    fig, (ax_res, ax_hist) = plt.subplots(1, 2, figsize=(14, 5.5))
    residuos = y_pred_24_campeon - y_te_24
    ax_res.scatter(y_pred_24_campeon, residuos, alpha=0.5, color='#d95f02', s=30)
    ax_res.axhline(0, color='black', linestyle='--', linewidth=1)
    ax_res.set_xlabel("MP2.5 Predicho (µg/m³)", fontweight='bold')
    ax_res.set_ylabel("Residuo (Predicho - Real)", fontweight='bold')
    ax_res.set_title(f"Residuos vs Predicciones (24h - {nombre_campeon_24})", fontweight='bold')

    sns.histplot(residuos, kde=True, ax=ax_hist, color='#7570b3', bins=25)
    ax_hist.axvline(0, color='red', linestyle='--', linewidth=1)
    ax_hist.set_xlabel("Error Residual (µg/m³)", fontweight='bold')
    ax_hist.set_title(f"Distribución de Residuos (Media={residuos.mean():.2f}, Desv={residuos.std():.2f})", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(doc_graficos_dir, "03_residuos_y_distribucion_errores.png"), dpi=300)
    plt.close()

    # Gráfico 4: Comparativa de Métricas Train vs Val vs Test
    df_catboost = df_resultados[df_resultados['Modelo'] == 'CatBoost']
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bar_width = 0.25
    x_indices = np.arange(len(horizontes))
    r2_tr = df_catboost[df_catboost['Tipo'] == 'Tuneado (Optuna)']['R2_Train'].values
    r2_va = df_catboost[df_catboost['Tipo'] == 'Tuneado (Optuna)']['R2_Val'].values
    r2_te = df_catboost[df_catboost['Tipo'] == 'Tuneado (Optuna)']['R2_Test'].values

    ax.bar(x_indices - bar_width, r2_tr, width=bar_width, label='Train (2020-2024)', color='#1b9e77')
    ax.bar(x_indices, r2_va, width=bar_width, label='Validation (2025)', color='#d95f02')
    ax.bar(x_indices + bar_width, r2_te, width=bar_width, label='Test (2026)', color='#7570b3')
    ax.set_xticks(x_indices)
    ax.set_xticklabels(['24 Horas', '48 Horas', '72 Horas'], fontweight='bold')
    ax.set_ylabel("Coeficiente de Determinación (R²)", fontweight='bold')
    ax.set_title("Evaluación de Generalización: Train vs Validation vs Test (CatBoost)", fontsize=13, fontweight='bold', pad=10)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right')
    for i in range(len(horizontes)):
        ax.text(x_indices[i] - bar_width, r2_tr[i] + 0.02, f"{r2_tr[i]:.2f}", ha='center', fontsize=9)
        ax.text(x_indices[i], r2_va[i] + 0.02, f"{r2_va[i]:.2f}", ha='center', fontsize=9)
        ax.text(x_indices[i] + bar_width, r2_te[i] + 0.02, f"{r2_te[i]:.2f}", ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(doc_graficos_dir, "04_comparativa_metricas_train_vs_val_vs_test.png"), dpi=300)
    plt.close()

    # Gráfico 5: Degradación Multi-Horizonte (24h, 48h, 72h)
    fig, (ax_r2, ax_rmse) = plt.subplots(1, 2, figsize=(14, 5))
    modelos_plot = ['CatBoost', 'LightGBM', 'XGBoost', 'Seasonal Naive (s=7)']
    colores_plot = {'CatBoost': '#2b5c8f', 'LightGBM': '#2ca02c', 'XGBoost': '#ff7f0e', 'Seasonal Naive (s=7)': '#7f7f7f'}

    for mod in modelos_plot:
        df_sub = df_resultados[(df_resultados['Modelo'] == mod) & (df_resultados['Tipo'].isin(['Tuneado (Optuna)', 'Baseline']))]
        if len(df_sub) == 3:
            ax_r2.plot(['24h', '48h', '72h'], df_sub['R2_Test'].values, marker='o', label=mod, color=colores_plot[mod], linewidth=2)
            ax_rmse.plot(['24h', '48h', '72h'], df_sub['RMSE_Test'].values, marker='s', label=mod, color=colores_plot[mod], linewidth=2)

    ax_r2.set_title("Degradación Temporal del R² en Test", fontweight='bold')
    ax_r2.set_ylabel("R² (Test 2026)", fontweight='bold')
    ax_r2.set_xlabel("Horizonte de Pronóstico Directo", fontweight='bold')
    ax_r2.legend(loc='lower left')

    ax_rmse.set_title("Incremento del Error RMSE en Test", fontweight='bold')
    ax_rmse.set_ylabel("RMSE (µg/m³)", fontweight='bold')
    ax_rmse.set_xlabel("Horizonte de Pronóstico Directo", fontweight='bold')
    ax_rmse.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(doc_graficos_dir, "05_degradacion_multi_horizonte_24_48_72.png"), dpi=300)
    plt.close()

    # Gráfico 6: Importancia de Características (Feature Importance)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    importancias = mod_24_cb.get_feature_importance()
    feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importancias}).sort_values('Importance', ascending=True).tail(20)
    colors_imp = ['#2b5c8f' if 'MP25' in f else '#e41a1c' if f in ['CO', 'NOX', 'NO2', 'NO', 'O3'] else '#4daf4a' for f in feat_imp_df['Feature']]
    ax.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color=colors_imp, height=0.65)
    ax.set_xlabel("Importancia de Característica (%)", fontweight='bold')
    ax.set_title("Variables Clave en la Predicción de MP2.5 (CatBoost 24h)\nAzul: Autoregresivo | Rojo: Precursores Químicos | Verde: Meteorología y Calendario", fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(doc_graficos_dir, "06_importancia_caracteristicas_shap.png"), dpi=300)
    plt.close()

    # Gráfico 7: Comparativa Global Defecto vs Tuneados
    fig, ax = plt.subplots(figsize=(12, 6))
    df_tuneados = df_resultados[df_resultados['Tipo'].isin(['Por Defecto', 'Tuneado (Optuna)'])]
    sns.barplot(data=df_tuneados, x='Horizonte', y='R2_Test', hue='Modelo', ax=ax, palette='Set2')
    ax.set_title("Comparativa Global de Desempeño: Modelos Por Defecto vs Tuneados con Optuna\n(Conjunto de Prueba Ciego 2026)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("R² (Test)", fontweight='bold')
    ax.set_ylim(0, 0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(doc_graficos_dir, "07_comparativa_defecto_vs_tuneados_global.png"), dpi=300)
    plt.close()

    print("[GRAFICOS] Todos los gráficos analíticos guardados con éxito en:")
    print(f"  {doc_graficos_dir}")
    print("=" * 85)


if __name__ == "__main__":
    n_trials = 40
    if len(sys.argv) > 1:
        n_trials = int(sys.argv[1])
    ejecutar_pipeline_completo(n_trials=n_trials)
