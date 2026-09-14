"""
Pipeline Completo y Autonomo de Optimizacion Bayesiana con Optuna y Walk-Forward Validation
Proyecto: Sistema Predictivo de Calidad del Aire (PM2.5) - Estacion Parque O'Higgins
Horizontes: 24h (t+1), 48h (t+2), 72h (t+3)
Modelos: LightGBM, XGBoost, CatBoost y Stacking Ensemble
Persistencia: SQLite (optuna_estudio.db)
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

from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def calcular_mape(y_true, y_pred):
    denom = np.where(y_true < 1.0, 1.0, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

def cargar_y_preparar_datos(data_path):
    df = pd.read_csv(data_path, sep=';', decimal=',')
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.sort_values('Fecha').reset_index(drop=True)
    
    # 1. Lags autorregresivos
    df['lag_1'] = df['MP25'].shift(1)
    df['lag_2'] = df['MP25'].shift(2)
    df['lag_3'] = df['MP25'].shift(3)
    df['lag_7'] = df['MP25'].shift(7)
    
    # 2. Ventanas moviles
    df['rolling_mean_3'] = df['MP25'].shift(1).rolling(window=3).mean()
    df['rolling_std_3'] = df['MP25'].shift(1).rolling(window=3, min_periods=2).std(ddof=1)
    df['rolling_mean_7'] = df['MP25'].shift(1).rolling(window=7).mean()
    
    # 3. Cinematica
    df['diff_1'] = df['lag_1'] - df['lag_2']
    
    # 4. Estacionalidad ciclica
    df['mes'] = df['Fecha'].dt.month
    df['dia_ano'] = df['Fecha'].dt.dayofyear
    df['dia_semana'] = df['Fecha'].dt.dayofweek
    df['es_fin_de_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
    
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    df['dia_ano_sin'] = np.sin(2 * np.pi * df['dia_ano'] / 365.25)
    df['dia_ano_cos'] = np.cos(2 * np.pi * df['dia_ano'] / 365.25)
    
    # Covariable fisica PM10 si existe
    if 'PM10' in df.columns:
        df['lag_1_pm10'] = df['PM10'].shift(1)
        df['rolling_mean_3_pm10'] = df['PM10'].shift(1).rolling(window=3).mean()
        
    # Targets directos para las 3 ventanas de tiempo
    df['target_24h'] = df['MP25'].shift(-1)
    df['target_48h'] = df['MP25'].shift(-2)
    df['target_72h'] = df['MP25'].shift(-3)
    
    df_clean = df.dropna().reset_index(drop=True)
    
    features = [
        'lag_1', 'lag_2', 'lag_3', 'lag_7',
        'rolling_mean_3', 'rolling_std_3', 'rolling_mean_7',
        'diff_1', 'mes_sin', 'mes_cos', 'dia_ano_sin', 'dia_ano_cos',
        'dia_semana', 'es_fin_de_semana'
    ]
    if 'lag_1_pm10' in df_clean.columns:
        features.extend(['lag_1_pm10', 'rolling_mean_3_pm10'])
        
    return df_clean, features

def optimizar_modelo(nombre_modelo, X_train_val, y_train_val, db_url, n_trials=500):
    tscv = TimeSeriesSplit(n_splits=5)
    
    def objective(trial):
        if 'LightGBM' in nombre_modelo or 'LGBM' in nombre_modelo:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 80, 450),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'min_child_samples': trial.suggest_int('min_child_samples', 15, 75),
                'subsample': trial.suggest_float('subsample', 0.65, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'random_state': 42,
                'verbosity': -1,
                'n_jobs': -1
            }
            model_cls = LGBMRegressor
            
        elif 'XGBoost' in nombre_modelo or 'XGB' in nombre_modelo:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 80, 450),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'min_child_weight': trial.suggest_int('min_child_weight', 2, 18),
                'subsample': trial.suggest_float('subsample', 0.65, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'random_state': 42,
                'n_jobs': -1
            }
            model_cls = XGBRegressor
            
        elif 'CatBoost' in nombre_modelo:
            params = {
                'iterations': trial.suggest_int('iterations', 80, 450),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'depth': trial.suggest_int('depth', 3, 7),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 15.0),
                'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_state': 42,
                'silent': True,
                'thread_count': -1
            }
            model_cls = CatBoostRegressor

        cv_scores = []
        for step, (train_idx, val_idx) in enumerate(tscv.split(X_train_val)):
            X_tr, y_tr = X_train_val.iloc[train_idx], y_train_val.iloc[train_idx]
            X_va, y_val = X_train_val.iloc[val_idx], y_train_val.iloc[val_idx]
            
            m = model_cls(**params)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_va)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            cv_scores.append(rmse)
            
            # Poda temprana de combinaciones malas
            trial.report(rmse, step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
        return float(np.mean(cv_scores))

    study_name = f"optuna_{nombre_modelo}"
    pruner = optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=2)
    study = optuna.create_study(
        study_name=study_name,
        storage=db_url,
        direction='minimize',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=pruner
    )
    
    print(f"[{nombre_modelo}] Iniciando optimizacion bayesiana ({n_trials} trials con Walk-Forward)...")
    study.optimize(objective, n_trials=n_trials, catch=(Exception,), n_jobs=1)
    print(f"[{nombre_modelo}] Mejor RMSE Walk-Forward: {study.best_value:.4f}")
    return study.best_params, study

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datos_dir = os.path.join(base_dir, "datos")
    app_dir = os.path.abspath(os.path.join(base_dir, "..", "App", "modelos"))
    if not os.path.exists(app_dir):
        app_dir = os.path.abspath(os.path.join(base_dir, "..", "App"))
    os.makedirs(app_dir, exist_ok=True)
    
    db_path = os.path.join(base_dir, "optuna_estudio.db")
    db_url = f"sqlite:///{db_path}"
    
    docs_dir = r"C:\Users\Usuario\Documents\Personal\Universidad\Semestres\8° Semestre\Titulacion\Documentos\Comparación de datos"
    graficos_dir = os.path.join(docs_dir, "graficos")
    os.makedirs(graficos_dir, exist_ok=True)
    
    print("=" * 70)
    print("PIPELINE DE OPTIMIZACION BAYESIANA (OPTUNA + WALK-FORWARD)")
    print("=" * 70)
    print(f"Base de datos SQLite: {db_path}")
    
    df, features = cargar_y_preparar_datos(data_path)
    n = len(df)
    idx_train = int(n * 0.70)
    idx_val = int(n * 0.85)
    
    X = df[features]
    fechas = df['Fecha']
    
    horizontes = {
        '24h': 'target_24h',
        '48h': 'target_48h',
        '72h': 'target_72h'
    }
    
    resultados_tuneados = []
    estudios_optuna = {}
    
    # 500 trials como solicitado en el prompt
    TRIALS_PER_STUDY = 500
    
    for h_name, h_target in horizontes.items():
        print(f"\n" + "=" * 70)
        print(f"  OPTIMIZANDO HORIZONTE: {h_name} ({h_target})")
        print("=" * 70)
        
        y = df[h_target]
        
        X_train, y_train = X.iloc[:idx_train], y.iloc[:idx_train]
        X_val, y_val = X.iloc[idx_train:idx_val], y.iloc[idx_train:idx_val]
        X_test, y_test = X.iloc[idx_val:], y.iloc[idx_val:]
        
        X_train_val = X.iloc[:idx_val]
        y_train_val = y.iloc[:idx_val]
        
        # 1. Optuna LightGBM
        p_lgb, st_lgb = optimizar_modelo(f"LGBM_{h_name}", X_train_val, y_train_val, db_url, n_trials=TRIALS_PER_STUDY)
        m_lgb = LGBMRegressor(**p_lgb, random_state=42, verbosity=-1, n_jobs=-1)
        m_lgb.fit(X_train, y_train)
        if h_name == '24h': estudios_optuna['LGBM_24h'] = st_lgb
        
        # 2. Optuna XGBoost
        p_xgb, st_xgb = optimizar_modelo(f"XGB_{h_name}", X_train_val, y_train_val, db_url, n_trials=TRIALS_PER_STUDY)
        m_xgb = XGBRegressor(**p_xgb, random_state=42, n_jobs=-1)
        m_xgb.fit(X_train, y_train)
        if h_name == '24h': estudios_optuna['XGB_24h'] = st_xgb
        
        # 3. Optuna CatBoost
        p_cat, st_cat = optimizar_modelo(f"CatBoost_{h_name}", X_train_val, y_train_val, db_url, n_trials=TRIALS_PER_STUDY)
        m_cat = CatBoostRegressor(**p_cat, random_state=42, silent=True, thread_count=-1)
        m_cat.fit(X_train, y_train)
        if h_name == '24h': estudios_optuna['CatBoost_24h'] = st_cat
        
        # 4. Stacking Optimizado
        estimators = [
            ('lgb_opt', m_lgb),
            ('xgb_opt', m_xgb),
            ('cat_opt', m_cat)
        ]
        stack_opt = StackingRegressor(estimators=estimators, final_estimator=RidgeCV(), cv=KFold(5, shuffle=False))
        stack_opt.fit(X_train, y_train)
        
        # Guardar modelo en 02_App_Publicada
        model_out_name = f"modelo_final_mp25_tuneado_{h_name}.joblib"
        joblib.dump(stack_opt, os.path.join(app_dir, model_out_name))
        print(f"[GUARDADO MODELO] {model_out_name} en {app_dir}")
        
        modelos_dict = {
            'LightGBM_Opt': m_lgb,
            'XGBoost_Opt': m_xgb,
            'CatBoost_Opt': m_cat,
            'Stacking_Opt': stack_opt
        }
        
        for m_lbl, mod in modelos_dict.items():
            for p_name, X_part, y_part in [('Train', X_train, y_train), ('Val', X_val, y_val), ('Test', X_test, y_test)]:
                preds = mod.predict(X_part)
                resultados_tuneados.append({
                    'Horizonte': f"{h_name} (t+{1 if h_name=='24h' else (2 if h_name=='48h' else 3)})",
                    'Modelo': m_lbl,
                    'Particion': p_name,
                    'RMSE': np.sqrt(mean_squared_error(y_part, preds)),
                    'MAE': mean_absolute_error(y_part, preds),
                    'MAPE': calcular_mape(y_part, preds),
                    'R2': r2_score(y_part, preds)
                })
                
    # Guardar features list
    joblib.dump(features, os.path.join(app_dir, 'features_list.joblib'))
    
    # Tabla consolidada de modelos tuneados
    df_tuneados = pd.DataFrame(resultados_tuneados)
    res_path = os.path.join(docs_dir, "metricas_modelos_tuneados_optuna.csv")
    df_tuneados.to_csv(res_path, index=False, sep=';', decimal=',')
    print(f"\n[OK] Metricas de modelos tuneados guardadas en: {res_path}")
    
    # 5. TABLA COMPARATIVA DEFINITIVA: DEFECTO vs TUNEADOS
    defecto_csv = os.path.join(docs_dir, "metricas_modelos_default.csv")
    if os.path.exists(defecto_csv):
        df_def = pd.read_csv(defecto_csv, sep=';', decimal=',')
        df_def['Tipo'] = 'Por Defecto'
        df_tuneados['Tipo'] = 'Optimizado Optuna'
        
        # Mapear nombres para comparacion limpia
        nombre_map = {
            'LightGBM_Opt': 'LightGBM',
            'XGBoost_Opt': 'XGBoost',
            'CatBoost_Opt': 'CatBoost',
            'Stacking_Opt': 'Stacking Ensemble'
        }
        df_tun_renamed = df_tuneados.copy()
        df_tun_renamed['Modelo'] = df_tun_renamed['Modelo'].replace(nombre_map)
        
        df_comparativo = pd.concat([df_def, df_tun_renamed], ignore_index=True)
        comp_path = os.path.join(docs_dir, "tabla_comparativa_defecto_vs_tuneados.csv")
        df_comparativo.to_csv(comp_path, index=False, sep=';', decimal=',')
        print(f"[OK] Tabla comparativa final guardada en: {comp_path}")
        
    # 6. GENERACION DE GRAFICOS DE OPTIMIZACION
    print("\n" + "=" * 70)
    print("GENERANDO GRAFICOS DE OPTIMIZACION BAYESIANA Y CURVAS DE APRENDIZAJE...")
    print("=" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # GRAFICO 7: Historia de Convergencia de Optuna (24h)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)
    for ax, (m_key, st) in zip(axes, estudios_optuna.items()):
        trials_df = st.trials_dataframe()
        trials_df = trials_df[trials_df['state'] == 'COMPLETE']
        if not trials_df.empty:
            ax.plot(trials_df['number'], trials_df['value'], color='#4a90e2', alpha=0.35, label='Trial')
            best_cum = np.minimum.accumulate(trials_df['value'])
            ax.plot(trials_df['number'], best_cum, color='#d62728', linewidth=2, label='Mejor Mínimo')
        ax.set_title(f"Convergencia Optuna - {m_key}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Número de Trial", fontsize=10)
        ax.set_ylabel("RMSE Walk-Forward (ug/m3)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', fontsize=9)
        
    plt.suptitle("Curvas de Convergencia Bayesiana (500 Trials - Horizonte 24h)", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    p7_path = os.path.join(graficos_dir, "07_convergencia_optuna_24h.png")
    plt.savefig(p7_path)
    plt.close()
    print(f"[GRAFICO 7] Guardado: {p7_path}")
    
    # GRAFICO 8: Comparativa R2 y MAE en Test (Defecto vs. Tuneados a 24h)
    if os.path.exists(defecto_csv):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
        mod_names = ['LightGBM', 'XGBoost', 'CatBoost', 'Stacking Ensemble']
        
        df_def_24 = df_def[(df_def['Horizonte'] == '24h (t+1)') & (df_def['Particion'] == 'Test')].set_index('Modelo')
        df_tun_24 = df_tun_renamed[(df_tun_renamed['Horizonte'] == '24h (t+1)') & (df_tun_renamed['Particion'] == 'Test')].set_index('Modelo')
        
        x = np.arange(len(mod_names))
        width = 0.35
        
        # R2
        r2_def = [df_def_24.loc[m, 'R2'] for m in mod_names]
        r2_tun = [df_tun_24.loc[m, 'R2'] for m in mod_names]
        ax1.bar(x - width/2, r2_def, width, label='Por Defecto', color='#e74c3c', alpha=0.85)
        ax1.bar(x + width/2, r2_tun, width, label='Optimizado (Optuna)', color='#2ecc71', alpha=0.85)
        ax1.set_title("Incremento de R² en Prueba Ciega (Mayor es mejor)", fontsize=11, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(mod_names, rotation=15, ha='right')
        ax1.set_ylim(0, 0.85)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper left', fontsize=9)
        
        # MAE
        mae_def = [df_def_24.loc[m, 'MAE'] for m in mod_names]
        mae_tun = [df_tun_24.loc[m, 'MAE'] for m in mod_names]
        ax2.bar(x - width/2, mae_def, width, label='Por Defecto', color='#e74c3c', alpha=0.85)
        ax2.bar(x + width/2, mae_tun, width, label='Optimizado (Optuna)', color='#2ecc71', alpha=0.85)
        ax2.set_title("Reducción de MAE en Prueba Ciega (Menor es mejor)", fontsize=11, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(mod_names, rotation=15, ha='right')
        ax2.set_ylabel("MAE (ug/m3)")
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right', fontsize=9)
        
        plt.suptitle("Impacto de la Optimización Bayesiana en Test (24 Horas)", fontsize=13, fontweight='bold', y=0.99)
        plt.tight_layout()
        p8_path = os.path.join(graficos_dir, "08_comparativa_defecto_vs_tuneados_24h.png")
        plt.savefig(p8_path)
        plt.close()
        print(f"[GRAFICO 8] Guardado: {p8_path}")
        
    print("=" * 70)
    print("PIPELINE AUTONOMO COMPLETADO CON EXITO.")
    print("=" * 70)

if __name__ == "__main__":
    main()
