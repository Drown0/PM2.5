import pandas as pd
import requests
import numpy as np
import io
import warnings
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib

warnings.filterwarnings('ignore')

def download_sinca_data(station_id=83, parameter='MP25'):
    """
    Intenta descargar datos del SINCA, si falla usa el archivo local de la Entrega 3.
    """
    # Intentamos la URL que suele funcionar en el portal nuevo
    url = f"https://sinca.mma.gob.cl/cgi-bin/ap_ex_csv.cgi?id={station_id}&param={parameter}&type=diario"
    try:
        response = requests.get(url, verify=False, timeout=10)
        if response.status_code == 200 and 'FECHA' in response.text:
            df = pd.read_csv(io.StringIO(response.text), sep=';', decimal=',', na_values=['', ' ', 'NaN'])
            print("Datos descargados exitosamente del SINCA.")
        else:
            raise Exception("URL no disponible o formato incorrecto")
    except Exception as e:
        print(f"Aviso: No se pudo conectar al SINCA ({e}). Usando datos locales de Entrega 3...")
        # Ruta al archivo local proporcionado en el contexto
        local_path = r'C:\Users\Usuario\Documents\Personal\Universidad\Semestres\7° Semestre\Minería de datos\Entrega 3\datos_000101_260508.csv'
        df = pd.read_csv(local_path, sep=';', decimal=',', na_values=['', ' ', 'NaN'])

    # Limpieza estándar para ambos casos
    df.columns = [c.strip() for c in df.columns]
    
    # Transformación de fechas (YYMMDD)
    df['FECHA_STR'] = df['FECHA (YYMMDD)'].astype(str).str.zfill(6)
    df['Fecha'] = pd.to_datetime(df['FECHA_STR'], format='%y%m%d')
    df = df.sort_values('Fecha').reset_index(drop=True)
    
    # Consolidar MP2.5
    df['MP25'] = df['Registros validados'].fillna(df['Registros preliminares']).fillna(df['Registros no validados'])
    
    # Eliminar nulos en el target para entrenamiento
    df = df.dropna(subset=['MP25'])
    
    return df[['Fecha', 'MP25']]

def engineer_features(df):
    """
    Aplica Ingeniería de Características avanzada.
    """
    df = df.copy()
    
    # 1. Lags (Rezagas)
    for i in range(1, 4):
        df[f'lag_{i}'] = df['MP25'].shift(i)
    
    # 2. Ventanas Móviles (Rolling Windows)
    df['rolling_mean_3'] = df['MP25'].shift(1).rolling(window=3).mean()
    df['rolling_std_3'] = df['MP25'].shift(1).rolling(window=3).std()
    df['rolling_mean_7'] = df['MP25'].shift(1).rolling(window=7).mean()
    
    # 3. Diferenciales (Momentum)
    df['diff_1'] = df['lag_1'] - df['lag_2']
    
    # 4. Características Temporales Cíclicas
    df['Mes'] = df['Fecha'].dt.month
    df['Dia_Semana'] = df['Fecha'].dt.dayofweek
    df['Es_FinDeSemana'] = df['Dia_Semana'].isin([5, 6]).astype(int)
    
    # Seno/Coseno para Mes (Ciclo de 12 meses)
    df['mes_sin'] = np.sin(2 * np.pi * df['Mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['Mes'] / 12)
    
    # 5. Limpieza de NaNs generados por lags/rolling
    df = df.dropna().reset_index(drop=True)
    
    return df

def train_professional_model(df):
    """
    Entrena un Stacking Ensemble de alto rendimiento.
    """
    features = ['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'rolling_std_3', 
                'rolling_mean_7', 'diff_1', 'mes_sin', 'mes_cos', 'Es_FinDeSemana']
    X = df[features]
    y = df['MP25']
    
    # División temporal (no aleatoria para series de tiempo)
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # Definición de modelos base
    estimators = [
        ('lgb', LGBMRegressor(n_estimators=200, learning_rate=0.05, verbosity=-1)),
        ('xgb', XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)),
        ('cat', CatBoostRegressor(n_estimators=200, learning_rate=0.05, depth=5, silent=True))
    ]
    
    # Stacking (Usa Ridge como meta-modelo para combinar predicciones)
    stack_reg = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(),
        cv=5
    )
    
    print("Entrenando Stacking Ensemble...")
    stack_reg.fit(X_train, y_train)
    
    # Evaluación
    preds = stack_reg.predict(X_test)
    r2 = stack_reg.score(X_test, y_test)
    print(f"R2 Score en Test: {r2:.4f}")
    
    return stack_reg, features

if __name__ == "__main__":
    print("Iniciando Pipeline de Datos...")
    data = download_sinca_data()
    print(f"Datos descargados: {len(data)} registros.")
    
    processed_data = engineer_features(data)
    print(f"Datos procesados con ingeniería de características.")
    
    model, feature_list = train_professional_model(processed_data)
    
    # Guardar modelo y lista de features para la Demo
    joblib.dump(model, 'modelo_final_mp25.joblib')
    joblib.dump(feature_list, 'features_list.joblib')
    print("Modelo guardado como 'modelo_final_mp25.joblib'")
