import pandas as pd
import requests
import numpy as np
import io
import warnings
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib

warnings.filterwarnings('ignore')

def download_sinca_data(from_year_yy=0):
    """
    Descarga la serie historica completa de PM2.5 directamente desde SINCA (Parque O'Higgins).
    Si falla la conexion, recurre a los datos locales de respaldo.
    """
    from datetime import datetime
    today_str = datetime.now().strftime('%y%m%d')
    from_str = f"{from_year_yy:02d}0101"

    url = (
        "https://sinca.mma.gob.cl/cgi-bin/APUB-MMA/apub.tsindico2.cgi?"
        "outtype=xcl&"
        "macro=./RM/D14/Cal/PM25//PM25.diario.diario.ic&"
        f"from={from_str}&"
        f"to={today_str}&"
        "path=/usr/airviro/data/CONAMA/&"
        "lang=esp&rsrc=&macropath="
    )
    try:
        response = requests.get(url, verify=False, timeout=30)
        if response.status_code == 200 and 'FECHA' in response.text:
            df = pd.read_csv(
                io.StringIO(response.text),
                sep=';',
                decimal=',',
                na_values=['', ' ', 'NaN'],
                dtype={'FECHA (YYMMDD)': str}
            )
            print(f"Datos descargados exitosamente del SINCA en tiempo real (hasta {today_str}).")
        else:
            raise Exception("URL no disponible o formato incorrecto")
    except Exception as e:
        print(f"Aviso: No se pudo conectar al SINCA ({e}). Usando datos locales de Entrega 3...")
        local_path = r'C:\Users\Usuario\Documents\Personal\Universidad\Semestres\7° Semestre\Minería de datos\Entrega 3\datos_000101_260508.csv'
        df = pd.read_csv(local_path, sep=';', decimal=',', na_values=['', ' ', 'NaN'], dtype={'FECHA (YYMMDD)': str})

    # Limpieza estandar para ambos casos
    df.columns = [c.strip() for c in df.columns]
    
    # Transformacion de fechas (YYMMDD)
    df['FECHA_STR'] = df['FECHA (YYMMDD)'].astype(str).str.split('.').str[0].str.zfill(6)
    df['Fecha'] = pd.to_datetime(df['FECHA_STR'], format='%y%m%d')
    df = df.sort_values('Fecha').reset_index(drop=True)
    
    # Consolidar MP2.5
    df['MP25'] = (
        df['Registros validados']
        .fillna(df.get('Registros preliminares', np.nan))
        .fillna(df.get('Registros no validados', np.nan))
    )
    
    # Eliminar nulos en el target para entrenamiento
    df = df.dropna(subset=['MP25'])
    
    # Tratamiento de outliers: Winsorizar al percentil 99
    # Los picos extremos (incendios, inversiones termicas) distorsionan el modelo
    p99 = df['MP25'].quantile(0.99)
    p01 = df['MP25'].quantile(0.01)
    df['MP25'] = df['MP25'].clip(lower=p01, upper=p99)
    print(f"Outliers tratados: valores limitados al rango [{p01:.1f}, {p99:.1f}] ug/m3")
    
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
    
    # Stacking con KFold sin shuffle (mantiene bloques temporales secuenciales).
    # Nota: TimeSeriesSplit no es compatible con StackingRegressor (requiere particiones completas).
    # La integridad temporal principal ya está garantizada por la división 80/20 cronológica.
    stack_reg = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(),
        cv=KFold(n_splits=5, shuffle=False)
    )
    
    print("Entrenando Stacking Ensemble...")
    stack_reg.fit(X_train, y_train)
    
    # Evaluación completa: Train vs Test para detectar Overfitting
    r2_train = stack_reg.score(X_train, y_train)
    r2_test = stack_reg.score(X_test, y_test)
    
    preds = stack_reg.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    print(f"\n{'='*50}")
    print(f"  DIAGNOSTICO DEL MODELO")
    print(f"{'='*50}")
    print(f"  R2 Train:  {r2_train:.4f}")
    print(f"  R2 Test:   {r2_test:.4f}")
    print(f"  Diferencia: {r2_train - r2_test:.4f}  {'[!] Posible Overfitting' if (r2_train - r2_test) > 0.10 else '[OK]'}")
    print(f"  MAE:  {mae:.2f} ug/m3 (error promedio)")
    print(f"  RMSE: {rmse:.2f} ug/m3 (penaliza errores grandes)")
    print(f"{'='*50}\n")
    
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
