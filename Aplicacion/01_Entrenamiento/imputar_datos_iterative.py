"""
Tratamiento de Valores Nulos mediante IterativeImputer (MICE)
Proyecto: Sistema Predictivo de Calidad del Aire (PM2.5) - Estacion Parque O'Higgins
Periodo: 2020 a la fecha actual
"""

import os
import io
import warnings
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

warnings.filterwarnings('ignore')

def obtener_datos_pm10(datos_dir, today_str):
    """
    Obtiene la serie de PM10 de la estacion Parque O'Higgins desde SINCA.
    Si ya existe localmente, la lee de disco.
    """
    pm10_path = os.path.join(datos_dir, "sinca_pm10_2020_presente_consolidado.csv")
    if os.path.exists(pm10_path):
        df10 = pd.read_csv(pm10_path, sep=';', decimal=',')
        df10['Fecha'] = pd.to_datetime(df10['Fecha'])
        print(f"[CACHE] Datos de PM10 cargados localmente desde: {pm10_path}")
        return df10
    
    macro = "./RM/D14/Cal/PM10//PM10.diario.diario.ic"
    url = (
        "https://sinca.mma.gob.cl/cgi-bin/APUB-MMA/apub.tsindico2.cgi?"
        "outtype=xcl&"
        f"macro={macro}&"
        f"from=200101&"
        f"to={today_str}&"
        "path=/usr/airviro/data/CONAMA/&"
        "lang=esp&rsrc=&macropath="
    )
    print(f"[SINCA] Descargando PM10 para correlacion fisica multivariada...")
    r = requests.get(url, verify=False, timeout=30)
    df10 = pd.read_csv(io.StringIO(r.text), sep=';', decimal=',', dtype={'FECHA (YYMMDD)': str})
    df10.columns = [c.strip() for c in df10.columns]
    df10['FECHA_STR'] = df10['FECHA (YYMMDD)'].astype(str).str.split('.').str[0].str.zfill(6)
    df10['Fecha'] = pd.to_datetime('20' + df10['FECHA_STR'], format='%Y%m%d', errors='coerce')
    
    col_val = 'Registros validados'
    col_pre = 'Registros preliminares'
    col_no_val = 'Registros no validados'
    
    df10['PM10'] = (
        df10[col_val]
        .fillna(df10.get(col_pre, np.nan))
        .fillna(df10.get(col_no_val, np.nan))
    )
    
    df10_clean = df10[['Fecha', 'PM10']].sort_values('Fecha').reset_index(drop=True)
    df10_clean.to_csv(pm10_path, sep=';', decimal=',', index=False)
    print(f"[GUARDADO] PM10 guardado en: {pm10_path}")
    return df10_clean


def imputar_con_iterative():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datos_dir = os.path.join(base_dir, "datos")
    pm25_path = os.path.join(datos_dir, "sinca_pm25_2020_presente_consolidado.csv")
    
    if not os.path.exists(pm25_path):
        raise FileNotFoundError(f"No se encontro el archivo base: {pm25_path}")
    
    df = pd.read_csv(pm25_path, sep=';', decimal=',')
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    today_str = datetime.now().strftime('%y%m%d')
    df10 = obtener_datos_pm10(datos_dir, today_str)
    
    # 1. Unir MP2.5 con PM10
    df = pd.merge(df, df10[['Fecha', 'PM10']], on='Fecha', how='left')
    
    # 2. Construir covariables temporales y fisicas
    df['mes'] = df['Fecha'].dt.month
    df['dia_ano'] = df['Fecha'].dt.dayofyear
    df['dia_semana'] = df['Fecha'].dt.dayofweek
    df['es_fin_de_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
    
    # Transformaciones ciclicas (seno/coseno)
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    df['dia_ano_sin'] = np.sin(2 * np.pi * df['dia_ano'] / 365.25)
    df['dia_ano_cos'] = np.cos(2 * np.pi * df['dia_ano'] / 365.25)
    
    # Variables de contexto temporal inmediato (rezago previo y posterior)
    df['lag_1'] = df['MP25'].shift(1)
    df['lead_1'] = df['MP25'].shift(-1)
    
    features_imputacion = [
        'MP25', 'PM10', 'lag_1', 'lead_1',
        'mes_sin', 'mes_cos', 'dia_ano_sin', 'dia_ano_cos',
        'dia_semana', 'es_fin_de_semana'
    ]
    
    nulos_antes = df['MP25'].isna().sum()
    indices_nulos = df[df['MP25'].isna()].index.tolist()
    
    print("=" * 65)
    print("PROCESO DE IMPUTACION CON ITERATIVE IMPUTER (BAYESIAN RIDGE)")
    print("=" * 65)
    print(f"Total registros:                  {len(df):,}")
    print(f"Valores nulos en MP2.5 iniciales: {nulos_antes} ({nulos_antes/len(df)*100:.2f}%)")
    print(f"Covariables empleadas:")
    print(f"  - Fisica: PM10 (Correlacion r=0.88 con MP2.5)")
    print(f"  - Contexto temporal: lag_1 (t-1) y lead_1 (t+1)")
    print(f"  - Estacionalidad ciclica: mes (sin/cos), dia del ano (sin/cos)")
    print(f"  - Actividad antropogenica: dia_semana, es_fin_de_semana")
    print("-" * 65)
    
    # 3. Aplicar IterativeImputer con BayesianRidge
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=30,
        random_state=42,
        min_value=0.0  # Las concentraciones de particulas no pueden ser negativas
    )
    
    matriz_imputada = imputer.fit_transform(df[features_imputacion])
    
    # Asignar serie imputada de MP2.5
    df['MP25_Original'] = df['MP25']
    df['MP25_Imputado'] = matriz_imputada[:, 0]
    df['Fue_Imputado'] = df['MP25_Original'].isna().astype(int)
    
    # Consolidar MP25 definitivo sin nulos
    df['MP25'] = df['MP25_Imputado']
    
    nulos_despues = df['MP25'].isna().sum()
    print(f"[OK] Imputacion completada con exito.")
    print(f"Valores nulos remanentes en MP2.5: {nulos_despues}")
    
    # 4. Auditoria y Diagnostico
    print("-" * 65)
    print("EJEMPLOS DE VALORES IMPUTADOS:")
    df_imputados_muestra = df.loc[indices_nulos[:8], ['Fecha', 'PM10', 'MP25_Imputado']]
    for _, row in df_imputados_muestra.iterrows():
        pm10_val = f"{row['PM10']:.1f}" if pd.notna(row['PM10']) else "N/D"
        print(f"  - {row['Fecha'].strftime('%d/%m/%Y')} -> MP2.5 Imputado: {row['MP25_Imputado']:.2f} ug/m3 (PM10: {pm10_val})")
    
    print("-" * 65)
    print("COMPARACION ESTADISTICA (ORIGINAL vs IMPUTADO):")
    stats_orig = df['MP25_Original'].describe(percentiles=[0.01, 0.50, 0.99])
    stats_imp = df['MP25'].describe(percentiles=[0.01, 0.50, 0.99])
    
    comp_df = pd.DataFrame({
        'Original': stats_orig,
        'Imputado': stats_imp
    })
    comp_df['Diferencia'] = comp_df['Imputado'] - comp_df['Original']
    for idx, row in comp_df.iterrows():
        print(f"  - {idx:<12}: Orig={row['Original']:>7.2f} | Imp={row['Imputado']:>7.2f} | Dif={row['Diferencia']:>+6.2f}")
    
    # 5. Guardar dataset imputado definitivo
    cols_salida = ['Fecha', 'MP25', 'MP25_Original', 'PM10', 'Fue_Imputado']
    output_path = os.path.join(datos_dir, "sinca_pm25_2020_presente_imputado.csv")
    df[cols_salida].to_csv(output_path, sep=';', decimal=',', index=False)
    print("=" * 65)
    print(f"[GUARDADO] Dataset imputado final guardado en: {output_path}")
    print("=" * 65)
    
    return df

if __name__ == "__main__":
    imputar_con_iterative()
