"""
Script de Extracción y Auditoría de Datos SINCA (MMA Chile)
Estación: Parque O'Higgins (Santiago de Chile)
Período: 2020 a la fecha actual
"""

import os
import io
import warnings
import requests
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')

def extraer_sinca_2020():
    today = datetime.now()
    today_str = today.strftime('%y%m%d')
    from_str = "200101"  # 1 de Enero de 2020
    
    macro = "./RM/D14/Cal/PM25//PM25.diario.diario.ic"
    url = (
        "https://sinca.mma.gob.cl/cgi-bin/APUB-MMA/apub.tsindico2.cgi?"
        "outtype=xcl&"
        f"macro={macro}&"
        f"from={from_str}&"
        f"to={today_str}&"
        "path=/usr/airviro/data/CONAMA/&"
        "lang=esp&rsrc=&macropath="
    )
    
    print("=" * 65)
    print("[SINCA] EXTRACCION DE DATOS (MINISTERIO DEL MEDIO AMBIENTE)")
    print("=" * 65)
    print(f"Estacion:    Parque O'Higgins (Codigo D14 / 273)")
    print(f"Parametro:   Material Particulado Fino (MP2.5) Diario")
    print(f"Rango:       01/01/2020 al {today.strftime('%d/%m/%Y')}")
    print(f"Conectando a: {url[:75]}...")
    
    response = requests.get(url, verify=False, timeout=30)
    if response.status_code != 200 or 'FECHA' not in response.text:
        raise RuntimeError(f"Error al descargar datos de SINCA. Status: {response.status_code}")
    
    print("[OK] Respuesta exitosa recibida del servidor SINCA.")
    
    # Crear directorio de datos si no existe
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datos_dir = os.path.join(base_dir, "datos")
    os.makedirs(datos_dir, exist_ok=True)
    
    raw_path = os.path.join(datos_dir, "sinca_pm25_2020_presente_raw.csv")
    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print(f"[GUARDADO] Archivo crudo guardado en: {raw_path}")
    
    # Parseo y estructuracion con Pandas
    df_raw = pd.read_csv(
        io.StringIO(response.text),
        sep=';',
        decimal=',',
        na_values=['', ' ', 'NaN'],
        dtype={'FECHA (YYMMDD)': str}
    )
    df_raw.columns = [c.strip() for c in df_raw.columns]
    
    # Procesamiento de Fechas
    df = df_raw.copy()
    df['FECHA_STR'] = df['FECHA (YYMMDD)'].astype(str).str.split('.').str[0].str.zfill(6)
    df['Fecha'] = pd.to_datetime('20' + df['FECHA_STR'], format='%Y%m%d', errors='coerce')
    df = df.sort_values('Fecha').reset_index(drop=True)
    
    # Consolidacion de Calidad
    col_val = 'Registros validados'
    col_pre = 'Registros preliminares'
    col_no_val = 'Registros no validados'
    
    n_total = len(df)
    n_val = df[col_val].notna().sum() if col_val in df.columns else 0
    n_pre = df[col_pre].notna().sum() if col_pre in df.columns else 0
    n_no_val = df[col_no_val].notna().sum() if col_no_val in df.columns else 0
    
    df['MP25'] = (
        df[col_val]
        .fillna(df.get(col_pre, np.nan))
        .fillna(df.get(col_no_val, np.nan))
    )
    
    n_validos_total = df['MP25'].notna().sum()
    n_nulos = df['MP25'].isna().sum()
    
    # Guardar version consolidada limpia
    clean_path = os.path.join(datos_dir, "sinca_pm25_2020_presente_consolidado.csv")
    df_clean = df[['Fecha', col_val, col_pre, col_no_val, 'MP25']]
    df_clean.to_csv(clean_path, index=False, sep=';', decimal=',')
    print(f"[GUARDADO] Archivo consolidado guardado en: {clean_path}")
    
    # Reporte estadistico
    print("\n" + "=" * 65)
    print("RESUMEN AUDITORIA DE DATOS EXTRAIDOS (2020 - PRESENTE)")
    print("=" * 65)
    print(f"Total de dias en la serie:         {n_total:,} dias")
    print(f"Fecha de inicio:                   {df['Fecha'].min().strftime('%d/%m/%Y')}")
    print(f"Fecha de termino:                  {df['Fecha'].max().strftime('%d/%m/%Y')}")
    print("-" * 65)
    print(f"Registros Validados:               {n_val:,} ({n_val/n_total*100:.1f}%)")
    print(f"Registros Preliminares:            {n_pre:,} ({n_pre/n_total*100:.1f}%)")
    print(f"Registros No Validados:            {n_no_val:,} ({n_no_val/n_total*100:.1f}%)")
    print(f"Registros con dato MP2.5:          {n_validos_total:,} ({n_validos_total/n_total*100:.1f}%)")
    print(f"Dias sin medicion (nulos):         {n_nulos:,} ({n_nulos/n_total*100:.1f}%)")
    print("-" * 65)
    print("Estadisticas Descriptivas de MP2.5 (ug/m3):")
    stats = df['MP25'].describe(percentiles=[0.01, 0.25, 0.50, 0.75, 0.90, 0.99])
    for k, v in stats.items():
        print(f"  - {k:<12}: {v:.2f}")
    print("=" * 65)
    
    return df_clean

if __name__ == "__main__":
    extraer_sinca_2020()
