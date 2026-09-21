# -*- coding: utf-8 -*-
"""
Script de Extracción Multivariada Oficial SINCA (MMA Chile)
Estación: Parque O'Higgins (Código D14 / 273), Santiago de Chile
Período: 01/01/2020 a la fecha actual

Descarga y consolida:
- Contaminantes químicos: MP2.5, MP10, CO, NOX, NO2, NO, O3, SO2
- Meteorología instrumental: Temperatura (media, mín, máx), Humedad relativa (media), Velocidad del viento (media)
"""

import os
import io
import warnings
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import urllib3

urllib3.disable_warnings()
warnings.filterwarnings('ignore')

def limpiar_flotante(val):
    if pd.isna(val) or val == '':
        return np.nan
    if isinstance(val, str):
        val = val.replace(',', '.').strip()
    try:
        f = float(val)
        return f if f >= 0 else np.nan  # Filtro físico básico (concentraciones >= 0)
    except:
        return np.nan

def descargar_variable_diaria(macro, label, from_date, to_date):
    url = (
        "https://sinca.mma.gob.cl/cgi-bin/APUB-MMA/apub.tsindico2.cgi?"
        "outtype=xcl&"
        f"macro={macro}&"
        f"from={from_date}&"
        f"to={to_date}&"
        "path=/usr/airviro/data/CONAMA/&"
        "lang=esp&rsrc=&macropath="
    )
    r = requests.get(url, verify=False, timeout=35)
    if r.status_code != 200 or 'FECHA' not in r.text:
        print(f"  [ERROR] Falló la descarga de {label} (Status: {r.status_code})")
        return None
    
    df = pd.read_csv(
        io.StringIO(r.text),
        sep=';',
        decimal=',',
        na_values=['', ' ', 'NaN'],
        dtype={'FECHA (YYMMDD)': str}
    )
    df.columns = [c.strip() for c in df.columns]
    
    # Formatear Fecha
    df['FECHA_STR'] = df['FECHA (YYMMDD)'].astype(str).str.split('.').str[0].str.zfill(6)
    df['Fecha'] = pd.to_datetime('20' + df['FECHA_STR'], format='%Y%m%d', errors='coerce')
    
    col_val = 'Registros validados'
    col_pre = 'Registros preliminares'
    col_no_val = 'Registros no validados'
    
    s_val = df[col_val].apply(limpiar_flotante) if col_val in df.columns else pd.Series(np.nan, index=df.index)
    s_pre = df[col_pre].apply(limpiar_flotante) if col_pre in df.columns else pd.Series(np.nan, index=df.index)
    s_no = df[col_no_val].apply(limpiar_flotante) if col_no_val in df.columns else pd.Series(np.nan, index=df.index)
    
    df[label] = s_val.fillna(s_pre).fillna(s_no)
    return df[['Fecha', label]].drop_duplicates(subset=['Fecha']).sort_values('Fecha').reset_index(drop=True)

def descargar_variable_horaria(macro, label, from_date, to_date):
    url = (
        "https://sinca.mma.gob.cl/cgi-bin/APUB-MMA/apub.tsindico2.cgi?"
        "outtype=xcl&"
        f"macro={macro}&"
        f"from={from_date}&"
        f"to={to_date}&"
        "path=/usr/airviro/data/CONAMA/&"
        "lang=esp&rsrc=&macropath="
    )
    r = requests.get(url, verify=False, timeout=60)
    if r.status_code != 200 or 'FECHA' not in r.text:
        print(f"  [ERROR] Falló la descarga horaria de {label} (Status: {r.status_code})")
        return None
    
    lines = [l for l in r.text.splitlines() if l.strip()]
    if len(lines) < 2:
        return None
    
    df = pd.read_csv(
        io.StringIO("\n".join(lines)),
        sep=';',
        decimal=',',
        na_values=['', ' ', 'NaN'],
        dtype={'FECHA (YYMMDD)': str, 'HORA (HHMM)': str}
    )
    df.columns = [c.strip() for c in df.columns]
    
    # Extraer columna de valor (tercera columna usualmente)
    val_cols = [c for c in df.columns if c not in ['FECHA (YYMMDD)', 'HORA (HHMM)', 'Unnamed: 3', '']]
    target_col = val_cols[0] if val_cols else df.columns[2]
    
    df['FECHA_STR'] = df['FECHA (YYMMDD)'].astype(str).str.split('.').str[0].str.zfill(6)
    df['Fecha'] = pd.to_datetime('20' + df['FECHA_STR'], format='%Y%m%d', errors='coerce')
    df['VALOR'] = df[target_col].apply(limpiar_flotante)
    
    # Agrupar a diario
    if label == 'TEMP':
        daily = df.groupby('Fecha')['VALOR'].agg(['mean', 'min', 'max']).reset_index()
        daily.columns = ['Fecha', 'TEMP_mean', 'TEMP_min', 'TEMP_max']
    elif label == 'RHUM':
        daily = df.groupby('Fecha')['VALOR'].agg(['mean']).reset_index()
        daily.columns = ['Fecha', 'RHUM_mean']
    elif label == 'WSPD':
        daily = df.groupby('Fecha')['VALOR'].agg(['mean', 'max']).reset_index()
        daily.columns = ['Fecha', 'WSPD_mean', 'WSPD_max']
    else:
        daily = df.groupby('Fecha')['VALOR'].agg(['mean']).reset_index()
        daily.columns = ['Fecha', f"{label}_mean"]
        
    return daily

def main():
    today = datetime.now()
    today_str = today.strftime('%y%m%d')
    from_str = "200101"  # 1 de Enero de 2020
    
    print("=" * 70)
    print("INICIANDO EXTRACCIÓN MULTIVARIADA SINCA (PARQUE O'HIGGINS)")
    print(f"Período: 01/01/2020 al {today.strftime('%d/%m/%Y')}")
    print("=" * 70)
    
    # 1. Variables diarias (Contaminantes gaseosos y material particulado)
    macros_diarias = {
        'MP25': './RM/D14/Cal/PM25//PM25.diario.diario.ic',
        'PM10': './RM/D14/Cal/PM10//PM10.diario.diario.ic',
        'CO':   './RM/D14/Cal/0004//0004.diario.diario.ic',
        'NOX':  './RM/D14/Cal/0NOX//0NOX.diario.diario.ic',
        'NO2':  './RM/D14/Cal/0003//0003.diario.diario.ic',
        'NO':   './RM/D14/Cal/0002//0002.diario.diario.ic',
        'O3':   './RM/D14/Cal/0008//0008.diario.diario.ic',
        'SO2':  './RM/D14/Cal/0001//0001.diario.diario.ic',
    }
    
    # 2. Variables meteorológicas horarias agregadas a diario
    macros_horarias = {
        'TEMP': './RM/D14/Met/TEMP//horario_000.ic',
        'RHUM': './RM/D14/Met/RHUM//horario_000.ic',
        'WSPD': './RM/D14/Met/WSPD//horario_000.ic',
    }
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datos_dir = os.path.join(base_dir, "datos")
    os.makedirs(datos_dir, exist_ok=True)
    
    master_df = None
    
    print("\n[1/2] Descargando contaminantes químicos diarios...")
    for label, macro in macros_diarias.items():
        print(f"  -> Descargando {label:5} ...", end=" ", flush=True)
        df_var = descargar_variable_diaria(macro, label, from_str, today_str)
        if df_var is not None:
            n_vals = df_var[label].notna().sum()
            n_tot = len(df_var)
            pct = (n_vals / n_tot) * 100 if n_tot > 0 else 0
            print(f"OK ({n_vals}/{n_tot} días válidos - {pct:.1f}%)")
            
            if master_df is None:
                master_df = df_var
            else:
                master_df = pd.merge(master_df, df_var, on='Fecha', how='outer')
        else:
            print("ERROR")
            
    print("\n[2/2] Descargando meteorología instrumental horaria y agregando a diario...")
    for label, macro in macros_horarias.items():
        print(f"  -> Descargando {label:5} ...", end=" ", flush=True)
        df_met = descargar_variable_horaria(macro, label, from_str, today_str)
        if df_met is not None:
            print(f"OK ({len(df_met)} días agregados)")
            if master_df is None:
                master_df = df_met
            else:
                master_df = pd.merge(master_df, df_met, on='Fecha', how='outer')
        else:
            print("ERROR")
            
    master_df = master_df.sort_values('Fecha').reset_index(drop=True)
    master_df = master_df.dropna(subset=['Fecha'])
    
    # Guardar dataset multivariado completo
    out_csv = os.path.join(datos_dir, "sinca_multivariado_2020_presente.csv")
    master_df.to_csv(out_csv, index=False, sep=';', decimal=',')
    
    print("\n" + "=" * 70)
    print("RESUMEN DE AUDITORÍA MULTIVARIADA SINCA")
    print("=" * 70)
    print(f"Archivo guardado en: {out_csv}")
    print(f"Rango temporal:      {master_df['Fecha'].min().date()} al {master_df['Fecha'].max().date()}")
    print(f"Total días en serie: {len(master_df)}")
    print("\nCobertura de datos por columna:")
    for col in master_df.columns:
        if col == 'Fecha':
            continue
        validos = master_df[col].notna().sum()
        nulos = master_df[col].isna().sum()
        pct = (validos / len(master_df)) * 100
        print(f"  - {col:15}: {validos:5} válidos | {nulos:4} nulos ({pct:5.1f}% cobertura)")
        
    print("\nCorrelación lineal (Pearson) preliminar con MP2.5:")
    corr = master_df.drop(columns=['Fecha']).corr(numeric_only=True)['MP25'].sort_values(ascending=False)
    for k, v in corr.items():
        if k != 'MP25':
            print(f"  * {k:15}: r = {v:+.4f}")
            
    print("=" * 70)
    print("PASO 1 COMPLETADO EXITOSAMENTE.")
    print("=" * 70)

if __name__ == "__main__":
    main()
