"""
Orquestador Maestro del Pipeline de Modelado Multivariado PM2.5
Proyecto: Sistema Predictivo de Calidad del Aire - Estación Parque O'Higgins (SINCA D14)

Este script ejecuta de manera secuencial y autónoma los 3 pasos metodológicos del proyecto:
  Paso 1: Ingesta y consolidación multivariada desde SINCA (Química + Meteorología)
  Paso 2: Saneamiento MICE en Train con PM10, eliminación de PM10, correlación (Clase 05) e imputación X (Clase 07)
  Paso 3: Evaluación de Baseline (SNaive), Boosting por Defecto y Optimización con Optuna (Paciencia 30)

Uso:
  python ejecutar_pipeline_completo.py [--trials N] [--skip-download]
"""

import os
import sys
import argparse
import subprocess

def ejecutar():
    parser = argparse.ArgumentParser(description="Ejecutar Pipeline Integral de PM2.5")
    parser.add_argument("--trials", type=int, default=40, help="Número de trials para Optuna por modelo y horizonte (defecto: 40)")
    parser.add_argument("--skip-download", action="store_true", help="Omitir la descarga de SINCA si el CSV multivariado ya existe")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 85)
    print("INICIANDO EJECUCIÓN DEL PIPELINE MAESTRO MULTIVARIADO PM2.5")
    print(f"Directorio de trabajo: {base_dir}")
    print("=" * 85)

    # -------------------------------------------------------------
    # PASO 1: Ingesta de datos SINCA
    # -------------------------------------------------------------
    if not args.skip_download:
        print("\n>>> EJECUTANDO PASO 1: Descarga y Consolidación de Datos Multivariados SINCA...")
        p1 = subprocess.run([sys.executable, "01_extraccion_datos_sinca.py"], cwd=base_dir)
        if p1.returncode != 0:
            print("[ERROR] Falló la extracción de datos de SINCA.")
            sys.exit(p1.returncode)
    else:
        print("\n>>> PASO 1 OMITIDO (--skip-download activado). Usando datos locales existentes.")

    # -------------------------------------------------------------
    # PASO 2: Saneamiento MICE, Correlación (Clase 05) e Imputación X (Clase 07)
    # -------------------------------------------------------------
    print("\n>>> EJECUTANDO PASO 2: Saneamiento MICE, Correlación e Imputación de X...")
    p2 = subprocess.run([sys.executable, "02_saneamiento_mice_y_correlacion.py"], cwd=base_dir)
    if p2.returncode != 0:
        print("[ERROR] Falló el procesamiento y saneamiento de datos.")
        sys.exit(p2.returncode)

    # -------------------------------------------------------------
    # PASO 3: Modelado, Early Stopping y Optimización Bayesiana Optuna
    # -------------------------------------------------------------
    print(f"\n>>> EJECUTANDO PASO 3: Benchmarking y Optimización Optuna ({args.trials} trials)...")
    p3 = subprocess.run([sys.executable, "03_evaluar_y_optimizar_modelos.py", str(args.trials)], cwd=base_dir)
    if p3.returncode != 0:
        print("[ERROR] Falló la evaluación u optimización de modelos.")
        sys.exit(p3.returncode)

    print("\n" + "=" * 85)
    print("PIPELINE MAESTRO COMPLETADO EXITOSAMENTE")
    print("Todos los modelos entrenados, gráficos académicos y tablas de métricas han sido actualizados.")
    print("=" * 85)

if __name__ == "__main__":
    ejecutar()
