# Pipeline de Modelado Predictivo Multivariado MP2.5

Este directorio contiene la infraestructura de código para la extracción, saneamiento, imputación, benchmarking y optimización bayesiana de modelos para la predicción de Material Particulado Fino ($PM_{2.5}$) en la estación Parque O'Higgins (SINCA Estación D14/273, Santiago de Chile).

---

## 🏗️ Arquitectura Modular del Pipeline

El pipeline se encuentra estructurado en 3 fases secuenciales estandarizadas:

1. **`01_extraccion_datos_sinca.py`**:
   - Descarga automatizada desde el servicio CGI oficial del MMA (SINCA).
   - Ingesta de 7 series químicas diarias ($PM_{2.5}$, $PM_{10}$, $CO$, $NO_X$, $NO_2$, $NO$, $O_3$).
   - Ingesta y agregación diaria de 3 series meteorológicas horarias ($TEMP$, $RHUM$, $WSPD$).
   - Consolidación en `datos/sinca_multivariado_2020_presente.csv`.

2. **`02_saneamiento_mice_y_correlacion.py`**:
   - **Reconstrucción del Target en Train:** Imputación de lagunas de $PM_{2.5}$ en el conjunto de entrenamiento (2020–2024) utilizando $PM_{10}$ mediante `IterativeImputer(BayesianRidge)`.
   - **Eliminación de $PM_{10}$ y $SO_2$:** $PM_{10}$ se descarta por completo para que ningún modelo futuro dependa de él en inferencia. $SO_2$ se descarta por falta de registros válidos en la estación.
   - **Selección de Variables por Correlación (Clase 05 - Diapo 75):** Análisis de correlación de Pearson y Spearman entre $PM_{2.5}$ y las variables químicas y meteorológicas. Generación de gráfico en `Documentos/Version 2 Experimentacion y Modelos/graficos/matriz_correlacion_clase05.png`.
   - **Imputación de $X$ con Cero Data Leakage (Clase 07 - Diapo 33):** `IterativeImputer` ajustado (`fit`) exclusivamente sobre Train y aplicado (`transform`) sobre Validación y Test.
   - **Ingeniería de Características y Partición Anual Estricta:** Generación de 32 variables predictivas (lags, medias móviles, momentum, gases y meteorología) divididas en Train (2020-2024), Validation (2025) y Test (2026).

3. **`03_evaluar_y_optimizar_modelos.py`**:
   - **Baseline:** Proyección de referencia con Seasonal Naive ($s=7$).
   - **Modelos por Defecto:** LightGBM, XGBoost y CatBoost.
   - **Optimización Bayesiana (Optuna):** Búsqueda hiperparamétrica orientada a maximizar directamente $R^2$, con **Early Stopping (paciencia = 30)** evaluado en el conjunto de Validación (2025).
   - **Evaluación Ciega en Test (2026):** Métricas rigurosas $R^2$, RMSE, MAE y MAPE sobre mediciones reales del sensor.
   - **Exportación:** Modelos serializados para la app (`Aplicacion/App/modelos/`), tabla comparativa oficial (`tabla_comparativa_defecto_vs_tuneados.csv`) y 7 gráficos analíticos de alta resolución (300 DPI).

---

## 🚀 Ejecución

### Ejecutar Todo el Pipeline con un solo comando:
```bash
python ejecutar_pipeline_completo.py --trials 40
```

### Ejecutar fases individuales:
```bash
# Paso 1: Ingesta de datos
python 01_extraccion_datos_sinca.py

# Paso 2: Saneamiento, Correlación e Imputación
python 02_saneamiento_mice_y_correlacion.py

# Paso 3: Optimización con Optuna y Evaluación
python 03_evaluar_y_optimizar_modelos.py 40
```
