# 🌬️ Plataforma de Monitoreo y Alerta Temprana MP2.5 - Estación Parque O'Higgins

**Versión V4 (Boosting Multivariado):** Sistema predictivo multi-ventana (**24h, 48h y 72h**) de la concentración diaria de Material Particulado Fino ($PM_{2.5}$) en la estación Parque O'Higgins (Santiago de Chile, SINCA D14/273). Impulsado por modelos individuales de Gradient Boosting (**LightGBM, XGBoost, CatBoost**) calibrados mediante **Optimización Bayesiana (Optuna)** con **Early Stopping (paciencia = 30)** sobre validación anual (2025) y sincronizados con la red oficial del **SINCA** (Ministerio del Medio Ambiente).

## 🚀 Acceso a la Demo en Vivo
Puedes acceder a la plataforma interactiva desplegada en Streamlit Cloud aquí:  
👉 **[https://pm25-prediccion-del-aire.streamlit.app/](https://pm25-prediccion-del-aire.streamlit.app/)**

---

## 🌟 Novedades y Mejoras Metodológicas en la Versión V4

1. **Descarte de Stacking Ensemble:** Se eliminó la complejidad innecesaria del ensamble por apilamiento en favor de modelos individuales de Gradient Boosting puros, interpretables y altamente optimizados.
2. **Saneamiento MICE del Target en Train y Retiro de PM10:** Reconstrucción de lagunas históricas de $PM_{2.5}$ en el conjunto de entrenamiento (2020–2024) utilizando $PM_{10}$ ($r = 0,88$) con `IterativeImputer(BayesianRidge)`. Posteriormente, **$PM_{10}$ y $SO_2$ se eliminan definitivamente del espacio de características**, garantizando que el modelo no dependa de $PM_{10}$ en inferencia operativa.
3. **Selección Multivariada por Correlación (Clase 05 - Diapo 75):** Integración de precursores químicos vehiculares e industriales ($CO, NO_X, NO_2, NO, O_3$) y variables meteorológicas de ventilación e inversión térmica ($WSPD, TEMP, RHUM$), respaldadas por una matriz de correlación de Pearson y Spearman.
4. **Imputación de Variables Exógenas sin Fuga de Información (Clase 07 - Diapo 33):** Imputador iterativo ajustado (`fit`) exclusivamente sobre el periodo de Train (2020–2024) y aplicado (`transform`) sobre Validación y Test (*Zero Data Leakage*).
5. **Partición Anual Estricta y Early Stopping:**
   * **Train:** 2020 a 2024 (1.820 días).
   * **Validation:** Año 2025 completo (365 días), utilizado para detener el boosting si pasan 30 iteraciones consecutivas sin mejora en $R^2$.
   * **Test:** Año 2026 a la fecha actual (263 días), evaluado **estrictamente contra mediciones reales del sensor SINCA**.
6. **Rendimiento Máximo Alcanzado:** El modelo campeón a 24 horas (**XGBoost Tuneado**) alcanzó **$R^2 = 0,6965$** y un error **$RMSE = 8,63\ \mu\text{g/m}^3$**, superando en más de un **340%** la persistencia estacional del Baseline Seasonal Naive ($R^2 = 0,1570$).

---

## 📁 Estructura del Repositorio Operativo

```text
PM2.5/
├── .gitignore
├── README.md                                  <- Documentación general V4
└── Aplicacion/
    ├── Entrenamiento/                         <- Pipeline de modelado secuencial y reproducible
    │   ├── 01_extraccion_datos_sinca.py       <- Descarga multivariada automática desde CGI SINCA
    │   ├── 02_saneamiento_mice_y_correlacion.py <- MICE en Train, retiro de PM10, correlación e imputación de X
    │   ├── 03_evaluar_y_optimizar_modelos.py  <- Baseline SNaive, Boosting defecto y Optuna con Early Stopping
    │   ├── ejecutar_pipeline_completo.py      <- Orquestador maestro para correr todo con 1 comando
    │   ├── README.md                          <- Guía técnica de la fase de entrenamiento
    │   ├── tabla_comparativa_defecto_vs_tuneados.csv <- Métricas oficiales consolidadas
    │   ├── datos/                             <- Datasets sinópticos y particiones Train/Val/Test
    │   └── modelos/                           <- Modelos serializados .joblib
    └── App/                                   <- Aplicación Streamlit en producción
        ├── app.py                             <- Interfaz interactiva V4 multi-ventana (24h, 48h, 72h)
        ├── requirements.txt                   <- Dependencias de producción
        ├── datos_respaldo.csv                 <- Respaldo local multivariado reciente (tolerancia a fallos)
        ├── tabla_comparativa_defecto_vs_tuneados.csv <- Tabla de métricas embebida en la app
        └── modelos/                           <- Modelos campeones cargados por la app
            ├── features_list.joblib           <- Vector de 32 variables de entrada
            ├── modelo_final_mp25_tuneado_24h.joblib <- Campeón 24h: XGBoost (R² = 0.6965)
            ├── modelo_final_mp25_tuneado_48h.joblib <- Campeón 48h: LightGBM (R² = 0.5293)
            └── modelo_final_mp25_tuneado_72h.joblib <- Campeón 72h: CatBoost (R² = 0.4909)
```

---

## 🔬 Matriz Comparativa de Resultados (Test Ciego 2026)

Evaluación en el conjunto de prueba independiente (263 días reales del año 2026):

| Horizonte | Modelo | Configuración | $R^2$ Test | RMSE ($\mu\text{g/m}^3$) | MAE ($\mu\text{g/m}^3$) | MAPE (%) | Estado / Evaluación |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **24h ($t+1$)** | Seasonal Naive ($s=7$) | Baseline Canónico | 0,1570 | 14,38 | 9,86 | 56,12% | Referencia Básica |
| **24h ($t+1$)** | LightGBM | Por Defecto | 0,6904 | 8,72 | 6,17 | 32,90% | Fuerte Desempeño Base |
| **24h ($t+1$)** | XGBoost | Por Defecto | 0,6376 | 9,43 | 6,55 | 33,64% | Línea Base Boosting |
| **24h ($t+1$)** | CatBoost | Por Defecto | 0,6638 | 9,08 | 6,19 | 32,05% | Regularizado de Fábrica |
| **24h ($t+1$)** | **XGBoost Tuneado 🏆** | **Optuna (Early Stop)** | **0,6965** | **8,63** | **5,92** | **31,48%** | **Campeón 24h ($R^2 \approx 0,70$)** |
| **24h ($t+1$)** | LightGBM Tuneado | Optuna (Early Stop) | 0,6846 | 8,80 | 6,03 | 31,91% | Alta Precisión |
| **24h ($t+1$)** | CatBoost Tuneado | Optuna (Early Stop) | 0,6774 | 8,90 | 6,17 | 32,53% | Muy Estable |
| | | | | | | | |
| **48h ($t+2$)** | Seasonal Naive ($s=7$) | Baseline Canónico | 0,2316 | 13,75 | 9,57 | 53,96% | Referencia Básica |
| **48h ($t+2$)** | LightGBM | Por Defecto | 0,4895 | 11,21 | 8,02 | 48,12% | Línea Base 48h |
| **48h ($t+2$)** | XGBoost | Por Defecto | 0,4579 | 11,55 | 8,05 | 46,67% | Línea Base 48h |
| **48h ($t+2$)** | CatBoost | Por Defecto | 0,5174 | 10,90 | 7,65 | 44,86% | Robusto |
| **48h ($t+2$)** | **LightGBM Tuneado 🏆** | **Optuna (Early Stop)** | **0,5293** | **10,76** | **7,57** | **45,27%** | **Campeón Global 48h** |
| **48h ($t+2$)** | XGBoost Tuneado | Optuna (Early Stop) | 0,5229 | 10,83 | 7,61 | 46,53% | Muy Competitivo |
| **48h ($t+2$)** | CatBoost Tuneado | Optuna (Early Stop) | 0,5183 | 10,89 | 7,65 | 43,87% | Menor Dispersión |
| | | | | | | | |
| **72h ($t+3$)** | Seasonal Naive ($s=7$) | Baseline Canónico | 0,2264 | 13,80 | 10,06 | 54,14% | Referencia Básica |
| **72h ($t+3$)** | LightGBM | Por Defecto | 0,4261 | 11,89 | 8,28 | 51,78% | Línea Base 72h |
| **72h ($t+3$)** | XGBoost | Por Defecto | 0,3511 | 12,64 | 8,99 | 55,84% | Línea Base 72h |
| **72h ($t+3$)** | CatBoost | Por Defecto | 0,4577 | 11,56 | 7,93 | 49,23% | Resistente |
| **72h ($t+3$)** | **CatBoost Tuneado 🏆** | **Optuna (Early Stop)** | **0,4909** | **11,20** | **7,90** | **48,32%** | **Campeón Global 72h** |
| **72h ($t+3$)** | XGBoost Tuneado | Optuna (Early Stop) | 0,4748 | 11,37 | 8,00 | 49,83% | Calibrado |
| **72h ($t+3$)** | LightGBM Tuneado | Optuna (Early Stop) | 0,4400 | 11,75 | 8,22 | 51,10% | Calibrado |

---

## ⚡ Ejecución Rápida del Pipeline

```bash
cd Aplicacion/Entrenamiento

# Ejecutar el pipeline completo de inicio a fin:
python ejecutar_pipeline_completo.py --trials 40
```
