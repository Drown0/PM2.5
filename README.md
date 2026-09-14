# 🌬️ Plataforma de Monitoreo y Alerta Temprana MP2.5 - Estación Parque O'Higgins

**Versión V3:** Sistema predictivo multi-ventana (**24h, 48h y 72h**) de la concentración diaria de Material Particulado Fino (MP2.5) en la estación Parque O'Higgins (Santiago, Chile), impulsado por técnicas de Gradient Boosting (**LightGBM, XGBoost, CatBoost**) combinadas mediante ensamble por apilamiento (**Stacking Regressor**), optimizadas bayesianamente con **Optuna** y sincronizadas en tiempo real con la red oficial del **SINCA** (Ministerio del Medio Ambiente).

## 🚀 Acceso a la Demo en Vivo
Puedes acceder a la plataforma interactiva desplegada en Streamlit Cloud aquí:  
👉 **[https://pm25-prediccion-del-aire.streamlit.app/](https://pm25-prediccion-del-aire.streamlit.app/)**

---

## 📁 Estructura del Repositorio Operativo

```text
PM2.5/
├── .gitignore
├── README.md
└── Aplicacion/
    ├── 01_Entrenamiento/
    │   ├── extraer_datos_sinca.py              # Ingesta automatizada desde servidor CGI Airviro
    │   ├── imputar_datos_iterative.py          # Saneamiento de nulos con MICE (BayesianRidge + PM10)
    │   ├── evaluar_modelos_default.py          # Evaluación base y diagnóstico de sobreajuste
    │   ├── optimizar_boosting_optuna.py        # Optimización bayesiana (4.500 trials) con Walk-Forward
    │   ├── optuna_estudio.db                   # Base de datos SQLite con los 4.500 trials persistidos
    │   └── datos/                              # Datasets históricos consolidados e imputados (2020–2026)
    └── 02_App_Publicada/
        ├── app.py                             # Interfaz interactiva Streamlit V3 multi-ventana
        ├── requirements.txt                   # Dependencias de producción
        ├── datos_respaldo.csv                 # Respaldo local de contingencia (auto-actualizable con PM2.5 y PM10)
        ├── features_list.joblib               # Vector de 16 características de ingeniería
        ├── modelo_final_mp25_tuneado_24h.joblib # Ensamble Stacking optimizado a 24 horas
        ├── modelo_final_mp25_tuneado_48h.joblib # Ensamble Stacking optimizado a 48 horas
        └── modelo_final_mp25_tuneado_72h.joblib # Ensamble Stacking optimizado a 72 horas
```

---

## 🧠 Arquitectura del Sistema Predictivo

### 1. Pronóstico Directo Multi-Horizonte (3 Ventanas de Tiempo)
En lugar de pronósticos autorregresivos recursivos que acumulan error paso a paso, el sistema opera bajo una formulación de **Pronóstico Directo Multi-Paso** con modelos especializados independientes:
* **Ventana 24 horas ($t+1$):** Proyección para el día de mañana.
* **Ventana 48 horas ($t+2$):** Proyección para pasado mañana.
* **Ventana 72 horas ($t+3$):** Proyección a tres días vista.

### 2. Junta de Expertos (Stacking Ensemble Regressor)
Cada horizonte cuenta con un ensamble heterogéneo apilado que combina:
* **XGBoost:** Regularización elástica L1/L2 para capturar picos agudos no lineales.
* **LightGBM:** Alta velocidad computacional y modelado sensible de gradientes estacionales.
* **CatBoost:** Árboles simétricos (*oblivious trees*) con máxima resistencia estructural al sobreajuste.
* **Meta-Modelo RidgeCV:** Regresión lineal con regularización $L_2$ que pondera de forma equilibrada las inferencias de los tres estimadores base.

### 3. Ingeniería de Características (16 Variables)
* **Lags temporales:** $\text{lag}_1$, $\text{lag}_2$, $\text{lag}_3$ y $\text{lag}_7$ (estacionalidad semanal).
* **Filtros estadísticos móviles:** Media y desviación estándar de 3 días (`ddof=1`), y media semanal de 7 días.
* **Cinemática discreta:** Diferencial de primer orden ($\text{diff}_1 = \text{lag}_1 - \text{lag}_2$).
* **Estacionalidad armónica:** $\sin/\cos(2\pi \cdot \text{Mes}/12)$ y $\sin/\cos(2\pi \cdot \text{DíaAño}/365,25)$.
* **Actividad antropogénica:** Día de la semana y máscara binaria `es_fin_de_semana`.
* **Covariables físicas:** Retardos pasados de la serie de material particulado grueso ($\text{PM10}$).

---

## 🔬 Comparativa Experimental: Modelos por Defecto vs. Optimizados con Optuna

Evaluación realizada sobre el **conjunto de prueba ciego independiente** (348 días fuera de muestra: mayo de 2025 a septiembre de 2026):

| Horizonte | Modelo | Configuración | $R^2$ Test | MAE ($\mu\text{g/m}^3$) | RMSE ($\mu\text{g/m}^3$) | MAPE (%) | Evaluación |
|---|---|---|---|---|---|---|---|
| **24h ($t+1$)** | LightGBM | Por Defecto | 0,3410 | 7,63 | 11,64 | 48,60% | Sobreajustado |
| **24h ($t+1$)** | LightGBM | **Optimizado Optuna** | **0,5198** | **6,67** | **9,94** | **43,54%** | **+52,4% en $R^2$** |
| **24h ($t+1$)** | XGBoost | **Optimizado Optuna** | **0,5225** | **6,71** | **9,91** | **43,82%** | **+23,4% en $R^2$** |
| **24h ($t+1$)** | CatBoost | **Optimizado Optuna** | **0,5367** | **6,57** | **9,76** | **42,48%** | **Mejor Individual 24h** |
| **24h ($t+1$)** | **Stacking** | **Optimizado Optuna** | **0,5353** | **6,61** | **9,78** | **42,47%** | **Ensamble Regularizado** |
| **48h ($t+2$)** | LightGBM | Por Defecto | 0,2497 | 8,07 | 12,42 | 52,43% | Sobreajustado |
| **48h ($t+2$)** | LightGBM | **Optimizado Optuna** | **0,4397** | **7,16** | **10,74** | **47,42%** | **+76,1% en $R^2$** |
| **48h ($t+2$)** | **Stacking** | **Optimizado Optuna** | **0,4846** | **6,90** | **10,30** | **45,25%** | **Mejor Modelo 48h** |
| **72h ($t+3$)** | LightGBM | Por Defecto | 0,2574 | 7,95 | 12,35 | 51,32% | Sobreajustado |
| **72h ($t+3$)** | LightGBM | **Optimizado Optuna** | **0,4078** | **7,34** | **11,03** | **48,91%** | **+58,4% en $R^2$** |
| **72h ($t+3$)** | **Stacking** | **Optimizado Optuna** | **0,4528** | **7,07** | **10,61** | **46,55%** | **Mejor Modelo 72h** |

---

## 🛡️ Sincronización en Tiempo Real y Tolerancia a Fallos

* **Ingesta Automática SINCA (Airviro CGI):** Conexión periódica al backend oficial (`apub.tsindico2.cgi`), consumiendo registros validados y preliminares de PM2.5 y PM10.
* **Caché en Memoria (`@st.cache_data(ttl=3600)`):** Garantiza tiempos de respuesta sub-segundo (< 0,05 s) sin saturar los servicios gubernamentales.
* **Mecanismo de Resiliencia (*Circuit Breaker*):** Conmutación transparente hacia `datos_respaldo.csv` si el servidor estatal no responde, actualizando automáticamente el respaldo en cada petición exitosa.
* **Clasificación Normativa PPDA (D.S. N° 31/2016):** Mapeo automático de predicciones a estados normativos (Bueno, Regular, Alerta, Pre-emergencia, Emergencia) con recomendaciones sanitarias específicas.
