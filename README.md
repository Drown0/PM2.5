# 🌬️ Predictor de Calidad del Aire MP2.5 - Estación Parque O'Higgins

Sistema de aprendizaje automático para la predicción de la concentración de Material Particulado Fino (MP2.5) con 24 horas de antelación en la estación Parque O'Higgins (Santiago, Chile), utilizando un ensamble por apilamiento (**Stacking Ensemble Regressor**) y sincronización en tiempo real con la red oficial del **SINCA** (Ministerio del Medio Ambiente).

## 🚀 Acceso a la Demo en Vivo
Puedes probar la aplicación interactiva desplegada en Streamlit Cloud aquí:  
👉 **[https://pm25-prediccion-del-aire.streamlit.app/](https://pm25-prediccion-del-aire.streamlit.app/)**

---

## 📁 Estructura del Proyecto Operativo

El repositorio contiene exclusivamente la aplicación funcional y su pipeline de entrenamiento:

```text
PM2.5/
├── .gitignore
├── README.md
└── Aplicacion/
    ├── 01_Entrenamiento/
    │   └── entrenamiento_avanzado.py      # Pipeline de ingesta, preprocesamiento y entrenamiento
    └── 02_App_Publicada/
        ├── app.py                         # Interfaz Streamlit con sincronización en tiempo real
        ├── requirements.txt               # Dependencias de producción
        ├── datos_respaldo.csv             # Respaldo local de contingencia (auto-actualizable)
        ├── modelo_final_mp25.joblib       # Ensamble persistido entrenado
        └── features_list.joblib           # Lista de variables del modelo
```

---

## 🧠 Metodología y Arquitectura del Modelo

El sistema utiliza una arquitectura de **Stacking Regressor** (junta de expertos) compuesta por tres algoritmos líderes de Gradient Boosting combinados mediante un meta-modelo regularizado:

1. **XGBoost:** Optimizado para capturar picos extremos y variaciones no lineales.
2. **LightGBM:** Alta eficiencia computacional y modelado de secuencias temporales.
3. **CatBoost:** Árboles simétricos (*oblivious trees*) con alta resistencia al sobreajuste frente a variables estacionales cíclicas.
4. **Meta-Modelo (RidgeCV):** Regresión lineal con regularización L2 para balancear y ponderar de forma óptima las salidas de los tres estimadores base.

### ⚙️ Ingeniería de Características (Variables del Modelo)
El modelo opera bajo un esquema autorregresivo univariado optimizado:
* **Lags temporales:** Valores de PM2.5 de $t-1$ (ayer), $t-2$ (anteayer) y $t-3$.
* **Ventanas móviles:** Promedio y desviación estándar de 3 días (`ddof=1`), y promedio semanal (7 días).
* **Cinemática:** Diferencial de primer orden (`lag_1 - lag_2`).
* **Transformaciones cíclicas:** $\sin(2\pi \cdot \text{Mes}/12)$ y $\cos(2\pi \cdot \text{Mes}/12)$ para modelar la estacionalidad anual.
* **Actividad antropogénica:** Variable binaria `Es_FinDeSemana` (reducción de tráfico e industrias).

---

## 🛡️ Robustez y Sincronización en Tiempo Real

* **Ingesta Directa SINCA (Airviro CGI):** Conexión automatizada al servicio oficial `apub.tsindico2.cgi`, descargando los registros consolidados hasta el día de ayer.
* **Caché Reactiva:** Implementación de `@st.cache_data(ttl=3600)` (1 hora de validez) para tiempos de respuesta instantáneos (< 0.05 s) sin saturar los servidores ministeriales.
* **Mecanismo de Resiliencia (Circuit Breaker):** Conmutación transparente a `datos_respaldo.csv` si el servidor estatal no está disponible, y auto-actualización silenciosa del respaldo en cada descarga exitosa.

---

## 📈 Métricas del Modelo

* **$R^2$ Train:** `0.6706` | **$R^2$ Test:** `0.6982` *(Sin sobreajuste)*
* **MAE (Error Absoluto Medio):** `5.65 µg/m³`
* **RMSE (Error Cuadrático Medio):** `8.31 µg/m³`
