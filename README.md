# 🌬️ Predictor de Calidad del Aire MP2.5 - Estación Parque O'Higgins

Sistema de aprendizaje automático para la predicción de la concentración de Material Particulado Fino (MP2.5) con 24 horas de antelación en la estación Parque O'Higgins (Santiago, Chile), utilizando un ensamble por apilamiento (**Stacking Ensemble Regressor**) y sincronización en tiempo real con la red oficial del **SINCA** (Ministerio del Medio Ambiente).

> [!IMPORTANT]
> **Propósito Académico:** Proyecto de Titulación (8° Semestre, Ingeniería Civil / Informática). Desarrollado para sentar las bases analíticas de un producto preventivo y una futura aplicación móvil ciudadana.

## 🚀 Acceso a la Demo en Vivo
Puedes probar la aplicación interactiva desplegada en Streamlit Cloud aquí:  
👉 **[https://pm25-prediccion-del-aire.streamlit.app/](https://pm25-prediccion-del-aire.streamlit.app/)**

---

## 📁 Estructura del Repositorio

El repositorio se organiza siguiendo las mejores prácticas de Ciencia de Datos y Desarrollo de Software:

```text
PM2.5/
├── .gitignore
├── README.md
├── Resumen_Proyecto.docx
├── docs/                                  # Documentación técnica formal de titulación
│   ├── Explicacion_Variables_y_Funcionamiento_Sistema.docx
│   ├── Funcion de descarga de archivos SINCA.docx
│   ├── Funcion de descarga de archivos SINCA.md
│   ├── Referencias_Bibliograficas.docx
│   └── Referencias_Bibliograficas.md
└── Aplicacion/
    ├── 01_Entrenamiento/
    │   └── entrenamiento_avanzado.py      # Pipeline de ingesta, preprocesamiento y entrenamiento
    └── 02_App_Publicada/
        ├── app.py                         # Interfaz Streamlit con sincronización en vivo
        ├── requirements.txt               # Dependencias de producción
        ├── datos_respaldo.csv             # Respaldo local de contingencia (auto-actualizable)
        ├── modelo_final_mp25.joblib       # Ensamble persistido entrenado
        └── features_list.joblib           # Lista de variables del modelo
```

---

## 🧠 Metodología y Arquitectura del Modelo

El sistema utiliza una arquitectura de **Stacking Regressor** (junta de expertos) compuesta por tres modelos base de alto rendimiento combinados mediante un meta-modelo regularizado:

1. **XGBoost:** Optimizado para capturar picos extremos y variaciones no lineales.
2. **LightGBM:** Alta eficiencia computacional y modelado de secuencias temporales.
3. **CatBoost:** Árboles simétricos (*oblivious trees*) con alta resistencia al sobreajuste frente a variables estacionales cíclicas.
4. **Meta-Modelo (RidgeCV):** Regresión lineal con regularización L2 para balancear y ponderar de forma óptima las salidas de los tres estimadores base.

### ⚙️ Ingeniería de Características (Variables del Modelo)
El modelo actual opera bajo un esquema **autorregresivo univariado**:
* **Lags temporales:** Valores de PM2.5 de $t-1$ (ayer), $t-2$ (anteayer) y $t-3$.
* **Ventanas móviles:** Promedio y desviación estándar de 3 días (`ddof=1`), y promedio semanal (7 días).
* **Cinemática:** Diferencial de primer orden (`lag_1 - lag_2`).
* **Transformaciones cíclicas:** $\sin(2\pi \cdot \text{Mes}/12)$ y $\cos(2\pi \cdot \text{Mes}/12)$ para modelar la estacionalidad invernal en Santiago.
* **Actividad antropogénica:** Variable binaria `Es_FinDeSemana` (reducción de tráfico e industrias).

---

## 🛡️ Robustez y Sincronización en Tiempo Real

* **Ingesta Directa SINCA (Airviro CGI):** Conexión automatizada al servicio oficial `apub.tsindico2.cgi`, descargando los registros consolidados hasta el día de ayer.
* **Caché Reactiva:** Implementación de `@st.cache_data(ttl=3600)` (1 hora de validez) para tiempos de respuesta instantáneos (< 0.05 s) sin saturar los servidores del Ministerio.
* **Mecanismo de Resiliencia (Circuit Breaker):** Conmutación transparente a `datos_respaldo.csv` si el servidor estatal no está disponible, y auto-actualización silenciosa del respaldo en cada descarga exitosa.

---

## 📈 Métricas y Diagnóstico del Modelo

Tras auditar y subsanar fugas de información (*Data Leakage*) con particiones secuenciales temporales y tratamiento de valores atípicos (winsorización 1%-99%):

* **$R^2$ Train:** `0.6706` | **$R^2$ Test:** `0.6982` *(Diferencia: -0.0276, sin sobreajuste)*
* **MAE (Error Absoluto Medio):** `5.65 µg/m³`
* **RMSE (Error Cuadrático Medio):** `8.31 µg/m³`

---

## 📚 Documentación Técnica Adicional

Para mayor detalle teórico, metodológico y referencias normativas, consultar la carpeta [`docs/`](./docs):
* **[Explicación de Variables y Funcionamiento Integral](./docs/Explicacion_Variables_y_Funcionamiento_Sistema.docx)**
* **[Arquitectura de Descarga SINCA e Ingeniería Inversa](./docs/Funcion%20de%20descarga%20de%20archivos%20SINCA.md)**
* **[Compendio de Referencias Bibliográficas (APA 7)](./docs/Referencias_Bibliograficas.md)**
