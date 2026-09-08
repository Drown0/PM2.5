# Sistema Predictivo de la Concentración de Material Particulado Fino (PM2.5) Mediante Modelos de Apilamiento Heterogéneo y Sincronización en Tiempo Real

**Subtítulo:** Estudio Aplicado en la Estación Parque O'Higgins, Santiago de Chile  
**Ámbito:** Memoria / Informe de Proyecto de Titulación — 8° Semestre  
**Fecha:** Septiembre 2026  

---

## Resumen

La contaminación atmosférica por material particulado fino (PM2.5) constituye una problemática crítica de salud pública en la cuenca de Santiago de Chile, acentuada por las condiciones orográficas de confinamiento y el fenómeno recurrente de inversión térmica invernal. Este trabajo presenta el diseño, entrenamiento, auditoría y despliegue de un sistema predictivo capaz de estimar con 24 horas de anticipación la concentración diaria de PM2.5 en la estación de monitoreo Parque O'Higgins (Código SINCA: 273). A partir de 25 años de registros continuos provistos por el Sistema de Información Nacional de Calidad del Aire (SINCA), se diseñó un pipeline de ingeniería de características temporales, autorregresivas y estacionales cíclicas. La arquitectura del modelo implementa un ensamble por apilamiento (*Stacking Regressor*) que integra tres algoritmos de aumento de gradiente (*XGBoost*, *LightGBM* y *CatBoost*), cuyas predicciones son sintetizadas por un meta-modelo de regresión Ridge regularizado con validación temporal estricta. El sistema alcanzó un coeficiente de determinación ($R^2$) de 0,6982 en el conjunto de prueba independiente, con un Error Absoluto Medio (MAE) de 5,65 µg/m³. Asimismo, se resolvió la ingesta automatizada de datos en tiempo real mediante la conexión directa al servicio oficial CGI del SINCA y se desarrolló una aplicación web funcional en Streamlit, sentando las bases teóricas y técnicas para una futura aplicación móvil de impacto ciudadano.

**Palabras clave:** PM2.5, Calidad del Aire, Stacking Ensemble, Machine Learning, Series de Tiempo, SINCA, Salud Pública.

---

## Abstract

Air pollution caused by fine particulate matter (PM2.5) represents a severe public health challenge in the Santiago Metropolitan Basin, Chile, aggravated by geographic confinement and winter thermal inversions. This study presents the design, training, auditing, and deployment of a predictive system capable of forecasting daily PM2.5 concentrations 24 hours in advance at the Parque O'Higgins monitoring station (SINCA ID: 273). Utilizing 25 years of continuous historical records from the National Air Quality Information System (SINCA), an autoregressive, moving-window, and cyclical seasonal feature engineering pipeline was developed. The model architecture implements a Stacking Ensemble Regressor combining three gradient boosting algorithms (XGBoost, LightGBM, and CatBoost) aggregated via a regularized Ridge meta-estimator under strict sequential temporal cross-validation. The system achieved a coefficient of determination ($R^2$) of 0.6982 on the unseen test set, with a Mean Absolute Error (MAE) of 5.65 µg/m³. Furthermore, automated real-time data ingestion was established via direct integration with SINCA's official CGI backend, supported by a functional Streamlit cloud web deployment, establishing the theoretical and technical foundation for a citizen-facing mobile application.

**Keywords:** PM2.5, Air Quality, Stacking Ensemble, Machine Learning, Time Series, SINCA, Public Health.

---

## 1. Introducción y Planteamiento del Problema

La exposición prolongada y aguda a elevadas concentraciones de material particulado respirable fino (PM2.5) representa uno de los principales factores de riesgo de morbilidad y mortalidad por enfermedades cardiovasculares y respiratorias en zonas urbanas densamente pobladas. En Chile, la cuenca geográfica de Santiago experimenta condiciones desfavorables de ventilación durante el otoño e invierno, provocando episodios críticos que obligan a la autoridad ambiental a decretar medidas de emergencia (Zapata González, 2017). Para monitorear esta situación, el Ministerio del Medio Ambiente (MMA) gestiona el Sistema de Información Nacional de Calidad del Aire (SINCA).

No obstante, la literatura especializada evidencia que gran parte de los esfuerzos predictivos previos se circunscriben a modelos de corto alcance (24 a 48 horas) o sufren de fragilidad operativa ante interrupciones de datos (Contreras & Da Costa, 2025; Lagos & Campos, 2025). Por otra parte, existe una brecha sustancial entre el monitoreo estatal de carácter reactivo y la necesidad de herramientas predictivas proactivas orientadas al ciudadano común, que permitan planificar actividades físicas escolares, deportes al aire libre y traslados con conocimiento previo y certeza probabilística.

---

## 2. Marco Teórico y Estado del Arte

Los métodos de pronóstico ambiental transitan desde modelos químicos-meteorológicos deterministas de alta demanda computacional hasta modelos estadísticos univariados lineales (como ARIMA). Sin embargo, la dispersión atmosférica de contaminantes en cuencas cerradas exhibe un comportamiento marcadamente no lineal, condicionado por emisiones vehiculares, actividad industrial, radiación solar y dinámicas de inversión térmica (Toro et al., 2019).

En este escenario, el paradigma de aprendizaje automático ha demostrado superioridad en precisión. En particular, los algoritmos basados en árboles de decisión con aumento de gradiente (*Gradient Boosting Decision Trees*):
* **XGBoost** (Chen & Guestrin, 2016): Destacado por su control de regularización y tratamiento de relaciones no lineales severas.
* **LightGBM** (Ke et al., 2017): Eficiente en series temporales extensas gracias a su división por hojas (*leaf-wise*).
* **CatBoost** (Prokhorenkova et al., 2018): Utiliza árboles simétricos (*oblivious trees*) que previenen el sobreajuste y procesan variables cíclicas con mínima varianza.

Asimismo, la teoría de generalización por apilamiento (*Stacked Generalization*) desarrollada por Wolpert (1992) demuestra que la combinación de múltiples estimadores diversos mediante un meta-modelo lineal regularizado (*Ridge*) reduce simultáneamente el sesgo y la varianza de la inferencia final, superando el rendimiento de cualquier algoritmo aislado.

---

## 3. Metodología

### 3.1. Fuente de Datos y Curaduría
Se extrajo la serie temporal completa de la estación Parque O'Higgins (Código SINCA: 273), abarcando más de 9.700 observaciones diarias entre enero del año 2000 y septiembre de 2026. Se aplicó consolidación jerárquica de registros validados y preliminares, junto con un tratamiento de valores atípicos mediante winsorización al percentil 1%-99% ($[6.0, 88.3]$ µg/m³), neutralizando el impacto de incendios forestales o fallas transitorias de sensores sin eliminar información temporal válida.

### 3.2. Formulación del Vector de Características
El modelo recibe 10 variables predictoras construidas a partir de la física del contaminante y el calendario:
1. **Autorregresivas:** $\text{lag}_1 = \text{PM2.5}(t-1)$, $\text{lag}_2 = \text{PM2.5}(t-2)$, $\text{lag}_3 = \text{PM2.5}(t-3)$.
2. **Estadísticas Móviles:** Media de 3 días (`rolling_mean_3`), desviación estándar muestral con $N-1$ grados de libertad (`rolling_std_3`, $\text{ddof}=1$), y media de 7 días (`rolling_mean_7`).
3. **Cinemática:** Derivada discreta de primer orden ($\text{diff}_1 = \text{lag}_1 - \text{lag}_2$).
4. **Cíclicas Estacionales:** $\text{mes\_sin} = \sin(2\pi \cdot \text{Mes}/12)$ y $\text{mes\_cos} = \cos(2\pi \cdot \text{Mes}/12)$.
5. **Máscara Antropogénica:** Variable binaria `Es_FinDeSemana`.

### 3.3. Partición Temporal y Validación Cruzada Libre de Fuga
Para evitar la fuga de información (*Data Leakage*), se utilizó una partición estrictamente cronológica: 80% inicial para entrenamiento y 20% más reciente para prueba. La validación cruzada interna del *Stacking* se configuró mediante bloques secuenciales sin reordenamiento aleatorio (`KFold(n_splits=5, shuffle=False)`).

---

## 4. Resultados Experimentales y Discusión

El modelo Stacking Ensemble alcanzó un coeficiente de determinación $R^2$ de **0,6706 en entrenamiento** y **0,6982 en prueba**. La diferencia negativa ($-0,0276$) descarta la presencia de sobreajuste (*overfitting*). En magnitudes físicas reales, el Error Absoluto Medio (**MAE**) fue de **5,65 µg/m³** y la Raíz del Error Cuadrático Medio (**RMSE**) alcanzó **8,31 µg/m³**.

Se identificó el fenómeno de **"inercia de rezago"**: al operar como un modelo univariado basado exclusivamente en mediciones pasadas de PM2.5, el sistema tarda cerca de 24 horas en acusar el impacto de perturbaciones climáticas repentinas, estableciendo un límite superior en torno al 70% de explicación de la varianza.

---

## 5. Clasificación Normativa Chilena (Decretos MMA)

El pronóstico continuo se mapea a las categorías oficiales del Plan de Prevención y Descontaminación Atmosférica (PPDA, Decreto Supremo N° 31/2016) y la norma primaria (Decreto Supremo N° 12/2011):
* **Bueno (0 a 50 µg/m³):** Condiciones seguras para deportes al aire libre.
* **Regular (51 a 79 µg/m³):** Grupos sensibles deben moderar esfuerzos prolongados.
* **Alerta (80 a 109 µg/m³):** Suspensión de deportes escolares; precaución para asmáticos y adultos mayores.
* **Pre-emergencia (110 a 169 µg/m³):** Paralización de fuentes industriales fijas y restricción vehicular extendida.
* **Emergencia ($\ge 170$ µg/m³):** Prohibición estricta de actividad física para la población general.

---

## 6. Discusión, Limitaciones y Propuesta de Titulación

La delimitación más relevante del modelo actual reside en la ausencia de variables exógenas como Humedad Relativa (%), Temperatura e Inversión Térmica (°C), y Velocidad del Viento (km/h). En Santiago, la humedad fomenta el crecimiento higroscópico de los aerosoles, mientras que el enfriamiento superficial atrapa la polución.

### Propuesta de Trabajo para la Titulación:
1. **Transición a Modelo Multivariado:** Incorporar variables meteorológicas en tiempo real y pronosticadas desde una API abierta (como Open-Meteo o la Dirección Meteorológica de Chile), permitiendo superar el 85% de certeza.
2. **Extensión del Horizonte Temporal:** Ampliar la ventana predictiva desde 24 horas hacia 7 a 14 días.
3. **Despliegue de Aplicación Móvil:** Desarrollar una app nativa en Flutter conectada a un backend en FastAPI, incorporando notificaciones push preventivas para la ciudadanía.

---

## 7. Conclusiones

Se desarrolló y auditó un sistema predictivo robusto, resiliente y de grado productivo para la estimación de PM2.5 en Santiago. La resolución de fallas técnicas de fuga de datos, el tratamiento de valores atípicos y la exitosa ingeniería inversa del protocolo de SINCA garantizan la operatividad del sistema hasta el día de hoy, conformando una base sólida para la culminación del proyecto de titulación.

---

## 8. Referencias Bibliográficas (Normas APA 7ª Edición)

* Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM. https://doi.org/10.1145/2939672.2939785
* Contreras, C. I., & Da Costa, V. V. (2025). *Sistema predictivo con modelos de series de tiempo y técnicas de Deep Learning de la calidad del aire (PM2.5)* [Tesis de pregrado, Universidad Bernardo O'Higgins]. Repositorio UBO, Santiago, Chile.
* Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67.
* Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, 3146–3154.
* Lagos, D., & Campos, D. (2025). Particulate Matter 2.5 Prediction in Osorno, Chile, Using a Discrete Markov Chain Model. *Environmental Research, Engineering and Management*, 81(3), 124–133.
* Ministerio del Medio Ambiente. (2011). *Decreto Supremo N° 12: Establece norma primaria de calidad ambiental para material particulado fino respirable MP2,5*. Biblioteca del Congreso Nacional de Chile.
* Ministerio del Medio Ambiente. (2016). *Decreto Supremo N° 31: Establece Plan de Prevención y Descontaminación Atmosférica para la Región Metropolitana de Santiago*. Biblioteca del Congreso Nacional de Chile.
* Ministerio del Medio Ambiente. (s. f.). *Sistema de Información Nacional de Calidad del Aire (SINCA)*. Gobierno de Chile. https://sinca.mma.gob.cl/
* Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
* Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. *Advances in Neural Information Processing Systems (NeurIPS 2018)*, 31, 6638–6648.
* Toro, R., Kahan, D. B., & Morales, R. G. (2019). Atmospheric chemistry and pollution dynamics in the Santiago Metropolitan Basin. *Atmospheric Environment*, 214, 116843.
* Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241–259. https://doi.org/10.1016/S0893-6080(05)80023-1
* Zapata González, E. A. (2017). *Estimación de la calidad del aire en la ciudad de Talca utilizando algoritmos de aprendizaje automático* [Tesis de magíster, Universidad de Talca]. Repositorio Dspace, Curicó, Chile.
