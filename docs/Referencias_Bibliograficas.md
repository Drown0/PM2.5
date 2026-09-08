# Compendio de Referencias Bibliográficas y Fuentes Técnicas

**Proyecto:** Sistema Predictivo de Calidad del Aire (PM2.5) — Estación Parque O'Higgins  
**Ámbito:** Proyecto de Titulación (8° Semestre)  
**Norma Editorial:** APA 7ª Edición con Justificación Técnica de Aplicación  
**Fecha:** Septiembre 2026  

---

## 1. Investigaciones Académicas y Tesis sobre Calidad del Aire en Chile

* **Contreras, C. I., & Da Costa, V. V. (2025).** *Sistema predictivo con modelos de series de tiempo y técnicas de Deep Learning de la calidad del aire (PM2.5)* [Tesis de pregrado, Universidad Bernardo O'Higgins]. Repositorio Institucional UBO, Santiago, Chile.
  > **Aplicación en el Proyecto:** Aporta el benchmark empírico sobre la efectividad de los modelos de Deep Learning y series de tiempo en la Región Metropolitana, evidenciando la barrera de las 24 horas y sirviendo como base teórica de comparación para nuestro modelo Stacking Ensemble.

* **Lagos, D., & Campos, D. (2025).** Particulate Matter 2.5 Prediction in Osorno, Chile, Using a Discrete Markov Chain Model. *Environmental Research, Engineering and Management*, 81(3), 124–133. https://doi.org/10.5755/j01.erem.81.3.37684
  > **Aplicación en el Proyecto:** Utilizado como antecedente para el modelado de transiciones de episodios críticos de contaminación en ciudades chilenas y la discusión sobre el horizonte temporal acotado de los modelos probabilísticos clásicos.

* **Zapata González, E. A. (2017).** *Estimación de la calidad del aire en la ciudad de Talca utilizando algoritmos de aprendizaje automático* [Tesis de magíster, Universidad de Talca]. Repositorio Digital Dspace, Curicó, Chile.
  > **Aplicación en el Proyecto:** Fundamenta el análisis de la "inercia de rezago" en la zona central de Chile, así como el impacto disruptivo de anomalías ambientales impredecibles (incendios forestales y quemas) en los modelos predictivos de Machine Learning.

* **Toro, R., Kahan, D. B., & Morales, R. G. (2019).** Atmospheric chemistry and pollution dynamics in the Santiago Metropolitan Basin: A multi-decadal perspective. *Atmospheric Environment*, 214, 116843. https://doi.org/10.1016/j.atmosenv.2019.116843
  > **Aplicación en el Proyecto:** Sustenta la justificación física de las transformaciones cíclicas anuales (seno y coseno de los meses) para modelar la estacionalidad invernal y las inversiones térmicas en la cuenca de Santiago.

---

## 2. Fuentes Institucionales, Datos Abiertos y Normativa Chilena

* **Ministerio del Medio Ambiente. (s. f.).** *Sistema de Información Nacional de Calidad del Aire (SINCA)*. Gobierno de Chile. https://sinca.mma.gob.cl/
  > **Aplicación en el Proyecto:** Fuente oficial primaria de los datos históricos (2000-2026) y en tiempo real. Provee las mediciones continuas de material particulado y gases de la estación Parque O'Higgins (Código SINCA: 273).

* **Ministerio del Medio Ambiente. (2011).** *Decreto Supremo N° 12: Establece norma primaria de calidad ambiental para material particulado fino respirable MP2,5*. Biblioteca del Congreso Nacional de Chile (BCN). https://www.bcn.cl/leychile/navegar?idNorma=1025000
  > **Aplicación en el Proyecto:** Define formalmente los umbrales de concentración diaria (0 a 50 µg/m³ Bueno, 51-79 Regular, 80-109 Alerta, 110-169 Pre-emergencia y $\ge 170$ Emergencia) que el sistema utiliza para clasificar las estimaciones numéricas.

* **Ministerio del Medio Ambiente. (2016).** *Decreto Supremo N° 31: Establece Plan de Prevención y Descontaminación Atmosférica (PPDA) para la Región Metropolitana de Santiago*. Biblioteca del Congreso Nacional de Chile (BCN).
  > **Aplicación en el Proyecto:** Marco legal que fundamenta las medidas operativas ciudadanas e industriales asociadas a cada episodio crítico pronosticado (restricción vehicular, suspensión de actividad física, fiscalización industrial).

* **Apertum IT AB. (2020).** *Airviro Air Quality Management System: Specifications and APUB-MMA Indico CGI Interface Manual*. Apertum Information Technology, Gothenburg, Sweden.
  > **Aplicación en el Proyecto:** Documentación técnica de ingeniería inversa utilizada para desentrañar el servicio CGI oficial de exportación de series temporales (`apub.tsindico2.cgi`) del portal SINCA.

---

## 3. Algoritmos de Machine Learning y Frameworks de Computación

* **Wolpert, D. H. (1992).** Stacked generalization. *Neural Networks*, 5(2), 241–259. https://doi.org/10.1016/S0893-6080(05)80023-1
  > **Aplicación en el Proyecto:** Publicación seminal que establece la teoría matemática del "Stacking Ensemble": combinación de múltiples estimadores base mediante un meta-modelo de generalización para reducir la varianza y el sesgo de inferencia.

* **Chen, T., & Guestrin, C. (2016).** XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). Association for Computing Machinery. https://doi.org/10.1145/2939672.2939785
  > **Aplicación en el Proyecto:** Referencia del primer modelo base del ensamble. XGBoost se implementó para capturar no-linealidades agudas y valores atípicos severos en la serie temporal.

* **Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017).** LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, 3146–3154.
  > **Aplicación en el Proyecto:** Fundamenta el segundo modelo base del ensamble. Aporta optimización por división de hojas (*leaf-wise*) y alta velocidad de procesamiento sobre datos secuenciales de gran volumen.

* **Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018).** CatBoost: Unbiased boosting with categorical features. *Advances in Neural Information Processing Systems (NeurIPS 2018)*, 31, 6638–6648.
  > **Aplicación en el Proyecto:** Sustenta el tercer modelo base del ensamble. Su arquitectura de árboles simétricos (*oblivious trees*) previene el sobreajuste y procesa eficientemente las variables temporales cíclicas (seno/coseno del mes).

* **Hoerl, A. E., & Kennard, R. W. (1970).** Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67. https://doi.org/10.1080/00401706.1970.10488634
  > **Aplicación en el Proyecto:** Fundamento matemático del meta-modelo RidgeCV. Aplica regularización L2 para balancear y ponderar de forma óptima los pesos asignados a XGBoost, LightGBM y CatBoost.

* **Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011).** Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
  > **Aplicación en el Proyecto:** Librería central utilizada para el ensamblador StackingRegressor, partición temporal KFold secuencial, métricas de error (MAE, RMSE, R²) y serialización del modelo con Joblib.

---

## 4. Arquitectura de Software, Despliegue Web y APIs Futuras

* **Evans, E. (2003).** *Domain-Driven Design: Tackling complexity in the heart of software*. Addison-Wesley Professional.
  > **Aplicación en el Proyecto:** Fundamento del patrón arquitectónico "Anti-Corruption Layer" (ACL) justificado en el informe técnico de descarga, utilizado para aislar la aplicación móvil de las dependencias crípticas del servidor ministerial legacy.

* **Streamlit Inc. (2024).** *Streamlit architecture and caching primitives: st.cache_data and resource lifecycle management*. Snowflake Inc. https://docs.streamlit.io/
  > **Aplicación en el Proyecto:** Documentación oficial utilizada para la implementación de la memoria caché con tiempo de vida de 1 hora (`@st.cache_data(ttl=3600)`), garantizando alta reactividad sin saturar los enlaces del MMA.

* **Open-Meteo GmbH. (2024).** *Open-Meteo Weather Forecast API Documentation: High-resolution historical reanalysis and hourly forecasts*. https://open-meteo.com/en/docs
  > **Aplicación en el Proyecto:** Fuente referenciada en la propuesta de trabajo de titulación para la transición a un modelo multivariado, permitiendo incorporar sin costo Humedad Relativa, Temperatura a 2m y Velocidad del Viento a escala horaria.
