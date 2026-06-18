# 🌬️ Predictor de Calidad del Aire MP2.5 - Estación Parque O'Higgins

Este proyecto desarrolla un sistema de minería de datos para la predicción de la concentración de Material Particulado Fino (MP2.5) con 24 horas de antelación, utilizando un modelo de **Stacking Ensemble**.

> [!IMPORTANT]
> **Propósito Académico:** Este trabajo es parte de la Entrega 5 del curso de Minería de Datos (7° Semestre). Su objetivo es demostrar la capacidad de transitar desde datos crudos hacia un producto funcional (MVP).

## 🚀 Acceso a la Demo en Vivo
Puedes probar la aplicación interactiva desplegada en Streamlit Cloud aquí:
**https://pm25-prediccion-del-aire.streamlit.app/**

---

## 📁 Estructura del Proyecto

El repositorio está organizado siguiendo el ciclo de vida de un proyecto de Ciencia de Datos:

- **`01_Entrenamiento/`**: Scripts de Python para la extracción, ingeniería de características y entrenamiento del modelo Stacking.
- **`02_App_Publicada/`**: Código fuente de la interfaz de usuario, archivos del modelo persistido (`.joblib`) y base de datos de respaldo.
- **`03_Material_Presentacion/`**: Guías de defensa técnica y recursos para la exposición final.

---

## 🧠 Metodología y Modelo

El sistema utiliza una arquitectura de **Stacking Regressor** que combina las fortalezas de tres algoritmos líderes:
1. **XGBoost:** Optimizado para capturar valores extremos y variaciones bruscas.
2. **LightGBM:** Alta eficiencia y precisión en patrones de series de tiempo.
3. **CatBoost:** Excelente manejo de variables temporales y estacionalidad.

> [!TIP]
> **Ingeniería de Características:** El modelo no solo usa datos pasados simples, sino que incorpora "Lags" de hasta 3 días, ventanas móviles de promedio semanal y transformaciones cíclicas (Seno/Coseno) para modelar con precisión el invierno en Santiago.

---

## 🛡️ Robustez y Resiliencia

> [!NOTE]
> **Sistema de Datos Híbrido:** La aplicación intenta sincronizarse en tiempo real con el **SINCA** (MMA Chile). En caso de que el servidor oficial no esté disponible, el sistema activa automáticamente un **respaldo local (CSV)** para garantizar la continuidad del servicio.

> [!CAUTION]
> **Limitaciones del Modelo:** Al ser un modelo basado en datos históricos y meteorológicos, presenta una "inercia de rezago". Eventos fortuitos repentinos (como incendios forestales) pueden causar desviaciones en la predicción de corto plazo.

---

## 📈 Resultados Principales

- **Coeficiente de Determinación ($R^2$):** **0.695** (Explicación del ~70% de la variabilidad).
- **Categorización:** El sistema traduce valores numéricos a estados normativos chilenos (Bueno, Alerta, Pre-emergencia, etc.).

> [!WARNING]
> **Uso de la Información:** Esta es una herramienta de planificación preventiva y educativa. Para decisiones legales o sanitarias oficiales, consulte siempre los decretos del Ministerio del Medio Ambiente.
