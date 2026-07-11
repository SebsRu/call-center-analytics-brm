# BRM Data Scientist Technical Test - [Tu Nombre]

## Overview
Este repositorio contiene la solución completa para la prueba técnica de Data Scientist de BRM. El enfoque adoptado no es puramente académico; es el desarrollo de un **Data Product** integral que conecta datos operativos, inteligencia predictiva y minería de voz para la toma de decisiones estratégicas.

## Project Structure
- `data/`: Contiene los datasets originales y los resultados procesados en formato CSV para su consumo por el dashboard.
- `Modulos 1-4/`: Contiene los notebooks de Jupyter (.ipynb) que documentan el pipeline completo:
    - `01_etl_data_analysis.ipynb`: Limpieza, transformación y EDA.
    - `02_nlp_sentiment.ipynb`: Modelos de clasificación de sentimientos.
    - `03_forecasting.ipynb`: Time series forecasting (XGBoost vs Prophet).
    - `04_speech_to_text.ipynb`: Transcripción de audios con Whisper.
- `app.py`: Dashboard interactivo en Streamlit que consolida los hallazgos.

## Live Demo
Puedes interactuar con el dashboard desplegado aquí:
👉 **(https://call-center-analytics-brm-brwsyewqlterpzhjnpwd9r.streamlit.app/)**

## Technical Stack
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning/AI:** `xgboost`, `prophet`, `transformers` (HuggingFace), `openai-whisper`
- **Visualization/Dashboard:** `streamlit`, `plotly`
- **Environment:** Python 3.x

## Strategic Value
Este dashboard permite a la gerencia de operaciones:
1. **Predecir:** Anticipar picos de demanda para evitar quiebres de SLA.
2. **Monitorizar:** Detectar anomalías en la calidad del servicio en tiempo real.
3. **Optimizar:** Identificar mediante NLP y transcripción de voz las consultas transaccionales de bajo valor, permitiendo su automatización y liberando capacidad humana.

---
*Desarrollado como entregable técnico para BRM.*
