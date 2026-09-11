# Tarea 2 · Predicción de falla cardíaca con MLOps local

[![CI](https://github.com/RubenEsg/Machine-Learning-Tarea-2/actions/workflows/ci.yml/badge.svg)](https://github.com/RubenEsg/Machine-Learning-Tarea-2/actions/workflows/ci.yml)

Proyecto integrador de Aprendizaje Automático (sección 10.12 del curso de Machine Learning del Dr. Lihki Rubio). Se desarrolla, evalúa y despliega un modelo de clasificación que estima el riesgo de enfermedad cardíaca (`HeartDisease` = 1 o 0) a partir de 11 variables clínicas.

**Integrantes:** Martínez Pulido Valerie · Basto Martínez Abrahan · Esguerra Fernández Rubén

| Entregable | Enlace |
|---|---|
| Informe (Jupyter Book) | https://rubenesg.github.io/Machine-Learning-Tarea-2/ |
| Notebook de la etapa 1 | [`notebooks/1_model_leakage_demo.ipynb`](notebooks/1_model_leakage_demo.ipynb) |
| Notebook de las etapas 2, 3 y 6 | [`notebooks/2_model_pipeline_cv.ipynb`](notebooks/2_model_pipeline_cv.ipynb) |

**Dataset:** [Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (Kaggle): 918 pacientes de cinco hospitales. Está incluido en `data/heart.csv`.

## Arquitectura y tecnologías utilizadas

- **scikit-learn**: `Pipeline` y `GridSearchCV` con validación cruzada.
- **Jupyter Notebooks**: análisis exploratorio, *data leakage*, comparación de modelos, entrenamiento y evaluación.
- **FastAPI + Docker**: API REST para predicciones, empaquetada en una imagen de Docker.
- **Kubernetes (Minikube)**: despliegue local con un `Deployment` y un `Service` tipo `LoadBalancer`.
- **GitHub Actions**: revisión de estilo (flake8) y pruebas (pytest) en cada *push*.
- **Evidently**: reporte de deriva de datos (*data drift*).
- **Jupyter Book**: informe publicado en GitHub Pages.

```text
data/heart.csv ──► notebooks 1 y 2 ──► app/model.joblib ──► API (FastAPI) ──► Docker ──► Kubernetes
                                                   │
                     GitHub Actions: flake8 + pytest   Evidently: drift_report.html
```

## Resultados

Se compararon 13 clasificadores del curso con `Pipeline` + `GridSearchCV`. Los mejores obtuvieron resultados muy parecidos, y se eligió **Random Forest**:

| Métrica (conjunto de prueba, 184 pacientes) | Valor |
|---|---|
| AUC | 0.928 |
| Accuracy | 0.891 |
| Sensibilidad / especificidad | 0.931 / 0.841 |

## Estructura del proyecto

```text
Machine-Learning-Tarea-2/
├── app/
│   ├── api.py                      # API FastAPI (Etapa 3)
│   └── model.joblib                # modelo entrenado que usa la API
├── data/
│   └── heart.csv                   # dataset
├── docker/
│   ├── Dockerfile                  # imagen de la API (Etapa 3)
│   └── requirements.txt
├── docs/                           # Jupyter Book (informe)
├── k8s/
│   ├── deployment.yaml             # Etapa 4
│   └── service.yaml                # Etapa 4
├── notebooks/
│   ├── 1_model_leakage_demo.ipynb  # Etapa 1
│   └── 2_model_pipeline_cv.ipynb   # Etapas 2, 3 y 6
├── src/                            # funciones compartidas por las pruebas y el monitoreo
├── monitoring/                     # script de monitoreo (Etapa 6)
├── tests/                          # pruebas que ejecuta el CI (Etapa 5)
├── .github/workflows/ci.yml        # Etapa 5
├── drift_report.html               # reporte de Evidently (Etapa 6)
├── model.joblib
└── README.md
```

## Partición de los datos

- Los datos se dividen **antes** de cualquier transformación: 80 % entrenamiento y 20 % prueba, estratificado y con `random_state=42`.
- La imputación, la codificación y el escalado van dentro del `Pipeline`, así que en la validación cruzada solo aprenden de los datos de entrenamiento.
- El conjunto de prueba solo se usa para reportar el resultado final, nunca para elegir el modelo.

## Cómo ejecutarlo

**1. Entorno** (Python 3.12 o 3.13):

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements-dev.txt
```

**2. Notebooks:** ejecuta en orden `1_model_leakage_demo.ipynb` y `2_model_pipeline_cv.ipynb`. El segundo genera `app/model.joblib` y `drift_report.html`.

**3. Pruebas (lo mismo que ejecuta el CI):**

```bash
flake8 app/ src/ tests/ monitoring/
pytest tests/
```

**4. API con Docker:**

```bash
docker build -t heart-api -f docker/Dockerfile .
docker run -p 8000:8000 heart-api
```

La documentación de la API queda en <http://localhost:8000/docs>.

**5. Kubernetes con Minikube:**

```bash
minikube start
minikube image load heart-api:latest
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get svc
minikube service heart-service --url
```

**6. Reporte de deriva:**

```bash
python monitoring/drift_report.py
```

**7. Jupyter Book:**

```bash
jupyter-book build . --config docs/_config.yml --toc docs/_toc.yml --path-output docs
ghp-import -n -p -f docs/_build/html
```

## Uso de la API

`POST /predict` recibe los 11 valores del paciente en el orden de las columnas del dataset:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"features": [54, "M", "ASY", 140, 239, 0, "Normal", 160, "N", 1.2, "Flat"]}'
```

Respuesta:

```json
{"heart_disease_probability": 0.776, "prediction": 1, "threshold": 0.5}
```

También acepta los campos con nombre (`{"Age": 54, "Sex": "M", ...}`). Otros endpoints: `GET /health` y `GET /model-info`.

## Limitaciones

- Son solo 918 pacientes de cinco hospitales; el modelo debería validarse con datos de la población donde se vaya a usar.
- Es un proyecto académico: el modelo no reemplaza el criterio médico.
