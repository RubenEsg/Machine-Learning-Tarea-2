# Etapa 0 — Estructura del proyecto

El enunciado pide una estructura modular para alojar el código, los notebooks, la API, los despliegues y el CI/CD. El repositorio [RubenEsg/Machine-Learning-Tarea-2](https://github.com/RubenEsg/Machine-Learning-Tarea-2) sigue esa estructura:

```text
Machine-Learning-Tarea-2/
├── app/
│   ├── api.py                          # API FastAPI (etapa 3)
│   ├── model.joblib                    # Pipeline entrenado que carga la API
│   └── model_metadata.json             # versiones, partición y métricas del modelo
├── docker/
│   ├── Dockerfile                      # imagen de la API (etapa 3)
│   └── requirements.txt                # dependencias con versiones fijadas
├── k8s/
│   ├── deployment.yaml                 # Deployment (etapa 4)
│   └── service.yaml                    # Service LoadBalancer 80 -> 8000 (etapa 4)
├── notebooks/
│   ├── 1_model_leakage_demo.ipynb      # etapa 1
│   └── 2_model_pipeline_cv.ipynb       # etapas 2, 3 (exportación) y 6
├── .github/
│   └── workflows/
│       └── ci.yml                      # integración continua (etapa 5)
├── drift_report.html                   # reporte de deriva (etapa 6)
├── model.joblib                        # copia del modelo
├── README.md
│
├── data/heart.csv                      # agregado: dataset de Kaggle
├── tests/                              # agregado: pruebas que ejecuta el CI
├── src/                                # agregado: código compartido por las pruebas y el monitoreo
├── monitoring/                         # agregado: script de monitoreo y ejemplo de reporte con deriva
└── docs/                               # agregado: este Jupyter Book
```

La parte superior es exactamente la estructura del enunciado. Se agregaron cinco carpetas:

- **`data/`**: el dataset versionado junto al código, para que todo sea reproducible.
- **`tests/`**: el `ci.yml` del enunciado ejecuta `pytest tests/`, pero esa carpeta no estaba en la estructura; sin ella el CI fallaría.
- **`src/`**: rutas, columnas, partición y monitoreo que comparten las pruebas y el script de monitoreo.
- **`monitoring/`**: el script de la etapa 6 para lotes nuevos de pacientes.
- **`docs/`**: el Jupyter Book con el informe.

Los dos notebooks son **autocontenidos**: cada uno define sus funciones, descarga el dataset si no lo encuentra (verificando su huella SHA-256) e instala las librerías que falten, así que se pueden ejecutar fuera del repositorio. Usan exactamente la misma partición de los datos, y el notebook 2 lo verifica con una huella de los índices de prueba.

## Flujo del proyecto

```text
data/heart.csv ──► notebooks 1 y 2 (EDA, leakage, GridSearchCV) ──► app/model.joblib
                                                                          │
          pytest + flake8 (GitHub Actions) ◄── código ──► Docker ──► Kubernetes (Minikube)
                                                                          │
                     Evidently: entrenamiento vs. datos de predicción ──► drift_report.html
```

## Cómo se publicó este libro

Con Jupyter Book 1, siguiendo el tutorial del curso (la versión 2 cambió los comandos):

```bash
pip install "jupyter-book==1.0.4.post1" ghp-import
jupyter-book build . --config docs/_config.yml --toc docs/_toc.yml --path-output docs
ghp-import -n -p -f docs/_build/html
```
