# Tarea 2 · Predicción de falla cardíaca con MLOps local

[![CI](https://github.com/RubenEsg/Machine-Learning-Tarea-2/actions/workflows/ci.yml/badge.svg)](https://github.com/RubenEsg/Machine-Learning-Tarea-2/actions/workflows/ci.yml)

Proyecto integrador de Aprendizaje Automático (sección 10.12 del curso de Machine Learning del Dr. Lihki Rubio). Desarrolla, evalúa y despliega un modelo de clasificación binaria que estima el riesgo de enfermedad cardíaca (`HeartDisease` = 1 o 0) a partir de 11 variables clínicas, con control de calidad y monitoreo integrados, usando solo herramientas locales y *open source*.

**Integrantes:** Martínez Pulido Valerie · Basto Martínez Abrahan · Esguerra Fernández Rubén

| Entregable | Enlace |
|---|---|
| Informe (Jupyter Book) | https://rubenesg.github.io/Machine-Learning-Tarea-2/ |
| Notebook de la etapa 1 (autocontenido) | [`notebooks/1_model_leakage_demo.ipynb`](notebooks/1_model_leakage_demo.ipynb) · [abrir en Google Colab](https://colab.research.google.com/github/RubenEsg/Machine-Learning-Tarea-2/blob/main/notebooks/1_model_leakage_demo.ipynb) |
| Notebook de las etapas 2, 3 y 6 (autocontenido) | [`notebooks/2_model_pipeline_cv.ipynb`](notebooks/2_model_pipeline_cv.ipynb) · [abrir en Google Colab](https://colab.research.google.com/github/RubenEsg/Machine-Learning-Tarea-2/blob/main/notebooks/2_model_pipeline_cv.ipynb) |

**Dataset:** [Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (Kaggle, fedesoriano): 918 pacientes de cinco bases de UCI (Cleveland, Hungría, Suiza, Long Beach VA y Stalog). Se incluye en `data/heart.csv` (SHA-256 `948420b0…a67aa49b`).

## Arquitectura y tecnologías utilizadas

- **scikit-learn**: `Pipeline` (imputación, codificación, escalado y modelo) y `GridSearchCV` con validación cruzada.
- **Jupyter Notebooks**: los dos del enunciado, autocontenidos. El primero tiene el análisis exploratorio, la demostración de *data leakage* y la comparación de 13 modelos; el segundo, el entrenamiento final, la evaluación, la exportación del modelo y el monitoreo.
- **Jupyter Book**: informe publicado en GitHub Pages.
- **FastAPI + Uvicorn**: API REST para predicciones en tiempo real.
- **Docker**: contenerización de la API (`python:3.10-slim`).
- **Kubernetes (Minikube)**: despliegue local con `Deployment` (2 réplicas y *health checks*) y `Service` tipo `LoadBalancer`.
- **GitHub Actions**: integración continua con *linting* (flake8), pruebas (pytest) y construcción y prueba de humo de la imagen Docker en cada *push*.
- **Evidently**: reportes de deriva de datos (*data drift*) entre entrenamiento y predicción.

```text
data/heart.csv ──► notebook (EDA, leakage, GridSearchCV) ──► app/model.joblib
                                                                  │
          pytest + flake8 (GitHub Actions) ◄── código ──► Docker ──► Kubernetes (Minikube)
                                                                  │
                     Evidently: entrenamiento vs. datos de predicción ──► drift_report.html
```

## Resultados principales

| Métrica del modelo desplegado (Random Forest) | Valor |
|---|---|
| AUC de validación cruzada (5 folds, solo entrenamiento) | 0.931 ± 0.025 |
| AUC de validación cruzada anidada (búsqueda de la etapa 2) | 0.921 ± 0.010 |
| **AUC en prueba** (184 pacientes que el modelo no vio) | **0.928** (IC 95 %: 0.886 a 0.963) |
| Accuracy en prueba | 0.891 |
| Sensibilidad / especificidad en prueba (umbral 0.5) | 0.931 / 0.841 |

Ranking de la etapa 1 (AUC de validación cruzada, 13 clasificadores del curso): XGBoost 0.9332 · Random Forest 0.9313 · Gradient Boosting 0.9308 · *bagging* 0.9297 · AdaBoost 0.9243 · MLP 0.9235 · k-NN 0.9230 · regresión logística 0.9227 · LinearSVC 0.9223 · SVC 0.9194 · Naive Bayes Bernoulli 0.9192 · Naive Bayes gaussiano 0.9180 · árbol de decisión 0.9120 · línea base 0.5000. Según el test de DeLong, en prueba solo el árbol de decisión y la línea base son significativamente peores que el primero del ranking. Se despliega Random Forest, el mejor candidato que depende solo de scikit-learn: XGBoost empata, pero su versión actual no es compatible con Python 3.10 y agregaría unos 450 MB a la imagen.

## Estructura del proyecto

```text
Machine-Learning-Tarea-2/
├── app/
│   ├── api.py                      # API FastAPI (Etapa 3)
│   ├── model.joblib                # Pipeline entrenado que carga la API
│   └── model_metadata.json         # versiones, partición y métricas del modelo
├── data/
│   └── heart.csv                   # dataset de Kaggle
├── docker/
│   ├── Dockerfile                  # imagen de la API (Etapa 3)
│   └── requirements.txt            # dependencias de la API, versiones fijadas
├── docs/                           # Jupyter Book (informe)
│   ├── _config.yml  _toc.yml
│   ├── informe.md                  # título, resumen, metodología, resultados y discusión
│   ├── etapa0_estructura.md  etapa3_api_docker.md  etapa4_kubernetes.md
│   ├── etapa5_ci.md  etapa6_monitoreo.md       # una página por etapa
│   └── images/                     # figuras del informe
├── k8s/
│   ├── deployment.yaml             # Deployment (Etapa 4)
│   └── service.yaml                # Service LoadBalancer 80 -> 8000 (Etapa 4)
├── notebooks/
│   ├── 1_model_leakage_demo.ipynb  # Etapa 1: EDA, preprocesamiento, data leakage y ranking
│   └── 2_model_pipeline_cv.ipynb   # Etapas 2, 3 (exportación) y 6 (monitoreo)
├── src/                            # código compartido por las pruebas y el monitoreo
│   ├── config.py                   #   rutas, columnas, semilla y partición
│   ├── data.py                     #   carga, validación y partición de datos
│   └── monitoring.py               #   reporte de deriva con Evidently
├── monitoring/
│   ├── drift_report.py             # CLI de monitoreo (Etapa 6)
│   └── drift_particion_secuencial.html  # ejemplo de un reporte con deriva
├── tests/                          # pruebas que ejecuta el CI (Etapa 5)
│   ├── test_api.py  test_data.py  test_model.py
├── .github/workflows/ci.yml        # integración continua (Etapa 5)
├── drift_report.html               # reporte de deriva entrenamiento vs. prueba (Etapa 6)
├── model.joblib                    # copia del modelo (estructura de la Etapa 0)
├── requirements-dev.txt            # entorno de desarrollo (notebook, pruebas, monitoreo, libro)
├── pytest.ini  .flake8  .dockerignore  .gitignore  .gitattributes
└── README.md
```

Frente a la estructura de la Etapa 0 se agregaron `data/`, `src/`, `tests/` (el `ci.yml` del enunciado ejecuta `pytest tests/`, pero esa carpeta no estaba en la estructura), `monitoring/` y `docs/`. Los dos notebooks sugeridos se unieron en uno solo, autocontenido: define todas sus funciones reutilizables, descarga el dataset si no lo encuentra (y verifica su SHA-256) e instala las librerías que falten, así que se puede ejecutar fuera del repositorio o en Google Colab.

## Estrategia de partición de los datos

1. **La partición se hace primero**, sobre los datos crudos: 80 % entrenamiento (734) y 20 % prueba (184), estratificada por `HeartDisease` y con `random_state=42`. Ningún paciente queda en ambos conjuntos.
2. **Todo lo que aprende de los datos va dentro del `Pipeline`**: la imputación de los ceros imposibles de `Cholesterol` y `RestingBP` (mediana), el *one-hot encoding* y el escalado. Así, en cada fold de validación cruzada se ajustan solo con la parte de entrenamiento del fold.
3. **Validación cruzada estratificada con barajado** (`StratifiedKFold(5, shuffle=True)`): el CSV está ordenado por hospital de origen y, sin barajar, cada fold sería un hospital distinto (el AUC por fold oscila entre 0.855 y 0.951).
4. **El conjunto de prueba nunca se usa para decidir**, solo para reportar. El preprocesamiento, el modelo, el escalado y los hiperparámetros se eligen con validación cruzada; el umbral de decisión se analiza con predicciones *out-of-fold* del entrenamiento. El análisis exploratorio también usa solo el entrenamiento.
5. **La misma partición en todas partes**: los dos notebooks, las pruebas (`src/data.py`) y el reporte de deriva usan los mismos criterios, y el notebook 2 verifica que su partición sea idéntica a la del notebook 1 con una huella de los índices de prueba (`6f866816e8c7`). Esa huella, guardada en `app/model_metadata.json`, también permite que `tests/test_model.py` verifique que el control de calidad se hace con pacientes que el modelo no vio. La única excepción es la reproducción literal del ejemplo de *data leakage* del enunciado (sección 6.1 del notebook 1), que usa su propia partición y solo ilustra el efecto de la fuga.

## Cómo reproducir el proyecto

Requisitos:

- **Entorno de desarrollo** (notebook, pruebas, monitoreo y libro): Python 3.12 o 3.13 (probado con 3.13). XGBoost 3.4.1 exige Python 3.12 o superior y numpy 2.2.6 no tiene versión para 3.14.
- **API, imagen Docker y CI:** Python 3.10 con `docker/requirements.txt` (no necesitas instalarlo aparte: lo trae la imagen `python:3.10-slim`).
- Para las etapas 3 y 4, además: [Docker Desktop](https://www.docker.com/products/docker-desktop/), [kubectl](https://kubernetes.io/docs/tasks/tools/) y [Minikube](https://minikube.sigs.k8s.io/docs/start/).

### 1. Entorno de desarrollo

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # si PowerShell lo bloquea: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements-dev.txt
```

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

En VS Code, selecciona el intérprete `.venv` como *kernel* de los notebooks.

### 2. Etapas 1, 2, 3 y 6: los notebooks

Ejecuta en orden `notebooks/1_model_leakage_demo.ipynb` (unos 4 minutos) y `notebooks/2_model_pipeline_cv.ipynb` (unos 4 minutos). El segundo regenera `app/model.joblib`, `model.joblib`, `app/model_metadata.json`, `drift_report.html` y `monitoring/drift_particion_secuencial.html`. Desde la terminal:

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/1_model_leakage_demo.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/2_model_pipeline_cv.ipynb
```

Los resultados son reproducibles: todas las semillas están fijadas.

### 3. Pruebas y estilo (lo mismo que ejecuta el CI)

```bash
flake8 app/ src/ tests/ monitoring/
pytest tests/ -v
```

### 4. Etapa 3: API local y Docker

Sin Docker:

```bash
uvicorn app.api:app --reload
```

Con Docker (desde la raíz del proyecto):

```bash
docker build -t heart-api -f docker/Dockerfile .
docker run -p 8000:8000 heart-api
```

La documentación interactiva queda en <http://localhost:8000/docs>.

### 5. Etapa 4: Kubernetes local (Minikube)

```bash
minikube start --driver=docker
docker build -t heart-api -f docker/Dockerfile .
minikube image load heart-api:latest      # copia la imagen local al clúster
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods                          # esperar 2 pods en estado Running (READY 1/1)
kubectl get svc
```

Para obtener la IP externa del `LoadBalancer`, ejecuta `minikube tunnel` en otra terminal (en Windows, como administrador) y usa `http://127.0.0.1/predict` (puerto 80). Otra opción es `minikube service heart-service --url`.

### 6. Etapa 5: integración continua

`.github/workflows/ci.yml` se ejecuta en cada *push* y *pull request*:

1. **build**: Python 3.10, instala `docker/requirements.txt`, ejecuta flake8 y pytest (29 pruebas: API y validación de entradas, datos y partición, y control de calidad del modelo con AUC ≥ 0.85).
2. **docker**: construye la imagen, la levanta y le envía una predicción de prueba.

Los resultados se ven en la pestaña **Actions** del repositorio.

### 7. Etapa 6: monitoreo de deriva

```bash
python monitoring/drift_report.py                                   # entrenamiento vs. prueba
python monitoring/drift_report.py --current lote_pacientes.csv      # entrenamiento vs. pacientes recibidos
python monitoring/drift_report.py --current lote.csv --fail-on-drift # código de salida 1 si hay deriva
```

Sin `--current`, el reporte se guarda en `drift_report.html` (la línea base del proyecto). Con `--current`, se guarda en `monitoring/drift_<nombre del CSV>.html` para no sobrescribir la línea base (o donde indique `--output`). Con la partición del proyecto solo 1 de 11 variables aparece con deriva (lo esperable por azar) y no hay deriva global. Con un lote de otro hospital (`monitoring/drift_particion_secuencial.html`) derivan 8 de 11 variables y el reporte declara deriva del dataset.

### 8. Jupyter Book (informe)

Se usa Jupyter Book 1, como en el tutorial del curso (la versión 2 cambió los comandos). El libro toma los notebooks directamente de `notebooks/` y muestra sus salidas sin volver a ejecutarlos; tiene el informe y una sección por etapa:

```bash
jupyter-book build . --config docs/_config.yml --toc docs/_toc.yml --path-output docs
ghp-import -n -p -f docs/_build/html      # publica en la rama gh-pages (GitHub Pages)
```

## Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | información básica |
| GET | `/health` | estado del servicio (lo usan las *probes* de Kubernetes) |
| GET | `/model-info` | modelo, hiperparámetros, versiones, partición y métricas |
| POST | `/predict` | probabilidad de enfermedad cardíaca y predicción con umbral 0.5 |

`POST /predict` acepta dos formatos equivalentes. Campos con nombre (recomendado, validados por Pydantic):

```json
{"Age": 54, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 140, "Cholesterol": 239,
 "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 160, "ExerciseAngina": "N",
 "Oldpeak": 1.2, "ST_Slope": "Flat"}
```

O el formato del enunciado, una lista con los valores en el orden de las columnas del dataset:

```json
{"features": [54, "M", "ASY", 140, 239, 0, "Normal", 160, "N", 1.2, "Flat"]}
```

Respuesta:

```json
{"heart_disease_probability": 0.776, "prediction": 1, "threshold": 0.5}
```

Ejemplos de uso:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"features": [54, "M", "ASY", 140, 239, 0, "Normal", 160, "N", 1.2, "Flat"]}'
```

```powershell
# En PowerShell "curl" es un alias de Invoke-WebRequest; usa curl.exe o Invoke-RestMethod:
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" `
  -Body '{"features": [54, "M", "ASY", 140, 239, 0, "Normal", 160, "N", 1.2, "Flat"]}'
```

`Cholesterol = 0` o `RestingBP = 0` significan "no medido" (convención del dataset) y el modelo los imputa. La validación es estricta: valores fuera de rango, categorías inexistentes, campos faltantes, tipos incorrectos (por ejemplo `true` o `"54"` en un campo numérico) y números no finitos (`1e400`, `NaN`) devuelven un error 422 que indica el campo, o la posición en la lista, con problemas.

## Correcciones y decisiones respecto al enunciado

| Punto del enunciado | Qué se hizo y por qué |
|---|---|
| El ejemplo usa `df.drop("target")` | En el dataset de Kaggle la variable objetivo es `HeartDisease`. |
| El ejemplo escala todo `X` con `MinMaxScaler` | El CSV tiene 5 columnas de texto; se codifican con one-hot (`ColumnTransformer` en el Pipeline). |
| En el "flujo correcto" del ejemplo sigue `leaky_feature` | Por eso ambos flujos dan AUC = 1.000. El notebook 1 lo demuestra y separa la fuga de preprocesamiento de la fuga por la variable objetivo; sin `leaky_feature` el AUC es 0.924 (con la partición del propio ejemplo). |
| "Tratar valores nulos" | No hay `NaN`, pero `Cholesterol` = 0 (172 filas) y `RestingBP` = 0 (1 fila) son faltantes codificados como 0; se imputan con la mediana dentro del Pipeline. |
| "Implementar todos los modelos vistos hasta esta sección" | Se implementan los 13 clasificadores que presenta el libro (k-NN, regresión logística, LinearSVC, Naive Bayes gaussiano y Bernoulli, árbol de decisión, *bagging*, Random Forest, AdaBoost, Gradient Boosting, XGBoost, SVC y MLP) más una línea base, todos con `Pipeline` + `GridSearchCV`. |
| Dos notebooks en la Etapa 0 | Se mantienen los dos, con los nombres del enunciado; cada uno es autocontenido (define sus funciones, descarga los datos si faltan e instala las librerías que falten). |
| `docker/requirements.txt` sin versiones | Se fijan las versiones: `model.joblib` solo es confiable con la misma versión de scikit-learn y numpy con que se entrenó. Se agrega `pandas`, que la API necesita. |
| `api.py` recibe `features: list` | Se acepta ese formato y además campos con nombre validados por tipo, rango y categoría, necesarios porque hay variables de texto. |
| `model.joblib` en la raíz (Etapa 0) y en `app/` (API) | Se guardan ambos; el Dockerfile copia `app/`, así que la API usa `app/model.joblib`. |
| Dockerfile del enunciado | Se mantiene su estructura y se agrega un usuario sin privilegios (UID 10001) y un `HEALTHCHECK` sobre `/health`. |
| `image: <TU_USUARIO_DOCKER>/heart-api` | Para Minikube se usa la imagen local `heart-api:latest` con `imagePullPolicy: IfNotPresent` (con la etiqueta `latest`, Kubernetes intentaría descargarla). Se agregan *probes*, límites de recursos y un `securityContext` que impide ejecutar como root. |
| `ci.yml` ejecuta `pytest tests/` | Se crea `tests/` con 29 pruebas. Se actualizan `actions/checkout@v4` y `actions/setup-python@v5`, se fijan las versiones de flake8, pytest y httpx2, y se agrega un job que construye y prueba la imagen Docker. |
| `from evidently.report import Report` | Esa API ya no existe en Evidently 0.7; se usa la equivalente actual (`from evidently import Report`). |
| `pip install -U jupyter-book` (tutorial del curso) | Hoy instala la versión 2, que ya no tiene los comandos del tutorial; se fija la 1.0.4. |
| README del enunciado (Data Lake, MinIO, Spark, Prophet, Streamlit, Airflow) | Describe otro proyecto; este README documenta el proyecto de falla cardíaca. |

## Verificación realizada

- Los dos notebooks se ejecutaron de principio a fin sin errores (resultados reproducibles), también por separado en una carpeta vacía, y cada cifra citada en sus textos y en el informe se contrastó con las salidas. La implementación del test de DeLong se valida en el notebook 1 contra el algoritmo de referencia de Sun y Xu (2014).
- Las pruebas de pytest y flake8 pasan con Python 3.13 y con Python 3.10 usando exactamente las dependencias de `docker/requirements.txt` (el entorno de la imagen y del CI).
- La API se probó con Uvicorn (ambos formatos de entrada, validación de errores y `/docs`), incluida una simulación del contenedor con Python 3.10 que solo contiene la carpeta `app/`.
- Los manifiestos de Kubernetes se validaron con `kubeconform -strict` contra los esquemas oficiales, y el workflow con `actionlint`.
- Un revisor independiente auditó las particiones y la fuga de datos, el código y la coherencia entre textos y resultados; sus observaciones se corrigieron.
- **Etapa 3 con Docker real** (Docker 29.7): la imagen `heart-api` (653 MB) se construye con el comando del enunciado. En el contenedor, la API responde en `/health`, `/predict` (ambos formatos) y `/docs`, rechaza entradas inválidas con 422, corre como usuario sin privilegios (UID 10001) y el `HEALTHCHECK` la marca `healthy`. GitHub Actions también construye la imagen y le hace una prueba de humo en cada *push*.
- **Etapa 4 en Minikube** (v1.39, Kubernetes v1.37): tras `minikube image load` y `kubectl apply`, el Deployment queda con 2/2 pods `Running` y listos (*probes* sobre `/health`), y la API responde a través del `Service` (`minikube service heart-service --url`). En 20 peticiones al puerto 80 del Service dentro del clúster, el tráfico se repartió entre las dos réplicas. Al eliminar un pod, Kubernetes lo reemplazó en segundos y el Deployment volvió a 2/2.

## Limitaciones

- 918 pacientes de cinco hospitales con distinta prevalencia de enfermedad; la falta de colesterol está asociada al hospital de origen. El modelo debe validarse con datos de la población donde se vaya a usar.
- Los mejores modelos están estadísticamente empatados: en prueba, Random Forest no difiere significativamente del mejor modelo de ninguna otra familia salvo el árbol de decisión individual. La regresión logística ofrece un desempeño equivalente (p = 0.77 en el test de DeLong) y es más interpretable.
- Es un proyecto académico: el modelo no reemplaza el criterio médico.
