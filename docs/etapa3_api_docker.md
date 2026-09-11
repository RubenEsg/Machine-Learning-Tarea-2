# Etapa 3 — API con FastAPI y Docker

El notebook 2 exporta el Pipeline completo (imputación, codificación, escalado y clasificador) a `app/model.joblib`. La API lo carga y recibe los datos crudos de un paciente, tal como vienen en el CSV.

## API REST

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | estado del servicio (lo usan las *probes* de Kubernetes y el `HEALTHCHECK` de Docker) |
| GET | `/model-info` | modelo, hiperparámetros, versiones, partición y métricas |
| POST | `/predict` | probabilidad de enfermedad cardíaca y predicción con umbral 0.5 |

`POST /predict` acepta campos con nombre o el formato del enunciado, una lista con los valores en el orden de las columnas del dataset:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"features": [54, "M", "ASY", 140, 239, 0, "Normal", 160, "N", 1.2, "Flat"]}'
# {"heart_disease_probability": 0.776, "prediction": 1, "threshold": 0.5}
```

La validación es estricta: valores fuera de rango, categorías inexistentes, campos faltantes, tipos incorrectos o números no finitos (`1e400`, `NaN`) devuelven un error 422 que indica el campo con problemas. La documentación interactiva queda en `http://localhost:8000/docs`.

```{literalinclude} ../app/api.py
:language: python
:caption: app/api.py
```

## Imagen Docker

La imagen usa Python 3.10, como pide el enunciado, con las mismas versiones de scikit-learn y numpy con que se entrenó el modelo. Así se evita que `model.joblib` falle al cargarse por diferencias de versión. Se ejecuta con un usuario sin privilegios e incluye un `HEALTHCHECK`.

```{literalinclude} ../docker/Dockerfile
:language: dockerfile
:caption: docker/Dockerfile
```

```{literalinclude} ../docker/requirements.txt
:language: text
:caption: docker/requirements.txt
```

```bash
docker build -t heart-api -f docker/Dockerfile .
docker run -p 8000:8000 heart-api
```

## Verificación

Se ejecutó en un equipo con Windows 11 y Docker 29.7:

```text
$ docker build -t heart-api -f docker/Dockerfile .
 => naming to docker.io/library/heart-api:latest                      (653 MB)

GET  /health                               -> {"status":"ok","model_loaded":true}
POST /predict  (formato del enunciado)     -> {"heart_disease_probability":0.776...,"prediction":1,"threshold":0.5}
POST /predict  (paciente de bajo riesgo)   -> {"heart_disease_probability":0.0046...,"prediction":0,"threshold":0.5}
POST /predict  (Sex = "X")                 -> HTTP 422
GET  /docs                                 -> HTTP 200

$ docker exec heart-api-prueba id
uid=10001(appuser) gid=10001(appuser) groups=10001(appuser)

$ docker inspect --format "{{.State.Health.Status}}" heart-api-prueba
healthy
```

Además, GitHub Actions construye la imagen y le hace una prueba de humo en cada *push* (etapa 5), y las pruebas automáticas de `tests/test_api.py` cubren ambos formatos de entrada y los casos de error.
