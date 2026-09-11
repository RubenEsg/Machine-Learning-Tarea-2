"""API REST para predecir el riesgo de enfermedad cardíaca (Etapa 3).

Ejecución local (desde la raíz del proyecto):
    uvicorn app.api:app --reload
Documentación interactiva: http://localhost:8000/docs
"""
import json
import math
from pathlib import Path
from typing import Any, List, Literal, Union

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.joblib"
METADATA_PATH = APP_DIR / "model_metadata.json"
THRESHOLD = 0.5

# Orden de las columnas en el dataset original: es el orden que debe
# respetar la lista del formato {"features": [...]} del enunciado.
FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
    "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
]

# El modelo es un Pipeline completo (imputación + codificación + escalado +
# clasificador), por eso recibe los datos crudos tal como vienen del CSV.
model = joblib.load(MODEL_PATH)
metadata = {}
if METADATA_PATH.exists():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predice la probabilidad de enfermedad cardíaca "
                "(HeartDisease = 1) a partir de 11 variables clínicas.",
    version="1.0.0",
)

EXAMPLE = {
    "Age": 54, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 140,
    "Cholesterol": 239, "FastingBS": 0, "RestingECG": "Normal",
    "MaxHR": 160, "ExerciseAngina": "N", "Oldpeak": 1.2, "ST_Slope": "Flat",
}
EXAMPLE_AS_LIST = [EXAMPLE[name] for name in FEATURES]


class PatientData(BaseModel):
    """Variables clínicas de un paciente (mismos nombres del dataset)."""

    # strict=True: no convierte tipos silenciosamente (por ejemplo, true no
    # se acepta como 1 ni "54" como 54). Un entero sí vale para un decimal.
    model_config = ConfigDict(extra="forbid", strict=True,
                              json_schema_extra={"examples": [EXAMPLE]})

    Age: int = Field(ge=1, le=120, description="Edad en años")
    Sex: Literal["M", "F"] = Field(description="Sexo")
    ChestPainType: Literal["TA", "ATA", "NAP", "ASY"] = Field(
        description="Tipo de dolor torácico")
    RestingBP: float = Field(
        ge=0, le=300,
        description="Presión arterial en reposo (mm Hg). 0 = no medida")
    Cholesterol: float = Field(
        ge=0, le=1000,
        description="Colesterol sérico (mg/dl). 0 = no medido")
    FastingBS: int = Field(
        ge=0, le=1,
        description="Glucosa en ayunas > 120 mg/dl (1) o no (0)")
    RestingECG: Literal["Normal", "ST", "LVH"] = Field(
        description="Electrocardiograma en reposo")
    MaxHR: float = Field(ge=40, le=250,
                         description="Frecuencia cardíaca máxima alcanzada")
    ExerciseAngina: Literal["Y", "N"] = Field(
        description="Angina inducida por ejercicio")
    Oldpeak: float = Field(ge=-5, le=10,
                           description="Depresión del segmento ST")
    ST_Slope: Literal["Up", "Flat", "Down"] = Field(
        description="Pendiente del segmento ST en el pico del ejercicio")


class FeaturesInput(BaseModel):
    """Formato del enunciado: lista con los 11 valores en el orden del
    dataset (ver FEATURES)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"features": EXAMPLE_AS_LIST}]},
    )

    features: List[Any] = Field(min_length=len(FEATURES),
                                max_length=len(FEATURES))


class Prediction(BaseModel):
    heart_disease_probability: float
    prediction: int
    threshold: float


def _json_safe(value):
    """Convierte floats no finitos (inf, nan) en texto: el JSON estándar de
    la respuesta no puede representarlos."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request,
                                   exc: RequestValidationError):
    """Responde 422 con el detalle de la validación, también cuando la
    entrada trae números como 1e400 o NaN (sin esto la API daría un 500)."""
    detail = _json_safe(jsonable_encoder(exc.errors()))
    return JSONResponse(status_code=422, content={"detail": detail})


def features_to_patient(values):
    """Valida la lista del formato del enunciado como un PatientData."""
    try:
        return PatientData(**dict(zip(FEATURES, values)))
    except ValidationError as exc:
        errors = []
        for error in exc.errors():
            field = error["loc"][0] if error["loc"] else None
            if field in FEATURES:  # ubica el error en la posición de la lista
                error = {**error, "loc": ("body", "features",
                                          FEATURES.index(field), field)}
            errors.append(error)
        raise RequestValidationError(errors) from exc


@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API",
            "docs": "/docs", "predict": "POST /predict"}


@app.get("/health")
def health():
    """Usado por Kubernetes (readiness/liveness probes)."""
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    return {"features": FEATURES, "threshold": THRESHOLD, **metadata}


@app.post("/predict", response_model=Prediction)
def predict(data: Union[PatientData, FeaturesInput]):
    """Acepta campos con nombre (recomendado) o {"features": [...]}."""
    if isinstance(data, FeaturesInput):
        data = features_to_patient(data.features)
    X = pd.DataFrame([data.model_dump()], columns=FEATURES)
    proba = float(model.predict_proba(X)[0, 1])
    return {"heart_disease_probability": proba,
            "prediction": int(proba > THRESHOLD),
            "threshold": THRESHOLD}
