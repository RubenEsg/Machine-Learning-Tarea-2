"""Pruebas de la API FastAPI (sin levantar un servidor real)."""
import json

import pytest
from fastapi.testclient import TestClient

from app.api import FEATURES, app

client = TestClient(app)

HIGH_RISK = {
    "Age": 63, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 150,
    "Cholesterol": 0, "FastingBS": 1, "RestingECG": "ST", "MaxHR": 105,
    "ExerciseAngina": "Y", "Oldpeak": 2.5, "ST_Slope": "Flat",
}
LOW_RISK = {
    "Age": 39, "Sex": "F", "ChestPainType": "ATA", "RestingBP": 120,
    "Cholesterol": 210, "FastingBS": 0, "RestingECG": "Normal",
    "MaxHR": 175, "ExerciseAngina": "N", "Oldpeak": 0.0, "ST_Slope": "Up",
}


def as_list(patient):
    return {"features": [patient[name] for name in FEATURES]}


def post_raw(body):
    """Envía el cuerpo tal cual (json.dumps no admite 1e400 ni NaN)."""
    return client.post("/predict", content=body,
                       headers={"Content-Type": "application/json"})


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info():
    body = client.get("/model-info").json()
    assert body["features"] == FEATURES
    assert body["threshold"] == 0.5


def test_predict_with_named_fields():
    response = client.post("/predict", json=HIGH_RISK)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["heart_disease_probability"] <= 1
    assert body["prediction"] == int(body["heart_disease_probability"] > 0.5)


def test_predict_with_features_list_gives_same_result():
    """Formato del enunciado: {"features": [...]} en el orden del dataset."""
    by_list = client.post("/predict", json=as_list(HIGH_RISK)).json()
    by_name = client.post("/predict", json=HIGH_RISK).json()
    assert by_list == by_name


def test_risk_ordering_is_clinically_sensible():
    high = client.post("/predict", json=HIGH_RISK).json()
    low = client.post("/predict", json=LOW_RISK).json()
    assert high["heart_disease_probability"] > 0.5
    assert low["heart_disease_probability"] < 0.5


@pytest.mark.parametrize("field, value", [
    ("Sex", "X"),            # categoría inexistente
    ("ChestPainType", "?"),
    ("Age", -3),             # fuera de rango
    ("MaxHR", "alto"),       # tipo incorrecto
    ("Age", True),           # booleano en un campo numérico
    ("FastingBS", True),
    ("FastingBS", 2),
    ("Age", "54"),           # número como texto
])
def test_invalid_values_are_rejected(field, value):
    payload = {**HIGH_RISK, field: value}
    assert client.post("/predict", json=payload).status_code == 422
    values = as_list(payload)
    assert client.post("/predict", json=values).status_code == 422


@pytest.mark.parametrize("raw_value", ["1e400", "NaN", "Infinity",
                                       "-Infinity"])
def test_non_finite_numbers_return_422_not_500(raw_value):
    named = json.dumps(HIGH_RISK).replace(
        '"Cholesterol": 0', f'"Cholesterol": {raw_value}')
    listed = json.dumps(as_list(HIGH_RISK)).replace(
        '"ASY", 150, 0,', f'"ASY", 150, {raw_value},')
    assert raw_value in named and raw_value in listed
    assert post_raw(named).status_code == 422
    assert post_raw(listed).status_code == 422


def test_missing_field_is_rejected():
    payload = {k: v for k, v in HIGH_RISK.items() if k != "Age"}
    assert client.post("/predict", json=payload).status_code == 422


def test_features_list_with_wrong_length_is_rejected():
    response = client.post("/predict", json={"features": [54, "M", "ASY"]})
    assert response.status_code == 422


def test_features_list_error_points_to_the_position():
    values = as_list(HIGH_RISK)
    values["features"][FEATURES.index("Sex")] = "X"
    response = client.post("/predict", json=values)
    assert response.status_code == 422
    locations = [error["loc"] for error in response.json()["detail"]]
    assert ["body", "features", FEATURES.index("Sex"), "Sex"] in locations
