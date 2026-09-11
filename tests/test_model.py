"""Pruebas del modelo serializado (control de calidad antes de desplegar)."""
import json

import joblib
import numpy as np
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import METADATA_PATH, MODEL_PATH, ZERO_AS_MISSING
from src.data import load_data, split_data, split_fingerprint

MIN_TEST_AUC = 0.85
MIN_TEST_ACCURACY = 0.80


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def holdout():
    _, X_test, _, y_test = split_data(load_data())
    return X_test, y_test


def test_model_is_full_pipeline(model):
    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["preprocessor", "scaler", "clf"]


def test_metadata_matches_partition(holdout):
    """El conjunto de prueba del control de calidad debe ser exactamente el
    que quedó fuera del entrenamiento del modelo."""
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    X_test, _ = holdout
    assert metadata["split"]["fingerprint"] == split_fingerprint(X_test)


def test_quality_gate_on_holdout(model, holdout):
    X_test, y_test = holdout
    proba = model.predict_proba(X_test)[:, 1]
    assert roc_auc_score(y_test, proba) >= MIN_TEST_AUC
    assert accuracy_score(y_test, model.predict(X_test)) >= MIN_TEST_ACCURACY


def test_probabilities_are_valid(model, holdout):
    X_test, _ = holdout
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1)


def test_zero_coded_values_are_imputed_with_training_median(model, holdout):
    """Cholesterol = 0 y RestingBP = 0 significan 'no medido': el Pipeline
    debe reemplazarlos por la mediana aprendida con el entrenamiento."""
    X_test, _ = holdout
    patient = X_test.iloc[[0]].copy()
    patient[ZERO_AS_MISSING] = 0
    preprocessor = model.named_steps["preprocessor"]
    medians = preprocessor.named_transformers_["imputer"].statistics_
    names = list(preprocessor.get_feature_names_out())
    transformed = preprocessor.transform(patient)
    for column, median in zip(ZERO_AS_MISSING, medians):
        assert median > 0
        assert transformed[0, names.index(f"imputer__{column}")] == median
    assert 0 <= model.predict_proba(patient)[0, 1] <= 1
