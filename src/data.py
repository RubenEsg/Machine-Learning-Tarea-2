"""Carga, validación y partición de los datos.

Regla de oro: la partición train/test se hace ANTES de cualquier
transformación que aprenda de los datos (imputación, escalado, codificación,
selección de variables). Todo eso vive dentro del Pipeline.
"""
import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CATEGORICAL_FEATURES, DATA_PATH, FEATURES, RANDOM_STATE, TARGET,
    TEST_SIZE,
)

EXPECTED_CATEGORIES = {
    "Sex": {"M", "F"},
    "ChestPainType": {"TA", "ATA", "NAP", "ASY"},
    "RestingECG": {"Normal", "ST", "LVH"},
    "ExerciseAngina": {"Y", "N"},
    "ST_Slope": {"Up", "Flat", "Down"},
}


def load_data(path=DATA_PATH):
    """Lee el CSV de Kaggle y valida su esquema."""
    df = pd.read_csv(path)
    validate_schema(df)
    return df


def validate_schema(df):
    """Verifica columnas, variable objetivo binaria y categorías conocidas."""
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el dataset: {sorted(missing)}")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError(f"{TARGET} debe ser binaria (0/1)")
    for col in CATEGORICAL_FEATURES:
        unknown = set(df[col].unique()) - EXPECTED_CATEGORIES[col]
        if unknown:
            raise ValueError(f"Categorías inesperadas en {col}: {unknown}")


def split_data(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Partición estratificada train/test sobre los datos crudos.

    Devuelve X_train, X_test, y_train, y_test. Con la misma semilla se obtiene
    exactamente la misma partición en todos los notebooks, pruebas y reportes.
    """
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def split_fingerprint(X_test):
    """Huella corta de los índices de prueba para verificar que dos
    procesos usan exactamente la misma partición."""
    ids = ",".join(str(i) for i in sorted(X_test.index))
    return hashlib.sha256(ids.encode()).hexdigest()[:12]
