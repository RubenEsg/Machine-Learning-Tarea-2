"""Pruebas del dataset y de la partición train/test."""
from src.config import FEATURES, RANDOM_STATE, TARGET, TEST_SIZE
from src.data import load_data, split_data, split_fingerprint


def test_dataset_schema():
    df = load_data()  # valida columnas, objetivo binario y categorías
    assert df.shape == (918, 12)
    assert df.isna().sum().sum() == 0
    assert set(df[TARGET].unique()) == {0, 1}


def test_split_sizes_and_no_overlap():
    X_train, X_test, y_train, y_test = split_data(load_data())
    assert len(X_train) == 734 and len(X_test) == 184
    assert list(X_train.columns) == FEATURES  # sin la variable objetivo
    # Ningún paciente puede estar en ambos conjuntos.
    assert set(X_train.index).isdisjoint(X_test.index)
    assert (X_train.index == y_train.index).all()
    assert (X_test.index == y_test.index).all()


def test_split_is_stratified():
    df = load_data()
    _, _, y_train, y_test = split_data(df)
    overall = df[TARGET].mean()
    assert abs(y_train.mean() - overall) < 0.01
    assert abs(y_test.mean() - overall) < 0.01


def test_split_is_reproducible():
    df = load_data()
    first = split_fingerprint(split_data(df)[1])
    second = split_fingerprint(split_data(df, TEST_SIZE, RANDOM_STATE)[1])
    assert first == second
