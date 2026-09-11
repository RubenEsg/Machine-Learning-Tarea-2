"""Monitoreo de deriva de datos (data drift) con Evidently - Etapa 6.

Nota: el enunciado usa la API antigua de Evidently
(``from evidently.report import Report``), que ya no existe en la versión
0.7. La API actual equivalente es ``from evidently import Report`` y
``report.run(...)`` devuelve un "snapshot" que se guarda con ``save_html``.
"""
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# FastingBS toma valores 0/1: se monitorea como variable categórica.
DRIFT_NUMERIC = [c for c in NUMERIC_FEATURES if c != "FastingBS"]
DRIFT_CATEGORICAL = CATEGORICAL_FEATURES + ["FastingBS"]


def build_drift_report(reference, current, output_path=None):
    """Compara la distribución de cada variable entre la referencia
    (datos de entrenamiento) y los datos actuales (datos de predicción)."""
    definition = DataDefinition(numerical_columns=DRIFT_NUMERIC,
                                categorical_columns=DRIFT_CATEGORICAL)
    snapshot = Report([DataDriftPreset()]).run(
        current_data=Dataset.from_pandas(current, data_definition=definition),
        reference_data=Dataset.from_pandas(reference,
                                           data_definition=definition),
    )
    if output_path is not None:
        snapshot.save_html(str(output_path))
    return snapshot


def drift_summary(snapshot):
    """Tabla por columna (prueba, p-valor o distancia, ¿deriva?) y resumen
    global del reporte."""
    rows, overall = [], {}
    for metric in snapshot.dict()["metrics"]:
        config, value = metric["config"], metric["value"]
        if config["type"].endswith("DriftedColumnsCount"):
            overall = {
                "columnas_con_deriva": int(value["count"]),
                "proporcion_con_deriva": value["share"],
                "umbral_proporcion": config["drift_share"],
                "deriva_del_dataset": value["share"] >= config["drift_share"],
            }
        elif config["type"].endswith("ValueDrift"):
            method = config["method"]
            # Para pruebas con p-valor hay deriva si p < umbral; para
            # distancias (datasets grandes) si la distancia >= umbral.
            if "p_value" in method:
                drift = value < config["threshold"]
            else:
                drift = value >= config["threshold"]
            rows.append({"columna": config["column"], "prueba": method,
                         "valor": value, "umbral": config["threshold"],
                         "deriva": "Sí" if drift else "No"})
    return pd.DataFrame(rows).set_index("columna"), overall
