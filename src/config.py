"""Configuración central: rutas, columnas y parámetros de partición."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "heart.csv"
APP_DIR = PROJECT_ROOT / "app"
MODEL_PATH = APP_DIR / "model.joblib"          # usado por la API y Docker
ROOT_MODEL_PATH = PROJECT_ROOT / "model.joblib"  # copia según la Etapa 0
METADATA_PATH = APP_DIR / "model_metadata.json"
DRIFT_REPORT_PATH = PROJECT_ROOT / "drift_report.html"

TARGET = "HeartDisease"

# Orden original de las columnas del CSV de Kaggle (sin la variable objetivo).
FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
    "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
]
NUMERIC_FEATURES = [
    "Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak",
]
CATEGORICAL_FEATURES = [
    "Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope",
]
# Un valor 0 en estas columnas es fisiológicamente imposible: es un faltante
# codificado como 0 ("no medido"). Se imputa DENTRO del Pipeline.
ZERO_AS_MISSING = ["RestingBP", "Cholesterol"]

# Partición: 80 % entrenamiento / 20 % prueba, estratificada por la clase.
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
