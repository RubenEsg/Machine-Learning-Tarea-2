"""Genera el reporte de deriva de datos (Evidently) entre los datos de
entrenamiento (referencia) y los datos de predicción (actual).

Uso, desde la raíz del proyecto:
    python monitoring/drift_report.py
        -> actual = partición de prueba (igual que el enunciado);
           escribe la línea base drift_report.html
    python monitoring/drift_report.py --current lote_produccion.csv
        -> actual = pacientes recibidos por la API (mismas columnas);
           escribe monitoring/drift_lote_produccion.html
    python monitoring/drift_report.py --current lote.csv --fail-on-drift
        -> devuelve código de salida 1 si hay deriva (para automatizar)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DRIFT_REPORT_PATH, FEATURES  # noqa: E402
from src.data import load_data, split_data  # noqa: E402
from src.monitoring import build_drift_report, drift_summary  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--current", type=Path,
                        help="CSV con los datos de predicción. Por defecto "
                             "se usa la partición de prueba.")
    parser.add_argument("--output", type=Path,
                        help="Ruta del reporte HTML. Por defecto: "
                             "drift_report.html (sin --current) o "
                             "monitoring/drift_<nombre del CSV>.html.")
    parser.add_argument("--fail-on-drift", action="store_true",
                        help="Termina con código 1 si hay deriva global.")
    args = parser.parse_args(argv)

    X_train, X_test, _, _ = split_data(load_data())
    if args.current is None:
        current = X_test
        output = args.output or DRIFT_REPORT_PATH
    else:
        current = pd.read_csv(args.current)[FEATURES]
        # No sobrescribe la línea base (drift_report.html) con un lote nuevo.
        output = args.output or (DRIFT_REPORT_PATH.parent / "monitoring"
                                 / f"drift_{args.current.stem}.html")

    snapshot = build_drift_report(X_train, current, output)
    summary, overall = drift_summary(snapshot)
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nColumnas con deriva: {overall['columnas_con_deriva']} de "
          f"{len(summary)} (umbral global: "
          f"{overall['umbral_proporcion']:.0%} de las columnas)")
    print("Deriva del dataset:",
          "SÍ" if overall["deriva_del_dataset"] else "NO")
    print(f"Reporte guardado en: {output}")
    if args.fail_on_drift and overall["deriva_del_dataset"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
