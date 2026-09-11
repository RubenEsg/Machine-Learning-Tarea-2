"""Código compartido por las pruebas automáticas y el script de monitoreo.

El análisis y el entrenamiento están en los notebooks
(notebooks/1_model_leakage_demo.ipynb y notebooks/2_model_pipeline_cv.ipynb),
que son autocontenidos y usan los mismos criterios de columnas, semilla y
partición definidos aquí.

Módulos:
    config      -> rutas, columnas, semillas y parámetros de partición.
    data        -> carga, validación y partición de los datos.
    monitoring  -> reporte de deriva de datos con Evidently.
"""
