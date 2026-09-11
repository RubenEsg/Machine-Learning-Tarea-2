# Etapa 6 — Monitoreo de deriva de datos con Evidently

Un modelo aprende con ciertos pacientes, pero en producción pueden llegar otros distintos: de otro hospital, de otra edad o con otros exámenes. Si eso pasa, sus métricas dejan de ser confiables aunque el código siga funcionando. El monitoreo de deriva (*data drift*) compara la distribución de cada variable entre los datos de entrenamiento (referencia) y los que llegan para predicción (actuales). Evidently usa la prueba de Kolmogorov-Smirnov para las variables numéricas y chi-cuadrado o Z para las categóricas, y declara deriva del dataset si al menos la mitad de las columnas cambian.

El análisis completo está en la sección 7 del notebook de la etapa 2.

## Resultados

| Comparación | Columnas con deriva | ¿Deriva del dataset? | Reporte |
|---|---|---|---|
| Entrenamiento vs. prueba (lo que pide el enunciado) | 1 de 11 (`Cholesterol`, p = 0.017) | No | [drift_report.html](https://rubenesg.github.io/Machine-Learning-Tarea-2/drift_report.html) |
| Simulación de pacientes de otro hospital (partición secuencial del archivo) | 8 de 11 | Sí | [drift_particion_secuencial.html](https://rubenesg.github.io/Machine-Learning-Tarea-2/drift_particion_secuencial.html) |

- **Entrenamiento vs. prueba:** una alarma entre 11 pruebas al 5 % es lo esperable por azar. Además tiene explicación: en prueba quedó una proporción algo mayor de colesteroles "no medidos" (23.4 % contra 17.6 %). La partición es representativa y este reporte queda como **línea base** del monitoreo.
- **Otro hospital:** como el archivo está ordenado por hospital de origen, comparar sus primeras 734 filas con las últimas 184 simula recibir una población distinta. El monitor la detecta, lo que confirma que reacciona a cambios reales.

## Uso en producción

El script del repositorio genera el mismo reporte comparando el entrenamiento con cualquier lote de pacientes recibidos por la API:

```bash
python monitoring/drift_report.py                                     # entrenamiento vs. prueba
python monitoring/drift_report.py --current lote_pacientes.csv        # entrenamiento vs. lote nuevo
python monitoring/drift_report.py --current lote.csv --fail-on-drift  # código 1 si hay deriva
```

Con `--fail-on-drift` termina con error cuando hay deriva, lo que permite automatizarlo, por ejemplo en una tarea programada o en el CI. Los reportes de lotes nuevos se guardan en `monitoring/` para no sobrescribir la línea base.

```{literalinclude} ../src/monitoring.py
:language: python
:caption: src/monitoring.py
```

**Nota sobre el enunciado:** su código usa `from evidently.report import Report`, que ya no existe en la versión actual de Evidently (0.7). Se usa la forma equivalente vigente, `from evidently import Report`.
