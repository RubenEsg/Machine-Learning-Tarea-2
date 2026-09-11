# Etapa 5 — Integración continua con GitHub Actions

[![CI](https://github.com/RubenEsg/Machine-Learning-Tarea-2/actions/workflows/ci.yml/badge.svg)](https://github.com/RubenEsg/Machine-Learning-Tarea-2/actions/workflows/ci.yml)

Cada vez que se suben cambios al repositorio (*push* o *pull request*), GitHub revisa el proyecto automáticamente y avisa si algo se rompió antes de que llegue a producción. El workflow tiene dos trabajos:

1. **build**, en Python 3.10 con las dependencias de `docker/requirements.txt` (el mismo entorno de la imagen):
   - revisión de estilo con **flake8**;
   - las pruebas automáticas de `tests/` con **pytest**.
2. **docker**: construye la imagen, la levanta y le envía una predicción (prueba de humo).

## Pruebas automáticas

| Archivo | Qué verifica |
|---|---|
| `tests/test_api.py` | `/health`, `/model-info` y `/predict` con ambos formatos; rechazo con 422 de categorías inexistentes, valores fuera de rango, tipos incorrectos, campos faltantes y números no finitos |
| `tests/test_data.py` | esquema del dataset; partición de 734/184 pacientes, sin pacientes compartidos, estratificada y reproducible |
| `tests/test_model.py` | el modelo es un Pipeline completo; **control de calidad**: falla si el AUC en prueba baja de 0.85 o la exactitud de 0.80; el modelo se evaluó con la misma partición (huella); probabilidades válidas; los ceros "no medidos" se imputan con la mediana del entrenamiento |

```{literalinclude} ../.github/workflows/ci.yml
:language: yaml
:caption: .github/workflows/ci.yml
```

## Diferencias con el `ci.yml` del enunciado

- Se creó la carpeta `tests/`, que el enunciado ejecuta pero no incluía en su estructura.
- Se actualizaron las acciones (`actions/checkout@v4` y `actions/setup-python@v5`) y se fijaron las versiones de flake8, pytest y httpx2.
- Se agregó el trabajo `docker`, que valida el `Dockerfile` en cada cambio.

El resultado de cada ejecución se ve en la insignia de arriba y en la pestaña [Actions](https://github.com/RubenEsg/Machine-Learning-Tarea-2/actions) del repositorio.
