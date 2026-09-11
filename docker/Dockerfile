# Imagen de la API de predicción. Construir desde la raíz del proyecto:
#   docker build -t heart-api -f docker/Dockerfile .
#   docker run -p 8000:8000 heart-api
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Primero las dependencias (capa cacheada mientras requirements no cambie).
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código de la API y modelo serializado (app/model.joblib).
COPY app/ ./app/

# La API no necesita privilegios: se ejecuta con un usuario sin root
# (UID numérico, requerido por runAsNonRoot en Kubernetes).
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin appuser
USER 10001

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
