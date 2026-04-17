FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PLITHOS_DB_PATH=/app/data/plithos.db \
    PLITHOS_VIOLENCE_MODEL=/app/modelnew.h5 \
    PLITHOS_VIOLENCE_ONNX=/app/modelnew.onnx \
    PLITHOS_SMOKING_MODEL="/app/weights (1).pt" \
    PLITHOS_FIRE_MODEL=/app/Fire_best.pt \
    PLITHOS_PERSON_MODEL=/app/yolov8n.pt

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY Requirements.txt ./Requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r Requirements.txt

COPY . .

RUN mkdir -p /app/data

EXPOSE 5000
VOLUME ["/app/data"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=5).read()" || exit 1

CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "5000"]
