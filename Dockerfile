FROM python:3.12-slim

# Tesseract for room labels/dimensions; Liberation fonts for plan annotation (Times-metric serif)
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr fonts-liberation libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[ui]"
COPY configs ./configs
COPY models ./models
COPY samples ./samples
COPY app ./app

ENV PYTHONUNBUFFERED=1 RYFC_ROOT=/app RYFC_WORK_DIR=/tmp/ryfc PORT=8000
EXPOSE 8000
# API by default; the UI image overrides CMD (see render.yaml)
CMD ["sh", "-c", "uvicorn ryfc.api:app --host 0.0.0.0 --port ${PORT}"]
