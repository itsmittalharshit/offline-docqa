FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface

WORKDIR /app

COPY requirements-lock.txt .
RUN pip install -r requirements-lock.txt \
 && python -m spacy download en_core_web_sm

COPY src/ src/
COPY eval/ eval/

EXPOSE 8000 8501
CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]