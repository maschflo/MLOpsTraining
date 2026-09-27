FROM python:3.14-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt --no-cache-dir

COPY api ./api
COPY model ./model

ENV MODEL_PATH="/app/model"
ENV MODEL_NAME="ai4i-failure"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0"]

