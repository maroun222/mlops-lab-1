FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--workers", "1", \
     "--backend-store-uri", "sqlite:////mlflow-data/mlflow.db", \
     "--default-artifact-root", "mlflow-artifacts:/", \
     "--serve-artifacts", \
     "--artifacts-destination", "/mlflow-data/mlruns", \
     "--allowed-hosts", "localhost:*,127.0.0.1:*,mlflow:5000"]