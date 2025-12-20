FROM ghcr.io/mlflow/mlflow:v2.21.3

RUN pip install --no-cache-dir psycopg2-binary boto3
