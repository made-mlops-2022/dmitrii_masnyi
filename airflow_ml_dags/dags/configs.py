from datetime import timedelta

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

VOLUME_PATH = "/Users/dmasny/made/mlops/dmitrii_masnyi/airflow_ml_dags/data:/data"
RAW_PATH = "/data/raw/{{ ds }}"
PROCESSED_PATH = "/data/processed/{{ ds }}"
ARTIFACTS_PATH = "/data/model_artifacts/{{ ds }}"
PREDICT_PATH = "/data/predictions/{{ ds }}"