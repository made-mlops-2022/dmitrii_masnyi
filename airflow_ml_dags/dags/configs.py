from datetime import timedelta

default_args = {
    "owner": "airflow",
    "email": ["dmasny.made@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

SOURCE = "/Users/dmasny/made/mlops/dmitrii_masnyi/airflow_ml_dags/data"
TARGET = "/data"
RAW_PATH = "/data/raw/{{ ds }}"
PROCESSED_PATH = "/data/processed/{{ ds }}"
ARTIFACTS_PATH = "/data/model_artifacts/{{ ds }}"
PREDICT_PATH = "/data/predictions/{{ ds }}"