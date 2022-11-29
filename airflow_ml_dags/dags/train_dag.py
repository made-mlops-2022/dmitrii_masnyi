import os
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.dates import days_ago
from docker.types import Mount

from configs import default_args, SOURCE, TARGET, RAW_PATH, PROCESSED_PATH, ARTIFACTS_PATH

def wait_for_file(file_name):
    return os.path.exists(file_name)

with DAG(
        dag_id="train",
        start_date=days_ago(5),
        schedule_interval="@daily",
        default_args=default_args,
        tags = ["HW3 mlops"],
) as dag:

    wait_data = PythonSensor(
        task_id="wait-for-data",
        python_callable=wait_for_file,
        op_args=["/opt/airflow/data/raw/{{ ds }}/data.csv"],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke"
    )

    wait_target = PythonSensor(
        task_id="wait-for-target",
        python_callable=wait_for_file,
        op_args=["/opt/airflow/data/raw/{{ ds }}/target.csv"],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke"
    )
 
    split_data = DockerOperator(
        image="airflow-split",
        command=f"--input-dir={RAW_PATH} --output-dir={PROCESSED_PATH}",
        # network_mode="bridge",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=SOURCE, target=TARGET, type='bind')]
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command=f"--input-dir={PROCESSED_PATH} --output-dir={ARTIFACTS_PATH}",
        network_mode="bridge",
        task_id="docker-airflow-fit-scaler",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=SOURCE, target=TARGET, type='bind')]
    )

    train = DockerOperator(
        image="airflow-train",
        command=f"--data-dir={PROCESSED_PATH} --artifacts-dir={ARTIFACTS_PATH} --output-dir={ARTIFACTS_PATH}",
        network_mode="bridge",
        task_id="docker-airflow-fit-model",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=SOURCE, target=TARGET, type='bind')]
    )

    validate = DockerOperator(
        image="airflow-validate",
        command=f"--data-dir={PROCESSED_PATH} --artifacts-dir={ARTIFACTS_PATH} --output-dir={ARTIFACTS_PATH}",
        network_mode="bridge",
        task_id="docker-airflow-validate",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=SOURCE, target=TARGET, type='bind')]
    )

    [wait_target, wait_data] >> split_data >> preprocess >> train >> validate