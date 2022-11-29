import os
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.dates import days_ago
from docker.types import Mount

from configs import default_args, SOURCE, TARGET, RAW_PATH, ARTIFACTS_PATH, PREDICTS_PATH

def wait_for_file(file_name):
    return os.path.exists(file_name)

with DAG(
        dag_id="predict",
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

    wait_model = PythonSensor(
        task_id="wait-for-model",
        python_callable=wait_for_file,
        op_args=["/opt/airflow/data/models/{{ ds }}/model.pkl"],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke"
    )
    wait_transformer = PythonSensor(
        task_id="wait-for-transformer",
        python_callable=wait_for_file,
        op_args=["/opt/airflow/data/models/{{ ds }}/transformer.pkl"],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke"
    )
 
    predict = DockerOperator(
        image="airflow-predict",
        command=f"--data-dir={RAW_PATH} --artifacts-dir={ARTIFACTS_PATH} --output-dir={PREDICTS_PATH}",
        network_mode="bridge",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=SOURCE, target=TARGET, type='bind')]
    )

    
    [wait_data, wait_model, wait_transformer] >> predict 