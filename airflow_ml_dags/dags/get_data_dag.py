from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

from configs import default_args, SOURCE, TARGET, RAW_PATH

with DAG(
        dag_id="get_data",
        start_date=days_ago(5),
        schedule_interval="@daily",
        default_args=default_args,
        tags = ["HW3 mlops"],
) as dag:
    get_data = DockerOperator(
        image="airflow-get-data",
        command=RAW_PATH,
        task_id="docker-airflow-get-data",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=SOURCE, target=TARGET, type='bind')],
    )

    get_data