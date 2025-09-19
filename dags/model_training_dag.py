from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess
import sys
import os

# Path to your Lore repo
LORE_PATH = r"C:\MY WORK\open-source\lore"

# Default args for DAG
default_args = {
    'owner': 'mohsinkhan85090',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Task 1: Train model
def train_model():
    # Run the main script (adjust if your training function is elsewhere)
    subprocess.run([sys.executable, os.path.join(LORE_PATH, 'lore', '__main__.py'), 'train'], check=True)

# Task 2: Evaluate model
def evaluate_model():
    subprocess.run([sys.executable, os.path.join(LORE_PATH, 'lore', '__main__.py'), 'evaluate'], check=True)

# Define DAG
with DAG(
    'lore_model_training',
    default_args=default_args,
    description='Automate Lore model training using Airflow',
    schedule_interval='@daily',
    start_date=datetime(2025, 9, 19),
    catchup=False
) as dag:

    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )

    evaluate_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model
    )

    # Task dependencies
    train_task >> evaluate_task
