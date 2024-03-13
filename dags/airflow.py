# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
#from src.pylab1 import load_train_test_valid_files,feature_engineering,preprocess_zero_imput_norm,preprocess_mean_input_norm
#import tensorflow as tf
from src.demo import load_train_files,load_test_files,feature_engineering,mean_imputation, training, model
from airflow import configuration as conf

# Enable pickle support for XCom, allowing data to be passed between tasks
conf.set('core', 'enable_xcom_pickling', 'True')

# Define default arguments for your DAG
default_args = {
    'owner': 'Pavss',
    'start_date': datetime(2023, 9, 17),
    'retries': 0, # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5), # Delay before retries
}

# Create a DAG instance named 'your_python_dag' with the defined default arguments
dag = DAG(
    'early_sepsis_prediction_pipeline1',
    default_args=default_args,
    description='Data Pipeline',
    schedule_interval=None,  # Set the schedule interval or use None for manual triggering
    catchup=False,
)

from airflow.operators.bash_operator import BashOperator

install_requirements = BashOperator(
    task_id='install_requirements',
    bash_command='pip install tensorflow',
    dag=dag
)
load_train_files_task = PythonOperator(
    task_id='load_train_files',
    python_callable=load_train_files,
    provide_context=True,
    dag=dag,
    execution_timeout=timedelta(minutes=360),
)

load_test_files_task = PythonOperator(
    task_id='load_test_files',
    python_callable=load_test_files,
    provide_context=True,
    dag=dag,
    execution_timeout=timedelta(minutes=360),
)

feature_engineering_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=feature_engineering,
    provide_context=True,
    dag=dag,
    execution_timeout=timedelta(minutes=360),
)

mean_imputation_task = PythonOperator(
    task_id='mean_imputation',
    python_callable=mean_imputation,
    provide_context=True,
    dag=dag,
    execution_timeout=timedelta(minutes=360),
)

training_task = PythonOperator(
    task_id='training',
    python_callable=training,
    provide_context=True,
    dag=dag,
    op_kwargs={'tf':'from tensorflow.keras.preprocessing.sequence import pad_sequences',
               'tf1':'import tensorflow as tf'},
    execution_timeout=timedelta(minutes=360),
)

model_task = PythonOperator(
    task_id='model',
    python_callable=model,
    provide_context=True,
    dag=dag,
    execution_timeout=timedelta(minutes=360),
)


# Set task dependencies
install_requirements>>load_train_files_task >> load_test_files_task >> feature_engineering_task >> mean_imputation_task>> training_task >>model_task

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.cli()