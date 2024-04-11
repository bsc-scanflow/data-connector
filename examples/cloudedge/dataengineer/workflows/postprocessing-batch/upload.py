import mlflow
import click
import logging
import pandas as pd
import os
import random
import statistics

import sys
sys.path.insert(0, '/scanflow/scanflow')
from scanflow.client import ScanflowTrackerClient

@click.command(help="Postprocessing: upload qos")
@click.option("--name", default=None, type=str)
def upload(name):

    logging.info("Workflow step: {}".format(name))

    client = ScanflowTrackerClient()

    #log
    mlflow.set_tracking_uri(client.get_tracker_uri(True))
    logging.info("Connecting tracking server uri: {}".format(mlflow.get_tracking_uri()))

    qos_list_base = [25, 22, 20, 19, 15, 14]
    # Adjust each QoS value by adding a random number between 0 and 2
    qos_list = [value + random.uniform(0, 5) for value in qos_list_base]
    logging.info("dumy qosvalue: {}".format(qos_list))

    mlflow.set_experiment("predictor")
    with mlflow.start_run():
        for i, number in enumerate(qos_list):
            # Log each number as a separate metric with a unique key (appending the index)
            mlflow.log_metric(key=f"qos_{i}", value=number)

        # Find the maximum QoS value and its index
        max_qos = max(qos_list)
        max_qos_index = qos_list.index(max_qos)
        std_dev = statistics.stdev(qos_list)
        population_std_dev = statistics.pstdev(qos_list)
        
        # Log the maximum QoS value
        mlflow.log_metric(key="max_qos", value=max_qos)
        mlflow.log_metric(key="std_dev", value=std_dev)
        mlflow.log_metric(key="population_std_dev", value=population_std_dev)
        # Log the index of the maximum QoS value as a parameter (or metric)
        mlflow.log_param(key="max_qos_index", value=max_qos_index)
        
if __name__ == '__main__':
    upload()