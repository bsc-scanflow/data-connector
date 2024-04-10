import mlflow
import click
import logging
import pandas as pd
import os

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

    qos_list = [25, 22, 20, 19, 15, 14]
    logging.info("dumy qosvalue: {}".format(qos_list))

    mlflow.set_experiment("predictor")
    with mlflow.start_run():
        for i, number in enumerate(qos_list):
            # Log each number as a separate metric with a unique key (appending the index)
            mlflow.log_metric(key=f"qos_{i}", value=number)
    
if __name__ == '__main__':
    upload()