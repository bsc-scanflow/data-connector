import mlflow
from mlflow.tracking import MlflowClient
import click
import logging
import pandas as pd
import os

import sys
sys.path.insert(0, '/scanflow/scanflow')
from scanflow.client import ScanflowTrackerClient

@click.command(help="Retrieve real-time data from Prometheus")
# from main scanflow
@click.option("--app_name", default=None, type=str)
@click.option("--team_name", default=None, type=str)
# from local scanflow
# TODO: Add PromCSV parameters
def download(app_name, team_name):

    client = ScanflowTrackerClient(verbose=True)
    mlflow.set_tracking_uri(client.get_tracker_uri(True))
    
    mlflowclient = MlflowClient(client.get_tracker_uri(True))
    logging.info("Connecting tracking server uri: {}".format(mlflow.get_tracking_uri()))

    
if __name__ == '__main__':
    download()