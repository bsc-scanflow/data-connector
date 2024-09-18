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
# TODO: Add PromCSV parameters
def data_retrieval(app_name, team_name):

    # Create a ScanflowTrackerClient to manage any interaction with the Scanflow Tracker (?)
    client = ScanflowTrackerClient(verbose=True)
    # Why is this command needed?
    mlflow.set_tracking_uri(client.get_tracker_uri(True))
    # Create a MLFlow client
    mlflowclient = MlflowClient(client.get_tracker_uri(True))
    logging.info("Connecting tracking server uri: {}".format(mlflow.get_tracking_uri()))

    #client.save_app_artifacts()
    #client.download_artifacts()
    
if __name__ == '__main__':
    data_retrieval()