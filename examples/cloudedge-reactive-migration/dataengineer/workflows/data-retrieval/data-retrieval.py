import mlflow
from mlflow.tracking import MlflowClient
import click
import logging
import pandas as pd
import os

from promcsv import PromCSV
import json

import sys
sys.path.insert(0, '/scanflow/scanflow')
from scanflow.client import ScanflowTrackerClient

def retrieve_prometheus_data(promcsv_config:click.File):
    """
    Retrieve Prometheus query results
    """
    logging.info("Retrieving Prometheus data...")
    with promcsv_config as f:
        query_config = json.load(f)

    prom_query = PromCSV(**query_config)

    # Launch the PromCSV queries
    logging.info("Launching Prometheus queries...")
    prom_query.retrieve_results()

    # Save the PromCSV results
    logging.info("Saving results to CSV file...")
    prom_query.pivot_results()
    prom_query.concat_results()
    csv_filename = prom_query.to_csv()
    
    logging.info(f"CSV results stored in: {csv_filename}")
    return csv_filename


def store_query_results(app_name:str, team_name:str , query_results:str):
    """
    Store the query results as an MLflow artifact
    ScanflowTrackerClient expects several environment variables to be already set:
    - AWS_ACCESS_KEY_ID: Username to access the MinIO object store
    - AWS_SECRET_ACCESS_KEY: Password to access the MinIO object store
    - MLFLOW_S3_ENDPOINT_URL: URL to the MLflow MinIO object store API
    - AWS_ENDPOINT_URL: URL to the MinIO object store API (same as MLFLOW_S3_ENDPOINT_URL)
    - SCANFLOW_TRACKER_URI: URI to the Scanflow Tracker (MLflow)
    - SCANFLOW_SERVER_URI: Same as SCANFLOW_TRACKER_URI
    - SCANFLOW_TRACKER_LOCAL_URI: Same as SCANFLOW_TRACKER_URI
    """
    logging.info("Storing query results...")
    # DEBUG:
    logging.info(f"App name: {app_name}, Team name: {team_name}")
    # Create a ScanflowTrackerClient to manage any interaction with the Scanflow Tracker (?)
    client = ScanflowTrackerClient(verbose=True)
    # Why is this command needed?
    mlflow.set_tracking_uri(client.get_tracker_uri(True))
    # Create a MLFlow client
    mlflowclient = MlflowClient(client.get_tracker_uri(True))
    logging.info("Connecting tracking server uri: {}".format(mlflow.get_tracking_uri()))

    # Set the MLflow experiment where to upload artifacts
    mlflow.set_experiment("reactive-predictor")
    # Use this method to upload the CSV as an MLFlow experiment artifact
    #client.save_app_artifacts()

    # If we wanted to download artifacts from the MLFlow experiment we'd use this method
    #client.download_artifacts()

@click.command(help="Retrieve real-time data from Prometheus")
# from main scanflow
@click.option("--app_name", default=None, type=str)
@click.option("--team_name", default=None, type=str)
# PromCSV arguments
@click.option("--promcsv_config", default="/app/data-retrieval/promql_queries.json", type=click.File('rb'))
def main(app_name, team_name, promcsv_config):
    
    # Retrieve application metrics from Prometheus
    csv_results = retrieve_prometheus_data(promcsv_config)

    # Store CSV results as an experiment's artifact
    store_query_results(app_name, team_name, csv_results)

if __name__ == '__main__':
    main()