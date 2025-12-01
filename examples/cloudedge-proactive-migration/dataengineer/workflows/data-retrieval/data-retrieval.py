import mlflow
from mlflow.tracking import MlflowClient
import click
import logging
import pandas as pd
import os

from promcsv import PromCSV
import json
from datetime import datetime

import sys

sys.path.insert(0, '/scanflow/scanflow')
from scanflow.client import ScanflowTrackerClient


def retrieve_prometheus_data(promcsv_config: click.File, output_path: str = None) -> str:
    """
    Retrieve Prometheus query results
    return: path to the generated CSV file
    """
    logging.info("Retrieving Prometheus data...")
    with promcsv_config as f:
        query_config = json.load(f)

    # Modify the PromCSV output directory so each execution creates a different subdir
    # - If output_path is provided, we ignore the value included in the promcsv_config file
    if output_path:
        query_config["output_dir"] = os.path.join(output_path, f"run_at_{round(datetime.now().timestamp())}")
    else:
        query_config["output_dir"] = os.path.join(query_config["output_dir"],
                                                  f"run_at_{round(datetime.now().timestamp())}")

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


def store_query_results(app_name: str, team_name: str, query_results: str) -> None:
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
    return: None
    """
    # Create a ScanflowTrackerClient to manage any interaction with the Scanflow Tracker
    client = ScanflowTrackerClient(verbose=True)

    # Upload the experiment's results as artifacts
    logging.info(f"Uploading query results dir {os.path.dirname(query_results)} as artifacts...")
    client.save_app_artifacts(
        app_name=app_name,
        team_name=team_name,
        app_dir=os.path.dirname(query_results)
    )


@click.command(help="Retrieve real-time data from Prometheus")
# from main scanflow
@click.option("--app_name", default=None, type=str)
@click.option("--team_name", default=None, type=str)
# PromCSV arguments
@click.option("--output_path", default="/workflow", type=str)
@click.option("--promcsv_config", default="/app/data-retrieval/promql_queries.json", type=click.File('rb'))
def main(app_name, team_name, output_path, promcsv_config):
    # Retrieve application metrics from Prometheus
    csv_results = retrieve_prometheus_data(
        promcsv_config=promcsv_config,
        output_path=output_path
    )

    # Store CSV results as an experiment's artifact
    store_query_results(
        app_name=app_name,
        team_name=team_name,
        query_results=csv_results
    )


if __name__ == '__main__':
    main()