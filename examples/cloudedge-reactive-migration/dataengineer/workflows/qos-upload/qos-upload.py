import mlflow
import click
import logging
import pandas as pd
import random
import statistics
from pathlib import Path
import sys
import json
sys.path.insert(0, '/scanflow/scanflow')
from scanflow.client import ScanflowTrackerClient

def get_modification_time(item: Path) -> float:
    """
    Return the modification timestamp of a Path object
    """
    return item.stat().st_mtime

def get_latest_file(files_path:str) -> Path:
    """
    Sort all available files from the files path and return the most recent one
    """
    path_object = Path(files_path)
    items = path_object.iterdir()
    sorted_items = sorted(items, key=get_modification_time, reverse=True)
    return sorted_items[0]

@click.command(help="Postprocessing: upload qos")
@click.option("--name", default=None, type=str)
@click.option("--app_name", default=None, type=str)
@click.option("--team_name", default=None, type=str)
@click.option("--csv_path", default="/workflow/migration_experiment", type=str)
@click.option("--csv_sep", default=";", type=str)
def upload(name: str, app_name: str, team_name: str, csv_path: str, csv_sep: str) -> None:
    """
    Upload the average QoS from the available latency values in the latest experiment results.
    ScanflowTrackerClient expects several environment variables to be already set:
    - AWS_ACCESS_KEY_ID: Username to access the MinIO object store
    - AWS_SECRET_ACCESS_KEY: Password to access the MinIO object store
    - MLFLOW_S3_ENDPOINT_URL: URL to the MLflow MinIO object store API
    - AWS_ENDPOINT_URL: URL to the MinIO object store API (same as MLFLOW_S3_ENDPOINT_URL)
    - SCANFLOW_TRACKER_URI: URI to the Scanflow Tracker (MLflow)
    - SCANFLOW_SERVER_URI: Same as SCANFLOW_TRACKER_URI
    - SCANFLOW_TRACKER_LOCAL_URI: Same as SCANFLOW_TRACKER_URI
    """
    logging.info("Workflow step: {}".format(name))

    # Retrieve latest experiment CSV file
    csv_filename = get_latest_file(csv_path)
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_filename, sep=csv_sep)

    # Filter the "pipelines_status_realtime_pipeline_latency" and "cluster" columns
    if {"pipelines_status_realtime_pipeline_latency", "cluster"}.issubset(df.columns):
        # Group by cluster: during migration we might have metrics from 2 different instances:
        # - The old one already unavailable
        # - The new one
        latency_df = df[[
            "cluster",
            "pipelines_status_realtime_pipeline_latency"
        ]].copy()

        # Convert latency column to numeric values
        latency_df["pipelines_status_realtime_pipeline_latency"] = pd.to_numeric(
            latency_df["pipelines_status_realtime_pipeline_latency"],
            errors='coerce'
        )
        # Drop NaN rows
        latency_df = latency_df.dropna()
        # Group latency values by 'cluster'
        cluster_grouped = latency_df.groupby("cluster")
        # Get the mean value for each "cluster"
        average_latency = cluster_grouped.mean()
        logging.info(f"Average latency values: {json.dumps(average_latency.to_dict(), indent=2)}")
        # qos_dict contains key-value entries, where "key" is the name of the cluster and "value" is the mean QoS latency
        qos_dict = average_latency.to_dict()["pipelines_status_realtime_pipeline_latency"]

        # Initialize the ScanflowTrackerClient for MLflow to retrieve the Tracker URI
        client = ScanflowTrackerClient()
        mlflow.set_tracking_uri(client.get_tracker_uri(True))
        logging.info("Connecting tracking server uri: {}".format(mlflow.get_tracking_uri()))

        # Set the experiment name
        # TODO: Verify that the experiment name is the expected one
        mlflow.set_experiment(f"{app_name}")

        # TODO: Start or attach to an existing run and log QoS metrics
        # TODO: Attach ALL QoS with indexed cluster_id. The max one might be from the previous app's cluster after a migration
        with mlflow.start_run():
            max_qos = 0
            max_cluster = ""
            for cluster, avg_qos in qos_dict.items():
                if avg_qos >= max_qos:
                    max_qos = avg_qos
                    max_cluster = cluster
            # Log the maximum QoS value
            mlflow.log_metric(key="max_qos", value=max_qos)
            mlflow.log_metric(key="cluster", value=max_cluster)
        
    else:
        sys.exit("Missing columns in CSV results")

        
if __name__ == '__main__':
    upload()