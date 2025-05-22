import mlflow
import click
import logging
import pandas as pd
import os
import shutil
import fnmatch
import sys
import json
# Required imports for HTTP requests
# - requests already available in scanflow-executor docker image
import requests

sys.path.insert(0, '/scanflow/scanflow')
from scanflow.client import ScanflowTrackerClient

def get_latest_file(files_path:str, file_ext:str = "*") -> str:
    """
    Sort all available files from the files path and return the most recent one
    - files_path: Experiment's root folder. Within it there's a subfolder for each experiment run
    """
    
    walk_results = os.walk(files_path)
    mtime = 0
    latest_file = ""

    # We only need the root and filenames list here, not the dirnames
    # TODO: Improve this operation as it will increase in time with the amount of experiment runs
    # - Previous experiment results aren't being purged as of now
    for root, dirnames, filenames in walk_results:
        for filename in fnmatch.filter(filenames, f"*.{file_ext}"):
            cur_filename = os.path.join(root, filename)
            if os.path.getmtime(cur_filename) > mtime:
                mtime = os.path.getmtime(cur_filename)
                latest_file = cur_filename
    
    logging.info(f"Latest CSV file found: {latest_file}")
    return latest_file

def get_latest_experiment_run_id(experiment_name: str = "", run_name: str = "", run_age: int = 300) -> str:
    """
    Return the latest experiment run id
    return: run_id hash
    """
    from datetime import datetime, timedelta

    # Get the experiment id
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    # Retrieve filtered experiment runs by run_name, ordered by descending end time --> First entry will be the most recent
    # WIP: reduce number of runs search to the last 5 minutes
    starttimeframe = int((datetime.now() - timedelta(seconds=run_age)).timestamp() * 1000)
    filter_string = f"run_name='{run_name}' and attributes.created > {str(starttimeframe)}"
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        order_by=["end_time DESC"]
    )
    logging.info(f"Number of experiments fetched: {len(runs_df.index)}")
    # First row is the most recently finalized one
    run_id = runs_df.loc[[0]]['run_id'][0]
    return run_id

def retrieve_qos_values_per_cluster(results_filename:str = "", csv_sep: str = ",") -> dict:
    """
    Calculate several Latency QoS aggregated values per each available cluster in the results filename
    return: a dictionary with keys for each computed QoS aggregation.
    """

    # Load the CSV file into a DataFrame
    df = pd.read_csv(results_filename, sep=csv_sep)
    # Fill empty cluster values with "no_cluster"
    # - This should only happen when launching queries against Prometheus API instead of Thanos aggregator API
    df.cluster = df.cluster.fillna("no_cluster")
    # Filter the "pipelines_status_realtime_pipeline_latency" and "cluster" columns
    if {"pipelines_status_realtime_pipeline_latency", "cluster"}.issubset(df.columns):

        latency_df = df[[
            "cluster",
            "pipelines_status_realtime_pipeline_latency"
        ]].copy()

        # Convert latency column to numeric values
        latency_df["pipelines_status_realtime_pipeline_latency"] = pd.to_numeric(
            latency_df["pipelines_status_realtime_pipeline_latency"],
            errors='coerce'
        )
        
    else:
        logging.info("Missing columns in CSV results. Uploading QoS = 0 for all cluster's id")
        latency_df = df[[
            "cluster"
        ]].copy()
        # Set all latency values to 0.0
        latency_df["pipelines_status_realtime_pipeline_latency"] = 0.0

    # Drop NaN rows
    latency_df = latency_df.dropna()
    # Group by cluster: during migration we might have metrics from 2 different instances:
    # - The old one already unavailable
    # - The new one
    cluster_grouped = latency_df.groupby("cluster")
    # Get the mean value for each "cluster"
    average_latency = cluster_grouped.mean()
    average_latency = average_latency.rename(columns={"pipelines_status_realtime_pipeline_latency": "avg"})
    # Get the max value for each "cluster"
    max_latency = cluster_grouped.max()
    max_latency = max_latency.rename(columns={"pipelines_status_realtime_pipeline_latency": "max"})
    # Merge all aggregated values by "cluster" and transpose it
    aggregated_latencies = pd.merge(average_latency, max_latency, on="cluster").transpose()

    logging.info(f"Aggregated latency values: {json.dumps(aggregated_latencies.to_dict(), indent=2)}")

    return aggregated_latencies.to_dict()


def purge_local_experiment_results(filename: str = "") -> None:
    """
    Locally remove the experiment's results folder
    - This file is already uploaded as an artifact in the MLflow experiment run
    """
    if os.path.exists(filename):
        shutil.rmtree(os.path.dirname(filename))



def send_qos_analysis_request(
    analysis_agent_uri: str,
    qos_aggr: str,
    nearbyone_service_name: str,
    nearbyone_env_email: str,
    nearbyone_env_password: str,
    nearbyone_env_name: str,
    nearbyone_organization_id: str) -> None:
    """
    Send a POST request to the Planner Agent sensor to analyse the Reactive migration QoS
    """
    # Compose the POST data and headers
    data = {
        "args": [],
        "kwargs": {
            "app_name": nearbyone_service_name,
            "qos_aggr": qos_aggr,
            "nearbyone_env_email": nearbyone_env_email,
            "nearbyone_env_password": nearbyone_env_password,
            "nearbyone_env_name": nearbyone_env_name,
            "nearbyone_organization_id": nearbyone_organization_id
        }
    }
    headers = {
        "Content-type": "application/json"
    }

    # Send the POST request
    try:
        r = requests.post(
            url=analysis_agent_uri,
            json=data,
            headers=headers
            )
        r.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)


@click.command(help="Postprocessing: upload qos")
@click.option("--name", default=None, type=str)
@click.option("--app_name", default=None, type=str)
@click.option("--team_name", default=None, type=str)
@click.option("--csv_path", default="/workflow", type=str)
@click.option("--csv_sep", default=";", type=str)
@click.option("--qos_aggr", default="avg", type=str)
@click.option("--purge_local_results", default=False, type=bool)
@click.option("--analysis_agent_uri", default=None, type=str)
@click.option("--nearbyone_service_name", default=None, type=str)
@click.option("--nearbyone_env_email", default=None, type=str)
@click.option("--nearbyone_env_password", default=None, type=str)
@click.option("--nearbyone_env_name", default=None, type=str)
@click.option("--nearbyone_organization_id", default=None, type=str)
def upload(
    name: str,
    app_name: str,
    team_name: str,
    csv_path: str,
    csv_sep: str,
    qos_aggr: str,
    purge_local_results:bool,
    analysis_agent_uri: str,
    nearbyone_service_name: str,
    nearbyone_env_email: str,
    nearbyone_env_password: str,
    nearbyone_env_name: str,
    nearbyone_organization_id: str) -> None:
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

    logging.info("Retrieving latest metrics CSV file...")
    # Retrieve latest experiment CSV file
    csv_filename = get_latest_file(csv_path, "csv")

    # Calculate several QoS aggregation values per each available cluster in the CSV file
    qos_dict = retrieve_qos_values_per_cluster(
        results_filename=csv_filename,
        csv_sep=csv_sep
    )

    # Initialize the ScanflowTrackerClient for MLflow to retrieve the Tracker URI
    logging.info("Retrieving latest experiment run id...")
    client = ScanflowTrackerClient()
    mlflow.set_tracking_uri(client.get_tracker_uri(True))
    
    # Set the experiment name
    mlflow.set_experiment(f"{app_name}")
    
    # Retrieve the latest experiment run id where to attach the QoS metrics and cluster parameters
    run_id = get_latest_experiment_run_id(
        experiment_name=app_name,
        run_name=team_name
    )

    # Attach ALL aggregated QoS values with indexed cluster_id to the latest experiment run. There's a chance that the highest one might be from the previous app's cluster
    # after a migration, so the sensor needs to evaluate all cluster's in search for the one where the application is really running.
    logging.info("Uploading aggregated QoS values...")
    with mlflow.start_run(run_id=run_id):
        # NEW: Log all the aggregated values
        for idx, (cluster, aggr_qos) in enumerate(qos_dict.items()):
            # Log the cluster ID for that index
            mlflow.log_param(key=f"cluster_{idx}", value=cluster)
            for aggr_key, aggr_value in aggr_qos.items():
                # Log each aggregated metric available, using its key and cluster index to name them
                mlflow.log_metric(key=f"{aggr_key}_qos_{idx}", value=aggr_value)

        # Once done, set parameter "last_stage_completed" to "qos-upload"
        mlflow.log_param(key="last_stage_completed", value="qos-upload")

    # Send the QoS analysis request
    logging.info("Sending QoS analysis request...")
    send_qos_analysis_request(
        analysis_agent_uri=analysis_agent_uri,
        qos_aggr=qos_aggr,
        nearbyone_service_name=nearbyone_service_name,
        nearbyone_env_email=nearbyone_env_email,
        nearbyone_env_password=nearbyone_env_password,
        nearbyone_env_name=nearbyone_env_name,
        nearbyone_organization_id=nearbyone_organization_id
    )

    # Purge local CSV files if requested
    # - This is useful to avoid large lists of files when looking for the latest one
    if purge_local_results:
        purge_local_experiment_results(
            filename=csv_filename
        )
    

if __name__ == '__main__':
    upload()