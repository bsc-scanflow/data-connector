import mlflow
from mlflow.tracking import MlflowClient
import click
import logging
import pandas as pd
import os
import json
import sys
sys.path.insert(0, '/scanflow/scanflow')
from scanflow.client import ScanflowTrackerClient

@click.command(help="load input data set")
# from main scanflow
@click.option("--app_name", default=None, type=str)
@click.option("--team_name", default=None, type=str)
# from local scanflow
@click.option("--model_version", default=None, type=int)
@click.option("--config", type=str, required=True, help="Configuration file path")

def download(app_name, team_name, model_version,config):
    config = load_config(config)
    download_config = config.get("download")
    model_name=download_config["model_name"]
    logging.info(model_name)
    
    for x in model_name:
        logging.info(x)
    for model in model_name:

        client = ScanflowTrackerClient(verbose=True)
        mlflow.set_tracking_uri(client.get_tracker_uri(True))

        mlflowclient = MlflowClient(client.get_tracker_uri(True))
        logging.info("Connecting tracking server uri: {}".format(mlflow.get_tracking_uri()))

        if model_version is not None:
            mv = mlflowclient.get_model_version(model, model_version)
        else:
            mv = mlflowclient.get_latest_versions(model, stages=["Production"])

        if not os.path.exists("/workflow/model"):
            os.makedirs(f"/workflow/model")

        if app_name is not None and team_name is not None:
            artifacts_dir = mlflowclient.download_artifacts(
	    	                          mv.run_id,
                                      path = f"{model}",
                                      dst_path = "/workflow/model")

        logging.info("Artifacts downloaded in: {}".format(artifacts_dir))
        logging.info("Artifacts: {}".format(os.listdir(artifacts_dir)))

def load_config(config_path):
    with open(config_path, "r") as file:
        return json.load(file)

if __name__ == '__main__':
    download()