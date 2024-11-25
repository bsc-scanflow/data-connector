import mlflow
import click
import os
import logging
import json
import sys
sys.path.insert(0, '/scanflow/scanflow')

from scanflow.client import ScanflowTrackerClient

@click.command(help="Modeling")
@click.option("--experiment_name", default='PatchMixer', type=str)
@click.option("--checkpoints", default='/checkpoints/', type=str)
@click.option("--model_name", default='model_name', type=str)
@click.option("--parameters", type=str, help="Model parameters as a JSON string (optional)")

def modeling(experiment_name, checkpoints, model_name, parameters):
    # Log model and data preparation models.
    client = ScanflowTrackerClient(verbose=True)
    mlflow.set_tracking_uri(client.get_tracker_uri(True))
    logging.info("Connecting tracking server uri: {}".format(mlflow.get_tracking_uri()))
        
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        # Path to the directory containing model files
        checkpoint_dir = os.path.join(checkpoints, model_name)
        
        # Log each file in the directory
        if os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir):
            for root, dirs, files in os.walk(checkpoint_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, checkpoint_dir)
                    mlflow.log_artifact(file_path, artifact_path=f"{model_name}/{relative_path}")
        else:
            logging.error(f"Checkpoint directory {checkpoint_dir} does not exist or is not a directory.")

        # Log model parameters
        if parameters:
            try:
                params_dict = json.loads(parameters)
                for key, value in params_dict.items():
                    mlflow.log_param(key, value)
            except json.JSONDecodeError:
                logging.error("Failed to decode parameters. Ensure it's a valid JSON string.")
        else:
            logging.warning("No parameters provided to log.")



if __name__ == '__main__':
    modeling()