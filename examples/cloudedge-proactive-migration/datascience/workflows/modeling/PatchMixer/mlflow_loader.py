import mlflow
import click
import os
import logging
import json
import sys
sys.path.insert(0, '/scanflow/scanflow')

from scanflow.client import ScanflowTrackerClient

# @click.command(help="Modeling")
# @click.option("--experiment_name", default='PatchMixer', type=str)
# @click.option("--checkpoints", default='/checkpoints/', type=str)
# @click.option("--model_name", default='model_name', type=str)
# @click.option("--parameters", type=str, help="Model parameters as a JSON string (optional)")

def modeling(experiment_name, checkpoints, model_name, parameters):
    # Log model and data preparation models.
    client = ScanflowTrackerClient(verbose=True)
    mlflow.set_tracking_uri(client.get_tracker_uri(True))
    logging.info("Connecting tracking server uri: {}".format(mlflow.get_tracking_uri()))

    # # Check if the experiment exists
    # experiment = mlflow.get_experiment_by_name(experiment_name)
    # if experiment is None:
    #     # Create the experiment if it doesn't exist
    #     experiment_id = mlflow.create_experiment(experiment_name)
    # else:
    #     # Retrieve the experiment ID if it exists
    #     experiment_id = experiment.experiment_id
    print(f"Start experiment:{experiment_name}")
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run():
        print("Starting mlflow run:")
        # Path to the directory containing model files
        checkpoint_dir = os.path.join(os.getcwd(), checkpoints, model_name)
        print(checkpoint_dir)
        # Log each file in the directory
        if os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir):
            for root, dirs, files in os.walk(checkpoint_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, checkpoint_dir)
                    print(f"Saving model: {file} in path {file_path} as an artifact: {model_name}/{relative_path}.")
                    mlflow.log_artifact(file_path, artifact_path=f"{model_name}/{relative_path}")
        else:
            logging.error(f"Checkpoint directory {checkpoint_dir} does not exist or is not a directory.")

        # Log parameters to MLFlow
        try:
            for key, value in parameters.items():
                print(f"Logging model parameters.")
                mlflow.log_param(key, value)
        except Exception as e:
            logging.error(f"Failed to log parameters to MLFlow: {e}")
            
                



# if __name__ == '__main__':
#     modeling()