import os
import logging
import sys
import mlflow
from mlflow import MlflowClient
sys.path.insert(0, '/scanflow/scanflow')
from scanflow.client import ScanflowTrackerClient

class MLflowConnections:
    def __init__(self, experiment_name=None, checkpoints=None, model_name=None, 
                 parameters=None, model_version=None, models_to_download=None):
        """
        Initialize MLflow Connections class with optional parameters
        
        :param experiment_name: Name of the MLflow experiment
        :param checkpoints: Directory containing model checkpoints
        :param model_name: Name of the model
        :param parameters: Dictionary of model parameters
        """
        self.client = ScanflowTrackerClient(verbose=True)
        self.mlflow_uri = self.client.get_tracker_uri(True)
        
        # Set common attributes
        self.experiment_name = experiment_name
        self.checkpoints = checkpoints
        self.model_name = model_name
        self.model_version = model_version
        self.models_to_download = models_to_download
        self.parameters = parameters or {}
        
        # Configure MLflow tracking
        mlflow.set_tracking_uri(self.mlflow_uri)
        logging.info(f"Connecting tracking server uri: {mlflow.get_tracking_uri()}")

    def load(self):
        """
        Log model and data preparation models to MLflow
        """
        if not self.experiment_name or not self.checkpoints or not self.model_name:
            logging.error("Missing required parameters for modeling")
            return

        print(f"Start experiment: {self.experiment_name}")
        mlflow.set_experiment(experiment_name=self.experiment_name)
        
        with mlflow.start_run():
            print("Starting MLflow run:")
            
            # Path to the directory containing model files
            checkpoint_dir = os.path.join(os.getcwd(), self.checkpoints, self.model_name)
            print(checkpoint_dir)
            
            # Log each file in the directory
            if os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir):
                for root, dirs, files in os.walk(checkpoint_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        print(f"Saving model: {file} in path {file_path} as an artifact: {self.model_name}.")
                        mlflow.log_artifact(file_path, artifact_path=f"{self.model_name}")
            else:
                logging.error(f"Checkpoint directory {checkpoint_dir} does not exist or is not a directory.")
            
            # Log parameters to MLFlow
            try:
                for key, value in self.parameters.items():
                    print(f"Logging model parameter: {key} = {value}")
                    mlflow.log_param(key, value)
            except Exception as e:
                logging.error(f"Failed to log parameters to MLFlow: {e}")

    def download(self):
        """
        Download model artifacts from MLflow based on models found in the checkpoints directory

        :param app_name: Application name
        :param team_name: Team name
        :param model_version: Specific model version to download
        """

        # If no models found, log and return
        if not self.models_to_download:
            logging.warning("No model directories found in the checkpoints path")
            return

        # Initialize MLflow client
        mlflow_client = MlflowClient(self.mlflow_uri)

        # Create download directory if it doesn't exist
        download_dir = "/workflow/model"
        os.makedirs(download_dir, exist_ok=True)

        # Download each discovered model
        for model in self.models_to_download:
            try:
                model_dir = os.path.join(self.checkpoints, self.model_name, model)
                # Get model version
                if self.model_version is not None:
                    mv = mlflow_client.get_model_version(model_dir, self.model_version)
                else:
                    # Get the latest production version
                    versions = mlflow_client.get_latest_versions(model_dir, stages=["Production"])
                    mv = versions[0] if versions else None

                # Skip if no model version found
                if mv is None:
                    logging.warning(f"No production version found for model {model_dir}")
                    continue

                # Download artifacts
                artifacts_dir = mlflow_client.download_artifacts(
                    mv.run_id,
                    path=f"{model_dir}",
                    dst_path=download_dir
                )
                logging.info(f"Contents of download directory: {os.listdir(download_dir)}")
                logging.info(f"Artifacts for {model_dir} downloaded in: {artifacts_dir}")

            except Exception as e:
                logging.error(f"Error downloading model {model_dir}: {e}")

        # Log summary of downloaded models
        logging.info(f"Total models downloaded: {len(self.models_to_download)}")
        logging.info(f"Downloaded models: {self.models_to_download}")

    def execute(self, action='load'):
        """
        Execute either load or download action
        
        :param action: 'load' or 'download'
        """
        if action == 'load':
            self.load()
        elif action == 'download':
            self.download()
        else:
            logging.error(f"Invalid action: {action}. Choose 'modeling' or 'download'.")
