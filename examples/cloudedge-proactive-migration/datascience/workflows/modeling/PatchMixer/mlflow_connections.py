import os
import logging
import sys
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import torch
import joblib
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, PatchMixer
from mlflow import MlflowClient
sys.path.insert(0, '/scanflow/scanflow')
from scanflow.client import ScanflowTrackerClient

class MLflowConnections:
    def __init__(self, args, model_version=None):
        # experiment_name=None, checkpoints=None, model_name=None, 
                #  parameters=None, , models_to_download=None, action='load'):
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
        self.args=args
        self.model_name = (
            f"loss_flag{args.loss_flag}_lr{args.learning_rate}_dm{args.d_model}_"
            f"{args.model_id}_{args.model_id}_{args.data}_ft{args.features}_sl{args.seq_len}_"
            f"pl{args.pred_len}_p{args.patch_len}s{args.stride}_random{args.random_seed}_0"
        )
        self.app_name = args.app_name
        self.team_name = args.team_name  
        self.experiment_name = args.model_id
        self.checkpoints = args.checkpoints
        self.model_version = model_version
        self.action = args.action
        self.models_to_download = args.models_to_download
        self.parameters = {
            "loss_flag": args.loss_flag,
            "learning_rate": args.learning_rate,
            "d_model": args.d_model,
            "model_id": args.model_id,
            "model": args.model,
            "data": args.data,
            "features": args.features,
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "patch_len": args.patch_len,
            "stride": args.stride,
            "random_seed": args.random_seed,
        }
        
        # Configure MLflow tracking
        mlflow.set_tracking_uri(self.mlflow_uri)
        logging.info(f"Connecting tracking server uri: {mlflow.get_tracking_uri()}")

    def load(self):
        """
        Log model and data preparation models to MLflow. One mlflow run per model. Load the production models to DataEngineer Tracker.
        """
        if not self.experiment_name or not self.checkpoints or not self.model_name:
            logging.error("Missing required parameters for modeling")
            return

        print(f"Start experiment: {self.experiment_name}")
        mlflow.set_experiment(experiment_name=self.experiment_name)
        model_list=[]
        
        # Path to the directory containing model files
        checkpoint_dir = os.path.join(os.getcwd(), self.checkpoints, self.model_name)
        # We check if the directory exists
        if os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir):
            # Iterate through the directory
            for root, dirs, files in os.walk(checkpoint_dir):
                with mlflow.start_run():
                    # Iterate through the multiple model files.
                    for file in files:

                        # logging.info(f"Starting MLflow run to load {file}:")
                        file_path = os.path.join(root, file)
                        clean_model_name = os.path.splitext(file)[0]
                        artifact_path=f"{self.model_name}/{clean_model_name}"
                        model_list.append(artifact_path)
                        # Specific handling for .pth files (PyTorch models)
                        # Innitialize model class
                        try:
                            if file.endswith('.pth'):  # PyTorch model
                                model_dict = {
                                    'Autoformer': Autoformer,
                                    'Transformer': Transformer,
                                    'Informer': Informer,
                                    'DLinear': DLinear,
                                    'NLinear': NLinear,
                                    'Linear': Linear,
                                    'PatchTST': PatchTST,
                                    'PatchMixer': PatchMixer,
                                }
                                model = model_dict[self.args.model].Model(self.args)
                                model.load_state_dict(torch.load(file_path))
                                logging.info(f"Saving model: {file_path} as {artifact_path}.")
                                mlflow.pytorch.log_model(
                                    pytorch_model=model,
                                    artifact_path=artifact_path,
                                    registered_model_name=artifact_path
                                )                                
                            elif file.endswith('.pkl'):  # Scikit-learn model
                                logging.info(f"Saving model: {file_path} as {artifact_path}.")
                                # Log scikit-learn model
                                sk_model = joblib.load(file_path)
                                mlflow.sklearn.log_model(
                                    sk_model=sk_model,  # Actual model object
                                    artifact_path=artifact_path,
                                    registered_model_name=artifact_path
                                )
                        except Exception as e:
                            logging.error(f"Failed to log model {file}: {e}")     
                    
                    # Log parameters to MLFlow
                    try:
                        for key, value in self.parameters.items():
                            print(f"Logging model parameter: {key} = {value}")
                            mlflow.log_param(key, value)
                    except Exception as e:
                        logging.error(f"Failed to log parameters to MLFlow: {e}")                   
                        
            else:
                logging.error(f"Checkpoint directory {checkpoint_dir} does not exist or is not a directory.")
            
        for model in model_list:
            # Retrieve the saved models from the registry. Upload them to the DataEngineer MLFlow tracker for prediction.  
            self.client.save_app_model(app_name=self.app_name,
                team_name= self.team_name,
                model_name=model)

    def download(self):
        """
        Download model artifacts from MLflow based on models found in the checkpoints directory

        :param app_name: Application name
        :param team_name: Team name
        :param model_version: Specific model version to download
        """

        # If no models found, log and return
        if not self.models_to_download:
            logging.warning("No model given to download.")
            return

        # Initialize MLflow client
        mlflow_client = MlflowClient(self.mlflow_uri)

        # Create download directory if it doesn't exist
        download_dir = "/workflow/model"
        os.makedirs(download_dir, exist_ok=True)

        # Download each discovered model
        for model in self.models_to_download:
            try:
                # Get model version
                if self.model_version is not None:
                    mv = mlflow_client.get_model_version(model, self.model_version)
                else:
                    # Get the latest production version
                    versions = mlflow_client.get_latest_versions(model)
                    mv = versions[0] if versions else None

                # Skip if no model version found
                if mv is None:
                    logging.warning(f"No production version found for model {model}")
                    continue

                # Download artifacts
                artifacts_dir = mlflow_client.download_artifacts(
                    mv.run_id,
                    path=f"{model}",
                    dst_path=download_dir
                )
                logging.info(f"Contents of download directory: {os.listdir(download_dir)}")
                logging.info(f"Artifacts for {model} downloaded in: {artifacts_dir}")

            except Exception as e:
                logging.error(f"Error downloading model {model}: {e}")

    def execute(self):
        """
        Execute either load or download action
        
        :param action: 'load' or 'download'
        """
        if self.action == 'load':
            self.load()
        elif self.action == 'download':
            self.download()
        else:
            logging.error(f"Invalid action: {self.action}. Choose 'modeling' or 'download'.")
