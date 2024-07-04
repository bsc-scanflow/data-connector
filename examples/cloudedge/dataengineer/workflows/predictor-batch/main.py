from mlflow.tracking import MlflowClient
import mlflow.pytorch
import mlflow
import click
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import json
import joblib
import os
from LSTM import LSTMModel 
sys.path.insert(0, '/scanflow/scanflow')
from scanflow.client import ScanflowTrackerClient

@click.command(help="batch predictions")
@click.option("--model_name", default='model_LSTM', type=str)
@click.option("--input_dir", help="Directory containing preprocessed data",
              default='/workflow/preprocessed_data/preprocessed_data.csv', type=str)
@click.option("--config", type=str, required=True, help="Configuration file path")
@click.option("--model_dir", help="Directory containing models",
              default='/workflow/model', type=str)

def inference(model_name, input_dir, config, model_dir):
    config = load_config(config)
    prediction_config = config.get("prediction")

    # Set up logging
    client = ScanflowTrackerClient(verbose=True)
    mlflow.set_tracking_uri(client.get_tracker_uri(True))
    mlflowclient = MlflowClient(client.get_tracker_uri(True))
    logging.info("Connecting tracking server uri: {}".format(mlflow.get_tracking_uri()))

    # Set up MLflow experiment
    mlflow.set_experiment("LSTM_inference")
    with mlflow.start_run(run_name='LSTM_batch_prediction'):
        data = pd.read_csv(input_dir)

        for cluster_id in data[prediction_config['cluster_id']].unique():
            cluster_data = data[data[prediction_config['cluster_id']] == cluster_id]
            dataloader_ = prepare_sequences(cluster_data,cluster_id, prediction_config["seq_length"], prediction_config["features"], prediction_config["target"], model_dir)

            # Load model directly from the file system
            model_folder_path = os.path.join(model_dir, f"{model_name}_{cluster_id}.pt")
            model = mlflow.pytorch.load_model(model_folder_path)
            logging.info(f"Loaded model for cluster {cluster_id} from {model_path}")

            predictions = predict(model, dataloader_)

            # Save predictions to CSV and log artifacts
            df_preds = pd.DataFrame(predictions, columns=['predictions'])
            prediction_path = f'predictions_{cluster_id}.csv'
            df_preds.to_csv(prediction_path, index=False)
            mlflow.log_artifact(prediction_path, artifact_path="predictions")

            mlflow.log_metric(key=f'num_predictions_{cluster_id}', value=len(df_preds))

def prepare_sequences(data,cluster_id, seq_length, features, target, model_dir):
    scaler_x, scaler_y = load_scalers(cluster_id, model_dir)         
    data[features] = scaler_x.transform(data[features])
    data[target] = scaler_y.transform(data[target])
    # Prepare sequences
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[features].iloc[i:(i + seq_length)].values)
        Y.append(data[target].iloc[i + seq_length].values)
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    mlflow.log_artifacts('.', 'processed_data')
    
    return DataLoader(TensorDataset(X_tensor, Y_tensor), batch_size=64, shuffle=False)

def load_scalers(cluster_id, model_dir):
    # Find directories for x and y scalers based on cluster_id pattern
    scaler_x_dir = next((os.path.join(model_dir, d) for d in os.listdir(model_dir) if f"LSTM_scaler_x_{cluster_id}" in d), None)
    scaler_y_dir = next((os.path.join(model_dir, d) for d in os.listdir(model_dir) if f"LSTM_scaler_y_{cluster_id}" in d), None)

    if scaler_x_dir is None or scaler_y_dir is None:
        raise FileNotFoundError(f"Scaler directories for cluster {cluster_id} not found in {model_dir}")

    # Path to the actual model.pkl file
    scaler_x_path = os.path.join(scaler_x_dir, "model.pkl")
    scaler_y_path = os.path.join(scaler_y_dir, "model.pkl")

    # Load the joblib models
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    return scaler_x, scaler_y


def predict(model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[0]
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            predictions.extend(predicted.cpu().numpy())
    return predictions

def load_config(config_path):
    with open(config_path, "r") as file:
        return json.load(file)

if __name__ == '__main__':
    inference()
