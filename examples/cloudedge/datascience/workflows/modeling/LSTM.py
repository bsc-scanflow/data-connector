from __future__ import annotations

import logging
import os
import joblib
import matplotlib.pyplot as plt
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
import mlflow.pytorch
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import sys
sys.path.insert(0, '/scanflow/scanflow')
from scanflow.client import ScanflowTrackerClient


class LSTMModel(pl.LightningModule):
    def __init__(
        self,
        num_layers,
        hidden_size,
        input_size,
        output_size,
        learning_rate=0.001,
    ):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers  # number of recurrent layers in the lstm
        self.hidden_size = hidden_size  # neurons in each lstm layer
        self.input_size = input_size  # number of input features
        self.output_size = output_size  # number of output classes
        self.learning_rate = learning_rate  # learning rate

        # LSTM model
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )  # lstm
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected
        self.fc_2 = nn.Linear(128, output_size)  # fully connected last layer
        self.relu = nn.ReLU()  # activation function

        self.loss_fn = nn.MSELoss()

    def forward(self, x, h_0=None, c_0=None):
        # Initialize hidden and cell states if not provided
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
                x.device
            )
        if c_0 is None:
            c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
                x.device
            )

        # Forward propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        # Select the last hidden state
        hn_last = hn[-1]  # hn is of shape (num_layers, batch, hidden_size)
        # Passing the last hidden state through the fully connected layers
        out = self.relu(hn_last)
        out = self.fc_1(out)  # first dense
        out = self.relu(out)  # relu
        out = self.fc_2(out)  # final output
        return out, hn, cn

    def training_step(self, batch, batch_idx):
        x, y = batch
        out, _, _ = self(x)
        loss = self.loss_fn(out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out, _, _ = self(x)
        loss = self.loss_fn(out, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class LSTMDataModulePerCluster(pl.LightningDataModule):
    def __init__(
        self,
        data,
        input_features,
        target_features,
        clusters,
        model_path,
        seq_length=10,
        batch_size=16,
        train_split=0.8,
        scaler_x=None,
        scaler_y=None,
    ):

        super(LSTMDataModulePerCluster, self).__init__()
        self._data = data
        self.input_features = input_features
        self.target_features = target_features
        self.clusters = clusters
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.train_split = train_split
        self.model_path = model_path
        
        # Create scalers for this specific cluster
        if scaler_x is None or scaler_y is None:
            self.scaler_x = MinMaxScaler()
            self.scaler_y = MinMaxScaler()
        else:
            self.scaler_x = scaler_x
            self.scaler_y = scaler_y

    # Function to create sequences.
    def create_sequences(self, data):
        xs = []
        ys = []
        for i in range(len(data) - self.seq_length):
            x = data[
                i : (i + self.seq_length), : -len(self.target_features)
            ]  # Select input features
            y = data[
                i + self.seq_length, -len(self.target_features) :
            ]  # ] Select targets
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def setup(self, stage=None):
        grouped_data = self._data[self._data["cluster"] == self.clusters]
        grouped_data = grouped_data[
            self.input_features + self.target_features
        ].values

        # Normalize data separately for this cluster
        grouped_data[:, : -len(self.target_features)] = (
            self.scaler_x.fit_transform(
                grouped_data[:, : -len(self.target_features)]
            )
        )
        grouped_data[:, -len(self.target_features) :] = (
            self.scaler_y.fit_transform(
                grouped_data[:, -len(self.target_features) :]
            )
        )

        x, y = self.create_sequences(grouped_data)

        split_idx = int(len(x) * self.train_split)
        X_train, X_test = x[:split_idx], x[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        self.train_data = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        self.val_data = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
        )

        # Save the fitted scalers
        joblib.dump(
            self.scaler_x,
            os.path.join(
                self.model_path, f"LSTM_scaler_x_{self.clusters}.pkl"
            ),
        )
        joblib.dump(
            self.scaler_y,
            os.path.join(
                self.model_path, f"LSTM_scaler_y_{self.clusters}.pkl"
            ),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=11,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=11,
            shuffle=False,
        )


class Training_LSTM:
    def __init__(self, data, model_archive, **kwargs):
        self._data = data

        self.input_features = kwargs.get("input_features")
        self.target_features = kwargs.get("target_features")
        self.clusters = kwargs.get("clusters")
        self.seq_length = kwargs.get("seq_length")
        self.batch_size = kwargs.get("batch_size")
        self.train_split = kwargs.get("train_split")
        self.epochs = kwargs.get("epochs")
        self.learning_rate = kwargs.get("learning_rate")
        self._num_layers = kwargs.get(
            "num_layers"
        )  # number of recurrent layers in the lstm
        self._hidden_size = kwargs.get(
            "hidden_size"
        )  # neurons in each lstm layer
        self._input_size = kwargs.get("input_size")  # number of input features
        self._output_size = kwargs.get(
            "output_size"
        )  # number of output classes
        self.model_archive = model_archive

        # Validation for required parameters
        required_params = [
            "seq_length",
            "input_features",
            "target_features",
            "clusters",
            "train_split",
            "batch_size",
            "epochs",
            "learning_rate",
            "_num_layers",
            "_hidden_size",
            "_input_size",
            "_output_size",
        ]
        for param in required_params:
            if getattr(self, param) is None:
                raise ValueError(f"'{param}' parameter must be provided.")

    # # Function to initialize training.
    def initialize_training(self):
        client = ScanflowTrackerClient(verbose=True)
        mlflow.set_tracking_uri(client.get_tracker_uri(True))
        logging.info("Connecting tracking server uri: {}".format(mlflow.get_tracking_uri()))

        mlflow.set_experiment("LSTM Experiment")
        models_dict = {}  # To store models for each cluster

        # Iterate through each cluster
        for cluster_id in self._data[self.clusters].unique():
            # Instantiate DataModule for the specific cluster
            data_module = LSTMDataModulePerCluster(
                self._data,
                self.input_features,
                self.target_features,
                cluster_id,
                self.model_archive,
                self.seq_length,
                self.batch_size,
                self.train_split,
            )

            model = LSTMModel(
                num_layers=self._num_layers,
                hidden_size=self._hidden_size,
                input_size=self._input_size,
                output_size=self._output_size,
                learning_rate=self.learning_rate,
            )

            # trainer = pl.Trainer(max_epochs=self.epochs, logger=mlflow_logger)
            trainer = pl.Trainer(max_epochs=self.epochs, logger=False)

            with mlflow.start_run(run_name=f"Cluster_{cluster_id}") as run:
                
                # mlflow.pytorch.autolog()
                trainer.fit(model,data_module)
                # Save the trained model to the dictionary
                models_dict[cluster_id] = model

                mlflow.pytorch.log_model(
                    model, 
                    artifact_path=f"model_LSTM_{cluster_id}.pt", 
                    registered_model_name=f"model_LSTM_{cluster_id}.pt"
                    )

                scaler_x = joblib.load(
                os.path.join(
                    self.model_archive, f"LSTM_scaler_x_{cluster_id}.pkl"
                )
                )

                scaler_y = joblib.load(
                    os.path.join(
                        self.model_archive, f"LSTM_scaler_y_{cluster_id}.pkl"
                    )
                )

                mlflow.log_metric('epochs', self.epochs)
                mlflow.sklearn.log_model(scaler_x, artifact_path= f"LSTM_scaler_x_{cluster_id}",registered_model_name = f"LSTM_scaler_x_{cluster_id}")
                mlflow.sklearn.log_model(scaler_y, artifact_path= f"LSTM_scaler_y_{cluster_id}",registered_model_name = f"LSTM_scaler_y_{cluster_id}")

                train_loss = trainer.callback_metrics.get('train_loss')
                val_loss = trainer.callback_metrics.get('val_loss')

                if train_loss is not None:
                    mlflow.log_metric('train_loss', train_loss.item())

                if val_loss is not None:
                    mlflow.log_metric('val_loss', val_loss.item())
            mlflow.end_run()

        return models_dict


class Inference:
    def __init__(self, test_data, model_path, **kwargs):
        super(Inference, self).__init__()
        self._data = test_data
        self.input_features = kwargs.get("input_features")
        self.target_features = kwargs.get("target_features")
        self.clusters = kwargs.get("clusters")
        self.seq_length = kwargs.get("seq_length")
        self.batch_size = kwargs.get("batch_size")
        self.train_split = kwargs.get("train_split")
        self.epochs = kwargs.get("epochs")
        self.learning_rate = kwargs.get("learning_rate")
        self._num_layers = kwargs.get(
            "num_layers"
        )  # number of recurrent layers in the LSTM
        self._hidden_size = kwargs.get(
            "hidden_size"
        )  # neurons in each LSTM layer
        self._input_size = kwargs.get("input_size")  # number of input features
        self._output_size = kwargs.get(
            "output_size"
        )  # number of output classes
        self.model_path = model_path

    @staticmethod
    def make_predictions(model, data_module):
        predictions = []
        actuals = []
        model.eval()
        with torch.no_grad():
            for batch in data_module.val_dataloader():
                x, y = batch
                out, _, _ = model(x)
                predictions.append(out.cpu().numpy())
                actuals.append(y.cpu().numpy())
        return np.concatenate(predictions), np.concatenate(actuals)

    @staticmethod
    def evaluate_model(predictions, actuals):
        mae = mean_absolute_error(actuals, predictions)
        mse = mean_squared_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        print(f"MAE: {mae}, MSE: {mse}, R²: {r2}")
        return mae, mse, r2

    def plot_results(self, predictions, actuals, cluster_id):
        plt.figure(figsize=(15, 5))
        plt.plot(actuals, label="Actual")
        plt.plot(predictions, label="Predicted")
        plt.legend()
        plt.title("Actual vs Predicted")
        plt.savefig(
            os.path.join(
                self.model_path, f"actual_vs_predicted_{cluster_id}.png"
            )
        )

        residuals = actuals - predictions
        plt.figure(figsize=(15, 5))
        plt.plot(residuals)
        plt.title("Residuals")
        plt.savefig(
            os.path.join(self.model_path, f"residuals_{cluster_id}.png")
        )

    # Function toinitialize inference
    def intialize_inference(self):
        for cluster_id in self._data[self.clusters].unique():
            scaler_x = joblib.load(
                os.path.join(
                    self.model_path, f"LSTM_scaler_x_{cluster_id}.pkl"
                )
            )
            scaler_y = joblib.load(
                os.path.join(
                    self.model_path, f"LSTM_scaler_y_{cluster_id}.pkl"
                )
            )
            # Instantiate DataModule for the specific cluster
            data_module = LSTMDataModulePerCluster(
                self._data,
                self.input_features,
                self.target_features,
                cluster_id,
                model_path=self.model_path,
                seq_length=self.seq_length,
                batch_size=self.batch_size,
                train_split=self.train_split,
                scaler_x=scaler_x,
                scaler_y=scaler_y,
            )

            data_module.setup()

            # Load the trained model (assuming model is already defined and trained)
            model = LSTMModel(
                self._num_layers,
                self._hidden_size,
                self._input_size,
                self._output_size,
                self.learning_rate,
            )
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        self.model_path, f"model_LSTM_{cluster_id}.pt"
                    )
                )
            )
            # Make predictions and evaluate
            predictions, actuals = self.make_predictions(model, data_module)
            self.plot_results(predictions, actuals, cluster_id)
            self.evaluate_model(predictions, actuals)
