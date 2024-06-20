#!/usr/bin/env python3
from __future__ import annotations

import os

import pandas as pd
import torch

from .model_types.LSTM import Inference
from .model_types.LSTM import Training_LSTM


class Training:
    def __init__(self, **kwargs):
        self.archive = kwargs.get("archive")
        self.key_values = kwargs.get("key_values")
        self.features = kwargs.get("features")
        self.model_type = kwargs.get("model_type")
        self.model_archive = kwargs.get("model_archive")

        if not os.path.exists(self.model_archive):
            os.makedirs(self.model_archive)

        if not self.archive:
            raise ValueError("'archive' parameter must be provided.")
        if not self.key_values:
            raise ValueError("'key_values' parameters must be provided.")
        if not self.features:
            raise ValueError("'features' parameters must be provided.")
        if not self.model_type:
            raise ValueError("'model_type' parameters must be provided.")

        # Begin trainig models.
        self.train()

    def train(self):
        # Check what models are specified and act accordingly
        if "LSTM" in self.model_type:
            config_lstm = self.model_type.get("LSTM")
            print("Training LSTM model...")
            data = self.preprocess_LSTM(
                self.key_values, self.features, self.archive
            )
            train_LSTM = Training_LSTM(data, self.model_archive, **config_lstm)
            models_dict = train_LSTM.initialize_training()

            for cluster_id, model in models_dict.items():
                torch.save(
                    models_dict[cluster_id].state_dict(),
                    os.path.join(
                        self.model_archive, f"model_LSTM_{cluster_id}.pt"
                    ),
                )
            config_inference = config_lstm.get("inference")
            if config_inference:
                print("Inference starting:")
                test_data = self.preprocess_LSTM(
                    self.key_values, self.features, config_inference
                )
                inference = Inference(
                    test_data, self.model_archive, **config_lstm
                )
                print(inference.intialize_inference())

        if "SVM" in self.model_type:
            # self.preprocess_for_svm()
            # Add SVM model training code here
            print("Training SVM model...")

    ### LSTM model specific functions
    @staticmethod
    def preprocess_LSTM(
        key_values, features, archive_path
    ):  # Implement the preprocessing specific to LSTM model

        files = os.listdir(archive_path)  # Get the list of files in the folder
        csv_files = [
            file for file in files if file.endswith(".csv")
        ]  # Filter out non-csv files
        csv_files.sort(
            key=lambda x: os.path.getmtime(os.path.join(archive_path, x)),
            reverse=True,
        )  # Sort the csv files by modification time in descending order

        most_recent_csv = os.path.join(
            archive_path, csv_files[0]
        )  # Get the path of the most recent csv file
        data = pd.read_csv(most_recent_csv, sep=";")  # Read the csv file
        # Drop columns except key_values and features
        data = data[key_values + features]
        # Convert timestamp to datetime
        data[key_values[0]] = data[key_values[0]].astype("datetime64[s]")
        # Use groupby with pd.Grouper to resample and calculate the mean for each 15-second interval and each cluster
        data = (
            data.groupby(
                [
                    pd.Grouper(key=key_values[0], freq="15s"),
                    key_values[1],
                ]
            )
            .mean()
            .reset_index()
        )
        # Sort by cluster and timestamp
        data = data.sort_values(by=[key_values[1], key_values[0]])
        return data
