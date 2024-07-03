import mlflow
import click
import logging
import pandas as pd
import os
import json
import sys


sys.path.insert(0, '/scanflow/scanflow')
from scanflow.client import ScanflowTrackerClient

@click.command(help="preprocessing")
@click.option("--config", type=str, required=True, help="Configuration file path")

def main(config):
    config = load_config(config)
    logging.info("Workflow step: Preprocessing")

    preprocessing_config = config.get("preprocessing")

    if preprocessing_config:
        data = preprocess_LSTM(
                preprocessing_config["key_values"],
                preprocessing_config["features"],
                preprocessing_config["archive"]
                )
        output_dir = preprocessing_config["output_dir"]
        save_preprocessed_data(data, preprocessing_config["key_values"][1], output_dir)

def load_config(config_path):
    with open(config_path, "r") as file:
        return json.load(file)

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
    return data.sort_values(by=[key_values[1], key_values[0]])


def save_preprocessed_data(data, cluster_key, output_dir):
    prep_dir=os.path.join(output_dir,"preprocessed_data")   
    if not os.path.exists(prep_dir):
        os.makedirs(prep_dir)
        logging.info(f"Created directory {prep_dir} for storing preprocessed data.")

    file_path = os.path.join(prep_dir, "preprocessed_data.csv")
    try:
        data.to_csv(file_path, index=False)
        logging.info(f"Saved preprocessed data.")
    except IOError as e:
        logging.error(f"Failed to save data:{e}")

if __name__ == '__main__':
    main()

