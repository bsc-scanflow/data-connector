#!/usr/bin/env python3
from __future__ import annotations

import os

import pandas as pd

## TDD: Test Driven Development


class Preprocessing:
    """
    This class initializes the required parameters for preprocessing the test files and provides the required methods to
    preprocess them and save them in the archive_preprocessed directory.

    """

    def __init__(self, **kwargs):

        self.directory = kwargs.get("directory")
        self.archive = kwargs.get("archive")
        self.pipeline = kwargs.get("pipeline")
        self.cluster = kwargs.get("cluster")

        data_structure_dict = kwargs.get("data_structure")
        # Creating self.dtypes
        self.dtypes = {}
        for key, section in data_structure_dict.items():
            for item in section:
                self.dtypes[item[0]] = item[1]

        # Creating self.key_values
        self.key_values = self.retrieve_metrics(
            data_structure_dict.get("key_values")
        )

        # Creating self.pipeline_values
        self.pipeline_values = self.retrieve_metrics(
            data_structure_dict.get("pipeline_values")
        )

        # Creating self.node_values
        self.node_values = self.retrieve_metrics(
            data_structure_dict.get("node_values")
        )

        # Creating self.pipeline_telemetry
        self.pipeline_telemetry = self.retrieve_metrics(
            data_structure_dict.get("pipeline_telemetry")
        )

        if not self.directory:
            raise ValueError("'directory' parameter must be provided.")
        if not self.archive:
            raise ValueError("'archive' parameter must be provided.")
        if not self.pipeline:
            raise ValueError("'pipeline' parameter must be provided.")
        if not self.cluster:
            raise ValueError("'cluster' parameter must be provided.")

    # First, we need to check how many "valid" files we have in the test directory, we check how many are csv.
    # How do we know the directory: parameter given by the user.

    @staticmethod
    def retrieve_metrics(values: list):
        return [metric for metric, _ in values]

    @staticmethod
    def get_files(directory):
        """
        Returns a list of CSV filenames in the specified directory if it exists and contains CSV files.
        Raises an OSError if the directory does not exist.
        Returns None if the directory is empty or contains no CSV files.
        """
        try:
            list_files = os.listdir(directory)
            csv_files = [file for file in list_files if file.endswith(".csv")]
            return csv_files
        except FileNotFoundError:
            raise OSError("Directory does not exist.")

    # Second, we need to ensure they share the same structure.

    def check_file_structure(self, directory):
        """
        Iterates through the CSV files in the specified directory and checks if their structure (columns) matches
        the one passed as an argument to the __init__ function: self.structure.
        We have 4 options of the structure:
            1. We have all the metrics (pipeline + node).
            2. We are missing the pipeline metrics.
            3. We are missing the node/pipeline server metrics.
                3.1 We are missing the node metrics.
                3.2 We are missing the pipeline server metrics.
                3.3 We are missing both node and pipeline server metrics.
            4. We are missing both pipeline and node/pipeline server metrics.
        For cases 1 and 2, we can still preprocess the data. For cases 3 and 4, we cannot, raises a ValueError.
        """
        files = self.get_files(directory)
        csv_files_structured = []

        for file in files:
            file_path = os.path.join(directory, file)
            with open(file_path, "r") as f:
                header = f.readline().strip().split(";")
                # Option 1: We have all the metrics (pipeline + node).
                if (
                    header
                    == self.key_values
                    + self.pipeline_values
                    + self.node_values
                    + self.pipeline_telemetry
                ):
                    csv_files_structured.append(file)
                # Option 2: We are missing the pipeline metrics.
                elif (
                    header
                    == self.key_values
                    + self.node_values
                    + self.pipeline_telemetry
                ):
                    header += self.pipeline_values
                    csv_files_structured.append(file)
                # Option 3: We are missing the node metrics.
                elif header != self.key_values + self.pipeline_values:
                    raise ValueError(
                        f"Error! The file'{file}' is missing node/pipeline server metrics. Check observability stack."
                    )
                # Option 4: We are missing both pipeline and node metrics.
                elif header == self.key_values:
                    raise ValueError(
                        f"Error! The file'{file}' is missing node and pipeline metrics. Check observability stack."
                    )
                else:
                    raise ValueError(
                        f"The structure of file '{file}' does not match the expected structure."
                    )

        return csv_files_structured

    # Third, we need to preprocess them: Nulls, duplicates, merges, rearrange, rename....

    def preprocess_files(self):

        # Input files is a list of csv's.
        csv_files_structured = self.check_file_structure(self.directory)
        dataset = None

        for file in csv_files_structured:
            file_path = os.path.join(self.directory, file)
            data = pd.read_csv(file_path, sep=";")
            # Set dtypes as specified in the initialization self.dtypes.
            data = data.astype(self.dtypes)

            # Downcasting behavior handling
            data = data.infer_objects(copy=False)

            # Start with the telemetry data, drop the rows that contain all node values null at the same time.
            target_df = data[
                self.key_values + self.node_values + self.pipeline_telemetry
            ].dropna(
                subset=self.node_values + self.pipeline_telemetry, how="all"
            )

            # Pipeline data.
            # Drop duplicates keeping the last value, check if the pipeline_id is null, as it means no pipeline is running. Fill with 0.
            features_df = data.drop(
                columns=self.node_values + self.pipeline_telemetry
            ).copy()
            features_df.drop_duplicates(
                subset=self.key_values, keep="last", inplace=True
            )

            # Handle nulls in rows where features_df[self.pipeline] is null
            null_pipeline_mask = features_df[self.pipeline].isnull()
            for column in features_df.columns:
                if pd.api.types.is_numeric_dtype(features_df[column]):
                    features_df.loc[null_pipeline_mask, column] = (
                        features_df.loc[null_pipeline_mask, column].fillna(0)
                    )
                elif pd.api.types.is_datetime64_any_dtype(features_df[column]):
                    features_df.loc[null_pipeline_mask, column] = (
                        features_df.loc[null_pipeline_mask, column].fillna(
                            pd.Timestamp("1970-01-01")
                        )
                    )
                elif pd.api.types.is_object_dtype(features_df[column]):
                    features_df.loc[null_pipeline_mask, column] = (
                        features_df.loc[null_pipeline_mask, column].fillna("0")
                    )

            features_df.dropna(inplace=True)

            # Merge those rows of pipeline_id null with the telemetry data, and fill with 0 the cpu and mem usage of pipeline.
            filtered_df = features_df.loc[
                features_df[self.pipeline] == 0, self.key_values
            ]
            filtered_target_df = pd.merge(
                target_df, filtered_df, on=self.key_values
            )
            filtered_target_df.loc[:, self.pipeline_telemetry] = 0
            filtered_target_df.drop_duplicates(
                subset=self.key_values, keep="first", inplace=True
            )
            filtered_target_df.set_index(self.key_values, inplace=True)

            # Update telemetry data with the new data of null pipelines.
            target_df.set_index(self.key_values, inplace=True)
            target_df.update(filtered_target_df)
            target_df.reset_index(drop=False, inplace=True)

            target_df.drop_duplicates(
                subset=self.key_values, keep="first", inplace=True
            )
            merged_df = pd.merge(features_df, target_df, on=self.key_values)

            # Handle NaN values with linear interpolation within each cluster
            columns_with_nan = merged_df.columns[
                merged_df.isna().any()
            ].tolist()

            for column in columns_with_nan:
                merged_df[column] = merged_df.groupby(self.cluster)[
                    column
                ].transform(
                    lambda x: x.interpolate(
                        method="linear", limit_direction="forward"
                    )
                )
            merged_df.dropna(inplace=True)

            if merged_df.empty:
                continue
            else:
                if dataset is None:
                    dataset = merged_df
                else:
                    dataset = pd.concat(
                        [dataset, merged_df], ignore_index=True
                    ).reset_index(drop=True)

        def calculate_fps(combined_df):
            data = []

            for cluster in combined_df.cluster.unique():
                # Subset data for the current cluster
                df = combined_df[combined_df["cluster"] == cluster]
                # Calculate the difference with the previous row
                df["elapsed_time_diff"] = df[self.pipeline_values[4]].diff()
                df["frame_count_diff"] = df[self.pipeline_values[5]].diff()

                # Calculate the frame rate
                df["frame_rate"] = df["frame_count_diff"] / (
                    df["elapsed_time_diff"]
                )  # Convert elapsed time to seconds

                # Handle the first row which will have NaN values due to diff()
                df["frame_rate"].fillna(0, inplace=True)

                data.append(df)

            # Combine the dataframes
            df = pd.concat(data).reset_index(drop=True)

            # Drop intermediate columns
            df.drop(
                columns=[
                    "frame_count_diff",
                    "elapsed_time_diff",
                    self.pipeline_values[-1],
                ],
                inplace=True,
            )
            df.rename(
                columns={"frame_rate": self.pipeline_values[-1]}, inplace=True
            )
            return df

        @staticmethod
        def clean_outliers_hardcoded(df, column, limit_value):
            # Adjust normalization based on conditions
            df[column] = df[column].clip(0, limit_value)
            return df

        @staticmethod
        def normalize(series, min_value, max_value):
            return (series - min_value) / (max_value - min_value)

        @staticmethod
        def calculate_qos(
            data,
            min_fps,
            max_fps,
            min_lat,
            max_lat,
            fps_col,
            latency_col,
            w_fps=0.5,
            w_latency=0.5,
        ):
            # Normalize FPS and Latency
            data["Normalized FPS"] = normalize(
                data[fps_col], min_fps, max_fps
            ).clip(0, 1)
            data["Normalized Latency"] = normalize(
                data[latency_col], min_lat, max_lat
            ).clip(0, 1)

            # Calculate QoS
            data["QoS"] = (w_fps * data["Normalized FPS"]) + (
                w_latency * (1 - data["Normalized Latency"])
            )
            data["QoS"] = data["QoS"].clip(0, 1)  # Ensure QoS is within [0, 1]
            # Adjust normalization based on conditions
            data["QoS"] = data.apply(
                lambda row: (
                    0
                    if row[fps_col] > max_fps or row[fps_col] < min_fps
                    else row["QoS"]
                ),
                axis=1,
            )
            data["QoS"] = data.apply(
                lambda row: (
                    0
                    if row[latency_col] > max_lat or row[latency_col] < min_lat
                    else row["QoS"]
                ),
                axis=1,
            )
            # Drop intermediate columns
            data.drop(
                columns=["Normalized FPS", "Normalized Latency"], inplace=True
            )
            return data

        if dataset is not None:
            dataset.drop_duplicates(
                subset=self.key_values, keep="first", inplace=True
            )
            dataset = calculate_fps(dataset)
            dataset = clean_outliers_hardcoded(
                dataset, self.pipeline_values[6], 1
            )
            # Calculate QoS with equal weights
            qos_data = calculate_qos(
                dataset,
                20,
                30,
                0.001,
                0.2,
                self.pipeline_values[-1],
                self.pipeline_values[6],
            )

            qos_data.sort_values(self.key_values[0], inplace=True)
            return qos_data
        else:
            return None

    # Fourth, we need to save them in the archive_preprocessed directory.
    def merge_current_datasets(self, dataset):
        """
        Merges the current preprocessed dataset with the last version stored in archive_preprocessed into a single DataFrame.
        """
        # Find the latest dataset in the archive
        list_of_files = self.get_files(self.archive)
        if list_of_files:
            latest_file = max(
                (os.path.join(self.archive, file) for file in list_of_files),
                key=os.path.getctime,
            )
            latest_dataset = pd.read_csv(latest_file, sep=";")
        else:
            latest_dataset = None

        # dataset is the DataFrame to be merged
        if dataset is not None:
            if latest_dataset is not None:
                # Ensure both columns are of the same type for comparison
                latest_dataset[self.key_values[0]] = pd.to_datetime(
                    latest_dataset[self.key_values[0]], errors="coerce"
                )
                dataset[self.key_values[0]] = pd.to_datetime(
                    dataset[self.key_values[0]], errors="coerce"
                )
                # Concatenate with the latest dataset and remove duplicates
                if max(latest_dataset[self.key_values[0]]) >= max(
                    dataset[self.key_values[0]]
                ):
                    raise ValueError(
                        "Data cannot be overwritten, no new timestamps are being added."
                    )
                elif (
                    latest_dataset.columns.tolist() != dataset.columns.tolist()
                ):
                    raise ValueError(
                        "The structure of the preprocessed data does not match the previous data."
                    )
                elif latest_dataset.dtypes.tolist() != dataset.dtypes.tolist():
                    raise ValueError(
                        "The dtypes of the preprocessed data does not match the previous data."
                    )
                else:
                    updated_dataset = pd.concat(
                        [latest_dataset, dataset], ignore_index=True
                    )
                    updated_dataset.drop_duplicates(
                        subset=self.key_values, keep="first", inplace=True
                    )
            elif latest_dataset is None:
                updated_dataset = dataset

            # Save the merged dataset with a timestamp in the filename
            timestamp_str = max(updated_dataset[self.key_values[0]])
            filename = f"preprocessed_data_{timestamp_str}.csv"
            updated_dataset.to_csv(
                os.path.join(self.archive, filename), index=False, sep=";"
            )
            return filename, updated_dataset
        else:
            return None, None

    # Sixth, we need to delete the files in the test directory.
    def purge_old_files(self, filename, dataset):
        """
        Deletes the files in the test directory after they have been preprocessed and stored in the archive_preprocessed directory.
        """
        if filename is not None and dataset is not None:
            if filename in os.listdir(self.archive):
                files = self.check_file_structure(self.directory)
                for file in files:
                    os.remove(os.path.join(self.directory, file))
            else:
                raise ValueError(
                    "The files have not been saved to the archive_preprocessed directory."
                )
