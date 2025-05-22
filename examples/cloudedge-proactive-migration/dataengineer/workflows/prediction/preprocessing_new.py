from __future__ import annotations

import os
from typing import Tuple, Dict

import pandas as pd


class Preprocessing:
    def __init__(self, **kwargs):
        self.output = kwargs.get("output")
        self.input = kwargs.get("input")
        self.pipeline = kwargs.get("pipeline")
        self.cluster = kwargs.get("cluster")
        self.inference = kwargs.get("inference")

        data_structure_dict = kwargs.get("data_structure")
        # Creating attribute lists and dtypes
        self.dtypes = {
            item[0]: item[1]
            for key, section in data_structure_dict.items()
            for item in section
        }
        self.key_values = self.retrieve_metrics(
            data_structure_dict.get("key_values")
        )
        self.pipeline_values = self.retrieve_metrics(
            data_structure_dict.get("pipeline_values")
        )
        self.node_values = self.retrieve_metrics(
            data_structure_dict.get("node_values")
        )
        self.pipeline_telemetry = self.retrieve_metrics(
            data_structure_dict.get("pipeline_telemetry")
        )

        if not all(
            [self.output, self.input, self.pipeline, self.cluster]
        ):
            raise ValueError("Missing required parameters")
        
        if self.inference == 1:
            inference_dataset=self.preprocess_files()
            aggregated_dataset = self.aggregate_by_cluster(inference_dataset)
            cluster_dfs = self.prepare_inference_data(aggregated_dataset)
            valid_clusters = self.check_nulls_in_numerical_columns(cluster_dfs)
            valid_clusters = self.check_data_completeness(valid_clusters)
            self.purge_input_files()
            self.save_inference_data(valid_clusters)
        else:
            self.inference = 0

    @staticmethod
    def retrieve_metrics(values: list):
        return [metric for metric, _ in values]

    @staticmethod
    def get_files(input):
        try:
            list_files = os.listdir(input)
            csv_files = [file for file in list_files if file.endswith(".csv")]
            return csv_files
        except FileNotFoundError:
            raise OSError("Directory does not exist.")

    def preprocess_files(self):
        dataset = None
        csv_files = self.get_files(self.input)
        current_day = None

        for file in csv_files:
            # Read the CSV
            file_path = os.path.join(self.input, file)
            data = pd.read_csv(file_path, sep=";")

            # Convert data types according to config
            data = data.astype(self.dtypes)

            if self.inference == 0:
                # Handle timestamp modification
                min_timestamp = pd.Timestamp(data[self.key_values[0]].min())
                file_day = min_timestamp.date()

                # Set target day
                if current_day is None:
                    current_day = file_day
                    target_day = current_day
                else:
                    current_day = current_day + pd.Timedelta(days=1)
                    target_day = current_day

                # Calculate time adjustment to start at 9:00 AM
                target_start = pd.Timestamp.combine(
                    target_day, pd.Timestamp("09:00:00").time()
                )
                time_adjustment = target_start - min_timestamp

                # Adjust all timestamps in this file
                data[self.key_values[0]] = (
                    pd.to_datetime(data[self.key_values[0]]) + time_adjustment
                )
            else:
                # Convert timestamp to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(data[self.key_values[0]]):
                    data[self.key_values[0]] = pd.to_datetime(data[self.key_values[0]], unit='s')
        
            # Split data into two parts:
            # 1. Rows with pipeline data (where pipeline_id is not null)
            pipeline_mask = ~data[self.pipeline].isna()
            pipeline_rows = data[pipeline_mask].copy()

            # 2. Rows with telemetry data (where node_values or pipeline_telemetry are not all null)
            telemetry_columns = self.node_values + self.pipeline_telemetry
            telemetry_mask = ~data[telemetry_columns].isna().all(axis=1)
            telemetry_rows = data[telemetry_mask].copy()

            # Keep only one telemetry row per timestamp+cluster combination
            telemetry_rows = telemetry_rows[
                self.key_values + telemetry_columns
            ].drop_duplicates(subset=self.key_values, keep="first")

            # Merge telemetry data back to pipeline rows
            merged_df = pd.merge(
                pipeline_rows.drop(
                    columns=telemetry_columns, errors="ignore"
                ),  # Remove telemetry columns if they exist
                telemetry_rows,
                on=self.key_values,
                how="left",
            )

            # Handle any remaining NaN values through interpolation within each cluster
            for column in merged_df.columns[merged_df.isna().any()]:
                if column in telemetry_columns:
                    merged_df[column] = merged_df.groupby(self.cluster)[
                        column
                    ].transform(
                        lambda x: x.interpolate(
                            method="linear", limit_direction="forward"
                        )
                        .fillna(method="bfill")
                        .fillna(method="ffill")
                    )

            # Append to main dataset
            if dataset is None:
                dataset = merged_df
            else:
                dataset = pd.concat([dataset, merged_df], ignore_index=True)

        if dataset is not None:
            # Final sorting and cleaning
            dataset.sort_values(self.key_values[0], inplace=True)
            dataset.reset_index(inplace=True, drop=True)
            return dataset
        return None

    def aggregate_by_cluster(self, df):
        """
        Aggregates data by timestamp and cluster, dropping specified columns and creating a pipeline count column.

        Args:
            df: pandas DataFrame containing the preprocessed data

        Returns:
            DataFrame aggregated by timestamp and cluster with averaged metrics
        """
        # Create number_pipelines column before grouping
        pipeline_counts = (
            df.groupby([self.key_values[0], self.cluster])
            .size()
            .reset_index(name="number_pipelines")
        )

        # Columns to drop
        columns_to_drop = [
            "pipelines_status_start_time",
            "pipelines_status_frame_count",
            "pipelines_status_count_pipeline_latency",
            "pipelines_status_elapsed_time",
            "pipelines_status_sum_pipeline_latency",
            "pipeline_id",
        ]

        # Drop specified columns
        df_cleaned = df.drop(columns=columns_to_drop)

        # Group by timestamp and cluster and calculate mean for all numeric columns
        aggregated_df = (
            df_cleaned.groupby([self.key_values[0], self.cluster])
            .mean()
            .reset_index()
        )

        # Merge with pipeline counts
        final_df = pd.merge(
            aggregated_df,
            pipeline_counts,
            on=[self.key_values[0], self.cluster],
        )

        # Reorder columns to put pipelines_status_realtime_pipeline_latency at the end
        columns = [
            col
            for col in final_df.columns
            if col != "pipelines_status_realtime_pipeline_latency"
        ]
        columns.append("pipelines_status_realtime_pipeline_latency")

        final_df = final_df[columns]

        return final_df

    def prepare_training_data(
        self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepares the preprocessed data for training by:
        1. Making timestamps unique by shifting second cluster to start after first cluster
        2. Splitting the data into train/val/test sets by cluster

        Args:
            df: Input DataFrame containing preprocessed data
            test_size: Proportion of data to use for testing (default: 0.2)
            val_size: Proportion of remaining data to use for validation (default: 0.1)

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_dfs = []
        val_dfs = []
        test_dfs = []

        # Sort clusters to ensure consistent ordering
        clusters = sorted(df[self.cluster].unique())

        # Process first cluster as is
        first_cluster = clusters[0]
        first_cluster_df = df[df[self.cluster] == first_cluster].copy()
        first_cluster_df = first_cluster_df.sort_values(self.key_values[0])

        # Get the last timestamp of first cluster
        last_timestamp = pd.Timestamp(
            first_cluster_df[self.key_values[0]].max()
        )
        next_day_start = pd.Timestamp.combine(
            (last_timestamp + pd.Timedelta(days=1)).date(),
            pd.Timestamp("09:00:00").time(),
        )

        # Process second cluster with adjusted timestamps
        second_cluster = clusters[1]
        second_cluster_df = df[df[self.cluster] == second_cluster].copy()
        second_cluster_df = second_cluster_df.sort_values(self.key_values[0])

        # Calculate time adjustment for second cluster
        min_second_timestamp = pd.Timestamp(
            second_cluster_df[self.key_values[0]].min()
        )
        time_adjustment = next_day_start - min_second_timestamp

        # Adjust timestamps for second cluster
        second_cluster_df[self.key_values[0]] = (
            pd.to_datetime(second_cluster_df[self.key_values[0]])
            + time_adjustment
        )

        # Combine clusters
        adjusted_df = pd.concat([first_cluster_df, second_cluster_df])

        # Process each cluster for train/val/test split
        for cluster in clusters:
            cluster_df = adjusted_df[
                adjusted_df[self.cluster] == cluster
            ].copy()
            cluster_df = cluster_df.sort_values(self.key_values[0])

            # Calculate split indices
            total_rows = len(cluster_df)
            test_split_idx = int(total_rows * (1 - test_size))
            val_split_idx = int(test_split_idx * (1 - val_size))

            # Split the data while maintaining temporal order
            test_df = cluster_df[test_split_idx:]
            val_df = cluster_df[val_split_idx:test_split_idx]
            train_df = cluster_df[:val_split_idx]

            # Append to respective lists
            train_dfs.append(train_df)
            val_dfs.append(val_df)
            test_dfs.append(test_df)

        # Concatenate splits maintaining cluster ordering
        final_train_df = pd.concat(train_dfs, axis=0)
        final_val_df = pd.concat(val_dfs, axis=0)
        final_test_df = pd.concat(test_dfs, axis=0)

        # Reset indices
        final_train_df = final_train_df.reset_index(drop=True)
        final_val_df = final_val_df.reset_index(drop=True)
        final_test_df = final_test_df.reset_index(drop=True)

        # Rename timestamp column to 'date' in final datasets
        final_train_df = final_train_df.rename(
            columns={self.key_values[0]: "date"}
        )
        final_val_df = final_val_df.rename(
            columns={self.key_values[0]: "date"}
        )
        final_test_df = final_test_df.rename(
            columns={self.key_values[0]: "date"}
        )

        return final_train_df, final_val_df, final_test_df
    
    def prepare_inference_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Prepares the preprocessed data for inference by:
        1. Renaming timestamp column to 'date'
        2. Splitting data by cluster for separate inference

        Args:
            df: Input DataFrame containing preprocessed data

        Returns:
            Dictionary of DataFrames by cluster ID
        """
        if df is None or df.empty:
            return {}
            
        # Rename timestamp column to 'date' to match training format
        df = df.rename(columns={self.key_values[0]: "date"})
        
        # Split data by cluster
        clusters = df[self.cluster].unique()
        cluster_dfs = {}
        
        for cluster_id in clusters:
            cluster_df = df[df[self.cluster] == cluster_id].copy()
            cluster_df = cluster_df.sort_values("date").reset_index(drop=True)
            cluster_dfs[cluster_id] = cluster_df
            
        return cluster_dfs

    def save_inference_data(self, cluster_dfs: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """
        Save the processed data as separate files for each cluster ID.
        
        Args:
            cluster_dfs: Dictionary of DataFrames by cluster ID
            
        Returns:
            Dictionary mapping cluster ID to its output file path
        """
        if not cluster_dfs:
            return {}
            
        # Format current time for filename
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            
        output_files = {}
        for cluster_id, cluster_df in cluster_dfs.items():
            # Create a sanitized cluster ID for the filename
            safe_cluster_id = str(cluster_id).replace("-", "_")
            
            # Create output filename
            filename = f"inference_data_{safe_cluster_id}_{timestamp}.csv"
            output_path = os.path.join(self.output, filename)
            
            # Save DataFrame to CSV
            cluster_df.to_csv(output_path, index=False)
            output_files[cluster_id] = output_path
            
        return output_files
    
    def check_nulls_in_numerical_columns(self, cluster_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Check for null values in numerical columns of each cluster DataFrame.
        Filters out clusters that have null values in numerical columns.

        Args:
            cluster_dfs: Dictionary of DataFrames by cluster ID

        Returns:
            Filtered dictionary of DataFrames (removes clusters with null values)
        """
        if not cluster_dfs:
            print("No cluster data available to check for nulls")
            return {}

        valid_clusters = {}
        invalid_clusters = []

        for cluster_id, df in cluster_dfs.items():
            # Get all numerical columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

            # Check for nulls in each column
            has_nulls = False
            null_cols = []

            for col in numeric_cols:
                null_count = df[col].isna().sum()
                if null_count > 0:
                    has_nulls = True
                    null_cols.append(f"{col} ({null_count} nulls)")

            if has_nulls:
                print(f"Warning: Cluster {cluster_id} has null values in numerical columns: {', '.join(null_cols)}")
                print(f"Cluster {cluster_id} will be excluded from inference")
                invalid_clusters.append(cluster_id)
            else:
                valid_clusters[cluster_id] = df

        if invalid_clusters:
            print(f"Removed {len(invalid_clusters)} clusters with null values: {invalid_clusters}")
        else:
            print("All clusters passed null value check")

        return valid_clusters

    def check_data_completeness(self, cluster_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Check if each cluster dataset is complete (at least 10 rows covering 5 minutes).

        Args:
            cluster_dfs: Dictionary of DataFrames by cluster ID
            interval_seconds: Expected time interval between consecutive rows (default: 30 seconds)

        Returns:
            Filtered dictionary with only the complete clusters
        """
        if not cluster_dfs:
            print("No cluster data available to check completeness")
            return {}

        valid_clusters = {}
        invalid_clusters = []

        for cluster_id, df in cluster_dfs.items():
            # Sort by timestamp
            df = df.sort_values("date")

            # Calculate time range
            min_time = df["date"].min()
            max_time = df["date"].max()
            total_seconds = (max_time - min_time).total_seconds()

            # Get actual row count
            actual_rows = len(df)

            # Check if we have at least 10 rows and 4.5 minutes of data
            is_complete = (total_seconds >= 270) and (actual_rows >= 10)

            if is_complete:
                valid_clusters[cluster_id] = df
            else:
                print(f"  - Insufficient data: cluster will be excluded")
                invalid_clusters.append(cluster_id)

        if invalid_clusters:
            print(f"Removed {len(invalid_clusters)} incomplete clusters")

        return valid_clusters
    
    def purge_input_files(self):
        """
        Delete all CSV files from the input directory.
        Only used during inference mode.
        """
        if self.inference != 1:
            return

        try:
            csv_files = self.get_files(self.input)
            deleted_count = 0

            for file in csv_files:
                file_path = os.path.join(self.input, file)
                os.remove(file_path)
                deleted_count += 1

            print(f"Purged {deleted_count} CSV files from input directory: {self.input}")

        except Exception as e:
            print(f"Error purging files: {e}")