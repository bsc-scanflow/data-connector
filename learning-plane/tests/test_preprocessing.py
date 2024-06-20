from __future__ import annotations

import os
import shutil
import unittest
from unittest import TestCase
from unittest.mock import MagicMock
from unittest.mock import mock_open
from unittest.mock import patch

import pandas as pd
from app.preprocessing.preprocessing import Preprocessing
from tests.test_tools import config
from tests.test_tools import dataframe_1
from tests.test_tools import dataframe_2
from tests.test_tools import dataframe_3


# TDD: Test Driven Development
class TestPreprocessingFiletype1(TestCase):

    def setUp(self):
        # Initializing needed folders:
        # Create the folder if it doesn't exist
        os.makedirs(
            os.path.join(os.getcwd(), "tests", "archive", "empty_archive"),
            exist_ok=True,
        )

        # Create the folder if it doesn't exist
        os.makedirs(
            os.path.join(os.getcwd(), "tests", "scraped_data"), exist_ok=True
        )

        self.preprocessor = Preprocessing(**config["preprocessing_1"])

    def tearDown(self):
        # Delete the contents of the 'empty_archive' directory
        empty_archive_dir = os.path.join(
            os.getcwd(), "tests", "archive", "empty_archive"
        )
        if os.path.exists(empty_archive_dir):
            shutil.rmtree(empty_archive_dir)

        # Delete the contents of the 'scraped_data' directory
        scraped_data_dir = os.path.join(os.getcwd(), "tests", "scraped_data")
        if os.path.exists(scraped_data_dir):
            shutil.rmtree(scraped_data_dir)

    @patch("os.listdir")
    def test_empty_directory(self, mock_listdir):
        """Test handling of an empty directory."""
        # Set up the mock to simulate an empty directory
        mock_listdir.return_value = []
        # Test get_files method
        self.assertEqual(
            self.preprocessor.get_files(self.preprocessor.directory), []
        )
        mock_listdir.assert_called_with(os.path.join("dummy_path"))

    def test_nonexistent_directory(self):
        """Test handling of a nonexistent directory."""
        # Assert that OSError is raised when the directory does not exist
        with self.assertRaises(OSError) as context:
            self.preprocessor.get_files(self.preprocessor.directory)

        # Optionally check the message of the exception
        self.assertEqual(str(context.exception), "Directory does not exist.")

    @patch("os.listdir")
    def test_directory_with_files(self, mock_listdir):
        """Test handling of a directory with files."""
        # Set up the mock to simulate a directory with files
        mock_listdir.return_value = ["file1.csv", "file2.csv"]
        # Test get_files method
        self.assertEqual(
            self.preprocessor.get_files(self.preprocessor.directory),
            ["file1.csv", "file2.csv"],
        )

    def test_initialization(self):
        """Test initialization of the Preprocessing class."""
        # Test if the initialization correctly sets the attributes
        self.assertEqual(
            self.preprocessor.directory,
            os.path.join("dummy_path"),
        )
        self.assertEqual(
            self.preprocessor.archive,
            os.path.join("dummy_archive"),
        )

    @patch(
        "os.listdir",
        return_value=["data.csv", "info.txt", "report.csv", "image.png"],
    )
    def test_mixed_file_types(self, mock_listdir):
        """Test handling of a directory with mixed file types."""
        expected_files = ["data.csv", "report.csv"]
        files = self.preprocessor.get_files(self.preprocessor.directory)
        self.assertEqual(files, expected_files)

    @patch("os.listdir", return_value=["info.txt", "image.png"])
    def test_nocsv_with_files(self, mock_listdir):
        """Test handling of a directory with no CSV files."""
        # Test get_files method
        self.assertEqual(
            self.preprocessor.get_files(self.preprocessor.directory), []
        )

    @patch("os.listdir")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="timestamp;pipeline_id;CPU;MEM\n",
    )
    def test_correct_structure(self, mock_file, mock_listdir):
        """Test handling of a file with correct structure."""
        mock_listdir.return_value = ["test.csv"]
        # No exception should be raised
        self.preprocessor.check_file_structure(self.preprocessor.directory)

    @patch("os.listdir")
    @patch(
        "builtins.open", new_callable=mock_open, read_data="column1;column3\n"
    )
    def test_incorrect_structure(self, mock_file, mock_listdir):
        """Test handling of a file with incorrect structure."""
        mock_listdir.return_value = ["wrong_structure.csv"]
        with self.assertRaises(ValueError):
            self.preprocessor.check_file_structure(self.preprocessor.directory)

    @patch("os.listdir")
    def test_empty_directory_check(self, mock_listdir):
        """Test file structure check for an empty directory."""
        mock_listdir.return_value = []
        # Method should return None and not raise an exception
        self.assertEqual(
            self.preprocessor.check_file_structure(
                self.preprocessor.directory
            ),
            [],
        )

    @patch("os.listdir")
    def test_no_csv_files(self, mock_listdir):
        """Test file structure check for a directory with no CSV files."""
        mock_listdir.return_value = ["test.txt", "data.pdf"]
        # Method should return None and not raise an exception
        self.assertEqual(
            self.preprocessor.check_file_structure(
                self.preprocessor.directory
            ),
            [],
        )

    def test_empty_file_list(self):
        """Test preprocessing with an empty file list."""
        self.preprocessor.get_files = MagicMock(return_value=[])
        self.assertIsNone(self.preprocessor.preprocess_files())

    def tests_non_existing_archive(self):
        """Test handling of a non-existing archive directory."""
        # Assert that OSError is raised when the directory does not exist
        with self.assertRaises(OSError) as context:
            self.preprocessor.merge_current_datasets(dataframe_2)

        # Optionally check the message of the exception
        self.assertEqual(str(context.exception), "Directory does not exist.")

    def tests_empty_archive(self):
        """Test merging datasets with an empty archive."""
        # Delete the CSV file
        csv_file = os.path.join(
            os.getcwd(),
            "tests",
            "archive",
            "empty_archive",
            "preprocessed_data_5.csv",
        )
        if os.path.exists(csv_file):
            os.remove(csv_file)

        # Initialize the Preprocessing class
        dataset = dataframe_2
        self.preprocessor.archive = os.path.join(
            "tests", "archive", "empty_archive"
        )

        # Merge fake preprocessed dataset and store it into archive/empty_archive.
        self.preprocessor.merge_current_datasets(dataset)

        # check the stored dataset has the same name.
        self.assertEqual(
            os.listdir(
                os.path.join(os.getcwd(), "tests", "archive", "empty_archive")
            )[0],
            "preprocessed_data_5.csv",
        )
        # check the stored dataset is the same as the fake one.
        pd.testing.assert_frame_equal(pd.read_csv(csv_file, sep=";"), dataset)

        # Delete the CSV file
        os.remove(csv_file)

    def test_preprocessed_data_empty(self):
        """Test handling of an empty preprocessed dataset."""
        dataset = None
        self.preprocessor.archive = os.path.join(
            "tests", "archive", "empty_archive"
        )

        with self.assertRaises(ValueError) as context:
            self.preprocessor.merge_current_datasets(dataset)

        # Optionally check the message of the exception
        self.assertEqual(
            str(context.exception), "No data after preprocessing."
        )

    def tests_correct(self):
        """Test merging datasets with files in the archive and new preprocessed data."""
        # Initialize the Preprocessing class
        current_directory = os.getcwd()
        dataset = dataframe_3
        self.preprocessor.archive = os.path.join(
            "tests", "archive", "populated_archive"
        )

        # Delete the new CSV file
        csv_file = os.path.join(
            current_directory,
            "tests",
            "archive",
            "populated_archive",
            "preprocessed_data_7.csv",
        )

        if os.path.exists(csv_file):
            os.remove(csv_file)

        # Merge fake preprocessed dataset and store it into archive/populated_archive.
        self.preprocessor.merge_current_datasets(dataset)

        latest_file = max(
            (
                os.path.join(
                    current_directory,
                    "tests",
                    "archive",
                    "populated_archive",
                    file,
                )
                for file in os.listdir(
                    os.path.join(
                        current_directory,
                        "tests",
                        "archive",
                        "populated_archive",
                    )
                )
            ),
            key=os.path.getctime,
        )

        # check the stored dataset has the same name.
        self.assertEqual(latest_file, csv_file)

        expected_dataset = pd.concat([dataframe_2, dataframe_3]).reset_index(
            drop=True
        )

        # check the stored dataset is the same as the fake one.
        pd.testing.assert_frame_equal(
            pd.read_csv(csv_file, sep=";"), expected_dataset
        )

        # Delete the new CSV file
        os.remove(csv_file)

    def test_max_timestamp_not_new(self):
        """
        Test if the max timestamp was already found in previous files
        We expect preprocessed data to be always "new", thus in the future.
        If the max timestamp is "not new", flag an error "Data cannot be overwritten,
        no new timestamps are being added."
        """
        dataset = dataframe_2
        self.preprocessor.archive = os.path.join(
            "tests", "archive", "populated_archive"
        )

        with self.assertRaises(ValueError) as context:
            self.preprocessor.merge_current_datasets(dataset)

        # Optionally check the message of the exception
        self.assertEqual(
            str(context.exception),
            "Data cannot be overwritten, no new timestamps are being added.",
        )

    def test_structure_not_match(self):
        """
        Test if the structure of the preprocessed data does not match the one of previous
        shouldn't add anything and should flag an error "The structure of the preprocessed data does not match the
        previous data."
        :return:
        """
        dataset = dataframe_3
        dataset = dataset[["timestamp", "pipeline_id", "MEM", "CPU"]]
        self.preprocessor.archive = os.path.join(
            "tests", "archive", "populated_archive"
        )

        with self.assertRaises(ValueError) as context:
            self.preprocessor.merge_current_datasets(dataset)

        # Optionally check the message of the exception
        self.assertEqual(
            str(context.exception),
            "The structure of the preprocessed data does not match the previous data.",
        )

    def test_dtypes_not_match(self):
        """
        Test if the dtypes of the preprocessed data does not match the one of previous
        shouldn't add anything and should flag an error "The dtypes of the preprocessed data
        does not match the previous data."
        :return:
        """
        # Delete the new CSV file
        csv_file = os.path.join(
            os.getcwd(),
            "tests",
            "archive",
            "populated_archive",
            "preprocessed_data_7.csv",
        )
        if os.path.exists(csv_file):
            os.remove(csv_file)
        dataset = dataframe_3
        dataset = dataset.astype({"CPU": "str"})
        self.preprocessor.archive = os.path.join(
            "tests", "archive", "populated_archive"
        )

        with self.assertRaises(ValueError) as context:
            self.preprocessor.merge_current_datasets(dataset)

        # Optionally check the message of the exception
        self.assertEqual(
            str(context.exception),
            "The dtypes of the preprocessed data does not match the previous data.",
        )

    def test_purge_if_files_in_archive(self):
        """
        Files are in the archive_preprocessed directory.
        Files are deleted from the test directory.
        :return:
        """
        # Initialize the Preprocessing class
        self.preprocessor.directory = os.path.join("tests", "scraped_data")
        self.preprocessor.archive = os.path.join(
            "tests", "archive", "populated_archive"
        )

        dataset1 = dataframe_2
        dataset2 = dataframe_3
        dataset_preprocessed = dataframe_3

        # We save the files in case they are not there, so we can test.
        for x in [dataset1, dataset2]:
            x.to_csv(
                os.path.join(
                    self.preprocessor.directory,
                    f"raw_data_{max(x[self.preprocessor.key_values[0]])}.csv",
                ),
                index=False,
                sep=";",
            )

        timestamp_str = max(
            dataset_preprocessed[self.preprocessor.key_values[0]]
        )
        # filename is the output we would get from merge_current_datasets
        filename = f"preprocessed_data_{timestamp_str}.csv"
        dataset_preprocessed.to_csv(
            os.path.join(self.preprocessor.archive, filename),
            index=False,
            sep=";",
        )

        # purge the files should see that filename is in the archive and delete all the CSVs from self.directory.
        self.preprocessor.purge_old_files(filename, dataset_preprocessed)

        for data in [dataset1, dataset2]:
            file = os.path.join(
                self.preprocessor.directory,
                f"raw_data_{max(data[self.preprocessor.key_values[0]])}.csv",
            )
            self.assertFalse(os.path.exists(file))
            if os.path.exists(file):
                os.remove(file)

        # Check if the file still exists
        file = os.path.join(
            self.preprocessor.directory, "this_should_not_be_deleted.py"
        )
        with open(file, "w") as f:
            f.write("This file should not be deleted.")
        self.assertTrue(os.path.exists(file))
        if os.path.exists(file):
            os.remove(file)

        csv_file = os.path.join(
            os.getcwd(),
            "tests",
            "archive",
            "populated_archive",
            "preprocessed_data_7.csv",
        )
        if os.path.exists(csv_file):
            os.remove(csv_file)

    def test_purge_if_no_files_in_archive(self):
        """
        Files are not in the archive_preprocessed directory.
        Files are not deleted from the test directory.
        Raise an error. "The files have not been saved to the archive_preprocessed directory."
        :return:
        """
        # Initialize the Preprocessing class
        self.preprocessor.directory = os.path.join("tests", "scraped_data")
        self.preprocessor.archive = os.path.join(
            "tests", "archive", "empty_archive"
        )

        dataset1 = dataframe_2
        dataset2 = dataframe_3
        dataset_preprocessed = pd.concat(
            [dataframe_2, dataframe_3]
        ).reset_index(drop=True)

        # We save the files in case they are not there, so we can test.
        for x in [dataset1, dataset2]:
            x.to_csv(
                os.path.join(
                    self.preprocessor.directory,
                    f"raw_data_{max(x[self.preprocessor.key_values[0]])}.csv",
                ),
                index=False,
                sep=";",
            )

        with self.assertRaises(ValueError) as context:
            self.preprocessor.purge_old_files(
                "preprocessed_data_7.csv", dataset_preprocessed
            )

        # Optionally check the message of the exception
        self.assertEqual(
            str(context.exception),
            "The files have not been saved to the archive_preprocessed directory.",
        )
        # Check if the files still exists
        self.assertTrue(
            os.path.exists(
                os.path.join(self.preprocessor.directory, "raw_data_5.csv")
            )
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(self.preprocessor.directory, "raw_data_7.csv")
            )
        )
        # remove files:
        for data in [dataset1, dataset2]:
            file = os.path.join(
                self.preprocessor.directory,
                f"raw_data_{max(data[self.preprocessor.key_values[0]])}.csv",
            )
            if os.path.exists(file):
                os.remove(file)

    def test_purge_if_files_in_archive_but_not_fully(self):
        """
        Files are in the archive_preprocessed directory but not fully, data inside is missing.
        Compare the dataset output from the merge_current_datasets with the files in the archive_preprocessed directory.
        If the dataset is not in the archive_preprocessed directory, raise an error
        "There has been an error while saving the preprocessed dataset."
        :return:
        """
        # Initialize the Preprocessing class
        self.preprocessor.directory = os.path.join("tests", "scraped_data")
        self.preprocessor.archive = os.path.join(
            "tests", "archive", "populated_archive"
        )

        dataset1 = dataframe_2
        dataset2 = dataframe_3
        dataset_preprocessed = pd.concat(
            [dataframe_2, dataframe_3]
        ).reset_index(drop=True)

        # We save the files in case they are not there, so we can test.
        for x in [dataset1, dataset2]:
            x.to_csv(
                os.path.join(
                    self.preprocessor.directory,
                    f"raw_data_{max(x[self.preprocessor.key_values[0]])}.csv",
                ),
                index=False,
                sep=";",
            )

        timestamp_str = max(
            dataset_preprocessed[self.preprocessor.key_values[0]]
        )
        # filename is the output we would get from merge_current_datasets
        filename = f"preprocessed_data_{timestamp_str}.csv"
        dataset1.to_csv(
            os.path.join(self.preprocessor.archive, filename),
            index=False,
            sep=";",
        )

        with self.assertRaises(ValueError) as context:
            self.preprocessor.purge_old_files(filename, dataset_preprocessed)

        self.assertEqual(
            str(context.exception),
            "There has been an error while saving the preprocessed dataset, preprocessed data does not match archived data.",
        )

        # Check if the files still exists
        self.assertTrue(
            os.path.exists(
                os.path.join(self.preprocessor.directory, "raw_data_5.csv")
            )
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(self.preprocessor.directory, "raw_data_7.csv")
            )
        )
        # remove files:
        for data in [dataset1, dataset2]:
            file = os.path.join(
                self.preprocessor.directory,
                f"raw_data_{max(data[self.preprocessor.key_values[0]])}.csv",
            )
            if os.path.exists(file):
                os.remove(file)


class TestPreprocessingFiletype2(TestCase):

    def setUp(self):
        self.preprocessor = Preprocessing(**config["preprocessing_2"])

    def test_no_nulls(self):
        """Test that the preprocessing results in no null values in the dataframe."""
        # Execute the preprocess_files method
        result_df = self.preprocessor.preprocess_files()

        # Assertions to verify no nulls and correct data processing
        self.assertFalse(result_df.isnull().values.any())

    def test_assembly_of_dataframe(self):
        """Test that the preprocessing correctly assembles the dataframe."""
        # Execute the preprocess_files method
        result_df = self.preprocessor.preprocess_files()

        # Expected DataFrame
        expected_df = dataframe_1.astype(self.preprocessor.dtypes)

        # Check the result against the expected DataFrame
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_no_duplicates_in_dataframe(self):
        """Test that the preprocessing results in no duplicate entries in the dataframe."""
        # Method that should handle the preprocessing
        result_df = self.preprocessor.preprocess_files()

        # The expected number of unique timestamps should be 2 ('1' from df2 should overwrite '1' from df1)
        self.assertEqual(
            len(result_df),
            (result_df[self.preprocessor.key_values[0]].nunique()),
        )


if __name__ == "__main__":
    unittest.main()
