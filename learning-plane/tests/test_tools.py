# preprocessor_initializers.py
from __future__ import annotations

import json

import pandas as pd


def load_json_from_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


# Define common variables for tests classes
config = load_json_from_file("tests/fixtures/config_tests.json")
dataframe_1 = pd.read_csv("tests/fixtures/dataframe_1.csv", sep=",")
dataframe_2 = pd.read_csv("tests/fixtures/dataframe_2.csv", sep=",")
dataframe_3 = pd.read_csv("tests/fixtures/dataframe_3.csv", sep=",")
