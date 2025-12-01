#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import logging

from preprocessing import Preprocessing

def log_env_variables():
    logging.info("Logging environment variables:")
    for key, value in os.environ.items():
        logging.info(f"{key}: {value}")
        print(f"{key}: {value}")  # Ensure printing as well

def main(config) -> str:
    log_env_variables()

    # Retrieve config file
    preprocessing_config = config.get("preprocessing")

    if preprocessing_config:

        print("Preprocessing new data.")
        # Instantiate the Preprocessing class
        prepro_obj = Preprocessing(**preprocessing_config)

        # Preprocess files.
        dataset = prepro_obj.preprocess_files()

        # Merge old preprocessed data with new preprocessed data.
        filename, updated_dataset = prepro_obj.merge_current_datasets(dataset)

        # Delete old files.
        prepro_obj.purge_old_files(filename, updated_dataset)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Process config file and input args."
    )
    # Train and verbose flags
    parser.add_argument(
        "--train",
        dest="Train",
        action="store_true",
        help="Enable training mode",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Enable verbose mode",
    )
    parser.add_argument(
        "--stress", dest="stress", action="store_true", help="Enable stress"
    )
    parser.add_argument(
        "--config", type=str, help="Config path", required=True
    )

    return parser.parse_args()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    args = parse_args()

    with open(args.config, "r") as file:
        config = json.load(file)
    main(config)
