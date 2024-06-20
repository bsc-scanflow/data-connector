#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from train import Training

def main(config) -> str:

    training_config = config.get("training")

    if training_config:
        print("Training of models starting:")

        # Instantiate the Training class
        Training(**training_config)

        print("Models trained and stored.")


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
    args = parse_args()

    with open(args.config, "r") as file:
        config = json.load(file)
    main(config)
