import mlflow
from mlflow.tracking import MlflowClient
import click
import logging
import pandas as pd
import os

import sys
sys.path.insert(0, '/scanflow/scanflow')
from scanflow.client import ScanflowTrackerClient

@click.command(help="preprocessing")
# from main scanflow
@click.option("--name", default=None, type=str)
def main(name):
    logging.info("Workflow step: {}".format(name))
    
if __name__ == '__main__':
    main()