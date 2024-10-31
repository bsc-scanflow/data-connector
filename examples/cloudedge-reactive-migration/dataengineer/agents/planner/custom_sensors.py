import mlflow.entities
from .custom_rules import *
from .custom_actuators import *
from typing import List, Optional
import logging
logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

import time

from scanflow.agent.sensors.sensor import sensor
import mlflow

def tock():
    print('Tock! The time is: %s' %  time.strftime("'%Y-%m-%d %H:%M:%S'"))

#example 1: count number of predictions in last 5 min
# TODO: parameterize the node names (use app_name for this and team_name for the run names)
@sensor(nodes=["cloudedge-migration-experiment-ci"])
async def reactive_watch_qos(runs: List[mlflow.entities.Run], args, kwargs):
    print(args)
    print(kwargs)
    
    migration_result = "No migration"

    # Only take into account the latest run and only if parameter "analysed" is set to false
    if runs:
        latest_run = runs[0]
        if not "analysed" in latest_run.data.params:
            
            # Initialize an MLflow RunStatus object just for convenience
            mlflow_run_status = mlflow.entities.RunStatus()
            # Add an availability/termination run timeout
            timeout = kwargs["timeout"] if "timeout" in kwargs else 60
            
            start_time = time.time()
            elapsed_time = time.time() - start_time
            
            # Wait for experiment parameters and metrics to be available
            logging.info(f"Waiting for run metrics and params to be available...")
            while "max_cluster" not in latest_run.data.params and elapsed_time < timeout:
                logging.debug(f"Run {latest_run.info.run_id} still hasn't logged the QoS values. Waiting...")
                time.sleep(1)
                elapsed_time = time.time() - start_time
                # Retrieve again the latest run info from backend
                latest_run = mlflow.get_run(run_id=latest_run.info.run_id)

            if elapsed_time > timeout:
                logging.info("Current run still hasn't finished. Can't retrieve QoS values, exiting...")
                return "No QoS values available"
            
            # Retrieve the available experiment metrics and parameters
            max_cluster = latest_run.data.params["max_cluster"]
            max_qos = latest_run.data.metrics["max_qos"]
            # Index not required as of now
            max_idx = latest_run.data.metrics["max_idx"]

            # Check if there's a max_cluster ID value, or it is set to None
            if (max_cluster != "None" and max_qos >= 0) and qos_constraints(max_qos):

                # Proceed to migrate the application
                logging.info(f"Maximum QoS value {max_qos} found in Cluster ID {max_cluster} is above SLA. Executing application migration...")

                migration_result = migrate_application(
                    app_name=kwargs["app_name"],
                    current_cluster_id=max_cluster,
                    nearbyone_env_name=kwargs["nearbyone_env_name"],
                    nearbyone_org_id=kwargs["nearbyone_organization_id"],
                    nearbyone_username=kwargs["nearbyone_env_username"],
                    nearbyone_password=kwargs["nearbyone_env_password"]
                )
                # Workaround: Sensor return value is expected to be str
                migration_result = json.dumps(
                    obj=migration_result,
                    indent=2
                )
            elif max_qos < 0:
                logging.info("No QoS values available! Either the application is not deployed or there are no dlpipelines running")
                migration_result = "No QoS values available! Either the application is not deployed or there are no dlpipelines running"
            else:
                # Nothing to do
                logging.info("QoS below SLA. No migration required.")
                migration_result = "QoS below SLA. No migration required"
            
            # Mark the experiment as already analysed
            # TODO: check if set_experiment is enough to avoid active run vs environment run issues
            logging.info(f"Experiment id: {latest_run.info.experiment_id}")
            logging.info(f"Experiment run id: {latest_run.info.run_id}")
            
            # Set latest experiment as the active one
            mlflow.set_experiment(experiment_id=latest_run.info.experiment_id)

            logging.info(f"Run status: {latest_run.info.status}")

            start_time = time.time()
            elapsed_time = time.time() - start_time

            run_status = mlflow_run_status.from_string(latest_run.info.status)
            logging.info(f"Waiting for run {latest_run.info.run_id} to terminate...")
            while not mlflow_run_status.is_terminated(run_status) and (elapsed_time < timeout):
                logging.debug(f"Run status is {latest_run.info.status}, waiting until run is terminated...")
                time.sleep(1)
                elapsed_time = time.time() - start_time
                # Retrieve again the latest run info from backend
                latest_run = mlflow.get_run(run_id=latest_run.info.run_id)
                run_status = mlflow_run_status.from_string(latest_run.info.status)

            if elapsed_time > timeout:
                logging.info("Current run still hasn't finished. Can't log `analysed` parameter, exiting...")
                return "Latest run not finished"

            logging.info(f"Setting experiment run {latest_run.info.run_id} as analysed...")
            with mlflow.start_run(
                run_id=latest_run.info.run_id,
                experiment_id=latest_run.info.experiment_id
                ):
                mlflow.log_param(
                    key="analysed",
                    value="True"
                )
        else:
            # No new experiment runs available to analyse
            logging.info("Last run already analysed. Skipping...")
            migration_result = "All runs already analysed"
    else:
        logging.info("No available runs in experiment. Skipping...")
        migration_result = "Experiment doesn't have runs"

    # TODO: Modify Sensor class so it allows logging data types other than str
    return migration_result
