import mlflow.entities
from custom_rules import *
from custom_actuators import *
from typing import List
import logging
import time
from scanflow.agent.sensors.sensor import sensor
import mlflow
import json


logger = logging.getLogger("custom_sensor")
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger.setLevel(logging.INFO)


def tock():
    print('Tock! The time is: %s' % time.strftime("'%Y-%m-%d %H:%M:%S'"))

def improved_migration_algorythm(args, kwargs)-> None:
    """
    :param args
    :param kwargs
    :return None
    """

    # Initialize NearbyActuator

    # Workaround - Hardcoded dict of available clusters
    # - TODO: use NearbyActuator for this
    sites_dict = {
        "edge-id": "edge",
        "cloud-id": "cloud"
    }

    # Filter QoS with currently running applications
    running_services = []
    # - For each registered cluster_id (params starting with "cluster_*") in Mlflow run --> latest_run.data.params:
    #   - Search the service that starts with the given application name (i.e. "DLStreamer Pipeline Server ....")
    #       - If found:
    #           - Initialize a new object: {"cluster_id": str, "service": ServiceChain or whatever, "qos": float, "cluster_type": str}
    #           - Save the Service (it might be deleted once the new one has been properly deployed) --> "service"
    #           - Store its QoS and cluster_id
    #               - Parse the "cluster_X" param name, get the index and use it to retrieve the "qos_X" metric from Mlflow --> latest_run.data.metrics["qos_X"] --> "qos"
    #           - Set 'edge/cloud' label based on the sites_dict --> "cluster_type"
    #           - Append the object to the running_services list
    #       - Else:
    #           - Skip this entry

    # Go through the previous list of running services and apply the migration rules

    pass

# TODO: parameterize the node names (use app_name for this and team_name for the run names)
@sensor(nodes=["cloudedge-migration-experiment-ci"])
async def reactive_watch_qos(runs: List[mlflow.entities.Run], args, kwargs):
    print(args)
    print(kwargs)

    # migration_result = "No migration"

    # Only take into account the latest run and only if parameter "analysed" is set to false
    if runs:
        latest_run = runs[0]
        if "analysed" not in latest_run.data.params:

            # Initialize an MlFlow RunStatus object just for convenience
            mlflow_run_status = mlflow.entities.RunStatus()
            # Add an availability/termination run timeout
            timeout = kwargs["timeout"] if "timeout" in kwargs else 60

            start_time = time.time()
            elapsed_time = time.time() - start_time

            # Wait for experiment parameters and metrics to be available
            logger.info(f"Waiting for run metrics and params to be available...")
            while "max_cluster" not in latest_run.data.params and elapsed_time < timeout:
                logger.debug(f"Run {latest_run.info.run_id} still hasn't logged the QoS values. Waiting...")
                time.sleep(1)
                elapsed_time = time.time() - start_time
                # Retrieve again the latest run info from backend
                latest_run = mlflow.get_run(run_id=latest_run.info.run_id)

            if elapsed_time > timeout:
                logger.info("Current run still hasn't finished. Can't retrieve QoS values, exiting...")
                return "No QoS values available"

            # Retrieve the available experiment metrics and parameters
            max_cluster = latest_run.data.params["max_cluster"]
            max_qos = latest_run.data.metrics["max_qos"]
            # Index not required as of now
            # max_idx = latest_run.data.metrics["max_idx"]

            # Check if there's a max_cluster ID value, or it is set to None
            if (max_cluster != "None" and max_qos >= 0) and qos_constraints(max_qos):

                # Proceed to migrate the application
                logger.info(
                    f"Maximum QoS value {max_qos} found in Cluster ID {max_cluster} "
                    f"is above SLA. Executing application migration..."
                )

                migration_result = migrate_application(
                    app_name=kwargs["app_name"],
                    current_cluster_id=max_cluster,
                    nearbyone_env_name=kwargs["nearbyone_env_name"],
                    nearbyone_org_id=kwargs["nearbyone_organization_id"],
                    nearbyone_email=kwargs["nearbyone_env_email"],
                    nearbyone_password=kwargs["nearbyone_env_password"]
                )
                # Workaround: Sensor return value is expected to be str
                migration_result = json.dumps(
                    obj=migration_result,
                    indent=2
                )
            elif max_qos < 0:
                migration_result = (
                    "No QoS values available! Either the application is not deployed"
                    "or there are no dlpipelines running"
                )
                logger.info(migration_result)

            else:
                # Nothing to do
                migration_result = "QoS below SLA. No migration required"
                logger.info(migration_result)

            # Mark the experiment as already analysed
            # TODO: check if set_experiment is enough to avoid active run vs environment run issues
            logger.info(f"Experiment id: {latest_run.info.experiment_id}")
            logger.info(f"Experiment run id: {latest_run.info.run_id}")

            # Set latest experiment as the active one
            mlflow.set_experiment(experiment_id=latest_run.info.experiment_id)

            logger.info(f"Run status: {latest_run.info.status}")

            start_time = time.time()
            elapsed_time = time.time() - start_time

            run_status = mlflow_run_status.from_string(latest_run.info.status)
            logger.info(f"Waiting for run {latest_run.info.run_id} to terminate...")
            while not mlflow_run_status.is_terminated(run_status) and (elapsed_time < timeout):
                logger.debug(f"Run status is {latest_run.info.status}, waiting until run is terminated...")
                time.sleep(1)
                elapsed_time = time.time() - start_time
                # Retrieve again the latest run info from backend
                latest_run = mlflow.get_run(run_id=latest_run.info.run_id)
                run_status = mlflow_run_status.from_string(latest_run.info.status)

            if elapsed_time > timeout:
                logger.info("Current run still hasn't finished. Can't log `analysed` parameter, exiting...")
                return "Latest run not finished"

            logger.info(f"Setting experiment run {latest_run.info.run_id} as analysed...")
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
            logger.info("Last run already analysed. Skipping...")
            migration_result = "All runs already analysed"
    else:
        logger.info("No available runs in experiment. Skipping...")
        migration_result = "Experiment doesn't have runs"

    # TODO: Modify Sensor class so it allows logging data types other than str
    return migration_result
