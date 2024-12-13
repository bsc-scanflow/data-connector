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


def improved_migration_algorythm(latest_run: mlflow.entities.Run, kwargs)-> str:
    """
    :param args
    :param kwargs
    :return None
    """

    # Initialize NearbyActuator
    # - Initialize environment variables required for KratosClient
    os.environ["NBY_ENV_EMAIL"] = kwargs["nearbyone_env_email"]
    os.environ["NBY_ENV_PASSWORD"] = kwargs["nearbyone_env_password"]
    os.environ["NBY_ORGANIZATION_ID"] = kwargs["nearbyone_organization_id"]
    os.environ["NBY_ENV_NAME"] = kwargs["nearbyone_env_name"]

    # - Initialize a NearbyOneActuator
    nearby_actuator = NearbyOneActuator(
    )

    # Workaround - Hardcoded dict of available clusters
    nearby_actuator.update_available_sites(description="Worker cluster")
    
    # - TODO: use NearbyOneActuator for this
    sites_dict = {
        "eb0e3eaa-b668-4ad6-bc10-2bb0eb7da259": "edge",
        "fd7816db-7948-4602-af7a-1d51900792a7": "cloud"
    }

    # - For each registered cluster_id (params starting with "cluster_*") in Mlflow run --> latest_run.data.params:
    cluster_dict = { name:id for (name, id) in latest_run.data.params.items() if name.startswith("cluster_") }
    qos_dict = { name:value for (name, value) in latest_run.data.metrics.items() if name.startswith("qos_") }

    # Initialize a list of running services
    running_services = []

    for cluster_name, cluster_id in cluster_dict.items():
        # - Search the service with the expected application name
        source_site: Site = nearby_actuator.get_site(site_id=cluster_id)
        source_service_name: str = f"{kwargs['app_name']} - {source_site.display_name}"
        source_service: ServiceChainResponseServiceChain = nearby_actuator.get_service(
            site=source_site,
            service_name=source_service_name
        )
        # - If found:
        if source_service:
            logger.info(f"Source service {source_service.name} found!")
            # Initialize an object with all the required information
            # TODO: use NearbyOne API to properly detect the cluster type from Site information (name, description, etc...)
            service_dict = {
                "service": source_service,
                "site": source_site,
                "qos": qos_dict[f"qos_{cluster_name.split('_')[1]}"],
                "cluster_type": sites_dict[cluster_id] if cluster_id in sites_dict else None
            }
            # Append the object to the list
            running_services.append(service_dict)

    # Initialize a dictionary with migration results
    migration_results = {}

    # Go through all the currently running service and check if they have to be migrated
    for service in running_services:

        # Verify QoS migration rules
        qos = service["qos"]
        cluster_type = service["cluster_type"]
        source_service = service["service"]
        source_site = service["site"]

        if qos_check(qos=qos, cluster_type=cluster_type):
            logger.info(f"Service {source_service.name} running on {cluster_type} with Qos {qos} requires migration.")
            # Migrate the application
            # TODO: don't use the current NearbyOneActuator.migrate_service() method as it duplicates already done job, like looking for the source site and service name                
            migration_result = nearby_actuator.migrate_service(app_name=kwargs['app_name'], source_cluster_id=source_site.id)
        else:
            # - Skip this entry
            logger.info(f"Service {source_service.name} running on {cluster_type} with Qos {qos} doesn't require migration.")
            migration_result = f"Service {source_service.name} running on {cluster_type} with Qos {qos} doesn't require migration."

        migration_results[source_service.name] = migration_result
    
    # Return all migration results as a string
    return json.dumps(
            obj=migration_results,
            indent=2
        )


def initial_migration_algorythm(latest_run: mlflow.entities.Run, kwargs)-> str:
    """
    Algorythm that just analyses the max QoS value, regardless of whether the application is still running on that cluster or not
    :param latest_run
    :return migration results dictionary
    """

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

    return migration_result


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

            # - Initial reactive migration algorythm
            #migration_result = initial_migration_algorythm(latest_run, kwargs=kwargs)

            # - Improved reactive migration algorythm
            migration_result = improved_migration_algorythm(latest_run=latest_run, kwargs=kwargs)

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
