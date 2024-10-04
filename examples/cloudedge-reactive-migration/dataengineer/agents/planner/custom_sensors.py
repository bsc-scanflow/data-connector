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
    
    migration_result = 0

    # Only take into account the latest run and only if parameter "analysed" is set to false
    if runs:
        if not "analysed" in runs[0].data.params:
            
            # Initialize an MLflow RunStatus object just for convenience
            mlflow_run_status = mlflow.entities.RunStatus()
            # Add an availability/termination run timeout
            timeout = kwargs["timeout"] if "timeout" in kwargs else 60
            
            start_time = time.time()
            elapsed_time = time.time() - start_time
            
            while "max_cluster" not in runs[0].data.params and elapsed_time < timeout:
                logging.info(f"Run {runs[0].info.run_id} still hasn't logged the QoS values. Waiting...")
                time.sleep(1)
                elapsed_time = time.time() - start_time

            if elapsed_time > timeout:
                logging.info("Current run still hasn't finished. Can't retrieve QoS values, exiting...")
                return migration_result
            
            # Retrieve the available experiment metrics and parameters
            max_cluster = runs[0].data.params["max_cluster"]
            max_qos = runs[0].data.metrics["max_qos"]
            # Index not required as of now
            max_idx = runs[0].data.metrics["max_idx"]

            # Check if there's a max_cluster ID value or it is set to None
            if max_cluster != "None" and qos_constraints(max_qos):

                # Proceed to migrate the application
                logging.info(f"Maximum QoS {max_qos} found in Cluster ID {max_cluster}. Waiting for migration decision...")
                # TODO: provide NearbyOne API url through kwargs
                migration_result = migrate_application(
                    app_name=kwargs["app_name"],
                    current_cluster_id=max_cluster,
                    nearbyone_url=kwargs["nearbyone_url"],
                    nearbyone_username=kwargs["nearbyone_username"],
                    nearbyone_password=kwargs["nearbyone_password"]
                )
                
            else:
                logging.info("QoS below SLA. No migration required.")
            
            # Mark the experiment as already analysed
            # TODO: check if set_experiment is enough to avoid active run vs environment run issues
            logging.info(f"Experiment id: {runs[0].info.experiment_id}")
            logging.info(f"Experiment run id: {runs[0].info.run_id}")
            #mlflow.set_experiment(runs[0].info.experiment_id)

            start_time = time.time()
            elapsed_time = time.time() - start_time
            logging.info(f"Run status type: {type(runs[0].info.status)}")
            while not mlflow_run_status.is_terminated(runs[0].info.status) and (elapsed_time < timeout):
                logging.info(f"Run status is {runs[0].info.status}, waiting until FINISHED")
                time.sleep(1)
                elapsed_time = time.time() - start_time

            if elapsed_time > timeout:
                logging.info("Current run still hasn't finished. Can't log `analysed` parameter, exiting...")
                return migration_result

            with mlflow.start_run(
                run_id=runs[0].info.run_id,
                experiment_id=runs[0].info.experiment_id
                ):
                mlflow.log_param(
                    key="analysed",
                    value="True"
                )
        else:
            logging.info("Last run already analysed. Skipping...")
    else:
        logging.info("No available runs in experiment. Skipping...")

    return migration_result