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
@sensor(nodes=["cloudskin-migration-experiment-ci"])
async def reactive_watch_qos(runs: List[mlflow.entities.Run], args, kwargs):
    print(args)
    print(kwargs)
    
    migration_result = {}

    # Only take into account the latest run and only if parameter "analysed" is set to false
    if runs:
        if not "analysed" in runs[0].data.params:
            # Retrieve the available experiment metrics and parameters
            max_cluster = runs[0].data.params["max_cluster"]
            # Check if there's a max_cluster ID value or it is set to None
            max_qos = runs[0].data.metrics["max_qos"]
            # Index not required as of now
            max_idx = runs[0].data.metrics["max_idx"]

            if max_cluster != "None" and qos_constraints(max_qos):

                # Proceed to create a NearbyOneActuator object for the migration
                logging.info(f"Maximum QoS {max_qos} found in Cluster ID {max_cluster}. Waiting for migration decision...")
                # TODO: provide NearbyOne API url through kwargs
                migration_result = migrate_application(
                    app_name=kwargs["app_name"],
                    current_cluster_id=max_cluster,
                    nearbyone_url=kwargs["nearbyone_url"],
                    nearbyone_username=kwargs["nearbyone_username"],
                    nearbyone_password=kwargs["nearbyone_password"]
                )
                pass
            else:
                logging.info("QoS below SLA. No migration required.")
            
            # Mark the experiment as already analysed
            with mlflow.start_run(run_id=runs[0].info.run_id):
                mlflow.log_param(
                    key="analysed",
                    value="True"
                )
        else:
            logging.info("Last run already analysed. Skipping...")
    else:
        logging.info("No available runs in experiment. Skipping...")

    return migration_result