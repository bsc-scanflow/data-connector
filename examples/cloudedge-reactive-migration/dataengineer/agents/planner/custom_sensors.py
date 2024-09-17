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
@sensor(nodes=["predictor-reactive"])
async def watch_qos(runs: List[mlflow.entities.Run], args, kwargs):
    print(args)
    print(kwargs)
    
    avg_qos = 0

    if runs:
        avg_qos = runs[0].data.metrics['avg_qos']
        avg_qos_index = runs[0].data.params['avg_qos_index']
        cluster_id = runs[0].data.metrics['cluster_id']
        
        logging.info("avg_qos {}, index {}, cluster_id {}".format(avg_qos, avg_qos_index, cluster_id))

        if qos_constraints(avg_qos):
            # Proceed to create a NearbyOneActuator object for the migration
            pass
        else:
            logging.info("QoS below SLA. No migration required.")
    else:
        logging.info("no data in last check")

    return avg_qos