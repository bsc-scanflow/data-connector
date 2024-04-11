from .custom_rules import *
from .custom_actuators import *
import numpy as np
from typing import List, Optional
import logging
logging.basicConfig(format='%(asctime)s -  %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

#fastapi
from fastapi import FastAPI, APIRouter, Depends
from fastapi import Response, status, HTTPException

from datetime import datetime
import time

from scanflow.agent.sensors.sensor import sensor

def tock():
    print('Tock! The time is: %s' %  time.strftime("'%Y-%m-%d %H:%M:%S'"))

#example 1: count number of predictions in last 5 min
@sensor(nodes=["predictor"])
async def watch_qos(runs: List[mlflow.entities.Run], args, kwargs):
    print(args)
    print(kwargs)
    
    max_qos = runs[0].data.metrics['max_qos']
    max_qos_index = runs[0].data.parameters['max_qos_index']
    
    logging.info("max_qos {}, index {}".format(max_qos, max_qos_index))

    if qos_constraints(max_qos):
       await call_migrate_app(max_qos_index)

    return max_qos 