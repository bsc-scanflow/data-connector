from pydantic import BaseModel
from pydantic.types import ImportString
from typing import Optional, List, Dict, Any
from datetime import datetime

class Trigger(BaseModel):
    type: str = None
    #1.interval
    weeks: int = 0
    days: int = 0
    hours: int = 0
    minutes: int = 0
    seconds: int = 0
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    timezone: Optional[str] = None
    jitter: Optional[int] = None
    #2.date
    run_date: str = None
    # Duplicated attribute
    #timezone: str = None
    #3.crontab
    crontab: str = None

class Sensor(BaseModel):
    name: str
    trigger: Trigger = None
    args: tuple = None
    kwargs: Dict[str, Any] = None
    next_run_time: datetime = None

class SensorCallable(Sensor):
    func: ImportString

class SensorOutput(BaseModel):
    id: str 
    name: str
    func_name: str
    trigger_str: str
    next_run_time: datetime = None