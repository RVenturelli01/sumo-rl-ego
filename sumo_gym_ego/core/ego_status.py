from enum import Enum

class EgoStatus(Enum):
    RUNNING = "running"
    ARRIVED = "arrived"
    COLLIDED = "collided"
    OFF_ROAD = "offroad"
    TELEPORTED = "teleported"
    REMOVED_UNKNOWN = "removed_unknown"
    TIMEOUT = "timeout"