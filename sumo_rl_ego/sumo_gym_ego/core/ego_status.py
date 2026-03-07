from enum import Enum

class EgoStatus(Enum):
    RUNNING = "running"
    ARRIVED = "arrived"
    COLLIDED = "collided"
    OFF_ROAD = "off_road"
    TELEPORTED = "teleported"
    REMOVED_UNKNOWN = "removed_unknown"