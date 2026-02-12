import traci
import numpy as np


class SumoSimulation:

    def __init__(self, config):
        self.config = config
        self.started = False
        self.off_road = False

        self.simulation_end = getattr(config, "simulation_end", 1000)
        self.time_step = getattr(config, "time_step", 0.1)

    # ======================
    # Lifecycle
    # ======================

    def start(self):
        traci.start(self.config.build_cmd())
        self.started = True
        self.off_road = False


    def step(self):
        traci.simulationStep()

        if "ego" not in traci.vehicle.getIDList():
            self.off_road = True
            print("Ego vehicle is off-road (not in simulation anymore).")


    def close(self):
        if self.started:
            traci.close()
            self.started = False

    def ego_exists(self, ego_id: str) -> bool:
        return ego_id in traci.vehicle.getIDList()

    # ======================
    # --- Vehicle State API
    # ======================

    def get_position(self, veh_id):
        return traci.vehicle.getPosition(veh_id)

    def get_angle(self, veh_id):
        return traci.vehicle.getAngle(veh_id)

    def get_speed(self, veh_id):
        return traci.vehicle.getSpeed(veh_id)

    def get_acceleration(self, veh_id):
        return traci.vehicle.getAcceleration(veh_id)

    def get_lane_position(self, veh_id):
        return traci.vehicle.getLanePosition(veh_id)

    def get_lane_position_lat(self, veh_id):
        return traci.vehicle.getLateralLanePosition(veh_id)

    def get_lane_index(self, veh_id):
        return traci.vehicle.getLaneIndex(veh_id)

    def get_lane_id(self, veh_id):
        return traci.vehicle.getLaneID(veh_id)

    def get_distance(self, veh_id):
        return traci.vehicle.getDistance(veh_id)

    def get_leader(self, veh_id):
        return traci.vehicle.getLeader(veh_id)

    def get_follower(self, veh_id):
        return traci.vehicle.getFollower(veh_id)

    # ======================
    # --- Lane Info
    # ======================

    def get_n_lanes(self, veh_id):
        lane_id = self.get_lane_id(veh_id)
        edge_id = traci.lane.getEdgeID(lane_id)
        return traci.edge.getLaneNumber(edge_id)

    def get_lane_max_speed(self, veh_id):
        lane_id = self.get_lane_id(veh_id)
        return traci.lane.getMaxSpeed(lane_id)

    # ======================
    # --- Control API
    # ======================

    def set_speed(self, veh_id, speed):
        traci.vehicle.setSpeed(veh_id, speed)

    def set_acceleration(self, veh_id, accel, duration=1):
        traci.vehicle.setAcceleration(veh_id, accel, duration)

    def apply_deceleration(self, veh_id, decel, dt):
        decel = np.clip(decel, None, 0)
        current_speed = self.get_speed(veh_id)
        new_speed = max(0, current_speed + decel * dt)
        self.set_speed(veh_id, new_speed)

    def apply_acceleration(self, veh_id, accel, dt):
        accel = np.clip(accel, 0, None)
        current_speed = self.get_speed(veh_id)
        max_speed = self.get_lane_max_speed(veh_id)
        new_speed = min(max(0, current_speed + accel * dt), max_speed)
        self.set_speed(veh_id, new_speed)

    # ======================
    # --- Lane Change
    # ======================

    def change_lane_left(self, veh_id, duration=10):
        lane = self.get_lane_index(veh_id)
        n_lanes = self.get_n_lanes(veh_id)

        if lane < n_lanes - 1:
            traci.vehicle.changeLane(veh_id, lane + 1, duration)
        else:
            self.off_road = True

    def change_lane_right(self, veh_id, duration=10):
        lane = self.get_lane_index(veh_id)

        if lane > 0:
            traci.vehicle.changeLane(veh_id, lane - 1, duration)
        else:
            self.off_road = True

    def set_off_road(self, veh_id=None):
            """
            Mark the simulation as off-road event.
            veh_id is optional and kept for interface consistency.
            """
            self.off_road = True

    # ======================
    # --- Safety / Events
    # ======================

    def has_crashed(self, veh_id):
        collisions = traci.simulation.getCollidingVehiclesIDList()
        return veh_id in collisions

    def is_off_road(self):
        return self.off_road

    def get_time(self):
        return traci.simulation.getTime()

    def get_vehicles_ids(self):
        return traci.vehicle.getIDList()
