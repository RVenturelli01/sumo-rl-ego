import traci


class SumoSimulation:
    def __init__(self, config):
        self.config = config
        self.started = False
        self.off_road = False # flag to track if ego has gone off-road

    # -------------------------
    # Lifecycle
    # -------------------------
    def reset(self):
        if self.started:
            traci.load(self.config.build_cmd()[1:])
            self.off_road = False  # reset off-road flag on new episode
        else:
            traci.start(self.config.build_cmd())
            self.started = True
            self.off_road = False

    def close(self):
        if self.started:
            traci.close()
            self.started = False


    # -------------------------
    # Simulation step
    # -------------------------
    def step(self):
        traci.simulationStep()


    # -------------------------
    # Ego utilities
    # -------------------------
    def ego_exists(self, ego_id):
        return ego_id in traci.vehicle.getIDList()

    def wait_for_vehicle(self, ego_id, max_steps=10000):
        wait = 0
        while not self.ego_exists(ego_id):
            traci.simulationStep()
            wait += 1
            if wait > max_steps:
                raise RuntimeError("Ego vehicle was not spawned.")

    def enable_rl_control(self, ego_id):
        traci.vehicle.setSpeedMode(ego_id, 7)
        traci.vehicle.setLaneChangeMode(ego_id, 0)


    # -------------------------
    # Safe action
    # -------------------------
    def get_ego_status(self, ego_id):
        exists = self.ego_exists(ego_id)

        collided = ego_id in traci.simulation.getCollidingVehiclesIDList()
        teleported = ego_id in traci.simulation.getStartingTeleportIDList()
        arrived = ego_id in traci.simulation.getArrivedIDList()
        off_road = self.off_road or (exists and traci.vehicle.getLaneID(ego_id) == "")

        return {
            "exists": exists,
            "collided": collided,
            "teleported": teleported,
            "off_road": off_road,
            "arrived": arrived,
        }


    def getTimeStep(self):
        return self.config.time_step


    def setSpeed(self, ego_id, speed):
        traci.vehicle.setSpeed(ego_id, speed)


    def getSpeed(self, ego_id):
        return traci.vehicle.getSpeed(ego_id)


    def changeLane(self, ego_id, lane_id_new, duration):
        lane_id = traci.vehicle.getLaneID(ego_id)
        edge_id = traci.lane.getEdgeID(lane_id)
        n_lanes = traci.edge.getLaneNumber(edge_id)

        if 0 <= lane_id_new < n_lanes:
            traci.vehicle.changeLane(ego_id, lane_id_new, duration)
        else:
            self.off_road = True  # flag off-road if lane change would go out of bounds


    def getLaneIndex(self, ego_id):
        return traci.vehicle.getLaneIndex(ego_id)


    def get_position(self, ego_id):
        return traci.vehicle.getPosition(ego_id)


    def get_angle(self, ego_id):
        return traci.vehicle.getAngle(ego_id)


    def get_speed(self, ego_id):
        return traci.vehicle.getSpeed(ego_id)


    def get_lane_position(self, ego_id):
        return traci.vehicle.getLanePosition(ego_id)


    def get_lane_position_lat(self, ego_id):
        return traci.vehicle.getLateralLanePosition(ego_id)


    def get_lane_index(self, ego_id):
        return traci.vehicle.getLaneIndex(ego_id)


    def get_distance(self, ego_id):
        return traci.vehicle.getDistance(ego_id)