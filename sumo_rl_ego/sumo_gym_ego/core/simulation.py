import traci

from sumo_rl_ego.sumo_gym_ego.core.ego_status import EgoStatus


class DomainProxy:
    def __init__(self, traci_domain, overrides=None):
        self._traci = traci_domain
        self._overrides = overrides or {}

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        attr = getattr(self._traci, name)
        if callable(attr):
            return attr
        return attr


class SumoSimulation:
    def __init__(self, config):
        self.config = config
        self.started = False
        self.off_road = False

        # ===============================
        # OVERRIDES
        # ===============================
        vehicle_overrides = {
            "changeLane": self._change_lane_safe,
            "setSpeed": self._set_speed_safe,
        }

        # ===============================
        # DOMAINS
        # ===============================
        self.vehicle = DomainProxy(traci.vehicle, vehicle_overrides)
        self.lane = DomainProxy(traci.lane)
        self.edge = DomainProxy(traci.edge)
        self.simulation = DomainProxy(traci.simulation)

    # =========================================================
    # Lifecycle
    # =========================================================
    def reset(self):
        cmd = self.config.build_cmd()

        if self.started:
            traci.load(cmd[1:])
        else:
            traci.start(cmd)
            self.started = True

        self.off_road = False

    def close(self):
        if self.started:
            traci.close()
            self.started = False

    def simulationStep(self):
        traci.simulationStep()

    # =========================================================
    # OVERRIDES
    # =========================================================

    def _set_speed_safe(self, veh_id, speed):
        if veh_id in traci.vehicle.getIDList():
            traci.vehicle.setSpeed(veh_id, speed)

    def _change_lane_safe(self, veh_id, lane_index_new, duration):
        if veh_id not in traci.vehicle.getIDList():
            return

        try:
            lane_id = traci.vehicle.getLaneID(veh_id)
            edge_id = traci.lane.getEdgeID(lane_id)
            n_lanes = traci.edge.getLaneNumber(edge_id)

            if 0 <= lane_index_new < n_lanes:
                traci.vehicle.changeLane(veh_id, lane_index_new, duration)
            else:
                self.off_road = True

        except traci.TraCIException:
            self.off_road = True

    # =========================================================
    # Helpers
    # =========================================================
    def wait_for_vehicle(self, ego_id, max_steps=10000):
        steps = 0
        while not self.ego_exists(ego_id):
            self.simulationStep()
            steps += 1
            if steps > max_steps:
                raise RuntimeError("Ego vehicle was not spawned.")


    def get_ego_status(self, ego_id):
        exists = self.ego_exists(ego_id)

        arrived = ego_id in traci.simulation.getArrivedIDList()

        collided = ego_id in traci.simulation.getCollidingVehiclesIDList()

        teleported = ego_id in traci.simulation.getStartingTeleportIDList()

        lane_id = traci.vehicle.getLaneID(ego_id)
        off_road = self.off_road or lane_id in ("", None)

        # assign status given the following priority
        if arrived:
            ego_status = EgoStatus.ARRIVED

        elif collided:
            ego_status = EgoStatus.COLLIDED

        elif off_road:
            ego_status = EgoStatus.OFF_ROAD

        elif teleported:
            ego_status = EgoStatus.TELEPORTED

        elif not exists:
            ego_status = EgoStatus.REMOVED_UNKNOWN

        else:
            ego_status = EgoStatus.RUNNING

        return ego_status


    def enable_rl_control(self, ego_id):
        traci.vehicle.setSpeedMode(ego_id, self.config.speed_mode)
        traci.vehicle.setLaneChangeMode(ego_id, self.config.lane_change_mode)
        traci.vehicle.setRoutingMode(ego_id, self.config.routing_mode)


    def ego_exists(self, ego_id):
        return ego_id in traci.vehicle.getIDList()
