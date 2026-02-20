import traci


class DomainProxy:
    def __init__(self, traci_domain, overrides=None):
        self._traci = traci_domain
        self._overrides = overrides or {}

    def __getattr__(self, name):
        # override locale
        if name in self._overrides:
            return self._overrides[name]

        # fallback a traci
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

    # ---- safe speed ----
    def _set_speed_safe(self, veh_id, speed):
        if veh_id in traci.vehicle.getIDList():
            traci.vehicle.setSpeed(veh_id, speed)

    # ---- safe lane change ----
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
    # Helpers tuoi
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

        lane_id = None
        if exists:
            try:
                lane_id = traci.vehicle.getLaneID(ego_id)
            except traci.TraCIException:
                exists = False

        collided = ego_id in traci.simulation.getCollidingVehiclesIDList()
        teleported = ego_id in traci.simulation.getStartingTeleportIDList()
        arrived = ego_id in traci.simulation.getArrivedIDList()
        off_road = self.off_road or lane_id in ("", None)

        ego_removed_unknown = (
            not exists
            and not collided
            and not teleported
            and not arrived
        )

        return {
            "exists": exists,
            "collided": collided,
            "teleported": teleported,
            "arrived": arrived,
            "off_road": off_road,
            "removed_unknown": ego_removed_unknown,
            }

    def enable_rl_control(self, ego_id):
        traci.vehicle.setSpeedMode(ego_id, 0)
        traci.vehicle.setLaneChangeMode(ego_id, 0)
        

    def ego_exists(self, ego_id):
        return ego_id in traci.vehicle.getIDList()
