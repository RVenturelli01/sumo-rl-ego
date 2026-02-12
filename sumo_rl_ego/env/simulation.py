import traci

class SumoSimulation:

    def __init__(self, config):
        self.config = config
        self.started = False

    def start(self):
        traci.start(self.config.build_cmd())
        self.started = True

    def step(self):
        traci.simulationStep()

    def close(self):
        if self.started:
            traci.close()
            self.started = False

    # --- Vehicle API ---
    def get_speed(self, veh_id):
        return traci.vehicle.getSpeed(veh_id)

    def set_acceleration(self, veh_id, accel):
        traci.vehicle.setAcceleration(veh_id, accel, 1)

    def get_leader(self, veh_id):
        return traci.vehicle.getLeader(veh_id)

    def check_collision(self, veh_id):
        collisions = traci.simulation.getCollidingVehiclesIDList()
        return veh_id in collisions
