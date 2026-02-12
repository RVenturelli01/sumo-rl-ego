import traci
import subprocess
import os


class SumoSimulation:
    def __init__(self, sumo_cmd):
        self.sumo_cmd = sumo_cmd
        self.started = False

    def start(self):
        if not self.started:
            traci.start(self.sumo_cmd)
            self.started = True

    def step(self):
        traci.simulationStep()

    def close(self):
        if self.started:
            traci.close()
            self.started = False

    def get_vehicle_state(self, veh_id):
        speed = traci.vehicle.getSpeed(veh_id)

        leader = traci.vehicle.getLeader(veh_id)
        if leader is None:
            distance = 100.0
        else:
            distance = leader[1]

        return {
            "speed": speed,
            "distance": distance
        }

    def apply_acceleration(self, veh_id, accel):
        current_speed = traci.vehicle.getSpeed(veh_id)
        new_speed = max(0.0, current_speed + accel)
        traci.vehicle.setSpeed(veh_id, new_speed)

    def has_collision(self):
        return traci.simulation.getCollidingVehiclesNumber() > 0
