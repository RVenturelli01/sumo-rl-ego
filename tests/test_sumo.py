import traci

SUMO_CMD = [
    "sumo",
    "-c", "scenarios/+_cross/cross.sumocfg"
]

traci.start(SUMO_CMD)

for step in range(20):
    traci.simulationStep()
    print("Step:", step)

traci.close()
