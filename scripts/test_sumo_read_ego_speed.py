import traci

SUMO_CMD = [
    "sumo-gui",
    "-c", "networks/+_cross/cross.sumocfg"
]

traci.start(SUMO_CMD)

for step in range(200):
    traci.simulationStep()

    print(traci.vehicle.getIDList())

    vehicles = traci.vehicle.getIDList()
    
    if "ego" in vehicles:
        speed = traci.vehicle.getSpeed("ego")
        print(f"Step {step} - Speed: {speed}")
    else:
        print("Ego not present")

traci.close()
