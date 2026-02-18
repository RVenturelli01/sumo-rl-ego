# sumo-rl-ego
A reinforcement learning framework built on top of SUMO with a custom Gymnasium environment for training and evaluating single-agent ego vehicle control policies.


# High level structure

```
Agent (RL)
    ↓ action
SumoEnv
    ├── SumoSimulation   → handles TraCI / libsumo
    ├── EgoVehicle       → translates action into vehicle control
    ├── Observation      → builds RL observation
    ├── Reward           → computes reward
    └── Kpi              → kpi tracker
    ↓
Observation, Reward, Terminated, Truncated, info → returned to Agent
```

made by Riccardo Venturelli
