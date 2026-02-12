from env.sumo_env import SumoEnv
from configs.config import SumoConfig

config = SumoConfig(
    net_file="networks/highway_fast/highway.net.xml",
    route_file="networks/highway_fast/highway.rou.xml",
    ego_id="ego",
    use_gui=True,
    auto_start=False, 
)

env = SumoEnv(config)

obs, _ = env.reset()

for _ in range(2000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

env.close()
