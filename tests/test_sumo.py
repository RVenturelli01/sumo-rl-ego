import os

import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("SUMO_RL_EGO_RUN_INTEGRATION") != "1",
    reason="Set SUMO_RL_EGO_RUN_INTEGRATION=1 to run SUMO-backed tests.",
)


def test_make_env_smoke():
    pytest.importorskip("traci")
    import sumo_gym_ego as sge

    env = sge.make_env("HighwayEgo-v0", seed=0, reward="fast", ego="discrete", use_gui=False)

    try:
        obs, _ = env.reset()
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        assert obs is not None
        assert next_obs is not None
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    finally:
        env.close()
