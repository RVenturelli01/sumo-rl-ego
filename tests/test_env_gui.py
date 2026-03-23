import os

import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("SUMO_RL_EGO_RUN_GUI_TESTS") != "1",
    reason="Set SUMO_RL_EGO_RUN_GUI_TESTS=1 to run GUI tests.",
)


def test_make_env_gui_smoke():
    pytest.importorskip("traci")
    import sumo_rl_ego as sre

    env = sre.make_env("HighwayEgo-v0", seed=0, reward="fast", ego="discrete", use_gui=True)

    try:
        obs, _ = env.reset()
        assert obs is not None
    finally:
        env.close()
