import traci
import time

from .step_display import TerminalDisplay


def play_policy(env, policy, manual=False, display=None, step_delay=None, max_episodes=None):
    """
    Run a policy in an environment with optional step display.

    Args:
        env: SumoEnv instance.
        policy: policy with .predict(obs) and .reset().
        manual: backward-compat shortcut — equivalent to display=TerminalDisplay(env, pause=True).
        display: a BaseStepDisplay instance for custom obs/action visualization.
        step_delay: seconds to sleep between steps (default: env.config.time_step).
        max_episodes: stop after this many episodes (None = run forever).
    """
    if manual and display is None:
        display = TerminalDisplay(env, pause=True)

    if step_delay is None:
        step_delay = 0.0 if manual else env.config.time_step

    episode_idx = 0

    while max_episodes is None or episode_idx < max_episodes:

        obs, _ = env.reset()
        policy.reset()

        traci.gui.setSchema("View #0", "real world")
        traci.gui.trackVehicle("View #0", "ego")
        traci.gui.setZoom("View #0", 1700)

        if display:
            display.on_episode_start(episode_idx)

        terminated = False
        truncated = False
        step = 0

        while not (terminated or truncated):
            action = policy.predict(obs)

            if display:
                display.on_step(obs, action, step)

            if not manual:
                time.sleep(step_delay)

            obs, reward, terminated, truncated, info = env.step(action)
            if display:
                display.on_reward(reward, info, step)
            step += 1

        print("Episode finished:", info.get("event"))

        if display:
            display.on_episode_end(info)
        elif manual and (max_episodes is None or episode_idx + 1 < max_episodes):
            input("Press Enter to reset...\n")

        episode_idx += 1
