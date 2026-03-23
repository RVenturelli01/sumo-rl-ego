import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

import sumo_rl_ego as sre


class ConstantPolicy(sre.Policy):
    def predict(self, obs):
        return 0


class FakeEnv:
    def __init__(self):
        self.current_step = 0
        self.reset_seeds = []

    def reset(self, seed=None):
        self.current_step = 0
        self.reset_seeds.append(seed)
        return 0, {}

    def step(self, action):
        self.current_step += 1
        terminated = self.current_step >= 3
        info = {
            "event": "done" if terminated else None,
            "metrics": {},
        }

        if terminated:
            info["metrics"]["episode"] = {
                "ep_length": self.current_step,
                "ep_avg_speed": 12.5,
            }
            info["log"] = {"episode/return": 3.0}

        return 0, 1.0, terminated, False, info


def test_run_episode():
    result = sre.run_episode(FakeEnv(), ConstantPolicy(), seed=5)

    assert result.return_ == 3.0
    assert result.length == 3
    assert result.event == "done"
    assert result.episode_metrics["ep_length"] == 3


def test_rollout_uses_incremented_seeds():
    env = FakeEnv()

    results = sre.rollout(env, ConstantPolicy(), n_episodes=3, seed=10)

    assert len(results) == 3
    assert env.reset_seeds == [10, 11, 12]


def test_evaluate_policy():
    result = sre.evaluate_policy(FakeEnv(), ConstantPolicy(), n_episodes=2, seed=0)

    assert result.return_mean == 3.0
    assert result.return_std == 0.0
    assert result.length_mean == 3
    assert result.event_counts == {"done": 2}


def test_play_policy_auto():
    results = sre.play_policy(FakeEnv(), ConstantPolicy(), seed=0, manual=False, max_episodes=1)

    assert len(results) == 1
    assert results[0].event == "done"
