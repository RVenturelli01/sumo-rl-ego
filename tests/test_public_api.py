import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

import sumo_gym_ego as sge
import sumo_rl_ego as sre
from sumo_gym_ego.sumo_envs import factory as env_factory
from sumo_gym_ego.sumo_envs.registry import ENV_REGISTRY
from sumo_rl_ego.workflows import evaluate_policy as evaluate_policy_from_workflows


class DummyEnv:
    def __init__(self, seed, **kwargs):
        self.seed = seed
        self.kwargs = kwargs


class DummyVecEnv:
    def __init__(self, env_fns):
        self.envs = [env_fn() for env_fn in env_fns]


def test_package_exports():
    assert sge.SumoEnv.__name__ == "SumoEnv"

    expected_sge = {"make_env", "make_vec_env"}
    assert expected_sge.issubset(set(sge.__all__))

    expected_sre = {"Policy", "load_policy", "list_policies", "run_episode", "play_policy"}
    assert expected_sre.issubset(set(sre.__all__))


def test_policy_subclass():
    class ConstantPolicy(sre.Policy):
        def predict(self, obs):
            return 3

    policy = ConstantPolicy()
    assert policy.predict(None) == 3


def test_policy_from_model_with_predict():
    class Model:
        def predict(self, obs, deterministic=True):
            return 2, {"deterministic": deterministic}

    policy = sre.policy_from_model(Model())
    assert policy.predict("obs") == 2


def test_policy_from_model_with_callable():
    policy = sre.policy_from_model(lambda obs: 1)
    assert policy.predict("obs") == 1


def test_list_policies_contains_builtin():
    assert "safe_speed_v1" in sre.list_policies()


def test_make_env(monkeypatch):
    monkeypatch.setitem(ENV_REGISTRY, "DummyEnv-v0", DummyEnv)

    env = sge.make_env("DummyEnv-v0", seed=7, reward="fast")

    assert env.seed == 7
    assert env.kwargs["reward"] == "fast"


def test_make_vec_env(monkeypatch):
    monkeypatch.setitem(ENV_REGISTRY, "DummyVecEnv-v0", DummyEnv)
    monkeypatch.setattr(env_factory, "SubprocVecEnv", DummyVecEnv)

    env = sge.make_vec_env("DummyVecEnv-v0", n_envs=3, base_seed=10, reward="fast")

    assert [subenv.seed for subenv in env.envs] == [10, 11, 12]
    assert all(subenv.kwargs["reward"] == "fast" for subenv in env.envs)


def test_list_envs_is_sorted(monkeypatch):
    monkeypatch.setitem(ENV_REGISTRY, "zzz-env", DummyEnv)
    monkeypatch.setitem(ENV_REGISTRY, "aaa-env", DummyEnv)

    names = sre.list_envs()
    assert names == sorted(names)


def test_named_internal_packages_are_available():
    assert evaluate_policy_from_workflows is sre.evaluate_policy
