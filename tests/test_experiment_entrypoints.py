import builtins

import pytest

pytest.importorskip("hydra")
pytest.importorskip("omegaconf")
pytest.importorskip("stable_baselines3")
pytest.importorskip("wandb")

from omegaconf import OmegaConf

from experiments import eval as eval_entrypoint
from experiments import play as play_entrypoint
from experiments import train as train_entrypoint


class FakeEnv:
    def close(self):
        pass


def test_play_entrypoint_uses_shared_loader_and_never_prompts(monkeypatch):
    cfg = OmegaConf.create(
        {
            "run": {"seed": 5},
            "wandb": {"enabled": False},
            "source": {"type": "policy_id", "policy_id": "safe_speed_v1", "model_path": None},
            "env": {"id": "HighwayEgo-v0", "kwargs": {}},
            "play": {"manual": False, "max_episodes": 2},
        }
    )
    calls = {}
    env = FakeEnv()

    def fail_input(*args, **kwargs):
        raise AssertionError("play entrypoint should not call input()")

    def fake_loader(loaded_cfg, use_gui):
        calls["loader"] = {"cfg": loaded_cfg, "use_gui": use_gui}
        return "policy", env

    def fake_play_policy(env, policy, seed, manual, max_episodes):
        calls["play_policy"] = {
            "env": env,
            "policy": policy,
            "seed": seed,
            "manual": manual,
            "max_episodes": max_episodes,
        }

    monkeypatch.setattr(builtins, "input", fail_input)
    monkeypatch.setattr(play_entrypoint, "print_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(play_entrypoint, "load_policy_and_env_from_cfg", fake_loader)
    monkeypatch.setattr(play_entrypoint.sre, "play_policy", fake_play_policy)

    play_entrypoint.main.__wrapped__(cfg)

    assert calls["loader"]["cfg"] is cfg
    assert calls["loader"]["use_gui"] is True
    assert calls["play_policy"] == {
        "env": env,
        "policy": "policy",
        "seed": 5,
        "manual": False,
        "max_episodes": 2,
    }


def test_eval_and_play_share_policy_env_loader(monkeypatch):
    cfg = OmegaConf.create(
        {
            "run": {"seed": 3},
            "wandb": {"enabled": False},
            "source": {"type": "policy_id", "policy_id": "safe_speed_v1", "model_path": None},
            "env": {"id": "HighwayEgo-v0", "kwargs": {}},
            "eval": {"n_episodes": 2, "max_steps": None},
            "play": {"manual": False, "max_episodes": 1},
        }
    )
    calls = []

    def fake_loader(loaded_cfg, use_gui):
        calls.append(use_gui)
        return "policy", FakeEnv()

    monkeypatch.setattr(eval_entrypoint, "print_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(play_entrypoint, "print_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_entrypoint, "load_policy_and_env_from_cfg", fake_loader)
    monkeypatch.setattr(play_entrypoint, "load_policy_and_env_from_cfg", fake_loader)
    monkeypatch.setattr(
        eval_entrypoint.sre,
        "evaluate_policy",
        lambda *args, **kwargs: type(
            "Result",
            (),
            {
                "return_mean": 1.0,
                "return_std": 0.0,
                "length_mean": 1.0,
                "length_std": 0.0,
                "event_counts": {"done": 2},
                "episodes": [],
            },
        )(),
    )
    monkeypatch.setattr(eval_entrypoint, "print_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_entrypoint, "log_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(play_entrypoint.sre, "play_policy", lambda *args, **kwargs: None)

    eval_entrypoint.main.__wrapped__(cfg)
    play_entrypoint.main.__wrapped__(cfg)

    assert calls == [False, True]


def test_train_entrypoint_uses_shared_training_adapter(monkeypatch):
    cfg = OmegaConf.create(
        {
            "run": {"seed": 9},
            "wandb": {"enabled": False},
            "source": {"type": "fresh", "model_path": None, "replay_buffer_path": None},
            "env": {"id": "HighwayEgo-v0", "n_envs": 2, "kwargs": {}},
            "model": {"algo": "DQN", "kwargs": {}, "overrides": {}},
            "learn": {"kwargs": {"total_timesteps": 10}},
            "save": {
                "config_path": "/tmp/config_resolved.yaml",
                "model_path": "/tmp/model.zip",
                "replay_buffer_path": None,
            },
        }
    )
    calls = {}

    class DummyModel:
        def learn(self, callback, **kwargs):
            calls["learn"] = {"callback_type": type(callback).__name__, "kwargs": kwargs}

    class DummyEnv:
        def close(self):
            calls["env_closed"] = True

    class DummyHydraRuntime:
        output_dir = "/tmp"

    class DummyHydraConfig:
        runtime = DummyHydraRuntime()

    monkeypatch.setattr(train_entrypoint, "print_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_entrypoint, "build_train_cfg", lambda raw_cfg: raw_cfg)
    monkeypatch.setattr(train_entrypoint, "build_train_env", lambda loaded_cfg: DummyEnv())
    monkeypatch.setattr(train_entrypoint, "build_train_model", lambda loaded_cfg, env: DummyModel())
    monkeypatch.setattr(
        train_entrypoint,
        "save_training_outputs",
        lambda loaded_cfg, model: calls.setdefault("saved", True),
    )
    monkeypatch.setattr(train_entrypoint.HydraConfig, "get", lambda: DummyHydraConfig())

    train_entrypoint.main.__wrapped__(cfg)

    assert calls["learn"]["callback_type"] == "SB3MetricsCallback"
    assert calls["learn"]["kwargs"] == {"total_timesteps": 10}
    assert calls["saved"] is True
    assert calls["env_closed"] is True
