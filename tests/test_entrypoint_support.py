from pathlib import Path

import pytest

pytest.importorskip("hydra")
pytest.importorskip("omegaconf")
pytest.importorskip("stable_baselines3")

from omegaconf import OmegaConf

from experiments import entrypoint_support as support


class DummyModel:
    def __init__(self):
        self.loaded_replay_buffer = None
        self.saved_replay_buffer = None
        self.learning_rate = None
        self.lr_schedule_reset = False
        self.saved_model_path = None

    def load_replay_buffer(self, path):
        self.loaded_replay_buffer = path

    def save_replay_buffer(self, path):
        self.saved_replay_buffer = path

    def save(self, path):
        self.saved_model_path = path

    def _setup_lr_schedule(self):
        self.lr_schedule_reset = True


class DummyAlgo:
    last_init = None
    last_load = None

    def __init__(self, env=None, **kwargs):
        DummyAlgo.last_init = {"env": env, "kwargs": kwargs}

    @classmethod
    def load(cls, path, env=None):
        cls.last_load = {"path": path, "env": env}
        return DummyModel()


def make_cfg(data):
    return OmegaConf.create(data)


def test_load_saved_run_config_prefers_resolved_config(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model_path = run_dir / "model.zip"
    model_path.write_text("model")
    (run_dir / "config_resolved.yaml").write_text(
        "model:\n"
        "  algo: DQN\n"
        "env:\n"
        "  id: DemoEnv-v0\n"
        "  n_envs: 2\n"
        "  kwargs:\n"
        "    reward: fast\n"
    )

    cfg = support.load_saved_run_config(model_path)

    assert cfg.model.algo == "DQN"
    assert cfg.env.id == "DemoEnv-v0"


def test_build_train_cfg_uses_saved_env_and_algo(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model_path = run_dir / "model.zip"
    model_path.write_text("model")
    (run_dir / "config_resolved.yaml").write_text(
        "model:\n"
        "  algo: PPO\n"
        "env:\n"
        "  id: SavedEnv-v0\n"
        "  n_envs: 4\n"
        "  kwargs:\n"
        "    reward: saved\n"
    )
    cfg = make_cfg(
        {
            "source": {"type": "model", "model_path": str(model_path), "replay_buffer_path": None},
            "model": {"algo": "DQN", "kwargs": {}, "overrides": {}},
            "env": {"id": "IgnoredEnv-v0", "n_envs": 1, "kwargs": {}},
        }
    )

    effective_cfg = support.build_train_cfg(cfg)

    assert effective_cfg.env.id == "SavedEnv-v0"
    assert effective_cfg.env.n_envs == 4
    assert effective_cfg.model.algo == "PPO"


def test_build_train_env_uses_shared_env_mapping(monkeypatch):
    calls = {}
    cfg = make_cfg(
        {
            "run": {"seed": 7},
            "source": {"type": "fresh"},
            "env": {"id": "TrainEnv-v0", "n_envs": 3, "kwargs": {"reward": "fast"}},
        }
    )

    def fake_make_vec_env(env_id, n_envs, base_seed, **kwargs):
        calls.update(
            {"env_id": env_id, "n_envs": n_envs, "base_seed": base_seed, "kwargs": kwargs}
        )
        return "vec-env"

    monkeypatch.setattr(support.sre, "make_vec_env", fake_make_vec_env)

    env = support.build_train_env(cfg)

    assert env == "vec-env"
    assert calls == {
        "env_id": "TrainEnv-v0",
        "n_envs": 3,
        "base_seed": 7,
        "kwargs": {"reward": "fast", "use_gui": False},
    }


def test_load_policy_and_env_from_cfg_supports_model_source(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model_path = run_dir / "model.zip"
    model_path.write_text("model")
    (run_dir / "config_resolved.yaml").write_text(
        "model:\n"
        "  algo: DUMMY\n"
        "env:\n"
        "  id: SavedEnv-v0\n"
        "  kwargs:\n"
        "    reward: saved\n"
    )
    cfg = make_cfg(
        {
            "run": {"seed": 11},
            "source": {"type": "model", "model_path": str(model_path), "policy_id": None},
            "env": {"id": "IgnoredEnv-v0", "kwargs": {}},
        }
    )
    calls = {}

    def fake_make_env(env_id, seed, **kwargs):
        calls["make_env"] = {"env_id": env_id, "seed": seed, "kwargs": kwargs}
        return "env"

    def fake_policy_from_model(model):
        calls["policy_from_model"] = model
        return "policy"

    monkeypatch.setattr(support.sre, "make_env", fake_make_env)
    monkeypatch.setattr(support.sre, "policy_from_model", fake_policy_from_model)
    monkeypatch.setitem(support.ALGO_REGISTRY, "DUMMY", DummyAlgo)

    policy, env = support.load_policy_and_env_from_cfg(cfg, use_gui=False)

    assert (policy, env) == ("policy", "env")
    assert calls["make_env"] == {
        "env_id": "SavedEnv-v0",
        "seed": 11,
        "kwargs": {"reward": "saved", "use_gui": False},
    }
    assert DummyAlgo.last_load == {"path": str(model_path), "env": "env"}
    assert isinstance(calls["policy_from_model"], DummyModel)


def test_load_policy_and_env_from_cfg_supports_policy_id_and_class(monkeypatch):
    cfg_policy_id = make_cfg(
        {
            "run": {"seed": 3},
            "source": {
                "type": "policy_id",
                "policy_id": "safe_speed_v1",
                "policy_class": None,
                "model_path": None,
            },
            "env": {"id": "EvalEnv-v0", "kwargs": {"reward": "fast"}},
        }
    )
    cfg_policy_class = make_cfg(
        {
            "run": {"seed": 4},
            "source": {
                "type": "policy_class",
                "policy_id": None,
                "policy_class": {"_target_": "tests.fake.Policy"},
                "model_path": None,
            },
            "env": {"id": "EvalEnv-v0", "kwargs": {"reward": "fast"}},
        }
    )
    env_calls = []

    def fake_make_env(env_id, seed, **kwargs):
        env_calls.append({"env_id": env_id, "seed": seed, "kwargs": kwargs})
        return f"env-{seed}"

    monkeypatch.setattr(support.sre, "make_env", fake_make_env)
    monkeypatch.setattr(support.sre, "load_policy", lambda policy_id: f"loaded:{policy_id}")
    monkeypatch.setattr(support, "instantiate", lambda cfg: f"instantiated:{cfg._target_}")

    policy_id_result = support.load_policy_and_env_from_cfg(cfg_policy_id, use_gui=False)
    policy_class_result = support.load_policy_and_env_from_cfg(cfg_policy_class, use_gui=True)

    assert policy_id_result == ("loaded:safe_speed_v1", "env-3")
    assert policy_class_result == ("instantiated:tests.fake.Policy", "env-4")
    assert env_calls == [
        {"env_id": "EvalEnv-v0", "seed": 3, "kwargs": {"reward": "fast", "use_gui": False}},
        {"env_id": "EvalEnv-v0", "seed": 4, "kwargs": {"reward": "fast", "use_gui": True}},
    ]


def test_build_train_model_supports_fresh_and_model_sources(monkeypatch, tmp_path):
    monkeypatch.setitem(support.ALGO_REGISTRY, "DUMMY", DummyAlgo)
    model_path = tmp_path / "model.zip"
    replay_buffer_path = tmp_path / "buffer.pkl"
    cfg_fresh = make_cfg(
        {
            "source": {"type": "fresh", "model_path": None, "replay_buffer_path": None},
            "model": {"algo": "DUMMY", "kwargs": {"gamma": 0.95}, "overrides": {}},
        }
    )

    fresh_model = support.build_train_model(cfg_fresh, env="train-env")

    assert isinstance(fresh_model, DummyAlgo)
    assert DummyAlgo.last_init == {"env": "train-env", "kwargs": {"gamma": 0.95}}

    cfg_model = make_cfg(
        {
            "source": {
                "type": "model",
                "model_path": str(model_path),
                "replay_buffer_path": str(replay_buffer_path),
            },
            "model": {
                "algo": "DUMMY",
                "kwargs": {},
                "overrides": {"learning_rate": 0.001},
            },
        }
    )

    loaded_model = support.build_train_model(cfg_model, env="loaded-env")

    assert isinstance(loaded_model, DummyModel)
    assert DummyAlgo.last_load == {"path": str(model_path), "env": "loaded-env"}
    assert loaded_model.loaded_replay_buffer == str(replay_buffer_path)
    assert loaded_model.learning_rate == 0.001
    assert loaded_model.lr_schedule_reset is True


def test_save_training_outputs_saves_config_model_and_replay_buffer(tmp_path):
    cfg = make_cfg(
        {
            "save": {
                "config_path": str(tmp_path / "outputs" / "config_resolved.yaml"),
                "model_path": str(tmp_path / "outputs" / "model.zip"),
                "replay_buffer_path": str(tmp_path / "outputs" / "replay.pkl"),
            }
        }
    )
    model = DummyModel()

    support.save_training_outputs(cfg, model)

    assert Path(cfg.save.config_path).exists()
    assert model.saved_model_path == Path(cfg.save.model_path)
    assert model.saved_replay_buffer == Path(cfg.save.replay_buffer_path)
