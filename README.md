# SUMO RL Ego

A modular reinforcement learning framework for ego-vehicle policy training and evaluation in [SUMO](https://eclipse.dev/sumo/), built on [Gymnasium](https://gymnasium.farama.org/) and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

---

## Key Contributions

1. **Gymnasium-Compatible SUMO Wrapper** — A modern `SumoEnv` (replacing deprecated Gym) with a plugin architecture: swap ego controllers, observation builders, reward functions, and metrics without touching core code.

2. **Plugin-Based Extensibility** — Every environment component (ego controller, observation builder, reward function, metrics tracker) is a plugin resolved at runtime via Hydra's `_target_` mechanism. Swap or add components without modifying core code.

3. **Hydra-Driven Experiment Pipeline** — Fully declarative YAML configs with composable groups (`env/`, `rl/`), CLI overrides, multirun sweeps, and automatic timestamped output directories.

4. **Standardized Benchmark Scenarios** — 19 curated SUMO scenarios (highways, roundabouts, T/+ intersections) for reproducible cross-method comparisons.

5. **Unified Policy Interface** — Both RL agents (`SB3Policy`) and handcrafted heuristics (`BasePolicy`) share the same `predict(obs) → action` interface for direct comparison.

---

## Repository Structure

```
sumo_rl_ego/
├─ sumo_gym_ego/            # Core Gymnasium wrapper
│  ├─ env.py                #   SumoEnv (reset / step / render)
│  ├─ core/                 #   SumoSimulation, SumoConfig, SimBus, EgoStatus, BaseEnvPlugin
│  ├─ ego/                  #   EgoController base + default implementation
│  ├─ observation/          #   ObservationBuilder base + default
│  ├─ reward/               #   RewardFunction base + default
│  └─ metrics/              #   MetricsTracker base + default
│
├─ plugins/                 # Drop-in plugin implementations
│  ├─ egos/                 #   HighwayDiscreteEgo, HighwayContinuousEgo, HighwayOneLaneDiscreteEgo
│  ├─ observations/         #   HighwayObs
│  ├─ rewards/              #   HighwayBasic/Fast/Lane/SmoothReward + composable components
│  ├─ metrics/              #   TerminationActionMetrics
│  └─ policies/             #   Handcrafted heuristic policies (MyPolicy)
│
├─ infra/                   # Experiment infrastructure
│  ├─ builders/             #   env_factory (build_env), model_factory (build_model, load_model)
│  ├─ trainer/              #   train() loop
│  ├─ policy/               #   BasePolicy (ABC), SB3Policy wrapper
│  └─ utils/                #   CustomLogsCallback, find_repo_root
│
└─ scenarios/               # Curated SUMO .sumocfg benchmarks
   ├─ highway_fast/         #   (+ highway_fast_modified, highway_slow_3lanes, …)
   ├─ roundabout/
   ├─ t_cross_3_way/        #   (+ empty, front_car, po, rand_speed, slow, …)
   ├─ t_cross_5_way/        #   (+ no_car, same_pr)
   └─ …                     #   19 scenarios total

experiments/                # Entry-point scripts + Hydra configs
├─ train.py  finetune.py  eval.py  play.py
└─ configs/
   ├─ exp/                  #   Root configs per entry point (train, finetune, eval, play)
   ├─ env/                  #   Environment groups (highway_fast, highway_lane, highway_smooth)
   └─ rl/                   #   Algorithm groups (dqn, ppo, dqn_finetune)

outputs/                    # Auto-generated timestamped run dirs (model, monitor, TensorBoard, config)
```

---

## User Guide

### Installation

```bash
pip install -e .
```

Requires a working [SUMO installation](https://sumo.dlr.de/docs/Installing/index.html) with `SUMO_HOME` set.

### Running Experiments

All entry points live in `experiments/` and are Hydra-powered. Run from the `sumo-rl-ego/` directory. Any config value can be overridden from the CLI:

```bash
# Example: training
python experiments/train.py                         
python experiments/train.py rl=ppo env=highway_lane seed=42
python experiments/train.py rl.model.learning_rate=1e-4 rl.training.total_timesteps=100000

# Multirun sweeps
python experiments/train.py --multirun rl.model.learning_rate=1e-3,3e-4,1e-4 seed=0,1,2
```

### Configuration

Configs are composed from independent groups merged at runtime:

| Group | Purpose | Examples |
|-------|---------|----------|
| `env/` | Scenario, ego, obs, reward, metrics | `highway_fast`, `highway_smooth` |
| `rl/` | Algorithm + hyperparameters | `dqn`, `ppo` |
| `exp/` | Root config per entry point (sets group defaults) | `train`, `eval` |

Example root config:

```yaml
# exp/train.yaml
env: highway_fast
rl: dqn
name: train_${rl.algorithm}_${name}
seed: 0
```

### Extending the Framework

To add a new plugin (e.g. a custom reward):

1. Create a class inheriting from the appropriate base (`RewardFunction`, `EgoController`, `ObservationBuilder`, `MetricsTracker`, or `BasePolicy`) under `plugins/`.
2. Reference it in an env config via its Hydra `_target_` path:
   ```yaml
   reward:
     _target_: sumo_rl_ego.plugins.rewards.my_reward.MyReward
     weight_progress: 1.0
   ```
3. Run as usual — no core code changes needed.

---

## Infra API Reference

The `sumo_rl_ego.infra` module exposes reusable building blocks for custom training pipelines.

#### Builders

| Function | Description |
|----------|-------------|
| `build_env(cfg, seed) → SumoEnv` | Instantiates a fully configured SUMO environment from a Hydra config (ego, obs, reward, metrics, scenario paths). |
| `build_model(env, cfg, seed)` | Creates a new SB3 model (DQN, PPO) with hyperparameters from config. |
| `load_model(env, cfg, load_path, seed)` | Loads a saved SB3 model from disk, optionally overriding hyperparameters. |

#### Training

| Function | Description |
|----------|-------------|
| `train(model, cfg)` | Runs `model.learn()` with training params from config and a `CustomLogsCallback` for TensorBoard metric bridging. |

#### Policy Interface

| Class | Description |
|-------|-------------|
| `BasePolicy` (ABC) | Abstract interface: `predict(obs) → action` and optional `reset()`. Inherit this for handcrafted policies. |
| `SB3Policy(BasePolicy)` | Wraps any saved SB3 model into the `BasePolicy` interface (deterministic prediction). |

#### Utilities

| Symbol | Description |
|--------|-------------|
| `CustomLogsCallback` | SB3 callback that forwards custom metrics from env `info["log"]` to the training logger. |
| `find_repo_root()` | Locates the repository root by searching parent directories for marker files. |
