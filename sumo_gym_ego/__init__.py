"""
SUMO Gym Ego

Modular Gymnasium environment for SUMO-based reinforcement learning.
"""

# Environment
from .core.env import SumoEnv

# Core components
from .core.config import SumoConfig
from .core.ego_status import EgoStatus

# Plugin system
from .core.base_plugins import (
    BaseEnvPlugin,
    BaseEgoController,
    BaseObservationBuilder,
    BaseRewardFunction,
    BaseMetricsTracker,
)

# Composite plugin implementations
from .core.composite_plugins import (
    CompositePlugin,
    CompositeReward,
    CompositeObservation,
    CompositeMetricsTracker,
)

# Submodules
from . import obs
from . import reward
from . import ego
from . import metrics
from . import sumo_envs

from .sumo_envs.factory import make_env, make_vec_env

__all__ = [
    "SumoEnv",
    "SumoConfig",
    "EgoStatus",
    "BaseEnvPlugin",
    "BaseEgoController",
    "BaseObservationBuilder",
    "BaseRewardFunction",
    "BaseMetricsTracker",
    "CompositePlugin",
    "CompositeReward",
    "CompositeObservation",
    "CompositeMetricsTracker",
    "obs",
    "reward",
    "ego",
    "metrics",
    "sumo_envs",
    "make_env",
    "make_vec_env",
]
