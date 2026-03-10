"""
SUMO Gym Ego

Modular Gymnasium environment for SUMO-based reinforcement learning.
"""

# Environment
from .core.env import SumoGymEgoEnv

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


__all__ = [
    "SumoGymEgoEnv",
    "SumoConfig",
    "EgoStatus",
    "BaseEnvPlugin",
    "BaseEgoController",
    "BaseObservationBuilder",
    "BaseRewardFunction",
    "BaseMetricsTracker",
    "CompositePlugins",
    "CompositeReward",  
    "CompositeObservation",
    "CompositeMetricsTracker",
    "obs",
    "reward",
    "ego",
    "metrics",
]