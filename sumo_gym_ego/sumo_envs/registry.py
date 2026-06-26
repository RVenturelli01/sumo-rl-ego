
ENV_REGISTRY = {}


def register_env(env_id):
    """
    Decorator to register a new environment class.
    """

    def decorator(cls):

        if env_id in ENV_REGISTRY:
            raise ValueError(f"Env '{env_id}' already registered")

        ENV_REGISTRY[env_id] = cls

        return cls

    return decorator


def list_envs():
    return sorted(ENV_REGISTRY.keys())
