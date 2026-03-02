class EnvContext:
    def __init__(self, config, sim):
        self.config = config
        self.sim = sim
        self._data = {}

    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(name)

    def inject(self, name, value):
        self._data[name] = value
