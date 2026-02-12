class SumoConfig:
    def __init__(
        self,
        net_file,
        route_file,
        use_gui=False,
        auto_start=True,
        step_length=0.1,
        max_steps=1000,
        ego_id="ego",
        target_speed=15.0,
        sumo_seed=42,
        delay=0,
    ):

        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.auto_start = auto_start
        self.step_length = step_length
        self.max_steps = max_steps
        self.ego_id = ego_id
        self.target_speed = target_speed
        self.sumo_seed = sumo_seed
        self.delay = delay

    @property
    def sumo_binary(self):
        return "sumo-gui" if self.use_gui else "sumo"

    def build_cmd(self):
        cmd = [
            self.sumo_binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "--step-length", str(self.step_length),
            "--seed", str(self.sumo_seed),
        ]

        if self.auto_start:
            cmd.append("--start")

        if self.delay > 0:
            cmd += ["--delay", str(self.delay)]

        return cmd
