class SumoConfig:
    def __init__(
        self,

        # cmd args for SUMO configuration
        sumocfg_file: str = None,
        use_gui: bool = False,
        auto_start: bool = True,
        no_step_log: bool = True,
        no_warnings: bool = True,
        extra_sumo_args: list | None = None,

        # ego id
        ego_id: str = "ego",

        # simulation parameters
        max_simulation_time: int = 200,
        time_step: float = 0.1,
        seed: int = 0,

        # mode configurations
        speed_mode: int = 0,
        lane_change_mode: int = 0,
        routing_mode: int = 0,
    ):
        self.sumocfg_file = sumocfg_file
        self.use_gui = use_gui
        self.auto_start = auto_start
        self.no_step_log = no_step_log
        self.no_warnings = no_warnings
        self.extra_sumo_args = extra_sumo_args or []

        self.ego_id = ego_id

        self.max_simulation_time = max_simulation_time
        self.max_steps = max_simulation_time*time_step
        self.time_step = time_step
        self.seed = seed

        self.speed_mode = speed_mode
        self.lane_change_mode = lane_change_mode
        self.routing_mode = routing_mode


    def build_cmd(self):
        cmd = []

        if self.use_gui:
            cmd.append("sumo-gui")
        else:
            cmd.append("sumo")

        cmd.extend([
            "-c", self.sumocfg_file,
            "--step-length", str(self.time_step),
            "--seed", str(self.seed),
        ])

        if self.auto_start:
            cmd.append("--start")

        if self.no_step_log:
            cmd.append("--no-step-log")

        if self.no_warnings:
            cmd.append("--no-warnings")

        cmd.extend(self.extra_sumo_args)

        return cmd
