import random


class SumoConfig:
    def __init__(
        self,

        # cmd args for SUMO configuration
        sumocfg_file: str,
        use_gui: bool = False,
        auto_start: bool = True,
        no_step_log: bool = True,
        no_warnings: bool = True,
        extra_sumo_args: list | None = None,

        # ego id
        ego_id: str = "ego",
        
        # simulation parameters
        max_steps: int = 1000,
        time_step: float = 0.1,
        seed: int = random.randint(0, 10_000_000),
    ):
        self.sumocfg_file = sumocfg_file
        self.use_gui = use_gui
        self.auto_start = auto_start
        self.no_step_log = no_step_log
        self.no_warnings = no_warnings
        self.extra_sumo_args = extra_sumo_args or []
        
        self.ego_id = ego_id

        self.max_steps = max_steps
        self.time_step = time_step
        self.seed = seed

    @property
    def sumo_binary(self):
        return "sumo-gui" if self.use_gui else "sumo"


    def build_cmd(self):
        cmd = [
            self.sumo_binary,
            "-c", self.sumocfg_file,
            "--step-length", str(self.time_step),
            "--seed", str(self.seed),
        ]

        if self.auto_start:
            cmd.append("--start")

        if self.no_step_log:
            cmd.append("--no-step-log")

        if self.no_warnings:
            cmd.append("--no-warnings")

        cmd.extend(self.extra_sumo_args)

        #print("Comando SUMO:", " ".join(cmd))
        return cmd
