from dataclasses import dataclass, field


@dataclass
class SumoConfig:
    # cmd args for SUMO configuration
    sumocfg_file: str = None
    use_gui: bool = False
    auto_start: bool = True
    no_step_log: bool = True
    no_warnings: bool = True
    extra_sumo_args: list = field(default_factory=list)

    # ego id
    ego_id: str = "ego"

    # simulation parameters
    max_simulation_time: int = 200
    time_step: float = 0.1
    seed: int = 0

    # mode configurations
    speed_mode: int = 0
    lane_change_mode: int = 0
    routing_mode: int = 0

    @property
    def max_steps(self):
        return self.max_simulation_time / self.time_step


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
