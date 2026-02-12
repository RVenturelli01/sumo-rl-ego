class SumoConfig:
    def __init__(
        self,
        sumocfg_file: str,
        use_gui: bool = False,
        auto_start: bool = True,
        ego_id: str = "ego",
        ego_class=None,
        obs_class=None,
        reward_class=None,
        done_class=None,
    ):
        self.sumocfg_file = sumocfg_file
        self.use_gui = use_gui
        self.auto_start = auto_start
        self.ego_id = ego_id

        # Class factories (Strategy injection)
        self.ego_class = ego_class
        self.obs_class = obs_class
        self.reward_class = reward_class
        self.done_class = done_class

    @property
    def sumo_binary(self):
        return "sumo-gui" if self.use_gui else "sumo"

    def build_cmd(self):
        cmd = [
            self.sumo_binary,
            "-c", self.sumocfg_file,
        ]

        if self.auto_start:
            cmd.append("--start")

        return cmd
