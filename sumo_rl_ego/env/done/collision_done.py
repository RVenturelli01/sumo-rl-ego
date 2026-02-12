class CollisionDone:

    def __init__(self, max_steps=1000):
        self.max_steps = max_steps

    def check(self, sim, ego, step_count):
        if sim.check_collision(ego.id):
            return True
        if step_count >= self.max_steps:
            return True
        return False
