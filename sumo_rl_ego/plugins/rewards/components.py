import numpy as np


def progress_reward(delta_x, max_delta_x, weight):
    return weight * np.clip(delta_x / max_delta_x, 0.0, 1.0)


def speed_tracking(speed, target_speed, weight):
    error = (speed - target_speed) / target_speed
    return weight * error**2


def jerk_penalty(acc, prev_acc, weight):
    jerk = acc - prev_acc
    return weight * jerk**2


def lane_change_penalty(action, penalty):
    if action in (3, 4):
        return penalty
    return 0.0
