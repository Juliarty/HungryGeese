import numpy as np
import math
from kaggle_environments.envs.hungry_geese.hungry_geese import  Observation


def get_eps(start_value, end_value, step, episodes):
    return end_value + (start_value - end_value) * (episodes - step) / episodes


def get_distance(x1, x2, max_coord):
    return np.min([int(math.fabs(x1 - x2)), int(max_coord - math.fabs(x1 - x2))])


def observation_copy(observation: Observation):
    return Observation(
            {
                'index': observation.index,
                'geese': [x.copy() for x in observation.geese],
                'step': observation.step,
                'food': observation.food.copy(),
                'remainingOverageTime': observation.remaining_overage_time
            })
