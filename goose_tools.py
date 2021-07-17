from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col, translate
import numpy as np
import math

configuration = Configuration({"rows": 7, "columns": 11, 'hunger_rate': 40, 'episodeSteps': 200})


def get_eps_based_on_step(start_value, end_value, step, episodes):
    return end_value + (start_value - end_value) * (episodes - step) / episodes


def get_distance(x1, x2, max_coord):
    return np.min([int(math.fabs(x1 - x2)), int(max_coord - math.fabs(x1 - x2))])


def record_game(agent, enemies, res_path="game.html"):
    env = make("hungry_geese", debug=True)
    env.reset()

    env.run([agent] + enemies)
    out = env.render(mode="html", width=500, height=400)

    with open(res_path, "w") as f:
        f.write(out)


def is_near_to_enemy(observation: Observation):
    rows = 7
    columns = 11
    index = observation.index
    if len(observation.geese[index]) == 0:
        return False    # There is no enemies in heaven
    head_y, head_x = row_col(observation.geese[index][0], columns)
    for i in range(len(observation.geese)):
        if i == index:
            continue
        for body_part in observation.geese[i]:
            body_y, body_x = row_col(body_part, columns)
            dist_x = get_distance(body_x, head_x, columns)
            dist_y = get_distance(body_y, head_y, rows)
            if dist_y == 1 or dist_x == 1:
                return True
    return False


def get_prev_actions(observation: Observation, prev_obs: Observation):
    if prev_obs is None:
        return None
    result = []
    for i in range(len(observation.geese)):
        if len(observation.geese[i]) == 0 or len(prev_obs.geese[i]) == 0:
            result.append(Action(1))
            continue
        head = observation.geese[i][0]

        prev_head = prev_obs.geese[i][0]

        for action in Action:
            if head == translate(prev_head, action, configuration.columns, configuration.rows):
                result.append(action)

    try:
        assert(len(result) == len(observation.geese))
    except:
        pass
    return result
