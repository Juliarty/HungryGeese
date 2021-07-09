from kaggle_environments import make
import numpy as np
import math


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