import itertools

from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Configuration, Observation, row_col
from goose_tools import get_eps_based_on_step, get_distance

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch


class AbstractFeatureTransform:
    FEATURES_CHANNELS_IN = 0
    FEATURES_HEIGHT = 0
    FEATURES_WIDTH = 0

    def __init__(self):
        ...

    @staticmethod
    def get_state(observation, prev_observation):
        ...

    @staticmethod
    def get_reward(observation, prev_observation):
        ...

    @staticmethod
    def adjust_action(observation: Observation, prev_observation: Observation, action: Action):
        return action


class YarFeatureTransform(AbstractFeatureTransform):
    FEATURES_CHANNELS_IN = 12
    FEATURES_HEIGHT = 7
    FEATURES_WIDTH = 11
    configuration = Configuration({"rows": 7, "columns": 11, 'hunger_rate': 40, 'episodeSteps': 200})

    @staticmethod
    def get_reward(observation: Observation, prev_observation: Observation):
        return get_yar_reward(observation, prev_observation, SimpleFeatureTransform.configuration)

    @staticmethod
    def get_state(observation: Observation, prev_observation: Observation):
        n_geese = len(observation.geese)
        prev_heads = [-1] * len(observation.geese)
        if prev_observation is not None:
            for i in range(n_geese):
                prev_heads[i] = prev_observation.geese[i][0] if len(prev_observation.geese[i]) != 0 else 1

        return get_features(observation, prev_heads, SimpleFeatureTransform.configuration)


class SimpleFeatureTransform(AbstractFeatureTransform):
    FEATURES_CHANNELS_IN = 12
    FEATURES_HEIGHT = 7
    FEATURES_WIDTH = 11
    configuration = Configuration({"rows": 7, "columns": 11, 'hunger_rate': 40, 'episodeSteps': 200})

    @staticmethod
    def get_reward(observation: Observation, prev_observation: Observation):
        return get_simple_reward(observation, prev_observation, SimpleFeatureTransform.configuration)

    @staticmethod
    def get_state(observation: Observation, prev_observation: Observation):
        n_geese = len(observation.geese)
        prev_heads = [-1] * len(observation.geese)
        if prev_observation is not None:
            for i in range(n_geese):
                prev_heads[i] = prev_observation.geese[i][0] if len(prev_observation.geese[i]) != 0 else 1

        return get_features(observation, prev_heads, SimpleFeatureTransform.configuration)


def get_yar_reward(observation: Observation, prev_obs: Observation, configuration: Configuration):
    index = observation.index
    if len(observation.geese[index]) > len(prev_obs.geese[index]):
        return 1

    if len(observation.geese[index]) == 0:
        return -2

    return 0


def get_simple_reward(observation: Observation, prev_obs: Observation, configuration: Configuration):
    index = observation.index
    if len(observation.geese[index]) > len(prev_obs.geese[index]):
        return len(observation.geese[index])

    if 1 < len(observation.geese[index]) < len(prev_obs.geese[index]):
        return -1

    if len(observation.geese[index]) == 0:
        return observation.step - configuration.episode_steps

    return 0


def get_reward(observation: Observation, prev_obs: Observation, configuration: Configuration):
    reward = 0
    index = observation.index

    if len(observation.geese[index]) > len(prev_obs.geese[index]):     # ate sth
        reward = len(observation.geese[index]) * 1.5
    elif len(observation.geese[index]) == 0 and len(prev_obs.geese[index]) >= 2:  # died like a hero
        reward = -1 * get_eps_based_on_step(10, 1, observation.step, configuration.episode_steps)
    elif len(observation.geese[index]) == 0 and len(prev_obs.geese[index]) == 1:  # died like a rat
         reward = -20

    if len(observation.geese[index]) > 0:
        # check whether he lost his chance to become bigger
        head_y, head_x = row_col(observation.geese[index][0], configuration.columns)
        for food in observation.food:
            food_y, food_x = row_col(food, configuration.columns)
            dist_x = get_distance(food_x, head_x, configuration.columns)
            dist_y = get_distance(food_y, head_y, configuration.rows)
            if dist_y == 0 and dist_x == 1 or \
               dist_y == 1 and dist_x == 0:
                reward -= 2
                break

    if len(observation.geese[index]) == 0:
        # check whether he has committed suicide
        head_y, head_x = row_col(prev_obs.geese[index][0], configuration.columns)
        died_alone = True
        for i in range(len(prev_obs.geese)):
            if i == index or len(prev_obs.geese[i]) == 0:
                continue
            enemy = prev_obs.geese[i][0]
            enemy_y, enemy_x = row_col(enemy, configuration.columns)
            dist_x = get_distance(enemy_x, head_x, configuration.columns)
            dist_y = get_distance(enemy_y, head_y, configuration.rows)
            if dist_y == 0 and dist_x == 1 or \
               dist_y == 1 and dist_x == 0:
                died_alone = False
                break
        if died_alone:
            reward -= 60
    # return torch.tensor([[reward]], device=device, dtype=torch.long)
    return reward


def ids2locations(ids, prev_head, step, rows, columns):
    state = np.zeros((4, rows * columns))
    if len(ids) == 0:
        return state
    state[0, ids[0]] = 1  # goose head
    if len(ids) > 1:
        state[1, ids[1:-1]] = 1  # goose body
        state[2, ids[-1]] = 1  # goose tail
    if step != 0:
        state[3, prev_head] = 1  # goose head one step before
    return state


def get_features(observation: Observation, prev_heads, configuration: Configuration):
    rows, columns = configuration.rows, configuration.columns
    geese = observation.geese
    index = observation.index
    step = observation.step
    # convert indices to locations
    locations = np.zeros((len(geese), 4, rows * columns))
    for i, g in enumerate(geese):
        locations[i] = ids2locations(g, prev_heads[i], step, rows, columns)
    if index != 0:  # swap rows for player locations to be in first channel
        locations[[0, index]] = locations[[index, 0]]
    # put locations into features
    features = np.zeros((12, rows * columns))
    for k in range(4):
        # склеить все части тела для каждого гуся (кроме предыдущей головы)
        features[k] = np.sum(locations[k][:3], 0)
    # склеить соответствующие состояния всех гусей (либо головы всех, либо тело, либо хвосты, либо пред. головы)
    for k in range(4):
        features[k + 4] = np.sum(locations[:, k], 0)

    # только пред. голова наешего героя
    features[-5, :] = locations[index][3]

    features[-4, observation.food] = 1  # food channel
    features[-3, :] = (step % configuration.hunger_rate) / configuration.hunger_rate  # hunger danger channel
    features[-2, :] = step / configuration.episode_steps  # timesteps channel
    features[-1, :] = float((step + 1) % configuration.hunger_rate == 0)  # hunger milestone indicator
    features = torch.Tensor(features).reshape(-1, rows, columns)
    # roll
    if len(geese[index]) > 0:
        head_row, head_col = row_col(geese[index][0], columns)
    else:
        head_row, head_col = row_col(prev_heads[index], columns)

    features = torch.roll(features, ((rows // 2) - head_row, (columns // 2) - head_col), dims=(-2, -1))
    return features


def plot_features(features, title="Figure"):
    fig, axs = plt.subplots(3, 4, figsize=(20, 10))
    for i in range(3):
        for j in range(4):
            sns.heatmap(features[i * 4 + j], ax=axs[i, j], cmap='Blues',
                        vmin=0, vmax=1, linewidth=2, linecolor='black', cbar=False)
    plt.title(title)
    plt.show()


def augment(states, actions):
    action_index = 1
    state_index = 2
    # random horizontal flip
    flip_mask = np.random.rand(len(states)) < 0.5

    states[flip_mask] = states[flip_mask].flip(-1)
    actions[flip_mask] = torch.where(actions[flip_mask] > 0, 4 - actions[flip_mask], 0)

    # random vertical flip (and also diagonal)
    flip_mask = np.random.rand(len(states)) < 0.5
    states[flip_mask] = states[flip_mask].flip(-2)
    actions[flip_mask] = torch.where(actions[flip_mask] < 3, 2 - actions[flip_mask], 3)

    # shuffle opponents channels
    permuted_axs = list(itertools.permutations([0, 1, 2]))
    permutations = [torch.tensor(permuted_axs[i]) for i in np.random.randint(6, size=len(states))]
    for i, p in enumerate(permutations):
        shuffled_channels = torch.zeros(3, states.shape[2], states.shape[3])
        shuffled_channels[p] = states[i, 1:4]
        states[:, 1:4] = shuffled_channels
    return states, actions
