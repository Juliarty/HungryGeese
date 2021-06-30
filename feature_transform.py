from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Configuration, Observation,row_col
from feature_proc import get_square_features, get_action_rotated_north_head, get_features
from goose_tools import get_eps, get_distance


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
    def get_reward(observation, prev_observation, env_reward):
        ...

    @staticmethod
    def adjust_action(observation: Observation, prev_observation: Observation, action: Action):
        return action


class SimpleFeatureTransform(AbstractFeatureTransform):
    FEATURES_CHANNELS_IN = 12
    FEATURES_HEIGHT = 7
    FEATURES_WIDTH = 11
    configuration = Configuration({"rows": 7, "columns": 11, 'hunger_rate': 40, 'episodeSteps': 200})

    @staticmethod
    def get_reward(observation: Observation, prev_observation: Observation, env_reward):
        return get_reward(observation, prev_observation, SimpleFeatureTransform.configuration)

    @staticmethod
    def get_state(observation: Observation, prev_observation: Observation):
        n_geese = len(observation.geese)
        prev_heads = [-1] * len(observation.geese)
        if prev_observation is not None:
            for i in range(n_geese):
                prev_heads[i] = prev_observation.geese[i][0] if len(prev_observation.geese[i]) != 0 else 1

        return get_features(observation, prev_heads, SimpleFeatureTransform.configuration)


class SquareWorld7x7FeatureTransform(AbstractFeatureTransform):
    FEATURES_CHANNELS_IN = 12
    FEATURES_HEIGHT = 7
    FEATURES_WIDTH = 7
    configuration = Configuration({"rows": 7, "columns": 7, 'hunger_rate': 40, 'episodeSteps': 200})

    @staticmethod
    def get_reward(observation: Observation, prev_observation: Observation, env_reward):
        return get_reward(prev_observation, observation, SquareWorld7x7FeatureTransform.configuration)

    @staticmethod
    def get_state(observation: Observation, prev_observation: Observation):
        prev_heads = get_heads(prev_observation, len(observation.geese))
        return get_square_features(observation, prev_heads, SquareWorld7x7FeatureTransform.configuration)

    @staticmethod
    def adjust_action(observation: Observation, prev_observation: Observation, action: Action):
        prev_heads = get_heads(prev_observation, len(observation.geese))
        return get_action_rotated_north_head(observation,
                                             prev_heads,
                                             action,
                                             SquareWorld7x7FeatureTransform.configuration.rows,
                                             SquareWorld7x7FeatureTransform.configuration.columns)


def get_heads(observation, n_geese):
    prev_heads = [-1] * n_geese
    if observation is not None:
        for i in range(n_geese):
            prev_heads[i] = observation.geese[i][0] if len(observation.geese[i]) != 0 else 1

    return prev_heads


def get_reward(observation: Observation, prev_obs: Observation, configuration: Configuration):
    reward = 0
    index = observation.index

    if len(observation.geese[index]) > len(prev_obs.geese[index]):     # ate sth
        reward = len(observation.geese[index]) * 1.5
    elif len(observation.geese[index]) == 0 and len(prev_obs.geese[index]) >= 2:  # died like a hero
        reward = -1 * get_eps(10, 1, observation.step, configuration.episode_steps)
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
        for i in range(len(prev_obs.geese)):
            if i == index or len(prev_obs.geese[i]) == 0:
                continue
            enemy = prev_obs.geese[i][0]
            enemy_y, enemy_x = row_col(enemy, configuration.columns)
            dist_x = get_distance(enemy_x, head_x, configuration.columns)
            dist_y = get_distance(enemy_y, head_y, configuration.rows)
            if dist_y == 0 and dist_x == 1 or \
               dist_y == 1 and dist_x == 0:
                reward -= 50
                break
    # return torch.tensor([[reward]], device=device, dtype=torch.long)
    return reward