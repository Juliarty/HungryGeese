from abc import ABC, abstractmethod
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Configuration, Observation, translate


class AbstractActionsGeneratorStrategy(ABC):

    @abstractmethod
    def get_actions(self, observation, prev_obs, prev_actions):
        pass


class DeepQGeneratorStrategy(AbstractActionsGeneratorStrategy):
    def __init__(self, q_estimator, get_state):
        self.q_estimator = q_estimator
        self.get_state = get_state

    # Takes all possible actions for our hero while taking best actions for his enemies
    def get_actions(self, observation, prev_obs, prev_actions):
        enemy_actions = [None] * len(observation['geese'])
        result = []
        index = observation['index']
        for i in range(len(observation['geese'])):
            if i == observation['index']:
                continue
            enemy_actions[i] = self._get_best_action_for_enemy(observation, prev_obs, i)

        for action in Action:
            if prev_actions is not None and action.value == prev_actions[index].opposite().value:
                continue
            actions = enemy_actions.copy()
            actions[index] = action
            result.append(actions)

        return result

    def _get_best_action_for_enemy(self, observation, prev_obs, enemy_index):
        if len(observation['geese'][enemy_index]) == 0:
            return Action(1)
        observation = Observation(observation)
        index = observation['index']
        observation['index'] = enemy_index
        if prev_obs is not None:
            prev_obs['index'] = enemy_index

        state = self.get_state(observation, prev_obs).unsqueeze(0)
        best_action = Action(int(self.q_estimator(state).max(1)[1].view(1)[0]) + 1)

        observation['index'] = index
        if prev_obs is not None:
            prev_obs['index'] = index

        return best_action
