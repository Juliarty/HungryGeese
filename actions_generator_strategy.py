import itertools
from abc import ABC, abstractmethod
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Configuration, Observation, translate, \
    GreedyAgent

from goose_tools import get_prev_actions


class AbstractActionsGeneratorStrategy(ABC):

    @abstractmethod
    def get_actions(self, observation, prev_obs, prev_actions):
        pass


class DeepQGeneratorStrategy(AbstractActionsGeneratorStrategy):
    configuration = Configuration({"rows": 7, "columns": 11, 'hunger_rate': 40, 'episodeSteps': 200})
    greedy = GreedyAgent(configuration)

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
            if len(observation['geese'][i]) == 0:
                enemy_actions[i] = Action(1)
                continue
            # enemy_actions[i] = self._get_best_action_for_enemy(observation, prev_obs, i)
            enemy_actions[i] = self._get_greedy(observation, i)

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
        prev_action = None if prev_obs is None else get_prev_actions(observation, prev_obs)[enemy_index]
        bad_actions = [action for action in Action if
                       not action_is_not_suicide(action, observation, prev_action, enemy_index)]

        q = self.q_estimator(state).squeeze()

        for i in range(4):
            if Action(i + 1) in bad_actions:
                q[i] -= 10000

        best_action = Action(int(q.max()[1]) + 1)

        observation['index'] = index
        if prev_obs is not None:
            prev_obs['index'] = index

        return best_action

    def _get_greedy(self, observation, enemy_index):
        observation = Observation(observation)
        index = observation['index']

        observation['index'] = enemy_index
        action = self.greedy(observation)
        observation['index'] = index

        return Action({'NORTH': 1, 'EAST': 2, 'SOUTH': 3, 'WEST': 4}[action])


class WideQGeneratorStrategy(AbstractActionsGeneratorStrategy):
    def __init__(self, q_estimator, get_state):
        self.q_estimator = q_estimator
        self.get_state = get_state

    # Takes all possible actions for our hero while taking best actions for his enemies
    def get_actions(self, observation, prev_obs, prev_actions):
        result = []
        index_to_actions = {}
        for i in range(len(observation.geese)):
            if len(observation.geese[i]) == 0:
                index_to_actions[i] = [Action(1)]
                continue
            prev_action = None if prev_actions is None else prev_actions[i]
            index_to_actions[i] = [action for action in Action if
                                   action_is_not_suicide(action, observation, prev_action, i)]
            if len(index_to_actions[i]) == 0:
                index_to_actions[i] = [Action(1)]

        for combination in itertools.product(*index_to_actions.values()):
            result.append(combination)
        return result


def action_is_not_suicide(action, observation, prev_action, agent_index):
    bodies = {position for goose in observation.geese for position in goose[:-1]}
    columns = 11
    rows = 7
    position = observation.geese[agent_index][0]
    bad_action_idx_list = [
        action for action in Action
        for new_position in [translate(position, action, columns, rows)]
        if (new_position in bodies or (prev_action is not None and action == prev_action.opposite()))]

    return action not in bad_action_idx_list
