import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese \
    import Observation, Configuration, Action, GreedyAgent as kGreedyAgent, translate, adjacent_positions
import torch
import random

from feature_transform import AnnFeatureTransform
from goose_tools import get_prev_actions
from oracle import Oracle
from actions_generator_strategy import DeepQGeneratorStrategy

class AbstractAgent:

    def __init__(self, net, get_state, eps_greedy):
        self.n_actions = 4
        self.net = net
        if net is not None:
            self.net.eval()

        self.eps_greedy = eps_greedy
        self.get_state = get_state

    def __call__(self, observation: Observation, configuration):
        pass

    def get_action_index(self, observation: Observation, configuration: Configuration):
        action = self.__call__(observation, configuration)
        i = 0
        while i < self.n_actions:
            if action == Action(i + 1).name:
                return i
            i += 1


class AgentFactory:
    def __init__(self, initial_net,
                 abstract_agent_class,
                 get_state):
        self.abstract_agent_class = abstract_agent_class
        self.get_state = get_state
        self.net = initial_net

    def create(self, eps_greedy, updated_net=None):
        if updated_net is not None:
            self.net = updated_net

        return self.abstract_agent_class(self.net, self.get_state, eps_greedy)


class GreedyAgent(AbstractAgent):
    def __init__(self, net, get_state, eps_greedy):
        super().__init__(net=net,
                         get_state=get_state,
                         eps_greedy=eps_greedy)
        self.eps = eps_greedy

    def __call__(self, observation, configuration):
        configuration = Configuration(configuration)
        observation = Observation(observation)
        self.greedy_agent = kGreedyAgent(configuration)

        if random.random() < self.eps:
            return Action(random.randrange(self.n_actions) + 1).name

        return self.greedy_agent(observation)


class RLAgent(AbstractAgent):
    def __init__(self, net, get_state, eps_greedy):
        super().__init__(net=net,
                         get_state=get_state,
                         eps_greedy=eps_greedy)
        self.prev_observation = None
        self.eps = eps_greedy

    def _get_eps_greedy_action(self, state):
        if random.random() > self.eps:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action = int(self.net(state.unsqueeze(0)).max(1)[1].view(1)[0])
        else:
            action = random.randrange(self.n_actions)

        # в Enum нумерация начинается с 1
        return Action(action + 1)

    def __call__(self, observation,  configuration):
        self.net.eval()
        observation = Observation(observation)

        state = self.get_state(observation,  self.prev_observation)
        action = self._get_eps_greedy_action(state)
        self.prev_observation = observation

        return action.name


class RLAgentWithRules(AbstractAgent):
    def __init__(self, net, get_state, eps_greedy):
        super().__init__(net=net,
                         get_state=get_state,
                         eps_greedy=eps_greedy)
        self.prev_observation = None
        self.eps = eps_greedy
        self.prev_action = Action(1)

    def __call__(self, observation,  configuration):
        self.net.eval()
        observation = Observation(observation)
        state = self.get_state(observation,  self.prev_observation)

        if random.random() > self.eps:
            action_index = self._get_not_opposite_action(state)
        else:
            action_index = self._get_not_opposite_random_action()

        self.prev_action = Action(action_index + 1)
        return Action(action_index + 1).name

    def _get_not_opposite_action(self, state):
        q_values = self.net(state.unsqueeze(0)).squeeze()
        q_values[(self.prev_action.opposite().value - 1)] = -10000
        return int(q_values.max(0)[1])

    def _get_not_opposite_random_action(self):
        opposite_action = self.prev_action.opposite().value
        return random.choice([i for i in range(4) if i != opposite_action])


class SmartGoose(AbstractAgent):
    def __init__(self, net, get_state, eps_greedy):
        super().__init__(net=net,
                         get_state=get_state,
                         eps_greedy=eps_greedy)
        self.prev_observation = None
        self.eps = eps_greedy
        self.prev_action = Action(1)

    def __call__(self, observation,  configuration):
        self.net.eval()
        observation = Observation(observation)
        configuration = Configuration(configuration)
        action_index = self._get_not_bad_action(observation, configuration)
        self.prev_action = Action(action_index + 1)
        return Action(action_index + 1).name

    def _get_not_bad_action(self, observation: Observation, configuration: Configuration):
        """
        Choose save action (he neither commits suicide nor does kamikadze moves), but probably moves to positions
        that are adjacent to other goose heads.
        :param state:
        :return:
        """
        # Don't move into any bodies
        bodies = {position for goose in observation.geese for position in goose[:-1]}

        position = observation.geese[observation.index][0]
        bad_action_idx_list = [
            action.value - 1 for action in Action
            for new_position in [translate(position, action, configuration.columns, configuration.rows)]
            if (new_position in bodies or (action == self.prev_action.opposite()))]

        opponents = [
            goose
            for index, goose in enumerate(observation.geese)
            if index != observation.index and len(goose) > 0
        ]

        head_adjacent_positions = {
            opponent_head_adjacent
            for opponent in opponents
            for opponent_head in [opponent[0]]
            for opponent_head_adjacent in adjacent_positions(opponent_head, configuration.columns, configuration.rows)
        }

        risky_action_idx_list = [
            action.value - 1 for action in Action
            for new_position in [translate(position, action, configuration.columns, configuration.rows)]
            if new_position in head_adjacent_positions]
        state = self.get_state(observation, self.prev_observation)

        q_values = self.net(state.unsqueeze(0)).squeeze()
        q_values[bad_action_idx_list] = -10000
        q_values[risky_action_idx_list] = -5000

        return int(q_values.max(0)[1])

    def _get_not_opposite_random_action(self):
        opposite_action = self.prev_action.opposite().value
        return random.choice([i for i in range(4) if i != opposite_action])


class GooseWarlock(AbstractAgent):
    def __init__(self, net, get_state, eps_greedy):
        super().__init__(net=net,
                         get_state=get_state,

                         eps_greedy=eps_greedy)
        self.prev_observation = None
        self.prev_actions = None
        self.eps = eps_greedy
        self.oracle = Oracle(net, DeepQGeneratorStrategy(net, get_state), get_state, AnnFeatureTransform.get_reward)

    def __call__(self, observation,  configuration):
        if observation.step == 0:
            self.reset()
        self.net.eval()
        observation = Observation(observation)
        self.prev_actions = get_prev_actions(observation, self.prev_observation)
        action = self.oracle.tell_best_action(observation, self.prev_observation, self.prev_actions, depth=4)

        self.prev_observation = observation

        return action.name

    def reset(self):
        self.prev_observation = None
        self.prev_actions = None


class EnemyFactorySelector:
    def __init__(self, agents,
                 probabilities):
        self.agents = agents
        self.probabilities = probabilities

    def create(self, eps_greedy):
        return random.choices(self.agents, self.probabilities, k=1)[0].create(eps_greedy)

