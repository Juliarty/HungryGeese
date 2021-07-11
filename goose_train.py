from replay_memory import ReplayMemory, get_priority_weight
from qestimator import AlexNetQEstimator, GooseNet, OneLayerNetQEstimator, RavenNet
from goose_agents import RLAgent, GreedyAgent, AgentFactory, EnemyFactorySelector, RLAgentWithRules
from feature_transform import SimpleFeatureTransform, augment
from goose_tools import get_eps_based_on_step, record_game
from goose_experience import collect_data, push_data_into_replay_randomly, load_experience

from tqdm import tqdm
from kaggle_environments import make, evaluate
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


class TrainGeese:
    _debug = False
    _win_rates = {'Score': [], 'Rank': []}
    _EPS_START = 0.45
    _EPS_END = 0.005
    _BATCH_SIZE = 128
    _GAMMA = 0.9

    _TARGET_UPDATE = 125
    _COLLECT_FROM_POLICY_NUM = 10
    _EVALUATE_FROM_POLICY_NUM = 3

    def __init__(self,
                 replay_memory: ReplayMemory,
                 q_estimator_class,
                 agent_factory,
                 abstract_feature_transform_class,
                 starting_net
                 ):
        """
        :param replay_memory:
        :param q_estimator_class:
        """
        self._env = make("hungry_geese", debug=TrainGeese._debug)
        self.result_net_path = ""

        self._replay_memory = replay_memory
        self._MEMORY_THRESHOLD = replay_memory.capacity

        self._get_state = abstract_feature_transform_class.get_state
        self._get_reward = abstract_feature_transform_class.get_reward
        self.feature_c = abstract_feature_transform_class.FEATURES_CHANNELS_IN
        self.feature_h = abstract_feature_transform_class.FEATURES_HEIGHT
        self.feature_w = abstract_feature_transform_class.FEATURES_WIDTH

        self._policy_net = q_estimator_class()
        self._target_net = q_estimator_class()
        self._result_net = q_estimator_class()

        if starting_net is not None:
            self._policy_net.load_state_dict(starting_net.state_dict())
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._result_net.load_state_dict(self._policy_net.state_dict())

        self._target_net.eval()
        self._result_net.eval()

        # self._optimizer = optim.RMSprop(self._policy_net.parameters())
        self._optimizer = torch.optim.Adam(self._policy_net.parameters(), 3e-4)

        self._agent_factory = agent_factory
        self._agent_factory.net = self._policy_net

        # self._enemy_factory = AgentFactory(self._policy_net, RLAgent, abstract_feature_transform_class, self._device)

        self._enemy_factory = EnemyFactorySelector(
            [AgentFactory(None, GreedyAgent, abstract_feature_transform_class.get_state),
             self._agent_factory], [0.85, 0.15])

        self._last_saved_score = 0
        self.n_episodes = 0

    def train(self, n_episodes):
        self.n_episodes = n_episodes
        self._reset_statistics()

        for episode in tqdm(range(n_episodes), desc='Training', leave=False):
            self._collect_data(get_eps_based_on_step(self._EPS_START, self._EPS_END, episode, n_episodes))
            self._optimize_model(get_eps_based_on_step(0.2, 1, episode, n_episodes))

            if episode % self._TARGET_UPDATE == 0:
                self._target_net.load_state_dict(self._policy_net.state_dict())
                self._update_result_net()

    def save_result(self):
        feature_str = "c{0}_h{1}_w{2}".format(self.feature_c, self.feature_h, self.feature_w)
        self.result_net_path = './results/{0}_{1}_{2}_DQN.net'.format(self.n_episodes,
                                                                      str(self._policy_net),
                                                                      feature_str)
        torch.save(self._target_net.state_dict(), self.result_net_path)

    def print_win_rates(self):
        t = np.arange(len(self._win_rates['Score']))
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('TimeSteps')
        ax1.set_ylabel('Score', color=color)
        ax1.plot(t, self._win_rates['Score'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Rank', color=color)  # we already handled the x-label with ax1
        ax2.plot(t, self._win_rates['Rank'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title('Performance of DQN-net vs 3 Greedy Agents')
        plt.show()

    def _optimize_model(self, beta):
        if len(self._replay_memory) < self._BATCH_SIZE:
            return
        priority_sample = self._replay_memory.get(self._BATCH_SIZE)

        idx = list(map(lambda x: x[0], priority_sample))
        transitions = list(map(lambda x: x[1], priority_sample))
        priorities = list(map(lambda x: x[2], priority_sample))

        state_batch = torch.stack(list(map(lambda x: x[0], transitions)))
        action_batch = torch.tensor(list(map(lambda x: x[1], transitions)))
        next_state_batch = torch.stack(list(map(lambda x: x[2], transitions)))
        reward_batch = torch.tensor(list(map(lambda x: x[3], transitions)), dtype=torch.float)
        is_not_done_batch = torch.tensor(list(map(lambda x: x[4] is False, transitions)), dtype=torch.bool)

        state_batch, action_batch = augment(state_batch, action_batch)

        if not is_not_done_batch.any():
            return

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        self._policy_net.train()
        state_action_values = self._policy_net(state_batch)\
            .gather(1, action_batch.unsqueeze(1))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self._BATCH_SIZE)
        next_state_values[is_not_done_batch] = self._target_net(next_state_batch[is_not_done_batch]).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self._GAMMA) + reward_batch

        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        for param in self._policy_net.parameters():
            param.grad.data *= get_priority_weight(priorities,
                                                   self._replay_memory,
                                                   beta)
            param.grad.data.clamp_(-1, 1)

        self._optimizer.step()

        with torch.no_grad():
            updated_priorities = torch.abs(state_action_values.squeeze(1) - expected_state_action_values).squeeze()
            self._replay_memory.update_priorities(idx, updated_priorities)

    def _evaluate(self, save_statistics=True):
        self._policy_net.eval()
        with torch.no_grad():
            agent = self._agent_factory.create(0, self._policy_net)
            enemies = [self._enemy_factory.create(0) for _ in range(3)]
            scores = evaluate("hungry_geese",
                              [agent] + enemies,
                              num_episodes=self._EVALUATE_FROM_POLICY_NUM, debug=TrainGeese._debug)
            evaluated_score = np.mean([r[0] for r in scores])

        if save_statistics:
            self._win_rates['Score'].append(evaluated_score)
            self._win_rates['Rank'].append(np.mean([sum(r[0] <= r_ for r_ in r if r_ is not None) for r in scores]))

        return evaluated_score

    def _reset_statistics(self):
        self._win_rates['Score'] = []
        self._win_rates['Rank'] = []

    def _update_result_net(self):
        evaluated_score = self._evaluate()
        # save agent if he outperforms our previous games
        if self._last_saved_score * 1.2 < evaluated_score:
            # check again whether it plays well
            second_evaluation_score = self._evaluate(save_statistics=False)
            if self._last_saved_score < second_evaluation_score:
                self._result_net.load_state_dict(self._policy_net.state_dict())
                self._last_saved_score = second_evaluation_score

    def _collect_data(self, eps_greedy):
        if self._replay_memory.is_there_new_experience():
            return

        data_list = collect_data(environment=self._env,
                                 agent_factory=self._agent_factory,
                                 enemy_factory=self._enemy_factory,
                                 agent_eps_greedy=eps_greedy,
                                 get_state=self._get_state,
                                 get_reward=self._get_reward,
                                 n_episodes=self._COLLECT_FROM_POLICY_NUM)
        push_data_into_replay_randomly(data_list, self._replay_memory, len(data_list) // 2)


def train(q_estimator_class, feature_transform_class, n_episodes, starting_net):
    replay_memory = prepare_replay_memory(150000)
    agent_factory = AgentFactory(None, RLAgentWithRules, feature_transform_class.get_state)
    train_geese = TrainGeese(replay_memory=replay_memory,
                             q_estimator_class=q_estimator_class,
                             agent_factory=agent_factory,
                             abstract_feature_transform_class=feature_transform_class,
                             starting_net=starting_net)

    train_geese.train(n_episodes)
    train_geese.save_result()
    train_geese.print_win_rates()


def prepare_replay_memory(capacity):
    replay_memory = ReplayMemory(capacity)
    data_list = load_experience("./experience/SimpleFeatureTransform/100k_40f_40d_20l.pickle")
    push_data_into_replay_randomly(data_list=data_list, replay_memory=replay_memory, n_moves=capacity // 5)
    return replay_memory


def record_game_against_greedy(q_estimator_class, feature_transform_class, net_path):
    net = q_estimator_class()
    net.load_state_dict(torch.load(net_path))
    agent_factory = AgentFactory(net, RLAgentWithRules, feature_transform_class.get_state)
    record_game(agent_factory.create(0), ["greedy"] * 3)


def record_game_against_himself(q_estimator_class, feature_transform_class, net_path):
    net = q_estimator_class()
    net.load_state_dict(torch.load(net_path))
    agent_factory = AgentFactory(net, RLAgentWithRules, feature_transform_class.get_state)
    record_game(agent_factory.create(0), [agent_factory.create(0) for _ in range(3)])


def record_game_against_one_greedy(q_estimator_class, feature_transform_class, net_path):
    net = q_estimator_class()
    net.load_state_dict(torch.load(net_path))
    agent_factory = AgentFactory(net, RLAgentWithRules, feature_transform_class.get_state)
    record_game(agent_factory.create(0), [agent_factory.create(0) for _ in range(2)] + ["greedy"])


if __name__ == '__main__':
    feature_transform_class = SimpleFeatureTransform
    q_estimator_class = GooseNet
    # net_path = "./results/60000_YarEstimator_c12_h7_w11_DQN.net"
    # net = q_estimator_class()
    # net.load_state_dict(torch.load(net_path))
    # # #
    # train(q_estimator_class=q_estimator_class,
    #       feature_transform_class=feature_transform_class,
    #       n_episodes=30000,
    #       starting_net=None)

    net_path = "./results/30000_YarEstimator_c12_h7_w11_DQN.net"
    record_game_against_greedy(q_estimator_class=q_estimator_class,
                                feature_transform_class=feature_transform_class,
                                net_path=net_path)
