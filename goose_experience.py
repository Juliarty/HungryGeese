import pickle
import random

import torch
from tqdm import tqdm

from goose_agents import GreedyAgent, AgentFactory, RLAgentWithRules
from feature_transform import SimpleFeatureTransform
from qestimator import GooseNet
from replay_memory import ReplayMemory
from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action


experience_directory = "./experience"


def collect_data(environment,
                 agent_factory,
                 enemy_factory,
                 agent_eps_greedy,
                 get_state,
                 get_reward,
                 n_episodes):
    result = []

    configuration = Configuration(environment.configuration)

    collect_range = tqdm(range(n_episodes), desc='Training', leave=False) if n_episodes > 100 else range(n_episodes)
    for _ in collect_range:
        done = False
        agent = agent_factory.create(eps_greedy=agent_eps_greedy)
        enemies = [enemy_factory.create() for _ in range(3)]
        trainer = environment.train([None] + enemies)

        prev_obs = None

        observation = Observation(trainer.reset())
        state = get_state(observation, prev_obs)
        while not done:
            action_index = agent.get_action_index(observation, configuration)

            prev_obs = observation
            prev_state = state
            observation, reward, done, _ = trainer.step(Action(action_index + 1).name)

            state = get_state(Observation(observation), prev_obs)
            reward = get_reward(Observation(observation), prev_obs, configuration)

            result.append((prev_state, action_index, state, reward, done))

    return result


def collect_long_data(environment,
                      agent_factory,
                      enemy_factory,
                      agent_eps_greedy,
                      get_state,
                      get_reward,
                      n_episodes):
    result = []

    configuration = Configuration(environment.configuration)

    collect_range = tqdm(range(n_episodes), desc='Collecting', leave=False) if n_episodes > 100 else range(n_episodes)
    for _ in collect_range:
        done = False
        agent = agent_factory.create(eps_greedy=agent_eps_greedy)
        enemies = [enemy_factory.create(0) for _ in range(3)]
        trainer = environment.train([None] + enemies)

        prev_obs = None

        observation = Observation(trainer.reset())
        state = get_state(observation, prev_obs)
        episode = []
        while not done:
            action_index = agent.get_action_index(observation, configuration)

            prev_obs = observation
            prev_state = state
            observation, reward, done, _ = trainer.step(Action(action_index + 1).name)

            state = get_state(Observation(observation), prev_obs)
            reward = get_reward(Observation(observation), prev_obs, configuration)

            episode.append((prev_state, action_index, state, reward, done))

        if len(episode) > 45:
            result += episode

    return result


def generate_greedy_experience(n_episodes, experience_name='greedy'):
    path = "{0}/{1}_{2}.pickle".format(experience_directory, experience_name, n_episodes)
    environment = make("hungry_geese", debug=False)

    agent_factory = AgentFactory(None, GreedyAgent, SimpleFeatureTransform.get_state)
    enemy_factory = AgentFactory(None, GreedyAgent, SimpleFeatureTransform.get_state)
    data = collect_long_data(environment=environment,
                        agent_factory=agent_factory,
                        enemy_factory=enemy_factory,
                        agent_eps_greedy=0.1,
                        get_state=SimpleFeatureTransform.get_state,
                        get_reward=SimpleFeatureTransform.get_reward,
                        n_episodes=n_episodes)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def generate_rl_agent_experience(net_path, q_estimator_class, n_episodes, experience_name='agent'):
    path = "{0}/{1}_{2}.pickle".format(experience_directory, experience_name, n_episodes)
    environment = make("hungry_geese", debug=False)
    net = q_estimator_class()
    net.load_state_dict(torch.load(net_path))

    agent_factory = AgentFactory(net, RLAgentWithRules, SimpleFeatureTransform.get_state)
    enemy_factory = AgentFactory(None, GreedyAgent, SimpleFeatureTransform.get_state)
    data = collect_long_data(environment=environment,
                             agent_factory=agent_factory,
                             enemy_factory=enemy_factory,
                             agent_eps_greedy=0.1,
                             get_state=SimpleFeatureTransform.get_state,
                             get_reward=SimpleFeatureTransform.get_reward,
                             n_episodes=n_episodes)

    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_experience(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def push_data_into_replay(data_list, replay_memory: ReplayMemory, n_moves):
    """
    Помещает случайно взятую выборку данных из data_list в replay_memory.
    :param data_list:
    :param replay_memory:
    :param n_moves: Количество ходов для занесения в replay_memory
    """
    if len(data_list) < n_moves:
        print("Data length is not enough.")
        n_moves = len(data_list)

    i = 0
    while i < n_moves:
        index = int(random.random() * len(data_list))
        replay_memory.push(ReplayMemory.MAXIMAL_PRIORITY, data_list[index])
        i += 1


if __name__ == '__main__':
    # generate_greedy_experience(30000, experience_name="greedy_long")
    # generate_rl_agent_experience("./results/25000_YarEstimator_c12_h7_w11_DQN.net",
    #                              GooseNet,
    #                              15000,
    #                              experience_name='yar_goose')
    t = load_experience("./experience/yar_goose_15000.pickle")
    ...