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
                 n_episodes,
                 epidode_condition=lambda x: True):
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
            reward = get_reward(Observation(observation), prev_obs)

            episode.append((prev_state, action_index, state, reward, done))

        if epidode_condition(episode):
            result += episode

    return result


def generate_experience(agent_factory, enemy_factory, n_episodes, episode_condition, move_condition, experience_name):
    path = "{0}/{1}_{2}.pickle".format(experience_directory, experience_name, n_episodes)
    environment = make("hungry_geese", debug=False)

    data = collect_data(environment=environment,
                        agent_factory=agent_factory,
                        enemy_factory=enemy_factory,
                        agent_eps_greedy=0.1,
                        get_state=lambda obs, prev_obs: obs,
                        get_reward=lambda obs, prev_obs: 0,
                        n_episodes=n_episodes,
                        epidode_condition=episode_condition)

    filtered_data = filter_experience(data, move_condition=move_condition)

    with open(path, 'wb') as f:
        pickle.dump(filtered_data, f)


def load_experience(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def push_data_into_replay_randomly(data_list, replay_memory: ReplayMemory, n_moves):
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


def filter_experience(experience_list, move_condition):
    result = []
    for move in experience_list:
        prev_obs, action, obs, _, _ = move
        prev_obs = Observation(prev_obs)
        obs = Observation(obs)

        is_cool = move_condition(obs, prev_obs)
        if is_cool:
            result.append(move)
            continue

    return result


def transform_data(experience_list, get_state, get_reward):
    """
    Преобразует "пятёрки" с наблюдениями и пустыми наградами в пятёрки с фичами и наградами.
    :param experience_list: 
    :param get_state: 
    :param get_reward: 
    :return: 
    """
    result = []
    for experience in experience_list:
        prev_obs, action, obs, _, is_done = experience
        prev_state = get_state(obs, prev_obs)
        state = get_state(obs, prev_obs)
        reward = get_reward(obs, prev_obs)
        result.append((prev_state, action, state, reward, is_done))

    return result


def generate_greedy_experience(n_episodes,
                               move_condition=lambda obs, prev_obs: True,
                               episode_condition=lambda x: True,
                               experience_name='greedy'):
    agent_factory = AgentFactory(None, GreedyAgent, None)
    enemy_factory = AgentFactory(None, GreedyAgent, None)

    generate_experience(agent_factory=agent_factory,
                        enemy_factory=enemy_factory,
                        n_episodes=n_episodes,
                        episode_condition=episode_condition,
                        move_condition=move_condition,
                        experience_name=experience_name)


def generate_rl_agent_experience(net_path, q_estimator_class, n_episodes, experience_name='agent'):
    net = q_estimator_class()
    net.load_state_dict(torch.load(net_path))
    agent_factory = AgentFactory(net, RLAgentWithRules, SimpleFeatureTransform.get_state)
    enemy_factory = AgentFactory(None, GreedyAgent, SimpleFeatureTransform.get_state)

    generate_experience(agent_factory=agent_factory,
                        enemy_factory=enemy_factory,
                        n_episodes=n_episodes,
                        episode_condition=lambda x: True,
                        move_condition=lambda obs, prev_obs: True,
                        experience_name=experience_name)


def prepare_simple_transform_data(n_moves=100000):
    """
        Prepare data for our goose brains.
        40% food
        40% deaths
        20% from the geese who lived long:)
    """
    path = './experience/SimpleFeatureTransform/100k_40f_40d_20l.pickle'
    result = []
    feed = load_experience("experience/greedy_feed_360k.pickle")
    deaths = load_experience('./experience/greedy_deaths_114k.pickle')
    long_episodes = load_experience('./experience/greedy_long_208k.pickle')
    result += random.choices(feed, k=int(n_moves * 0.4))
    result += random.choices(deaths, k=int(n_moves * 0.4))
    result += random.choices(long_episodes, k=int(n_moves * 0.2))

    result = transform_data(result, SimpleFeatureTransform.get_state, SimpleFeatureTransform.get_reward)

    with open(path, 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    prepare_simple_transform_data()
    # generate_greedy_experience(n_episodes=120000,
    #                            move_condition=lambda obs, prev_obs:
    #                            # len(obs['geese'][obs['index']]) > len(prev_obs['geese'][prev_obs['index']]),
    #                            len(obs['geese'][obs['index']]) == 0,
    #                            experience_name='greedy_death_without_suicide')
    # generate_greedy_experience(30000, experience_name="greedy_pure_long")
    # generate_rl_agent_experience("./results/25000_YarEstimator_c12_h7_w11_DQN.net",
    #                              GooseNet,
    #                              10000,
    #                              experience_name='yar_goose_pure_obs')



    ...
