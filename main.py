import math

from replay_memory import Transition, ReplayMemory
from dqn import DQN
from goose_agents import RLAgent, GreedyAgent
from feature_proc import get_square_features_params, get_square_features, get_reward, get_action_rotated_north_head, plot_features, get_eps, get_features_params, get_features

import itertools
from random import shuffle
from tqdm import tqdm
from kaggle_environments import make, evaluate
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Observation, Configuration
import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim


env = make("hungry_geese", debug=False)

# if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

BATCH_SIZE = 64
GAMMA = 0.9

TARGET_UPDATE = 150
MEMORY_THRESHOLD = 10000
MEMORY_GREEDY_EXAMPLES = MEMORY_THRESHOLD // 2000
N_EPISODES_TRAIN = 50000
N_ACTIONS = 4
COLLECT_FROM_POLICY_NUM = 3
EVALUATE_FROM_POLICY_NUM = 3

EPS_START = 0.85
EPS_END = 0.05

policy_net = DQN(*get_features_params(env.configuration),
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 outputs=N_ACTIONS,
                 device=device).to(device)

target_net = DQN(*get_features_params(env.configuration),
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 outputs=N_ACTIONS,
                 device=device).to(device)

result_net = DQN(*get_features_params(env.configuration),
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 outputs=N_ACTIONS,
                 device=device).to(device)


target_net.load_state_dict(policy_net.state_dict())
result_net.load_state_dict(policy_net.state_dict())

target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
# optimizer = torch.optim.Adam(policy_net.parameters(), 3e-4)
memory = ReplayMemory(MEMORY_THRESHOLD)


def optimize_model(step, max_step):
    if len(memory) < BATCH_SIZE:
        return
    priority_sample = memory.get(BATCH_SIZE)
    idx = list(map(lambda x: x[0], priority_sample))
    transitions = list(map(lambda x: x[1], priority_sample))
    priorities = list(map(lambda x: x[2], priority_sample))
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)

    non_final_next_states_list = [s for s in batch.next_state if s is not None]
    # torch.stack в данном случае не будет работать
    if len(non_final_next_states_list) == 0:
        return

    non_final_next_states = torch.stack(non_final_next_states_list)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(torch.stack(batch.state, dim=0, out=None)).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.squeeze(1)

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        max_annealing_param = annealing_parameter(len(memory), memory.memory.min_priority, memory.memory.total(), step, max_step)
        weights_c = list(map(lambda x: math.pow(max_annealing_param, -1) * annealing_parameter(len(memory), x, memory.memory.total(), step, max_step), priorities))
        param.grad.data *= np.average(weights_c)
        param.grad.data.clamp_(-1, 1)

    optimizer.step()

    with torch.no_grad():
        update_priorities(idx, torch.abs(state_action_values.squeeze(1) - expected_state_action_values).squeeze())


def annealing_parameter(N, priority, priority_sum, step, max_step):
    beta = get_eps(0.2, 1, step, max_step)
    prob = priority / priority_sum

    omega = math.pow(1 / N / prob, beta)

    return omega


def collect_data(player, players, k, get_state):
    shuffle(players)
    trainer = env.train([None] + players)
    configuration = Configuration(env.configuration)

    for ii in range(k):
        observation = Observation(trainer.reset())
        prev_obs = observation
        done = False
        prev_heads = [-1 for _ in range(4)]
        prev_state = get_state(observation, prev_heads, configuration)
        # start rollout
        while not done:
            # cache previous state
            # plot_features(prev_state)
            for i, g in enumerate(observation['geese']):
                if len(g) > 0:
                    prev_heads[i] = prev_obs['geese'][i][0]

            # make a move
            action_index = player.get_action_index(prev_obs, configuration)
            action_name = Action(action_index + 1).name

            # observe
            observation, reward, done, _ = trainer.step(action_name)
            observation = Observation(observation)
            if not done:
                state = get_state(observation, prev_heads, configuration)
            else:
                state = None

            reward = get_reward(prev_obs, observation, configuration, device)
            action = torch.tensor([[action_index]], device=device, dtype=torch.long)
            priority = calculate_priority(reward, prev_state, state, action_index, target_net, policy_net, GAMMA)
            memory.push(priority, prev_state, action, state, reward)

            prev_obs = observation
            prev_state = state


def update_priorities(idx_list, priorities):
    for i in range(len(idx_list)):
        memory.update(idx_list[i], priorities[i])


def calculate_priority(reward, prev_state, state, action, target, policy, gamma):
    # R_j + gamma_j * Q_target(S_j, argmax_a(S_j, a)) - Q(S_{j - 1}, A_{j - 1})
    with torch.no_grad():
        q1 = policy(prev_state.unsqueeze(0)).squeeze(0)[action]
        q2 = reward + gamma * np.argmax(target(state.unsqueeze(0)).squeeze(0)) if state is not None else reward
        return np.abs(q2 - q1)


def collect_greedy_data(k, get_state, eps=0.25):
    player = GreedyAgent(None, get_state, device, eps)
    trainer = env.train([None] + ['greedy'] * 3)
    configuration = Configuration(env.configuration)
    for ii in tqdm(range(k), desc='Episode', leave=False):
        observation = Observation(trainer.reset())
        prev_obs = observation
        done = False
        prev_heads = [-1 for _ in range(4)]
        prev_state = get_state(observation, prev_heads, configuration)

        # start rollout
        while not done:
            # cache previous state
            # plot_features(prev_state)
            for i, g in enumerate(observation['geese']):
                if len(g) > 0:
                    prev_heads[i] = prev_obs['geese'][i][0]

            # make a move
            action_index = player.get_eps_greedy(prev_obs, configuration)
            action_name = Action(action_index + 1).name

            # observe
            observation, reward, done, _ = trainer.step(action_name)
            observation = Observation(observation)
            if not done:
                state = get_state(observation, prev_heads, configuration)
            else:
                state = None

            reward = get_reward(prev_obs, observation, Configuration(configuration), device)
            action = torch.tensor([[action_index]], device=device, dtype=torch.long)

            max_priority = 100
            memory.push(max_priority, prev_state, action, state, reward)

            prev_obs = observation
            prev_state = state


def train(n_episodes=N_EPISODES_TRAIN):
    # we load into result_net the best parameters we have seen (we evaluate policy_net during training)
    # here we keep last saved evaluation result of the net we saved
    last_saved_score = 0
    win_rates = {'Score': [], 'Rank': []}
    get_state = get_features
    adjust_action = None
    print('\n-Collect greedy examples')
    collect_greedy_data(MEMORY_GREEDY_EXAMPLES, get_state, eps=0.35)

    print('\n-Start Training')
    for episode in tqdm(range(n_episodes), desc='Episode', leave=False):
        # update statistics
        target_net.eval()
        player = RLAgent(policy_net, get_state, device, get_eps(EPS_START, EPS_END, episode, n_episodes), adjust_action)
        players = ['greedy', 'greedy', RLAgent(policy_net, get_state, device, 0.1, adjust_action)]
        scores = evaluate("hungry_geese", [player] + players, num_episodes=EVALUATE_FROM_POLICY_NUM, debug=False)

        evaluated_score = np.mean([r[0] for r in scores])
        win_rates['Score'].append(evaluated_score)
        win_rates['Rank'].append(np.mean([sum(r[0] <= r_ for r_ in r if r_ is not None) for r in scores]))
        # collect data
        collect_data(player, players, COLLECT_FROM_POLICY_NUM, get_state)
        # perform training
        optimize_model(episode, n_episodes)
        # Update the target network, copying all weights and biases in DQN
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # save best agent so far
        if last_saved_score < evaluated_score:
            result_net.load_state_dict(policy_net.state_dict())

    return win_rates


def print_win_rates(win_rates):
    t = np.arange(len(win_rates['Score']))
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Score', color=color)
    ax1.plot(t, win_rates['Score'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Rank', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, win_rates['Rank'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Performance of DQN-net vs 3 Greedy Agents')
    plt.show()


def main():
    res = train()
    print_win_rates(res)
    # env.run([None, "greedy", "greedy", "greedy"])
    # env.render(mode="html", width=500, height=450)
    torch.save(result_net.state_dict(), './results/5x5_DQN.net')


if __name__ == '__main__':
    main()
